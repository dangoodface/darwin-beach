"""A process pool where each process owns one simulator.

Results come back in task order whatever the completion order, so a run does not
depend on scheduling. With `count <= 1` everything runs in this process, which is
what the tests and the reproducibility check use.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import NamedTuple

import numpy as np

from .sim import MatchResult, Simulator

Job = tuple[int, ...]  # (seed, slot a, slot b), or (seed, a, b, hold_a, hold_b) on kernel version 3


class SimInfo(NamedTuple):
    kernel_sha: str
    kernel_version: int
    genome_len: int
    scripted_slot: int | None


_sim: Simulator | None = None  # the worker process's own simulator


def make_simulator(kind: str, kernel_path: str = "") -> Simulator:
    if kind == "fake":
        from .sim import FakeSimulator

        return FakeSimulator()
    if kind == "kernel":
        from .kernel import Kernel  # Track K

        return Kernel(kernel_path)
    raise ValueError(f"unknown simulator kind {kind!r}")


def _init(kind: str, kernel_path: str) -> None:
    global _sim
    _sim = make_simulator(kind, kernel_path)


def _run_chunk(task: tuple[np.ndarray, list[Job], int]) -> list[MatchResult]:
    table, jobs, max_ticks = task
    assert _sim is not None
    _sim.load_genomes(table)
    return _sim.play_batch(jobs, max_ticks)


class Workers:
    def __init__(self, kind: str, kernel_path: str = "", count: int = 1, jobs_per_task: int = 128) -> None:
        self.kind, self.kernel_path = kind, kernel_path
        self.count, self.jobs_per_task = max(1, count), jobs_per_task
        self._pool = None
        self._local: Simulator | None = None
        if self.count > 1:
            # spawn, not fork: the simulator may hold a wasmtime store that must not be shared
            self._pool = mp.get_context("spawn").Pool(self.count, _init, (kind, kernel_path))
        else:
            self._local = make_simulator(kind, kernel_path)

    @property
    def simulator_info(self) -> SimInfo:
        """Read from a local instance."""
        sim = self._local or make_simulator(self.kind, self.kernel_path)
        return SimInfo(sim.kernel_sha, sim.kernel_version, sim.genome_len, getattr(sim, "scripted_slot", None))

    def play(self, table: np.ndarray, jobs: list[Job], max_ticks: int = 3000) -> list[MatchResult]:
        """Play `jobs` against `table` (slot i = row i); results in job order."""
        chunks = [jobs[i : i + self.jobs_per_task] for i in range(0, len(jobs), self.jobs_per_task)]
        tasks = [(table, c, max_ticks) for c in chunks]
        if self._pool is None:
            assert self._local is not None
            return [r for t in tasks for r in self._local_chunk(t)]
        return [r for part in self._pool.map(_run_chunk, tasks, chunksize=1) for r in part]

    def _local_chunk(self, task: tuple[np.ndarray, list[Job], int]) -> list[MatchResult]:
        table, jobs, max_ticks = task
        self._local.load_genomes(table)
        return self._local.play_batch(jobs, max_ticks)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "Workers":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
