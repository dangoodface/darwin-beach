"""The real simulator: the Rust kernel's wasm module, run through wasmtime.

Implements the Simulator protocol of sim.py over the exports in docs/ABI.md.
Memory crosses the boundary in bulk only: one copy for the genome table, one
for a batch of jobs, one for a batch of result blocks, never a parameter at a
time. The browser runs the same bytes, which is what makes replays exact.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path

import numpy as np
import wasmtime

from .sim import MASK64, N_DESC, MatchResult

JOB_DTYPE = np.dtype([("seed", "<u8"), ("a", "<u4"), ("b", "<u4")])
RESULT_DTYPE = np.dtype(
    [
        ("final_hash", "<u8"),
        ("ticks", "<u4"),
        ("points_a", "<u4"),
        ("points_b", "<u4"),
        ("flags", "<u4"),
        ("desc_a", "<f8", (N_DESC,)),
        ("desc_b", "<f8", (N_DESC,)),
    ]
)
assert JOB_DTYPE.itemsize == 16 and RESULT_DTYPE.itemsize == 152

FLAG_NAN, FLAG_MAX_TICKS, FLAG_BAD_JOB = 1, 2, 4
BAD_JOB = MatchResult(0, 0, 0, 0, FLAG_BAD_JOB, (0.0,) * N_DESC, (0.0,) * N_DESC, None)


ROOT = Path(__file__).resolve().parents[2]
CARGO_BUILD = ROOT / "kernel" / "target" / "wasm32-unknown-unknown" / "release" / "slime_kernel.wasm"


def find_kernel() -> Path | None:
    """The wasm to test or benchmark: $ARENA_KERNEL if set, else the cargo
    release build in this checkout, else None."""
    env = os.environ.get("ARENA_KERNEL")
    if env:
        return Path(env)
    return CARGO_BUILD if CARGO_BUILD.exists() else None


def _i64(x: int) -> int:
    """A u64 as wasmtime-py wants it for an i64 parameter: its signed twin."""
    x &= MASK64
    return x - (1 << 64) if x >= 1 << 63 else x


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= 1 << 31 else x


class Kernel:
    """One wasmtime instance of the kernel. Not shareable across processes:
    each worker builds its own (the pool uses spawn)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        wasm = self.path.read_bytes()
        self.kernel_sha = hashlib.sha256(wasm).hexdigest()
        engine = wasmtime.Engine()
        self._store = wasmtime.Store(engine)
        instance = wasmtime.Instance(self._store, wasmtime.Module(engine, wasm), [])
        exports = instance.exports(self._store)
        self._exports = exports
        self._memory = exports["memory"]
        self._play_match = exports["play_match"]
        self._play_batch = exports["play_batch"]

        def u32(name: str) -> int:
            return exports[name](self._store) & 0xFFFFFFFF

        self.kernel_version = u32("kernel_version")
        self.genome_len = u32("genome_len")
        self.scripted_slot = u32("scripted_slot")
        self.table_capacity = u32("table_capacity")
        self.batch_capacity = u32("batch_capacity")
        self.max_ticks_cap = u32("max_ticks_cap")
        self._table_ptr = u32("genome_table_ptr")
        self._jobs_ptr = u32("jobs_ptr")
        self._results_ptr = u32("results_ptr")
        self._tick_hash_ptr = u32("tick_hash_ptr")
        self._state_ptr = u32("state_ptr")
        self._loaded = 0

    # ---- memory, in bulk -------------------------------------------------

    def _base(self, offset: int, nbytes: int) -> int:
        if offset + nbytes > self._memory.data_len(self._store):
            raise IndexError("write or read past the end of kernel memory")
        return ctypes.addressof(self._memory.data_ptr(self._store).contents) + offset

    def _write(self, offset: int, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        ctypes.memmove(self._base(offset, array.nbytes), array.ctypes.data, array.nbytes)

    def _read(self, offset: int, dtype: np.dtype, count: int) -> np.ndarray:
        nbytes = dtype.itemsize * count
        return np.frombuffer(ctypes.string_at(self._base(offset, nbytes), nbytes), dtype=dtype)

    # ---- the Simulator protocol -----------------------------------------

    def load_genomes(self, table: np.ndarray) -> None:
        """Shape (n, genome_len), float32, slot i = row i; one copy for all rows."""
        table = np.ascontiguousarray(table, dtype="<f4")
        if table.ndim != 2 or table.shape[1] != self.genome_len:
            raise ValueError(f"expected (n, {self.genome_len}) float32, got {table.shape}")
        if table.shape[0] > self.table_capacity:
            raise ValueError(f"{table.shape[0]} genomes, the kernel holds {self.table_capacity}")
        self._write(self._table_ptr, table)
        self._loaded = table.shape[0]

    def _slot_ok(self, slot: int) -> bool:
        # The kernel only knows its 1,024 slots; rows past the last load would
        # be stale genomes, so they are bad jobs here, as in FakeSimulator.
        return slot == self.scripted_slot or 0 <= slot < self._loaded

    def play(
        self, seed: int, a: int, b: int, max_ticks: int = 3000, record_hashes: bool = False
    ) -> MatchResult:
        if not (self._slot_ok(a) and self._slot_ok(b)):
            return BAD_JOB
        ticks = self._play_match(self._store, _i64(seed), _i32(a), _i32(b), _i32(max_ticks), int(record_hashes))
        result = self._results(1)[0]
        if record_hashes and not result.flags & FLAG_BAD_JOB:
            hashes = self._read(self._tick_hash_ptr, np.dtype("<u8"), ticks & 0xFFFFFFFF)
            result = result._replace(tick_hashes=tuple(hashes.tolist()))
        return result

    def play_batch(self, jobs: list[tuple[int, int, int]], max_ticks: int = 3000) -> list[MatchResult]:
        """Jobs are (seed, a, b); one result per job, in job order."""
        results: list[MatchResult] = [BAD_JOB] * len(jobs)
        if not 1 <= max_ticks <= self.max_ticks_cap:
            return results
        todo = [i for i, (_, a, b) in enumerate(jobs) if self._slot_ok(a) and self._slot_ok(b)]
        while todo:
            chunk, todo = todo[: self.batch_capacity], todo[self.batch_capacity :]
            buf = np.empty(len(chunk), dtype=JOB_DTYPE)
            buf["seed"] = [jobs[i][0] & MASK64 for i in chunk]
            buf["a"] = [jobs[i][1] & 0xFFFFFFFF for i in chunk]
            buf["b"] = [jobs[i][2] & 0xFFFFFFFF for i in chunk]
            self._write(self._jobs_ptr, buf)
            done = self._play_batch(self._store, len(chunk), max_ticks) & 0xFFFFFFFF
            # Every slot was checked above, so the kernel only stops early on a
            # job it judges bad; record that block and run the rest again.
            got = self._results(min(done + 1, len(chunk)))
            for i, r in zip(chunk, got):
                results[i] = r
            todo = chunk[done + 1 :] + todo
        return results

    def _results(self, n: int) -> list[MatchResult]:
        blocks = self._read(self._results_ptr, RESULT_DTYPE, n)
        hashes = blocks["final_hash"].tolist()
        ticks = blocks["ticks"].tolist()
        points_a = blocks["points_a"].tolist()
        points_b = blocks["points_b"].tolist()
        flags = blocks["flags"].tolist()
        desc_a = blocks["desc_a"].tolist()
        desc_b = blocks["desc_b"].tolist()
        return [
            MatchResult(hashes[i], ticks[i], points_a[i], points_b[i], flags[i], tuple(desc_a[i]), tuple(desc_b[i]), None)
            for i in range(n)
        ]

    # ---- the interactive game, as the page drives it ---------------------

    def replay(self, seed: int, a: int, b: int, max_ticks: int = 3000) -> tuple[int, int]:
        """Plays a match through reset, policy_action, set_actions and step, the
        way the page replays one, and returns (ticks, final hash). It must
        agree with `play`; tests hold the two together."""
        call = {name: self._exports[name] for name in ("reset", "is_done", "policy_action", "set_actions", "step")}
        st = self._store
        call["reset"](st, _i64(seed))
        while not call["is_done"](st, max_ticks):
            action_a = call["policy_action"](st, 0, _i32(a))
            action_b = call["policy_action"](st, 1, _i32(b))
            if action_a < 0 or action_b < 0:
                raise ValueError(f"policy_action refused slot {a} or {b}")
            call["set_actions"](st, action_a, action_b)
            call["step"](st)
        ticks = int(self._read(self._state_ptr, np.dtype("<f8"), 15)[14])
        return ticks, self._exports["state_hash"](st) & MASK64
