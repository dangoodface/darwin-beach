"""Fitness: how a candidate does against the fixed probe set.

Every candidate plays every probe on every evaluation seed, on both sides.
Fitness is the mean score margin (points for minus points against); the
behaviour descriptors are the mean over those same matches, taken from the
candidate's own side.
"""

from __future__ import annotations

import numpy as np

from .genome import Genome
from .optim import EvalResult, Replay
from .workers import Job, Workers

TABLE_CAPACITY = 1024  # kernel table slots; probes plus one batch of candidates must fit


class Evaluator:
    def __init__(
        self,
        workers: Workers,
        probes: list[Genome],
        seeds: tuple[int, ...],
        indices: tuple[int, ...],
        max_ticks: int = 3000,
    ) -> None:
        self.workers, self.probes, self.seeds = workers, probes, seeds
        self.indices, self.max_ticks = indices, max_ticks
        self.matches_total = 0

    @property
    def matches_per_candidate(self) -> int:
        return len(self.probes) * len(self.seeds) * 2

    def evaluate(self, genomes: list[Genome]) -> list[EvalResult]:
        n_probe = len(self.probes)
        if n_probe + len(genomes) > TABLE_CAPACITY:
            raise ValueError(f"{n_probe + len(genomes)} genomes exceed the table capacity {TABLE_CAPACITY}")
        table = np.stack([g.params for g in self.probes + genomes]).astype(np.float32)
        jobs: list[Job] = []
        for i in range(len(genomes)):
            slot = n_probe + i
            for p in range(n_probe):
                for s in self.seeds:
                    jobs.append((s, slot, p))  # candidate on the left
                    jobs.append((s, p, slot))  # candidate on the right
        results = self.workers.play(table, jobs, self.max_ticks)
        self.matches_total += len(jobs)

        per = self.matches_per_candidate
        out: list[EvalResult] = []
        for i, g in enumerate(genomes):
            margins, descs = [], []
            block = results[i * per : (i + 1) * per]
            first = block[0]  # seed 0, probe 0, candidate on the left
            replay = Replay(self.seeds[0], self.probes[0].id, "A", first.final_hash, first.ticks,
                            (first.points_a, first.points_b))
            for k, r in enumerate(block):
                left = k % 2 == 0
                mine, theirs = (r.points_a, r.points_b) if left else (r.points_b, r.points_a)
                margins.append(mine - theirs)
                descs.append(r.desc_a if left else r.desc_b)
            d = np.mean(np.asarray(descs, dtype=np.float64), axis=0)
            out.append(EvalResult(g, tuple(float(d[j]) for j in self.indices), float(np.mean(margins)), replay))
        return out
