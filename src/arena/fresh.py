"""The fresh-serve check: elites against the scripted bot on serves nobody selected on.

The run draws a few seeds from a stream of its own (run.STREAM_FRESH) that no fitness,
descriptor or ladder match ever uses. At each export the best elite and every elite that
beats the scripted bot on the graded serves play those seeds again, on both sides, in a
simulator owned by the main process. A bot that only learned the graded serves loses here.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .sim import Simulator

FRESH_SEEDS = 8
CHUNK = 256  # genomes per table load; the kernel table holds 1024


class FreshResult(NamedTuple):
    won: int  # matches won out of 2 * seeds
    margin: float  # mean points for minus points against


class FreshCheck:
    def __init__(self, sim: Simulator, seeds: tuple[int, ...], max_ticks: int = 3000) -> None:
        if sim.scripted_slot is None:
            raise ValueError("the fresh check needs the simulator's scripted bot")
        self.sim, self.seeds, self.max_ticks = sim, seeds, max_ticks
        self._cache: dict[str, FreshResult] = {}  # a genome's result never changes: same seeds, same bot

    @property
    def matches(self) -> int:
        return 2 * len(self.seeds)

    def check(self, genomes: dict[str, np.ndarray]) -> dict[str, FreshResult]:
        """Results for `genomes` (id -> params), playing only the ones not seen before."""
        todo = [g for g in genomes if g not in self._cache]
        for i in range(0, len(todo), CHUNK):
            ids = todo[i : i + CHUNK]
            self.sim.load_genomes(np.stack([genomes[g] for g in ids]).astype(np.float32))
            bot = self.sim.scripted_slot
            jobs = [j for k in range(len(ids)) for s in self.seeds for j in ((s, k, bot), (s, bot, k))]
            res = self.sim.play_batch(jobs, self.max_ticks)
            per = self.matches
            for k, gid in enumerate(ids):
                diffs = []
                for m, r in enumerate(res[k * per : (k + 1) * per]):
                    left = m % 2 == 0  # jobs alternate: candidate on the left, then on the right
                    diffs.append(r.points_a - r.points_b if left else r.points_b - r.points_a)
                self._cache[gid] = FreshResult(sum(d > 0 for d in diffs), float(np.mean(diffs)))
        return {g: self._cache[g] for g in genomes}
