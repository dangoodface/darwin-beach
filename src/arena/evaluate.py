"""Fitness: how a candidate does against its opponents.

Every candidate plays every opponent on every evaluation seed, on both sides.

v0: the opponents are the fixed probe set. Fitness is the mean score margin
(points for minus points against); the behaviour descriptors are the mean over
those same matches, taken from the candidate's own side.

v1 (a ladder is given): the candidate also plays the ladder's champions, after
the probe set; scripted-v1 is in both and is played once. Descriptors still
come from the probe matches only, so a cell never moves when the ladder grows.
Fitness is the mean margin over the matches against the ladder members, and
the rating is the Elo that best explains those matches against the members'
frozen ratings. With a pool (run 6), each member's mean margin is weighted by
the ladder's rating weights, so beating the strong ones counts most.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .genome import Genome
from .sim import SCRIPTED_ID
from .optim import EvalResult, Replay
from .workers import Job, Workers

if TYPE_CHECKING:
    from .ladder import Ladder

TABLE_CAPACITY = 1024  # kernel table slots; probes, champions and one batch of candidates must fit


class Evaluator:
    def __init__(
        self,
        workers: Workers,
        probes: list[Genome],
        scripted_slot: int | None,
        seeds: tuple[int, ...],
        indices: tuple[int, ...],
        max_ticks: int = 3000,
        ladder: Ladder | None = None,
        fitness_kind: str = "weighted",
        softmin_top: int = 8,
        softmin_temp: float = 0.5,
        coach_gate: bool = False,
    ) -> None:
        """`fitness_kind` (28 October): "weighted" is the rating-weighted mean margin over the members (plain mean
        when the ladder's weight scale is 0); "softmin" is a smooth minimum, at temperature `softmin_temp` points,
        of the mean margins against the `softmin_top` highest-rated members, so the worst matchup is what
        improves. `coach_gate`: a candidate that loses to the coach on average scores that margin minus ten."""
        self.workers, self.probes, self.seeds = workers, probes, seeds
        self.fitness_kind, self.softmin_top, self.softmin_temp, self.coach_gate = fitness_kind, softmin_top, softmin_temp, coach_gate
        # the probe set: the scripted bot first (when the simulator has one), then the frozen genomes.
        # Each opponent is (id, table slot or the scripted slot); genome probes sit in slots 0..n-1.
        self.opponents: list[tuple[str, int]] = (
            [(SCRIPTED_ID, scripted_slot)] if scripted_slot is not None else []
        ) + [(g.id, i) for i, g in enumerate(probes)]
        self.indices, self.max_ticks = indices, max_ticks
        self.ladder = ladder
        self.matches_total = 0

    def _ladder_opponents(self) -> tuple[list[tuple[str, int]], list[Genome]]:
        """Members not already in the probe set, as opponents, and the genomes
        they need in the table (placed after the probes)."""
        if self.ladder is None:
            return [], []
        have = {oid for oid, _ in self.opponents}
        extra: list[tuple[str, int]] = []
        genomes: list[Genome] = []
        for m in self.ladder.members:
            if m.id in have or not m.fitness:  # promotion-only members (oracle rungs) are not played here
                continue
            if m.genome is None:
                extra.append((m.id, m.slot))
            else:
                extra.append((m.id, len(self.probes) + len(genomes)))
                genomes.append(m.genome)
        return extra, genomes

    @property
    def matches_per_candidate(self) -> int:
        return (len(self.opponents) + len(self._ladder_opponents()[0])) * len(self.seeds) * 2

    def evaluate(self, genomes: list[Genome]) -> list[EvalResult]:
        extra, champions = self._ladder_opponents()
        opponents = self.opponents + extra
        fixed = self.probes + champions
        if len(fixed) + len(genomes) > TABLE_CAPACITY:
            raise ValueError(f"{len(fixed) + len(genomes)} genomes exceed the table capacity {TABLE_CAPACITY}")
        table = np.stack([g.params for g in fixed + genomes]).astype(np.float32)
        jobs: list[Job] = []
        for i in range(len(genomes)):
            slot = len(fixed) + i
            for _, opp in opponents:
                for s in self.seeds:
                    jobs.append((s, slot, opp))  # candidate on the left
                    jobs.append((s, opp, slot))  # candidate on the right
        results = self.workers.play(table, jobs, self.max_ticks)
        self.matches_total += len(jobs)

        per_opp = len(self.seeds) * 2  # matches per opponent: opponent-major, then seed, then side
        per = len(opponents) * per_opp
        n_probe = len(self.opponents) * per_opp  # the probe set's matches come first in each block
        out: list[EvalResult] = []
        for i, g in enumerate(genomes):
            margins, descs = [], []
            block = results[i * per : (i + 1) * per]
            first = block[0]  # seed 0, probe 0 (the scripted bot when there is one), candidate on the left
            replay = Replay(self.seeds[0], self.opponents[0][0], "A", first.final_hash, first.ticks,
                            (first.points_a, first.points_b))
            for k, r in enumerate(block):
                left = k % 2 == 0
                mine, theirs = (r.points_a, r.points_b) if left else (r.points_b, r.points_a)
                margins.append(mine - theirs)
                descs.append(r.desc_a if left else r.desc_b)
            d = np.mean(np.asarray(descs[:n_probe], dtype=np.float64), axis=0)
            by_opp = tuple(float(np.mean(margins[k * per_opp : (k + 1) * per_opp]))
                           for k in range(len(self.opponents)))
            full = tuple(float(x) for x in d)
            if self.ladder is None:
                out.append(EvalResult(g, tuple(float(d[j]) for j in self.indices), float(np.mean(margins)),
                                      replay, by_opp, desc=full))
                continue
            fitness, rating = self._against_members(opponents, margins, per_opp)
            out.append(EvalResult(g, tuple(float(d[j]) for j in self.indices), fitness, replay, by_opp,
                                  rating=rating, desc=full))
        return out

    def _against_members(self, opponents: list[tuple[str, int]], margins: list[int], per_opp: int) -> tuple[float, float]:
        from .ladder import elo_mle

        rating_of = {m.id: m.rating for m in self.ladder.members}
        weight = self.ladder.weights()
        member_margins: list[int] = []
        means: dict[str, float] = {}
        records = []
        weighted = total = 0.0
        for o, (oid, _) in enumerate(opponents):
            if oid not in rating_of:
                continue
            block = margins[o * per_opp : (o + 1) * per_opp]
            member_margins.extend(block)
            means[oid] = float(np.mean(block))
            weighted += weight[oid] * means[oid]
            total += weight[oid]
            score = sum(1.0 if m > 0 else 0.5 if m == 0 else 0.0 for m in block)
            records.append((rating_of[oid], score, len(block)))
        if self.fitness_kind == "softmin":
            top = sorted((oid for oid in means if oid != SCRIPTED_ID), key=lambda oid: -rating_of[oid])[: self.softmin_top]
            fitness = softmin(np.array([means[oid] for oid in top]), self.softmin_temp)
        elif not self.ladder.weight_scale:  # every member counts the same: the plain mean, as before run 6
            fitness = float(np.mean(member_margins))
        else:
            fitness = weighted / total
        if self.coach_gate and SCRIPTED_ID in means and means[SCRIPTED_ID] < 0:
            fitness = means[SCRIPTED_ID] - 10.0
        return fitness, elo_mle(records)


def softmin(values: np.ndarray, temp: float) -> float:
    """A smooth minimum: -temp * log(mean(exp(-values / temp))); the minimum as temp goes to 0."""
    x = -np.asarray(values, dtype=np.float64) / temp
    m = float(x.max())
    return float(-temp * (m + np.log(np.mean(np.exp(x - m)))))
