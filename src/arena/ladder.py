"""The honest ladder (v1): frozen ratings, eras and promotion.

scripted-v1 anchors the scale at 1000 and is the champion of era 0. Every few
generations the best live elite challenges the champion over seeded matches on
both sides. It is promoted when it wins at least `threshold` of them (a win is
more points in a match, a tie counts half) and its rating against every member
beats the champion's. Its rating is then frozen and it joins the ladder; a new
era begins, and fitness, measured against the members, is re-scored.

Because a member's rating never changes and a newcomer must out-rate the
champion, the champion's rating only rises: a climbing curve cannot come from
a shifting scale or from beating a weak ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .genome import Genome
from .sim import SCRIPTED_ID
from .workers import Workers

ANCHOR_RATING = 1000.0
ELO_SCALE = 400.0


def expected_score(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / ELO_SCALE))


def elo_mle(records: Sequence[tuple[float, float, int]]) -> float:
    """The rating that best explains a record against fixed-rated opponents.

    `records` holds (opponent rating, score, games), where the score counts a
    win as 1 and a tie as 1/2. Maximum likelihood sets the expected total score
    equal to the actual total; one virtual draw against each opponent keeps a
    perfect or a pointless record finite. The expected total rises with the
    rating, so bisection finds the root.
    """
    if not records:
        raise ValueError("a rating needs at least one opponent")
    target = sum(s + 0.5 for _, s, _ in records)
    lo = min(r for r, _, _ in records) - 4000.0
    hi = max(r for r, _, _ in records) + 4000.0
    while hi - lo > 1e-9:
        mid = 0.5 * (lo + hi)
        if sum((n + 1) * expected_score(mid, r) for r, _, n in records) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Member:
    id: str
    kind: str  # "scripted" or "champion"
    rating: float  # frozen when it joined
    since_era: int
    genome: Genome | None = None  # None for the scripted bot
    promoted_at_gen: int | None = None
    winrate_vs_previous: float | None = None


@dataclass(frozen=True)
class Pair:
    """A block of seeded matches, `a` against `b`, each seed played on both sides."""

    a: str
    b: str
    n: int
    wins_a: int
    draws: int
    wins_b: int

    @property
    def score_a(self) -> float:
        return self.wins_a + 0.5 * self.draws

    @property
    def winrate_a(self) -> float:
        return self.score_a / self.n


@dataclass(frozen=True)
class Challenge:
    """One promotion check: the best live elite against the champion."""

    gen: int
    era: int  # the era it was played in
    challenger_id: str
    headtohead: Pair  # challenger (a) against the champion (b)
    vs_members: tuple[Pair, ...] = ()  # against every member, only when the head-to-head passed
    rating: float | None = None  # against every member, only when the head-to-head passed
    member: Member | None = None  # the new champion, when promoted
    cycle: tuple[tuple[str, float], ...] = ()  # its win rate against the last few members, when promoted

    @property
    def promoted(self) -> bool:
        return self.member is not None


class Ladder:
    def __init__(self, scripted_slot: int, seeds: tuple[int, ...], threshold: float = 0.55,
                 cycle_window: int = 5) -> None:
        self.scripted_slot = scripted_slot
        self.seeds = seeds
        self.threshold = threshold
        self.cycle_window = cycle_window
        self.members: list[Member] = [Member(SCRIPTED_ID, "scripted", ANCHOR_RATING, 0)]
        self.era = 0
        self.matches_total = 0
        # Matches are a pure function of (seed, genomes): a challenger that failed
        # against this champion would fail again, so it is not replayed this era.
        self._failed: set[str] = set()

    @property
    def champion(self) -> Member:
        return self.members[-1]

    def member_ids(self) -> set[str]:
        return {m.id for m in self.members}

    def play(self, workers: Workers, genome: Genome, opponent: Member, max_ticks: int = 3000) -> Pair:
        """`genome` against a member over every ladder seed, both sides."""
        table = [genome.params] + ([opponent.genome.params] if opponent.genome is not None else [])
        opp = 1 if opponent.genome is not None else self.scripted_slot
        jobs = [job for s in self.seeds for job in ((s, 0, opp), (s, opp, 0))]
        results = workers.play(np.stack(table).astype(np.float32), jobs, max_ticks)
        self.matches_total += len(jobs)
        wins = draws = losses = 0
        for k, r in enumerate(results):
            mine, theirs = (r.points_a, r.points_b) if k % 2 == 0 else (r.points_b, r.points_a)
            if mine > theirs:
                wins += 1
            elif mine == theirs:
                draws += 1
            else:
                losses += 1
        return Pair(genome.id, opponent.id, len(jobs), wins, draws, losses)

    def challenge(self, workers: Workers, challenger: Genome, gen: int, max_ticks: int = 3000) -> Challenge | None:
        """Plays the promotion check; on success the challenger joins and the era advances.

        Returns None when there is nothing new to check: the challenger is
        already a member or already failed against this champion.
        """
        if challenger.id in self.member_ids() or challenger.id in self._failed:
            return None
        champion = self.champion
        h2h = self.play(workers, challenger, champion, max_ticks)
        if h2h.winrate_a < self.threshold:
            self._failed.add(challenger.id)
            return Challenge(gen, self.era, challenger.id, h2h)
        vs = tuple(h2h if m is champion else self.play(workers, challenger, m, max_ticks) for m in self.members)
        rating = elo_mle([(m.rating, p.score_a, p.n) for m, p in zip(self.members, vs)])
        if rating <= champion.rating:
            self._failed.add(challenger.id)
            return Challenge(gen, self.era, challenger.id, h2h, vs, rating)
        member = Member(challenger.id, "champion", rating, self.era + 1, challenger, gen, h2h.winrate_a)
        cycle = tuple((p.b, p.winrate_a) for p in vs[-self.cycle_window:])
        played_in = self.era
        self.members.append(member)
        self.era += 1
        self._failed.clear()
        return Challenge(gen, played_in, challenger.id, h2h, vs, rating, member, cycle)
