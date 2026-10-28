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

Run 6 (27 October 2026) adds a pool: members present from era 0, rated once
by a round robin on the ladder seeds with the anchor fixed at 1000
(`rate_pool`). The champion is then the top-rated member, fitness may weight
the members by rating (`weights`), and a promotion may be required to hold
the anchor as well as win the head-to-head (`hold_anchor`): the night's
lesson was champions that beat each other and lost every match to the coach.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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


def joint_elo(ids: Sequence[str], pairs: Sequence["Pair"], anchor: str, start: dict[str, float] | None = None,
              rounds: int = 500, tol: float = 1e-7) -> dict[str, float]:
    """Ratings for everyone at once from blocks of matches among them, the anchor fixed at 1000.

    Each member's rating is re-fitted against the others' current ratings in turn
    (`elo_mle`, with its virtual draws) until no rating moves; the likelihood
    rises at every step, so the sweep converges when every member has played.
    """
    rating = {i: ANCHOR_RATING for i in ids}
    rating.update(start or {})
    rating[anchor] = ANCHOR_RATING
    recs: dict[str, list[tuple[str, float, int]]] = {i: [] for i in ids}
    for p in pairs:
        recs[p.a].append((p.b, p.score_a, p.n))
        recs[p.b].append((p.a, p.n - p.score_a, p.n))
    for _ in range(rounds):
        moved = 0.0
        for i in ids:
            if i == anchor or not recs[i]:
                continue
            new = elo_mle([(rating[o], s, n) for o, s, n in recs[i]])
            moved = max(moved, abs(new - rating[i]))
            rating[i] = new
        if moved < tol:
            break
    return rating


@dataclass(frozen=True)
class Member:
    id: str
    kind: str  # "scripted", "champion" or "pool"
    rating: float  # frozen when it joined
    since_era: int
    genome: Genome | None = None  # None for the scripted bot
    promoted_at_gen: int | None = None
    winrate_vs_previous: float | None = None
    label: str | None = None  # pool members: the name the page shows
    source: str | None = None  # pool members: the id the genome had where it came from
    slot: int | None = None  # members without a genome (the coach, an oracle rung): the simulator slot that plays them
    fitness: bool = True  # False: rated and faced in promotion checks, but not played by every candidate (costly rungs)


@dataclass(frozen=True)
class Pair:
    """A block of seeded matches, `a` against `b`, each seed played on both sides."""

    a: str
    b: str
    n: int
    wins_a: int
    draws: int
    wins_b: int
    margin_a: float = 0.0  # mean points for `a` minus points against, over the block

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
                 cycle_window: int = 5, pool: Sequence[Member] = (), hold_anchor: bool = False,
                 weight_scale: float = 0.0) -> None:
        self.scripted_slot = scripted_slot
        self.seeds = seeds
        self.threshold = threshold
        self.cycle_window = cycle_window
        self.hold_anchor = hold_anchor
        self.weight_scale = weight_scale
        # the anchor first, always; pool members carry provisional ratings until rate_pool
        self.members: list[Member] = [Member(SCRIPTED_ID, "scripted", ANCHOR_RATING, 0, slot=scripted_slot)] + list(pool)
        self.pool_pairs: tuple[Pair, ...] = ()
        self.era = 0
        self.matches_total = 0
        # Matches are a pure function of (seed, genomes): a challenger that failed
        # against this champion would fail again, so it is not replayed this era.
        self._failed: set[str] = set()

    @property
    def anchor(self) -> Member:
        return self.members[0]

    @property
    def champion(self) -> Member:
        """The top-rated member; the latest one when ratings tie. Without a pool
        that is the last member, since a newcomer must out-rate the champion."""
        return max(enumerate(self.members), key=lambda km: (km[1].rating, km[0]))[1]

    def member_ids(self) -> set[str]:
        return {m.id for m in self.members}

    def weights(self) -> dict[str, float]:
        """Fitness weight per member: 10 ** ((rating - top) / weight_scale), so the
        strongest counts 1 and the rest less; everyone counts 1 when the scale is 0."""
        if not self.weight_scale:
            return {m.id: 1.0 for m in self.members}
        top = max(m.rating for m in self.members)
        return {m.id: 10.0 ** ((m.rating - top) / self.weight_scale) for m in self.members}

    def play(self, workers: Workers, a: Genome | Member, b: Member, max_ticks: int = 3000) -> Pair:
        """`a` (a genome, or a member) against member `b` over every ladder seed, both sides. A member without
        a genome (the coach, an oracle rung; 28 October) plays from its slot."""
        table: list[np.ndarray] = []

        def slot_of(x: Genome | Member) -> int:
            g = x if isinstance(x, Genome) else x.genome
            if g is not None:
                table.append(g.params)
                return len(table) - 1
            if x.slot is None:
                raise ValueError(f"{x.id} has neither a genome nor a slot")
            return x.slot

        sa, sb = slot_of(a), slot_of(b)
        if not table:  # two scripted bots: the table still needs a row
            table.append(np.zeros(workers.simulator_info.genome_len, dtype=np.float32))
        jobs = [job for s in self.seeds for job in ((s, sa, sb), (s, sb, sa))]
        results = workers.play(np.stack(table).astype(np.float32), jobs, max_ticks)
        self.matches_total += len(jobs)
        wins = draws = losses = 0
        margin = 0
        for k, r in enumerate(results):
            mine, theirs = (r.points_a, r.points_b) if k % 2 == 0 else (r.points_b, r.points_a)
            margin += mine - theirs
            if mine > theirs:
                wins += 1
            elif mine == theirs:
                draws += 1
            else:
                losses += 1
        return Pair(a.id, b.id, len(jobs), wins, draws, losses, margin / len(jobs))

    def rate_pool(self, workers: Workers, max_ticks: int = 3000) -> tuple[Pair, ...]:
        """Every pool member against every other member and the anchor, on the ladder
        seeds; the ratings are the joint fit with the anchor at 1000. Returns the blocks."""
        pairs: list[Pair] = []
        for k, a in enumerate(self.members):
            if k == 0:
                continue  # the anchor plays as everyone's opponent
            for b in self.members[:k]:
                pairs.append(self.play(workers, a, b, max_ticks))
        rating = joint_elo([m.id for m in self.members], pairs, self.anchor.id)
        self.members = [m if k == 0 else replace(m, rating=rating[m.id]) for k, m in enumerate(self.members)]
        self.pool_pairs = tuple(pairs)
        return self.pool_pairs

    def add_member(self, workers: Workers, genome: Genome, kind: str = "exploiter", label: str | None = None,
                   max_ticks: int = 3000) -> tuple[Member, tuple[Pair, ...]]:
        """A newcomer (28 October: an exploiter trained outside the run) played against every member on the
        ladder seeds, rated against their frozen ratings, frozen and appended. A new era: fitness changes."""
        if genome.id in self.member_ids():
            raise ValueError(f"{genome.id[:8]} is already a member")
        vs = tuple(self.play(workers, genome, m, max_ticks) for m in self.members)
        rating = elo_mle([(m.rating, p.score_a, p.n) for m, p in zip(self.members, vs)])
        member = Member(genome.id, kind, rating, self.era + 1, genome, label=label)
        self.members.append(member)
        self.era += 1
        self._failed.clear()
        return member, vs

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
        held = not self.hold_anchor or vs[0].winrate_a >= self.threshold  # vs[0] is the anchor's block
        if rating <= champion.rating or not held:
            self._failed.add(challenger.id)
            return Challenge(gen, self.era, challenger.id, h2h, vs, rating)
        member = Member(challenger.id, "champion", rating, self.era + 1, challenger, gen, h2h.winrate_a)
        cycle = tuple((p.b, p.winrate_a) for p in vs[-self.cycle_window:])
        played_in = self.era
        self.members.append(member)
        self.era += 1
        self._failed.clear()
        return Challenge(gen, played_in, challenger.id, h2h, vs, rating, member, cycle)
