"""The league (28 October 2026): soft-minimum fitness with the coach gate, newcomers from the inbox, padding."""

import numpy as np
import pytest

from arena.config import Config
from arena.evaluate import Evaluator, softmin
from arena.genome import make_genome
from arena.ladder import Ladder, Member, Pair
from arena.pool import fit_params
from arena.sim import SCRIPTED_ID

K = "ab" * 32


def _member(i, rating):
    g = make_genome(np.full(8, i, dtype=np.float32), K, (), "pool", -1)
    return Member(g.id, "pool", rating, 0, g, label=f"m{i}")


def test_softmin_is_a_smooth_minimum():
    v = np.array([3.0, -1.0, 2.0])
    assert softmin(v, 0.01) == pytest.approx(-1.0, abs=0.02)  # within temp * log(3) of the minimum
    assert -1.0 < softmin(v, 0.5) < np.mean(v)


def test_softmin_fitness_over_top_members_and_the_coach_gate():
    members = [_member(1, 1900.0), _member(2, 1700.0), _member(3, 1200.0)]
    ladder = Ladder(0, (1, 2), pool=members)
    ev = Evaluator(None, [], 0, (1, 2), (0, 1), ladder=ladder, fitness_kind="softmin", softmin_top=2,
                   softmin_temp=0.01, coach_gate=True)
    opponents = [(SCRIPTED_ID, 0)] + [(m.id, k + 1) for k, m in enumerate(members)]
    per = 4  # margins per opponent: coach +2, m1 +1, m2 -1, m3 +5
    margins = [2] * per + [1] * per + [-1] * per + [5] * per
    fit, rating = ev._against_members(opponents, margins, per)
    assert fit == pytest.approx(-1.0, abs=1e-2)  # the worse of the top two by rating; m3 is ignored
    assert rating > 1000
    fit2, _ = ev._against_members(opponents, [-3] * per + [5] * per * 3, per)
    assert fit2 == pytest.approx(-13.0)  # loses to the coach: that margin minus ten, below anything that holds
    ev.coach_gate = False
    fit3, _ = ev._against_members(opponents, [-3] * per + [5] * per * 3, per)
    assert fit3 == pytest.approx(5.0, abs=1e-2)


def test_add_member_rates_a_newcomer_against_frozen_members_and_opens_an_era(monkeypatch):
    members = [_member(1, 1800.0), _member(2, 1500.0)]
    ladder = Ladder(0, (1, 2), pool=members)

    def play(self, workers, genome, opponent, max_ticks=3000):  # the newcomer wins 3 of 4 against everyone
        return Pair(genome.id, opponent.id, 4, 3, 0, 1)

    monkeypatch.setattr(Ladder, "play", play)
    new = make_genome(np.full(8, 9, dtype=np.float32), K, (), "exploiter", 7)
    m, vs = ladder.add_member(None, new, "exploiter", "exploiter-1")
    assert m.kind == "exploiter" and m.label == "exploiter-1" and m.since_era == 1 and ladder.era == 1
    assert len(vs) == 3 and ladder.members[-1] is m
    assert 1600.0 < m.rating < 1800.0  # 3 of 4 against the coach (1000), m1 (1800) and m2 (1500) at once
    assert ladder.champion is members[0]  # m1 still out-rates it
    with pytest.raises(ValueError, match="already a member"):
        ladder.add_member(None, new)


def test_fit_params_passes_equal_lengths_and_refuses_the_rest():
    p = np.arange(4, dtype=np.float32)
    assert fit_params(p, 4) is p
    with pytest.raises(ValueError, match="expected 4"):
        fit_params(np.arange(6, dtype=np.float32), 4)
    with pytest.raises(ValueError):  # shorter than the kernel: padded only when the helper accepts it (515 into 1027)
        fit_params(p, 8)
    assert len(fit_params(np.zeros(515, dtype=np.float32), 1027)) == 1027


def test_league_config_round_trip():
    c = Config.from_dict({"slug": "t", "seed": 1, "ladder": {"enabled": True, "fitness": "softmin", "softmin_top": 6,
                                                            "softmin_temp": 0.25, "coach_gate": True, "max_ticks": 6000,
                                                            "inbox_every": 20}})
    assert (c.ladder_fitness, c.ladder_softmin_top, c.ladder_softmin_temp, c.ladder_coach_gate,
            c.ladder_max_ticks, c.inbox_every) == ("softmin", 6, 0.25, True, 6000, 20)
    assert Config.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError, match="weighted or softmin"):
        Config.from_dict({"slug": "t", "seed": 1, "ladder": {"fitness": "hardmin"}})
