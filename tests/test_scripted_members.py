"""Ladder members without a genome (28 October 2026): the coach and the oracle rungs play from a slot."""

from types import SimpleNamespace

import numpy as np
import pytest

from arena.config import Config
from arena.genome import make_genome
from arena.ladder import Ladder, Member, Pair
from arena.pool import load_pool
from arena.sim import SCRIPTED_ID
from arena.workers import SimInfo

K = "cd" * 32


class FakeWorkers:
    def __init__(self):
        self.calls = []

    @property
    def simulator_info(self):
        return SimInfo(K, 3, 4, 99)

    def play(self, table, jobs, max_ticks=3000):
        self.calls.append((table.copy(), list(jobs)))
        return [SimpleNamespace(points_a=3, points_b=1) for _ in jobs]


def _genome_member(i, rating=1500.0):
    g = make_genome(np.full(4, i, dtype=np.float32), K, (), "pool", -1)
    return Member(g.id, "pool", rating, 0, g, label=f"g{i}")


def test_play_uses_a_slot_for_a_member_without_a_genome():
    g, o = _genome_member(1), Member("oracle-v1-rung2", "oracle", 2000.0, 0, None, label="rung 2", slot=7)
    ladder = Ladder(99, (1, 2), pool=[g, o])
    w = FakeWorkers()
    p = ladder.play(w, o, g)
    table, jobs = w.calls[-1]
    assert jobs == [(1, 7, 0), (1, 0, 7), (2, 7, 0), (2, 0, 7)] and table.shape == (1, 4)
    assert (p.a, p.b, p.n, p.wins_a, p.draws, p.wins_b) == (o.id, g.id, 4, 2, 0, 2)  # 3-1 for whoever is on the left
    p2 = ladder.play(w, o, ladder.anchor)
    table2, jobs2 = w.calls[-1]
    assert jobs2[0] == (1, 7, 99) and table2.shape == (1, 4) and p2.b == SCRIPTED_ID
    p3 = ladder.play(w, g.genome, o)
    assert w.calls[-1][1][0] == (1, 0, 7) and p3.a == g.id
    with pytest.raises(ValueError, match="neither a genome nor a slot"):
        ladder.play(w, Member("x", "oracle", 1000.0, 0), g)


def test_rate_pool_rates_scripted_members_too(monkeypatch):
    g, o = _genome_member(1), Member("oracle-v1-rung0", "oracle", 1000.0, 0, None, label="rung 0", slot=5)
    ladder = Ladder(99, (1,), pool=[g, o])
    played = []

    def play(self, workers, a, b, max_ticks=3000):
        played.append((a.id, b.id))
        strength = {SCRIPTED_ID: 0, g.id: 1, o.id: 2}
        wins = 2 if strength[a.id] > strength[b.id] else 0
        return Pair(a.id, b.id, 2, wins, 0, 2 - wins)

    monkeypatch.setattr(Ladder, "play", play)
    ladder.rate_pool(None)
    assert played == [(g.id, SCRIPTED_ID), (o.id, SCRIPTED_ID), (o.id, g.id)]
    assert ladder.anchor.rating == 1000.0 and ladder.members[2].rating > ladder.members[1].rating > 1000.0
    assert ladder.champion.id == o.id


def test_load_pool_accepts_a_named_bot():
    members = load_pool([{"bot": "oracle-v1-rung2", "label": "rung 2"}], K, 4, {SCRIPTED_ID: 99, "oracle-v1-rung2": 7})
    assert members[0].id == "oracle-v1-rung2" and members[0].kind == "oracle" and members[0].slot == 7
    assert members[0].genome is None and members[0].label == "rung 2"
    with pytest.raises(ValueError, match="no scripted bot"):
        load_pool([{"bot": "oracle-v1-rung2"}], K, 4, {SCRIPTED_ID: 99})


def test_config_accepts_bot_pool_entries():
    c = Config.from_dict({"slug": "t", "seed": 1, "ladder": {"enabled": True, "pool": [{"bot": "oracle-v1-rung2", "label": "r"}]}})
    assert c.ladder_pool[0]["bot"] == "oracle-v1-rung2"
    with pytest.raises(ValueError):
        Config.from_dict({"slug": "t", "seed": 1, "ladder": {"enabled": True, "pool": [{"label": "no path"}]}})
