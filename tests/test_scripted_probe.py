import json
from datetime import datetime, timezone

import pytest

from arena.config import Config
from arena.export import ExportObserver
from arena.run import run
from arena.sim import SCRIPTED_ID, SCRIPTED_SLOT, FakeSimulator
from arena.store import StoreObserver

NOW = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)


def _cfg(**kw):
    base = dict(seed=5, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10))
    base.update(kw)
    return Config(**base)


def _fp(ctx):
    return [(k, e.genome.id, e.fitness) for k, e in ctx.archive.cells().items()]


def test_fake_plays_the_scripted_slot_on_either_side():
    import numpy as np
    from arena.genome import random_params
    sim = FakeSimulator()
    sim.load_genomes(np.stack([random_params(np.random.default_rng(1))]))
    assert sim.scripted_slot == SCRIPTED_SLOT
    a, b = sim.play(9, 0, SCRIPTED_SLOT), sim.play(9, SCRIPTED_SLOT, 0)
    assert a == sim.play(9, 0, SCRIPTED_SLOT) and a.flags == 0 and b.flags == 0
    assert a.final_hash != b.final_hash
    assert sim.play(9, SCRIPTED_SLOT, SCRIPTED_SLOT).flags == 0
    assert sim.play(9, 0, 7).flags & 4  # a slot that is neither in the table nor scripted


def test_scripted_run_reproduces_and_replays_name_the_bot(tmp_path):
    so = StoreObserver(tmp_path)
    a = run(_cfg(), [so, ExportObserver(so, 1e9, tmp_path)], generations=30, now=NOW)
    b = run(_cfg(), generations=30)
    assert _fp(a) == _fp(b)
    out = tmp_path / "out" / a.run_id
    latest = json.loads((out / "latest.json").read_text())
    assert latest["cells"] and all(c["replay"]["opponent_id"] == SCRIPTED_ID for c in latest["cells"])
    assert not (out / "genomes" / f"{SCRIPTED_ID}.bin").exists()
    # probe genomes still get files; the scripted bot has none
    names = {p.stem for p in (out / "genomes").iterdir()}
    assert {p.id for p in a.probes} <= names


def test_scripted_probe_can_be_switched_off():
    ctx = run(_cfg(scripted_probe=False), generations=3)
    assert SCRIPTED_ID not in ctx.probe_ids and len(ctx.probe_ids) == 2
    assert ctx.cfg.scripted_probe is False


def test_probe_genomes_do_not_change_when_the_scripted_bot_is_added():
    on, off = run(_cfg(), generations=1), run(_cfg(scripted_probe=False), generations=1)
    assert [p.id for p in on.probes] == [p.id for p in off.probes]


def test_a_simulator_without_a_scripted_bot_is_refused(monkeypatch):
    monkeypatch.setattr(FakeSimulator, "scripted_slot", None)
    with pytest.raises(ValueError, match="no scripted bot"):
        run(_cfg(), generations=1)
