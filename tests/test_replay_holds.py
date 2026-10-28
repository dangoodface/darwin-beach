"""Replay records carry the holds they were played with (kernel version 3), so the page can
replay held bots: always present, 1 by default, and through the store into latest.json."""

from __future__ import annotations

import json

from arena.config import Config
from arena.export import export_run
from arena.optim import Replay
from arena.run import run
from arena.store import Store, StoreObserver, store_path


def _cfg(**kw):
    base = dict(seed=5, workers=1, batch_size=16, init_size=10, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10))
    base.update(kw)
    return Config(**base)


def _latest(tmp_path, ctx):
    st = Store(store_path(ctx.run_id, tmp_path))
    out = export_run(st, ctx.run_id, tmp_path)
    st.close()
    return json.loads((out / "latest.json").read_text())


def test_replays_carry_holds_of_1_by_default(tmp_path):
    assert Replay(1, "x", "A", 0, 10, (5, 0))[6:] == (1, 1)
    ctx = run(_cfg(), [StoreObserver(tmp_path)], generations=4)
    cells = _latest(tmp_path, ctx)["cells"]
    assert cells and all((c["replay"]["hold_a"], c["replay"]["hold_b"]) == (1, 1) for c in cells)


def test_a_held_replay_reaches_latest_json(tmp_path, monkeypatch):
    import arena.evaluate

    def held(*args, **kw):  # the evaluator's replay match, as if its candidate had played at hold 9
        return Replay(*args, **kw)._replace(hold_a=9)

    monkeypatch.setattr(arena.evaluate, "Replay", held)
    ctx = run(_cfg(), [StoreObserver(tmp_path)], generations=4)
    cells = _latest(tmp_path, ctx)["cells"]
    assert cells and all((c["replay"]["hold_a"], c["replay"]["hold_b"]) == (9, 1) for c in cells)


def test_an_old_store_without_the_columns_exports_holds_of_1(tmp_path):
    ctx = run(_cfg(), [StoreObserver(tmp_path)], generations=3)
    p = store_path(ctx.run_id, tmp_path)
    st = Store(p)
    for col in ("replay_hold_a", "replay_hold_b"):
        st.db.execute(f"ALTER TABLE evaluations DROP COLUMN {col}")
    st.db.commit()
    st.close()
    cells = _latest(tmp_path, ctx)["cells"]  # reopening adds the columns back, empty
    assert cells and all((c["replay"]["hold_a"], c["replay"]["hold_b"]) == (1, 1) for c in cells)
