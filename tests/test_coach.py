import json
from datetime import datetime, timezone

from arena.config import Config
from arena.export import coach_curve, export_run
from arena.run import run
from arena.store import Store, StoreObserver, store_path

NOW = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)


def _cfg(**kw):
    base = dict(seed=5, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10))
    base.update(kw)
    return Config(**base)


def _export(tmp_path, cfg, gens=60):
    ctx = run(cfg, [StoreObserver(tmp_path)], generations=gens, now=NOW)
    st = Store(store_path(ctx.run_id, tmp_path))
    export_run(st, ctx.run_id, tmp_path, now=NOW)
    out = tmp_path / "out" / ctx.run_id
    return ctx, st, json.loads((out / "latest.json").read_text()), json.loads((out / "curve.json").read_text())


def test_beats_coach_moves_and_matches_the_final_cells(tmp_path):
    ctx, _, latest, curve = _export(tmp_path, _cfg())
    series = [p["beats_coach"] for p in curve["points"]]
    assert len(series) == 60 and all(isinstance(x, int) for x in series)
    assert series[0] != series[-1] and series[-1] > 0  # the pretend coach is beatable, so the count moves
    # the last point equals a recount over the cells the page sees
    recount = sum(1 for c in latest["cells"] if c["margins"][0] > 0)
    assert latest["stats"]["beats_coach"] == recount == series[-1]
    assert latest["probes"][0] == "scripted-v1"


def test_beats_coach_is_null_without_a_scripted_probe(tmp_path):
    _, _, latest, curve = _export(tmp_path, _cfg(scripted_probe=False), gens=10)
    assert latest["stats"]["beats_coach"] is None
    assert all(p["beats_coach"] is None for p in curve["points"])


def test_beats_coach_is_null_for_a_store_without_margins(tmp_path):
    ctx = run(_cfg(), [StoreObserver(tmp_path)], generations=8, now=NOW)
    path = store_path(ctx.run_id, tmp_path)
    old = Store(path)
    old.db.execute("ALTER TABLE evaluations DROP COLUMN margins_json")
    old.db.commit()
    old.close()
    st = Store(path)
    export_run(st, ctx.run_id, tmp_path, now=NOW)
    out = tmp_path / "out" / ctx.run_id
    assert json.loads((out / "latest.json").read_text())["stats"]["beats_coach"] is None
    assert all(p["beats_coach"] is None for p in json.loads((out / "curve.json").read_text())["points"])


def test_coach_curve_swaps_a_cells_verdict_when_it_is_replaced():
    by_id = {"a": {"margins": [1.0]}, "b": {"margins": [-1.0]}, "c": {"margins": [2.0]}}
    events = [(0, (0, 0), "a", 0.0), (0, (0, 1), "c", 0.0), (1, (0, 0), "b", 1.0)]  # b replaces a, and loses the coach
    assert coach_curve(events, by_id, ["scripted-v1", "p"]) == {0: 2, 1: 1}
    assert coach_curve(events, by_id, ["p"]) is None
