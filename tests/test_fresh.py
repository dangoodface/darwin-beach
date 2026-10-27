"""The fresh-serve check: eight seeds no selection ever sees, played at each export."""
import json
from datetime import datetime, timezone

import numpy as np

from arena.config import Config
from arena.export import ExportObserver, export_run
from arena.run import STREAM_FRESH, run
from arena.sim import FakeSimulator
from arena.store import Store, StoreObserver, store_path

NOW = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)


def _cfg(**kw):
    base = dict(seed=5, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10))
    base.update(kw)
    return Config(**base)


def _run(tmp_path, gens=40, export=True):
    so = StoreObserver(tmp_path)
    obs = [so, ExportObserver(so, 0.0, tmp_path)] if export else [so]
    ctx = run(_cfg(), obs, generations=gens, now=NOW)
    out = tmp_path / "out" / ctx.run_id
    return ctx, out


def test_fresh_stream_is_appended_and_its_seeds_are_never_graded(tmp_path):
    assert STREAM_FRESH == 4
    old = np.random.SeedSequence(5).spawn(4)
    new = np.random.SeedSequence(5).spawn(5)
    for a, b in zip(old, new):  # the first four streams are untouched by appending the fifth
        assert np.array_equal(a.generate_state(4), b.generate_state(4))
    ctx, _ = _run(tmp_path, gens=3)
    assert len(ctx.fresh.seeds) == 8 and len(set(ctx.fresh.seeds)) == 8
    assert not set(ctx.fresh.seeds) & set(ctx.cfg.eval_seeds)


def test_the_check_does_not_change_the_run(tmp_path):
    a, _ = _run(tmp_path / "a", gens=30, export=True)
    b, _ = _run(tmp_path / "b", gens=30, export=False)
    assert a.archive.qd_score() == b.archive.qd_score() and a.archive.filled() == b.archive.filled()


def test_stats_cells_and_curve(tmp_path):
    ctx, out = _run(tmp_path)
    latest = json.loads((out / "latest.json").read_text())
    curve = json.loads((out / "curve.json").read_text())["points"]
    fr = latest["stats"]["fresh"]
    assert fr["seeds"] == 8 and fr["matches"] == 16
    checked = [c for c in latest["cells"] if c["fresh"] is not None]
    assert fr["checked"] == len(checked) > 0
    # checked = the best elite plus every elite with a positive margin against the scripted bot
    want = {c["genome_id"] for c in latest["cells"] if c["margins"][0] > 0} | {latest["stats"]["best_id"]}
    assert {c["genome_id"] for c in checked} == want
    assert all(0 <= c["fresh"]["won"] <= 16 and -5 <= c["fresh"]["margin"] <= 5 for c in checked)
    assert fr["beats_coach"] == sum(1 for c in checked if c["fresh"]["won"] > 8)
    best = next(c for c in latest["cells"] if c["genome_id"] == latest["stats"]["best_id"])
    assert fr["champion"] == best["fresh"]
    # the curve: null before the first export, then integers, ending at the stat
    series = [p["beats_coach_fresh"] for p in curve]
    assert series[-1] == fr["beats_coach"] and all(v is None or isinstance(v, int) for v in series)
    assert series[0] is not None  # exporting on every generation, so the first point has one


def test_a_checked_elite_matches_a_replay_on_fresh_seeds(tmp_path):
    ctx, out = _run(tmp_path, gens=20)
    latest = json.loads((out / "latest.json").read_text())
    cell = next(c for c in latest["cells"] if c["fresh"] is not None)
    params = np.frombuffer((out / "genomes" / f"{cell['genome_id']}.bin").read_bytes(), dtype=np.float32)
    sim = FakeSimulator()
    sim.load_genomes(params[None, :])
    diffs = []
    for s in ctx.fresh.seeds:
        r = sim.play(s, 0, sim.scripted_slot)
        diffs.append(r.points_a - r.points_b)
        r = sim.play(s, sim.scripted_slot, 0)
        diffs.append(r.points_b - r.points_a)
    assert cell["fresh"] == {"won": sum(d > 0 for d in diffs), "margin": float(np.mean(diffs))}


def test_without_a_checker_the_fields_are_null(tmp_path):
    ctx, _ = _run(tmp_path, gens=10, export=False)
    st = Store(store_path(ctx.run_id, tmp_path))
    out = export_run(st, ctx.run_id, tmp_path, now=NOW)
    latest = json.loads((out / "latest.json").read_text())
    assert latest["stats"]["fresh"] is None and all(c["fresh"] is None for c in latest["cells"])
    assert all(p["beats_coach_fresh"] is None for p in json.loads((out / "curve.json").read_text())["points"])


def test_summary_prints_the_fresh_figures(tmp_path):
    from arena.summary import exported_stats, render, summarize

    ctx, out = _run(tmp_path, gens=20)
    st = Store(store_path(ctx.run_id, tmp_path))
    d = summarize(st, ctx.run_id, with_candidates=False)
    d["exported"] = exported_stats(ctx.run_id, tmp_path)
    text = render(d)
    assert "graded serves:" in text and "fresh serves (8 new seeds, 16 matches each)" in text
