import json
from datetime import datetime, timezone

import numpy as np

from arena.config import Config
from arena.export import export_run
from arena.run import run
from arena.store import Store, StoreObserver, store_path

NOW = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)


def _cfg(**kw):
    base = dict(seed=5, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10))
    base.update(kw)
    return Config(**base)


class Rec:
    def __init__(self): self.results = []
    def on_start(self, ctx): pass
    def on_end(self, ctx, info): pass
    def on_generation(self, ctx, info): self.results += info.results


def test_margins_average_to_fitness_in_probe_order():
    rec = Rec()
    ctx = run(_cfg(), [rec], generations=10)
    assert len(ctx.probe_ids) == 3
    for r in rec.results:
        assert r.margins is not None and len(r.margins) == len(ctx.probe_ids)
        assert abs(np.mean(r.margins) - r.fitness) < 1e-9


def test_margins_do_not_change_the_run():
    # fitness and the archive are what they were before margins existed: same seed, same cells
    a, b = Rec(), Rec()
    c1, c2 = run(_cfg(), [a], generations=15), run(_cfg(), [b], generations=15)
    fp = lambda c: [(k, e.genome.id, e.fitness) for k, e in c.archive.cells().items()]
    assert fp(c1) == fp(c2) and [r.margins for r in a.results] == [r.margins for r in b.results]


def test_margins_reach_the_store_and_latest_json(tmp_path):
    ctx = run(_cfg(), [StoreObserver(tmp_path)], generations=15, now=NOW)
    st = Store(store_path(ctx.run_id, tmp_path))
    ents = st.entrants(ctx.run_id)
    assert all(e["margins"] is not None and len(e["margins"]) == 3 for e in ents)
    export_run(st, ctx.run_id, tmp_path, now=NOW)
    d = json.loads((tmp_path / "out" / ctx.run_id / "latest.json").read_text())
    assert d["probes"] == ctx.probe_ids and d["probes"][0] == "scripted-v1"
    for c in d["cells"]:
        assert len(c["margins"]) == len(d["probes"])
        assert abs(sum(c["margins"]) / len(c["margins"]) - c["fitness"]) < 1e-9


def test_old_store_without_the_column_opens_and_exports_null(tmp_path):
    ctx = run(_cfg(), [StoreObserver(tmp_path)], generations=8, now=NOW)
    path = store_path(ctx.run_id, tmp_path)
    old = Store(path)
    old.db.execute("ALTER TABLE evaluations DROP COLUMN margins_json")  # what a pre-T11 store looks like
    old.db.commit()
    old.close()
    st = Store(path)  # reopening adds the column back, nothing is rewritten
    cols = {r[1] for r in st.db.execute("PRAGMA table_info(evaluations)")}
    assert "margins_json" in cols
    assert all(e["margins"] is None for e in st.entrants(ctx.run_id))
    export_run(st, ctx.run_id, tmp_path, now=NOW)
    d = json.loads((tmp_path / "out" / ctx.run_id / "latest.json").read_text())
    assert d["cells"] and all(c["margins"] is None for c in d["cells"])
    assert d["probes"] == ctx.probe_ids


def test_append_only_triggers_survive_the_new_column(tmp_path):
    import sqlite3
    import pytest
    ctx = run(_cfg(), [StoreObserver(tmp_path)], generations=3, now=NOW)
    st = Store(store_path(ctx.run_id, tmp_path))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        st.db.execute("UPDATE evaluations SET margins_json=NULL")
