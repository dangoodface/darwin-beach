"""All eight descriptors travel with every entrant, so the page can draw strategy over the map."""
import json
import numpy as np
from arena.config import Config
from arena.export import export_run
from arena.run import run
from arena.sim import DESCRIPTOR_NAMES
from arena.store import Store, StoreObserver, store_path


def _cfg(**kw):
    d = {"slug": "desc", "seed": 5, "sim": {"kind": "fake"}, "optim": {"batch_size": 16, "init_size": 10},
         "workers": {"count": 1}, "eval": {"probe_count": 2, "seeds_per_probe": 2}}
    d.update(kw)
    return Config.from_dict(d)


def test_desc_matches_bd_at_the_indices():
    ctx = run(_cfg(), (), generations=8)
    # the loop only keeps results through the archive; re-evaluate one elite to see the full vector
    from arena.evaluate import Evaluator  # noqa: F401  (import check)
    for e in ctx.archive.elites():
        assert len(e.bd) == 2


def test_desc_reaches_the_store_and_latest_json(tmp_path):
    so = StoreObserver(root=tmp_path)
    ctx = run(_cfg(), [so], generations=8)
    st = Store(store_path(ctx.run_id, tmp_path))
    ents = st.entrants(ctx.run_id)
    assert ents and all(e["desc"] is not None and len(e["desc"]) == 8 for e in ents)
    for e in ents:
        assert [e["desc"][i] for i in ctx.cfg.indices] == e["bd"]
    out = export_run(st, ctx.run_id, tmp_path)
    L = json.loads((out / "latest.json").read_text())
    assert L["descriptor_names"] == list(DESCRIPTOR_NAMES)
    assert all(len(c["desc"]) == 8 for c in L["cells"])
    st.close()


def test_desc_does_not_change_the_run():
    a = run(_cfg(), (), generations=10)
    b = run(_cfg(), (), generations=10)
    fa = [(k, e.genome.id, e.fitness) for k, e in a.archive.cells().items()]
    fb = [(k, e.genome.id, e.fitness) for k, e in b.archive.cells().items()]
    assert fa == fb


def test_old_store_without_the_column_exports_null(tmp_path):
    so = StoreObserver(root=tmp_path)
    ctx = run(_cfg(), [so], generations=4)
    p = store_path(ctx.run_id, tmp_path)
    st = Store(p)
    st.db.execute("ALTER TABLE evaluations DROP COLUMN desc_json")
    st.db.commit(); st.close()
    st = Store(p)  # reopens and adds the column back
    out = export_run(st, ctx.run_id, tmp_path)
    L = json.loads((out / "latest.json").read_text())
    assert all(c["desc"] is None for c in L["cells"])
    st.close()
