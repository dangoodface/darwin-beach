import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest

from arena.config import Config
from arena.run import run
from arena.store import StoreObserver, Store, data_root, store_path

NOW = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)


def _cfg(**kw):
    base = dict(seed=11, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10))
    base.update(kw)
    return Config(**base)


@pytest.fixture
def done(tmp_path):
    ctx = run(_cfg(), [StoreObserver(tmp_path)], generations=20, now=NOW)
    return ctx, Store(store_path(ctx.run_id, tmp_path))


def test_run_row_keeps_probes_and_seeds(done):
    ctx, st = done
    row = st.run_row(ctx.run_id)
    assert json.loads(row["probe_ids_json"]) == [p.id for p in ctx.probes]
    assert json.loads(row["eval_seeds_json"]) == [str(s) for s in ctx.cfg.eval_seeds]
    assert json.loads(row["config_json"])["eval"]["eval_seeds"] == [str(s) for s in ctx.cfg.eval_seeds]
    assert row["kernel_sha"] == ctx.kernel_sha and row["era"] == 0
    assert row["run_id"].endswith("-slime-v0") and row["run_id"].startswith("20261027T090000Z")


def test_entrants_only_with_parents_and_tags(done):
    ctx, st = done
    ents = st.entrants(ctx.run_id)
    ids = {e["genome_id"] for e in ents}
    assert len(ids) == len(ents) > 0
    probes = {p.id for p in ctx.probes}
    for e in ents:
        assert e["genome_id"] not in probes
        assert len(e["parents"]) in (0, 1, 2)
    # every non-root parent is itself an entrant (the tree has no dangling parents)
    for e in ents:
        for p in e["parents"]:
            assert p in ids
    kinds = st.db.execute("SELECT DISTINCT era, kernel_sha FROM genomes").fetchall()
    assert kinds == [(0, ctx.kernel_sha)]


def test_occupants_match_live_archive(done):
    ctx, st = done
    live = {k: e.genome.id for k, e in ctx.archive.cells().items()}
    assert st.occupants(ctx.run_id) == live


def test_params_round_trip(done):
    ctx, st = done
    e = next(iter(ctx.archive.cells().values()))
    got = np.frombuffer(st.params(e.genome.id), dtype="<f4")
    assert np.array_equal(got, e.genome.params)


def test_generations_table(done):
    ctx, st = done
    gens = st.generations(ctx.run_id)
    assert [g["gen"] for g in gens] == list(range(20))
    assert gens[-1]["filled"] == ctx.archive.filled()
    assert abs(gens[-1]["qd_score"] - ctx.archive.qd_score()) < 1e-9
    assert gens[-1]["matches_total"] == 20 * 16 * 2 * 2 * 2


def test_append_only(done):
    _, st = done
    for sql in ("UPDATE evaluations SET fitness=0", "DELETE FROM genomes", "UPDATE runs SET slug='x'",
                "DELETE FROM archive_events", "UPDATE genome_params SET params=x''"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            st.db.execute(sql)


def test_wal_and_data_root(done, monkeypatch):
    _, st = done
    assert st.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    monkeypatch.setenv("ARENA_DATA", "/tmp/nope")
    with pytest.raises(ValueError):
        data_root()
