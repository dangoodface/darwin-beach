"""Validates a 50-generation FakeSimulator run against docs/DATA.md, file by file."""
import json
import re
from datetime import datetime, timezone

import pytest

from arena.config import Config
from arena.export import ExportObserver, export_run, layout, thin
from arena.run import run
from arena.store import Store, StoreObserver, store_path

NOW = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _cfg(**kw):
    base = dict(seed=5, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10))
    base.update(kw)
    return Config(**base)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("arena")
    so = StoreObserver(root)
    ctx = run(_cfg(), [so, ExportObserver(so, 1e9, root)], generations=50, now=NOW)
    out = root / "out" / ctx.run_id
    load = lambda n: json.loads((out / n).read_text())
    return ctx, root, out, load


def test_files_exist(exported):
    ctx, root, out, _ = exported
    for n in ("latest.json", "curve.json", "tree.json", "tree-full.json"):
        assert (out / n).is_file(), n
    idx = json.loads((root / "out" / "index.json").read_text())
    assert idx == [{"run_id": ctx.run_id, "title": "slime-v0",
                    "started_at": "2026-10-27T09:00:00Z", "latest_generation": 49}]


def test_latest_json(exported):
    ctx, _, out, load = exported
    d = load("latest.json")
    assert d["run_id"] == ctx.run_id and d["generation"] == 49 and d["era"] == 0
    assert d["kernel_sha"] == ctx.kernel_sha and d["kernel_version"] == 0
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", d["generated_at"])
    g = d["grid"]
    assert g["descriptors"] == ["court_position", "airtime"] and g["indices"] == [0, 1]
    assert g["bounds"] == [[0.0, 1.0], [0.0, 1.0]] and g["resolution"] == [10, 10]
    assert len(d["cells"]) == d["stats"]["filled"] == ctx.archive.filled()
    seen = set()
    for c in d["cells"]:
        assert 0 <= c["i"] < 10 and 0 <= c["j"] < 10 and (c["i"], c["j"]) not in seen
        seen.add((c["i"], c["j"]))
        assert HEX64.match(c["genome_id"]) and c["rating"] is None and c["era"] == 0
        assert len(c["bd"]) == 2 and all(0 <= v <= 1 for v in c["bd"])
        assert isinstance(c["fitness"], float) and isinstance(c["gen"], int)
        r = c["replay"]
        assert re.fullmatch(r"0x[0-9a-f]{16}", r["final_hash"])
        assert r["side"] == "A" and len(r["points"]) == 2 and r["ticks"] > 0
        assert isinstance(r["seed"], (int, str))
        if isinstance(r["seed"], str):
            assert int(r["seed"]) > 2**53
        assert (out / "genomes" / f"{c['genome_id']}.bin").stat().st_size == 515 * 4
        if r["opponent_id"] != "scripted-v1":  # the scripted bot has no genome file
            assert (out / "genomes" / f"{r['opponent_id']}.bin").is_file()
        else:
            assert not (out / "genomes" / "scripted-v1.bin").exists()
    s = d["stats"]
    assert s["champion_id"] in {c["genome_id"] for c in d["cells"]}
    best = max(c["fitness"] for c in d["cells"])
    assert next(c for c in d["cells"] if c["genome_id"] == s["champion_id"])["fitness"] == best
    assert s["matches_total"] == 50 * 16 * 3 * 2 * 2  # scripted bot + 2 probe genomes and s["champion_rating"] is None
    assert abs(s["qd_score"] - ctx.archive.qd_score()) < 1e-9


def d_lit_expected(ctx):
    return sum(1 for e in ctx.archive.cells().values() if e.fitness >= 0.0)


def test_curve_json(exported):
    ctx, _, _, load = exported
    d = load("curve.json")
    pts = d["points"]
    assert d["run_id"] == ctx.run_id and [p["gen"] for p in pts] == list(range(50))
    assert all(0 <= p["lit"] <= p["filled"] for p in pts) and pts[-1]["lit"] == d_lit_expected(ctx)
    assert set(pts[0]) == {"gen", "t", "filled", "lit", "beats_coach", "beats_coach_fresh", "qd_score", "champion_rating",
                           "best_fitness", "era", "matches_total"}
    assert all(b["filled"] >= a["filled"] and b["qd_score"] >= a["qd_score"] - 1e-9
               for a, b in zip(pts, pts[1:]))


def test_thin_keeps_last_and_caps():
    pts = [{"gen": i} for i in range(5000)]
    t = thin(pts)
    assert len(t) <= 2000 and t[-1]["gen"] == 4999 and t[0]["gen"] == 0
    assert thin(pts[:2000]) == pts[:2000]


def test_tree_json(exported):
    ctx, _, out, load = exported
    d = load("tree.json")
    full = load("tree-full.json")
    ids = {n["id"] for n in d["nodes"]}
    assert len(ids) == len(d["nodes"]) and len(full["nodes"]) >= len(d["nodes"])
    champs = [n for n in d["nodes"] if n["champion"]]
    assert len(champs) == 1 and champs[0]["alive"]
    lineage = d["champion_lineage"]
    assert lineage[-1] == champs[0]["id"] and set(lineage) <= ids
    by_id = {n["id"]: n for n in full["nodes"]}
    assert by_id[lineage[0]]["parents"] == [] or all(p not in by_id for p in by_id[lineage[0]]["parents"])
    for n in d["nodes"]:
        assert len(n["parents"]) in (0, 1, 2) and n["op"] in ("random", "mutate", "iso_line")
        assert 0.0 <= n["x"] <= 1.0 and 0.0 <= n["y"] <= 1.0
        assert n["rating"] is None and (n["cell"] is None or len(n["cell"]) == 2)
    for p, c in d["edges"]:
        assert p in ids and c in ids and p in by_id[c]["parents"]
    assert {n["id"] for n in d["nodes"] if n["alive"]} == {c["genome_id"] for c in load("latest.json")["cells"]}
    # every entrant's parents are entrants: the full tree is closed
    assert all(p in by_id for n in full["nodes"] for p in n["parents"])
    # parents are older than children, and x grows with generation
    assert all(by_id[p]["gen"] < n["gen"] for n in full["nodes"] for p in n["parents"])
    assert all(by_id[lineage[i]]["gen"] <= by_id[lineage[i + 1]]["gen"] for i in range(len(lineage) - 1))


def test_layout_is_deterministic_and_in_range(exported):
    _, root, _, _ = exported
    ctx = exported[0]
    st = Store(store_path(ctx.run_id, root))
    ents = st.entrants(ctx.run_id)
    a, b = layout(ents), layout(ents)
    assert a == b and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in a.values())


def test_content_addressed_files_are_written_once(exported):
    ctx, root, out, _ = exported
    p = next((out / "genomes").iterdir())
    before = p.stat().st_mtime_ns
    st = Store(store_path(ctx.run_id, root))
    export_run(st, ctx.run_id, root, now=NOW)
    assert p.stat().st_mtime_ns == before
    assert not list(out.rglob("*.tmp-*"))


def test_kernel_file_is_copied_and_checked(tmp_path):
    wasm = tmp_path / "k.wasm"
    wasm.write_bytes(b"\0asm\1\0\0\0")
    # a fake run whose recorded sha does not match the file must be refused
    root = tmp_path / "d"
    so = StoreObserver(root)
    ctx = run(_cfg(), [so], generations=2, now=NOW)  # store only, no export
    st = Store(store_path(ctx.run_id, root))
    st.db.execute("DROP TRIGGER runs_no_update")
    st.db.execute("UPDATE runs SET config_json = replace(config_json, '\"fake\"', '\"kernel\"')")
    st.db.execute("UPDATE runs SET config_json = replace(config_json, '\"kernel_path\": \"\"', ?)",
                  (f'"kernel_path": "{wasm}"',))
    st.db.commit()
    with pytest.raises(ValueError, match="does not match"):
        export_run(st, ctx.run_id, root, now=NOW)


def test_tree_full_is_throttled_but_always_written_at_the_end(tmp_path):
    so = StoreObserver(tmp_path)
    present_during_run = []

    class Spy:
        def on_start(self, ctx): pass
        def on_end(self, ctx, info): pass
        def on_generation(self, ctx, info):
            present_during_run.append((tmp_path / "out" / ctx.run_id / "tree-full.json").is_file())

    # exports on every generation, tree-full at most once an hour: only the final export writes it
    ctx = run(_cfg(), [so, ExportObserver(so, 0.0, tmp_path, 3600.0), Spy()], generations=5, now=NOW)
    out = tmp_path / "out" / ctx.run_id
    assert not any(present_during_run)
    assert (out / "tree.json").is_file() and (out / "tree-full.json").is_file()


def test_tree_full_with_a_zero_interval_is_written_every_time(tmp_path):
    so = StoreObserver(tmp_path)
    present = []

    class Spy:
        def on_start(self, ctx): pass
        def on_end(self, ctx, info): pass
        def on_generation(self, ctx, info):
            present.append((tmp_path / "out" / ctx.run_id / "tree-full.json").is_file())

    run(_cfg(), [so, ExportObserver(so, 0.0, tmp_path, 0.0), Spy()], generations=3, now=NOW)
    assert all(present)
