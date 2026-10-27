import json
from datetime import datetime, timezone

import pytest

from arena.config import Config
from arena.replay import open_simulator
from arena.run import run
from arena.store import Store, StoreObserver, store_path
from arena.summary import main, render, summarize

NOW = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)


class Ops:
    """Counts the candidates the loop really asked for, by operator."""

    def __init__(self): self.counts = {"random": 0, "mutate": 0, "iso_line": 0}
    def on_start(self, ctx): pass
    def on_end(self, ctx, info): pass
    def on_generation(self, ctx, info):
        for r in info.results:
            self.counts[r.genome.op] += 1


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    root = tmp_path_factory.mktemp("summary")
    cfg = Config(seed=9, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                 resolution=(10, 10))
    ops = Ops()
    ctx = run(cfg, [StoreObserver(root), ops], generations=60, now=NOW)
    st = Store(store_path(ctx.run_id, root))
    return ctx, root, st, ops


def test_the_facts_exist(done):
    ctx, _, st, _ = done
    d = summarize(st, ctx.run_id, sim=open_simulator(st, ctx.run_id))
    marks = d["fill"]["marks"]
    assert d["fill"]["capacity"] == 100 and [m["cells"] for m in marks] == [100] and marks[0]["all"]
    assert (marks[0]["gen"] is not None) == (ctx.archive.filled() == 100)
    assert d["fill"]["filled_now"] == ctx.archive.filled()
    assert d["fill"]["last_cell"]["gen"] <= 59
    pts = d["lit"]["points"]
    assert [p["gen"] for p in pts] == [0, 10, 50, 59]
    assert all(b["lit"] >= a["lit"] for a, b in zip(pts, pts[1:]))
    regs = d["regions"]["regions"]
    assert len(regs) == 9 and sum(r["capacity"] for r in regs) == 100
    assert sum(r["filled"] for r in regs) == ctx.archive.filled()
    assert d["regions"]["strongest"]["mean"] >= d["regions"]["weakest"]["mean"]
    assert len(d["jumps"]) == 3 and d["jumps"][0]["jump"] >= d["jumps"][1]["jump"] >= d["jumps"][2]["jump"]
    assert set(d["extremes"]) == {"most_airborne", "most_grounded", "closest_to_net", "deepest_back"}
    assert d["throughput"]["generations"] == 60
    assert d["throughput"]["matches_total"] == 60 * 16 * 3 * 2 * 2
    text = render(d)
    for n in "12345678":
        assert f"\n{n}. " in "\n" + text or text.startswith(f"{n}. ")
    json.dumps(d)  # the --json form must serialise


def test_champion_line_ends_on_a_random_genome(done):
    ctx, _, st, _ = done
    ch = summarize(st, ctx.run_id)["champion"]
    ln = ch["lineage"]
    assert ln["oldest_op"] == "random" and ln["line_length"] == sum(ln["operators"].values())
    assert ln["ancestors"] >= ln["line_length"] - 1
    assert ch["cell"] == list(ctx.archive.cell_of((ch["court_position"], ch["airtime"])))
    assert len(ln["fitness_line"]) <= 12 and ln["fitness_line"][-1] == ch["fitness"]


def test_champion_rivals_agree_with_the_store(done):
    ctx, _, st, _ = done
    ch = summarize(st, ctx.run_id, sim=open_simulator(st, ctx.run_id))["champion"]
    probes = ch["rivals"]["probes"]
    assert [p["probe"] for p in probes][0] == "scripted-v1" and len(probes) == 3
    for p, stored in zip(probes, ch["stored_margins"]):
        assert abs(p["margin"] - stored) < 1e-9 and 0 <= p["wins"] <= p["matches"] == 4
    assert ch["rivals"]["beats_the_coach"] == (probes[0]["margin"] > 0)


def test_operator_shares_sum_to_one_and_candidates_are_the_real_ones(done):
    ctx, _, st, ops = done
    o = summarize(st, ctx.run_id)["operators"]
    for k in ("entrants", "candidates"):
        assert abs(sum(v["share"] for v in o[k].values()) - 1.0) < 1e-9
    # replayed from the config, the candidates equal what the loop actually asked for
    assert {k: v["count"] for k, v in o["candidates"].items()} == ops.counts
    assert all(o["entrants"].get(k, {"count": 0})["count"] <= v["count"] for k, v in o["candidates"].items())


def test_lit_floor_override_and_cli(done, capsys):
    ctx, root, st, _ = done
    lo = summarize(st, ctx.run_id, lit_floor=-99.0)["lit"]["points"][-1]["lit"]
    hi = summarize(st, ctx.run_id, lit_floor=99.0)["lit"]["points"][-1]["lit"]
    assert lo == ctx.archive.filled() and hi == 0
    assert main([ctx.run_id, "--root", str(root)]) == 0
    assert "8. Throughput" in capsys.readouterr().out
    assert main([ctx.run_id, "--root", str(root), "--json", "--no-rematch", "--no-candidates"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["run_id"] == ctx.run_id and d["operators"]["candidates"] is None
    assert main(["nope", "--root", str(root)]) == 2
