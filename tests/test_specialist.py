"""The specialist trainer on the FakeSimulator: masks, windows, the ES step, the output contract."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from arena.genome import GENOME_LEN, N_HID, N_IN, genome_id, random_params
from arena import specialist as sp

NOW = datetime(2026, 10, 27, 14, 0, 0, tzinfo=timezone.utc)


def net_action(params, obs):
    """The kernel's policy in numpy (kernel/src/policy.rs): bit i pressed when output i > 0."""
    w1 = params[: N_HID * N_IN].reshape(N_HID, N_IN)
    b1 = params[N_HID * N_IN : N_HID * N_IN + N_HID]
    w2 = params[sp.W2 : sp.B2].reshape(3, N_HID)
    b2 = params[sp.B2 :]
    h = w1 @ (obs / 10.0) + b1
    h = h / (1.0 + np.abs(h))
    return (w2 @ h + b2) > 0.0


def cfg(**kw):
    c = sp.SpecialistConfig(sim_kind="fake", population=8, train_seeds=2, graded_seeds=3, fresh_seeds=2, graded_every=2,
                            windows={"long_ball": dict(speed_min=34.8, speed_scale=9.0, hit_height_max=3.8, hit_height_scale=1.6),
                                     "lob_artist": dict(speed_max=25.8, speed_scale=9.0, hit_height_min=5.5, hit_height_scale=1.6)})
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_repair_is_idempotent_and_the_pinned_jump_never_fires():
    rng = np.random.default_rng(1)
    for name, on in (("flat_earther", False), ("kangaroo", True)):
        st = sp.make_style(name, cfg())
        for _ in range(20):
            p = random_params(rng) * rng.uniform(0.5, 8.0)  # big weights too
            r = st.repair(p)
            assert np.array_equal(st.repair(r), r) and r.dtype == np.float32
            keep = np.ones(GENOME_LEN, bool); keep[sp.W2 + 2 * N_HID : sp.W2 + 3 * N_HID] = False; keep[sp.B2 + 2] = False
            assert np.array_equal(r[keep], p[keep])  # only the jump output changed
            presses = [net_action(r, rng.uniform(-30, 30, 12))[2] for _ in range(300)]
            assert all(presses) if on else not any(presses)


def test_the_twin_has_no_repair_window_or_extra_terms():
    c = cfg()
    for name in ("flat_earther", "kangaroo", "long_ball", "sprinter", "joga_bonito"):
        tw = sp.make_style(name, c, twin=True)
        p = random_params(np.random.default_rng(2))
        assert np.array_equal(tw.repair(p), p) and not tw.constrained and tw.descriptors == 8
    with pytest.raises(ValueError, match="its own twin"):
        sp.make_style("all_rounder", c, twin=True)


def test_window_penalty_is_proportional_and_the_check_strict():
    st = sp.make_style("long_ball", cfg())
    names = sp.DESCRIPTOR_NAMES
    ok = dict.fromkeys(names, 0.0) | {"mean_speed_after_hit": 36.0, "mean_hit_height": 3.0}
    assert st.window.violation(ok) == 0.0 and st.window.satisfied(ok)
    one = ok | {"mean_speed_after_hit": 34.8 - 9.0}  # one scale-width short on speed
    two = ok | {"mean_speed_after_hit": 34.8 - 18.0}
    assert abs(st.window.violation(one) - 1.0) < 1e-9 and abs(st.window.violation(two) - 2.0) < 1e-9
    assert not st.window.satisfied(ok | {"mean_hit_height": 3.9})
    lob = sp.make_style("lob_artist", cfg())
    assert lob.window.satisfied(dict.fromkeys(names, 0.0) | {"mean_speed_after_hit": 20.0, "mean_hit_height": 6.0})
    # scoring subtracts weight * violation
    d = np.zeros((1, 1, 1, 2, 8)); d[..., 4] = 34.8 - 9.0; d[..., 3] = 3.0
    m = sp.Matches(np.full((1, 1, 1, 2), 3.0), np.full((1, 1, 1, 2), 1.0), np.full((1, 1, 1, 2), 600.0), d)
    fit, pen = sp.score(st, cfg(), m, names)
    assert abs(pen[0] - 2.0) < 1e-9 and abs(fit[0] - (2.0 - 2.0)) < 1e-9


def test_sprinter_and_all_rounder_fitness():
    names = sp.DESCRIPTOR_NAMES
    d = np.zeros((1, 2, 1, 2, 8))
    mine = np.array([[[[4, 4]], [[2, 2]]]], float); theirs = np.array([[[[1, 1]], [[1, 1]]]], float)  # margins 3 and 1
    ticks = np.full((1, 2, 1, 2), 300.0)
    m = sp.Matches(mine, theirs, ticks, d)
    c = cfg(sprinter_lambda=0.01)
    assert sp.score(sp.make_style("all_rounder", c), c, m, names)[0][0] == 1.0
    plain = sp.score(sp.make_style("flat_earther", c), c, m, names)[0][0]
    assert plain == 2.0
    tpp = (300 / np.array([5, 5, 3, 3])).mean()
    assert abs(sp.score(sp.make_style("sprinter", c), c, m, names)[0][0] - (2.0 - 0.01 * tpp)) < 1e-9


def test_es_improves_a_toy_quadratic_at_the_real_constants():
    c = sp.SpecialistConfig()
    rng = np.random.default_rng(0)
    target = rng.standard_normal(GENOME_LEN) * 0.3
    f = lambda th: -np.mean((th - target) ** 2, axis=-1)
    theta, adam = np.zeros(GENOME_LEN), sp.Adam(GENOME_LEN, c.lr)
    start = f(theta)
    for _ in range(300):
        eps = rng.standard_normal((c.population // 2, GENOME_LEN))
        gp, gn = f(theta + c.sigma * eps), f(theta - c.sigma * eps)
        theta = theta + adam.step(sp.es_gradient(eps, gp, gn, c.sigma))
    assert f(theta) > start * 0.2  # the error falls by at least five times in 300 generations


def test_centred_ranks_and_the_gradient_points_uphill():
    assert np.allclose(sp.centred_ranks(np.array([5.0, 1.0, 3.0])), [0.5, -0.5, 0.0])
    eps = np.array([[1.0, 0.0], [0.0, 1.0]])
    g = sp.es_gradient(eps, np.array([2.0, 0.0]), np.array([0.0, 2.0]), 0.1)  # +eps0 and -eps1 are better
    assert g[0] > 0 and g[1] < 0


def _export_fixture(root):
    """Two fake exports with genomes and cells, as the trainer's rivals and warm starts read them."""
    rng = np.random.default_rng(3)
    ids = {}
    for run in ("r5", "r4"):
        gdir = root / "out" / run / "genomes"; gdir.mkdir(parents=True)
        cells = []
        for k in range(4):
            p = random_params(rng); gid = f"{run}-{k}"; (gdir / f"{gid}.bin").write_bytes(p.astype("<f4").tobytes())
            cells.append({"i": k, "j": 0, "genome_id": gid, "fitness": float(k), "lit": True,
                          "bd": [0.5, [0.0, 0.5, 0.95, 0.3][k]], "desc": [0.5, 0.5, 5, 4.0, 30.0, 3, 100, 3]})
        (root / "out" / run / "latest.json").write_text(json.dumps({"cells": cells, "stats": {"champion_id": f"{run}-3"}}))
        ids[run] = f"{run}-3"
    return ids


def test_result_json_schema_and_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DATA", str(tmp_path))
    ids = _export_fixture(tmp_path)
    c = cfg(start_run="r5", fallback_run="r4", rivals=[
        {"id": ids["r5"], "kind": "champion", "out_run": "r5"}, {"id": ids["r4"], "kind": "champion", "out_run": "r4"},
        {"id": "scripted-v1", "kind": "coach"}])
    out = sp.run_specialist("flat_earther", False, c, None, 1, generations=6, now=NOW, log=lambda *a: None)
    r = json.loads((out / "result.json").read_text())
    assert out.name == "20261027T140000Z-spec-flat_earther"
    assert set(r) == {"style", "twin", "constraint", "start", "rivals", "seeds", "generations", "matches_total",
                      "name", "seed", "gate", "lazy_cap",
                      "elapsed_s", "kernel_sha", "kernel_version", "final", "curve"}
    assert r["style"] == "flat_earther" and r["twin"] is False and r["constraint"]["kind"] == "pinned_output"
    assert r["start"] == {"genome_id": "r5-0", "run_id": "r5", "cell": [0, 0], "reason": "best lit cell that fits, of 1"}
    assert [x["kind"] for x in r["rivals"]] == ["champion", "champion", "coach"]
    assert len(r["seeds"]["graded"]) == 3 and len(r["seeds"]["fresh"]) == 2
    assert not set(r["seeds"]["fresh"]) & set(r["seeds"]["graded"])
    f = r["final"]
    assert set(f) == {"genome_id", "graded", "fresh", "descriptors_fresh_mean", "constraint_satisfied", "proof", "showcase",
                      "wins_all", "stalled", "points_per_match"}
    for sect, n in (("graded", 6), ("fresh", 4)):
        assert set(f[sect]) == {x["id"] for x in r["rivals"]}
        assert all(set(v) == {"won", "n", "margin"} and v["n"] == n for v in f[sect].values())
    assert set(f["descriptors_fresh_mean"]) == set(sp.DESCRIPTOR_NAMES)
    assert f["showcase"] and set(f["showcase"][0]) == {"seed", "side", "rival_id", "margin", "descriptors", "hold_a", "hold_b"}
    assert set(r["curve"][0]) == {"gen", "t", "fitness_best", "fitness_mean", "margin_by_rival", "constraint_penalty"}
    assert len(r["curve"]) == r["generations"] == 6
    # the final genome is on disk under its content hash, with the rivals and the start
    final = np.frombuffer((out / "genomes" / f"{f['genome_id']}.bin").read_bytes(), dtype=np.float32)
    assert genome_id(final, r["kernel_sha"]) == f["genome_id"] and np.array_equal(sp.make_style("flat_earther", c).repair(final), final)
    assert (out / "genomes" / "r5-0.bin").is_file() and (out / "genomes" / f"{ids['r4']}.bin").is_file()


def test_the_twin_shares_the_start_and_the_seeds_and_has_no_constraint(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DATA", str(tmp_path))
    ids = _export_fixture(tmp_path)
    c = cfg(start_run="r5", fallback_run="r4", rivals=[{"id": ids["r5"], "kind": "champion", "out_run": "r5"},
                                                      {"id": "scripted-v1", "kind": "coach"}])
    a = json.loads((sp.run_specialist("kangaroo", False, c, None, 1, 3, now=NOW, log=lambda *a: None) / "result.json").read_text())
    out = sp.run_specialist("kangaroo", True, c, None, 1, 3, now=NOW, log=lambda *a: None)
    b = json.loads((out / "result.json").read_text())
    assert out.name.endswith("-spec-kangaroo-twin") and b["twin"] is True and b["constraint"]["kind"] == "none"
    assert b["start"] == a["start"] and b["seeds"] == a["seeds"]
    assert b["final"]["constraint_satisfied"] is True and set(b["final"]["proof"]) == {"ticks_per_point"}


def test_joga_needs_ten_descriptors(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DATA", str(tmp_path))
    ids = _export_fixture(tmp_path)
    c = cfg(start_run="r5", fallback_run="r4", rivals=[{"id": "scripted-v1", "kind": "coach"}])
    with pytest.raises(SystemExit, match="10 descriptors"):
        sp.run_specialist("joga_bonito", False, c, None, 1, 2, now=NOW, log=lambda *a: None)
