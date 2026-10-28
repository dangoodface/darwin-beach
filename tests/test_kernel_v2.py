"""Kernel version 2: the juggle counters, and a game that has not moved by a bit.

Version 2 only adds two descriptors per side (juggled_points, max_streak), so it
must play every match to version 1's hash, ticks, points and first eight
descriptors, and night run 5's stored matches (kernel version 1) must replay to
their stored hashes, through play() and through the page's interactive loop.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from arena.genome import pad_params, random_params
from arena.kernel import ROOT, Kernel, find_kernel

V1 = ROOT / "artifacts/kernel/5487b2a61f66880a4bd468957b6f662cb0f5afd05ab8ae7ac6736a81b025674a.wasm"
V2 = ROOT / "artifacts/kernel/93c29e323d3d1d7874286754ae395de50a8e3932b20cd75aa77787ce4387ebfb.wasm"
FIXTURE = Path(__file__).parent / "fixtures" / "r5_replays.json"
KERNEL = find_kernel()
pytestmark = pytest.mark.skipif(KERNEL is None, reason="no kernel: build it (README) or set ARENA_KERNEL")


@pytest.fixture(scope="module")
def v2() -> Kernel:
    return Kernel(V2)


@pytest.fixture(scope="module")
def current() -> Kernel:
    return Kernel(KERNEL)


def test_version_2_plays_version_1s_matches_bit_for_bit(v2):
    v1 = Kernel(V1)
    assert (v1.kernel_version, v1.desc_count, v2.desc_count) == (1, 8, 10)
    rng = np.random.default_rng(27)
    table = np.stack([random_params(rng) * s for s in np.linspace(0.3, 3.0, 40)]).astype(np.float32)
    v1.load_genomes(table)
    v2.load_genomes(table)
    bot = v2.scripted_slot
    seeds = rng.integers(0, 2**64, size=300, dtype=np.uint64).tolist()
    jobs = [(s, i % 40, (i * 7 + 1) % 40) for i, s in enumerate(seeds[:240])]
    jobs += [(s, bot, i % 40) for i, s in enumerate(seeds[240:270])] + [(s, bot, bot) for s in seeds[270:]]
    old, new = v1.play_batch(jobs), v2.play_batch(jobs)
    for r1, r2 in zip(old, new):
        assert r1[:5] == r2[:5]  # final hash, ticks, points, flags
        assert r1.desc_a == r2.desc_a[:8] and r1.desc_b == r2.desc_b[:8]
        for d in (r2.desc_a, r2.desc_b):
            assert d[8] <= d[7] and d[9] <= d[2]  # juggled points within points, streak within hits
    assert max(max(r.desc_a[9], r.desc_b[9]) for r in new) >= 4  # some side strings four touches together


@pytest.mark.parametrize("which", ["v2", "current"])
def test_night_run_5s_stored_matches_replay(which, request):
    """Version 1 matches replay to their stored hashes under version 2 and under the
    kernel being built, whose wider brain takes the genomes padded (pad_params)."""
    k = request.getfixturevalue(which)
    fix = json.loads(FIXTURE.read_text())
    assert fix["kernel_version"] == 1 and len(fix["matches"]) >= 5
    for m in fix["matches"]:
        rows = [np.frombuffer(base64.b64decode(m["params_b64"]), dtype="<f4")]
        if m["opponent_params_b64"]:
            rows.append(np.frombuffer(base64.b64decode(m["opponent_params_b64"]), dtype="<f4"))
        if k.genome_len != rows[0].shape[0]:
            rows = [pad_params(r) for r in rows]
        k.load_genomes(np.stack(rows))
        opp = 1 if m["opponent_params_b64"] else k.scripted_slot
        a, b = (0, opp) if m["side"] == "A" else (opp, 0)
        r = k.play(int(m["seed"]), a, b)
        mine = [r.points_a, r.points_b] if m["side"] == "A" else [r.points_b, r.points_a]
        assert (f"0x{r.final_hash:016x}", r.ticks, mine) == (m["final_hash"], m["ticks"], m["points"])
        assert k.replay(int(m["seed"]), a, b) == (r.ticks, r.final_hash)  # the page's loop


def test_exported_descriptor_names_follow_the_runs_kernel(tmp_path):
    from arena.config import Config
    from arena.export import export_run
    from arena.run import run
    from arena.store import Store, StoreObserver, store_path

    for path, version, n in ((V1, 1, 8), (V2, 2, 10), (KERNEL, 3, 11)):
        cfg = Config(slug=f"names-v{version}", seed=3, sim_kind="kernel", kernel_path=str(path), workers=1,
                     batch_size=8, init_size=8, probe_count=1, seeds_per_probe=1)
        so = StoreObserver(tmp_path)
        ctx = run(cfg, [so], generations=2)
        st = Store(store_path(ctx.run_id, tmp_path))
        latest = json.loads((export_run(st, ctx.run_id, tmp_path) / "latest.json").read_text())
        st.close()
        assert latest["kernel_version"] == version
        assert len(latest["descriptor_names"]) == n and all(len(c["desc"]) == n for c in latest["cells"])
    assert latest["descriptor_names"][8:] == ["juggled_points", "max_streak", "spiked_points"]
