"""Kernel version 3: the wider brain (12-64-3) and per-side action holds.

The proof that the wider brain, and the SIMD that runs it, changed nothing is
the padding test: a version 2 genome padded with zeros (pad_params) plays, at
hold 1, the bit-identical match in version 3, hash, ticks, points, flags and
all ten descriptors. Night run 6's 1,000-match acceptance is the same check on
real genomes; this file keeps a version that runs anywhere.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from arena.genome import GENOME_LEN, GENOME_LEN_V3, N_HID, N_HID_V3, N_IN, N_OUT, pad_params, random_params
from arena.kernel import ROOT, Kernel, find_kernel
from arena.sim import FakeSimulator

V2 = ROOT / "artifacts/kernel/93c29e323d3d1d7874286754ae395de50a8e3932b20cd75aa77787ce4387ebfb.wasm"
FIXTURE = Path(__file__).parent / "fixtures" / "r5_replays.json"
KERNEL = find_kernel()
needs_kernel = pytest.mark.skipif(KERNEL is None, reason="no kernel: build it (README) or set ARENA_KERNEL")


def v2_genomes() -> np.ndarray:
    """Seven evolved genomes from night run 5 and 57 fresh ones at assorted scales."""
    fix = json.loads(FIXTURE.read_text())
    evolved = [np.frombuffer(base64.b64decode(m["params_b64"]), dtype="<f4") for m in fix["matches"]]
    rng = np.random.default_rng(1028)
    fresh = [random_params(rng) * s for s in np.linspace(0.3, 3.0, 57)]
    return np.stack(evolved + fresh).astype(np.float32)


def test_pad_params_layout():
    v2 = np.arange(1, GENOME_LEN + 1, dtype=np.float32)
    v3 = pad_params(v2)
    assert v3.shape == (GENOME_LEN_V3,) and v3.dtype == np.float32 and GENOME_LEN_V3 == 1027
    w1, b1, w2, b2 = np.split(v3, [N_HID_V3 * N_IN, N_HID_V3 * (N_IN + 1), N_HID_V3 * (N_IN + 1 + N_OUT)])
    w1, w2 = w1.reshape(N_HID_V3, N_IN), w2.reshape(N_OUT, N_HID_V3)
    assert (w1[:N_HID].ravel() == v2[: N_HID * N_IN]).all() and not w1[N_HID:].any()
    assert (b1[:N_HID] == v2[N_HID * N_IN : N_HID * (N_IN + 1)]).all() and not b1[N_HID:].any()
    assert (w2[:, :N_HID].ravel() == v2[N_HID * (N_IN + 1) : -N_OUT]).all() and not w2[:, N_HID:].any()
    assert (b2 == v2[-N_OUT:]).all()
    with pytest.raises(ValueError):
        pad_params(v3)


def test_random_params_fill_the_wider_brain():
    p = random_params(np.random.default_rng(0), GENOME_LEN_V3)
    w1 = p[: N_HID_V3 * N_IN]
    assert p.shape == (GENOME_LEN_V3,) and w1[N_HID * N_IN :].any()  # units 32 to 63 are live
    assert not p[N_HID_V3 * N_IN : N_HID_V3 * (N_IN + 1)].any()  # zero biases
    assert random_params(np.random.default_rng(0)).shape == (GENOME_LEN,)  # version 2 unchanged


@needs_kernel
def test_padded_genomes_play_version_2s_matches_bit_for_bit():
    v2, v3 = Kernel(V2), Kernel(KERNEL)
    assert (v2.genome_len, v3.genome_len, v3.kernel_version) == (515, 1027, 3)
    table = v2_genomes()
    v2.load_genomes(table)
    v3.load_genomes(np.stack([pad_params(row) for row in table]))
    rng = np.random.default_rng(3)
    n, bot = len(table), v3.scripted_slot
    seeds = rng.integers(0, 2**64, size=260, dtype=np.uint64).tolist()
    jobs = [(s, i % n, (i * 11 + 5) % n) for i, s in enumerate(seeds[:220])]
    jobs += [(s, i % n, bot) for i, s in enumerate(seeds[220:250])] + [(s, bot, bot) for s in seeds[250:]]
    old, new = v2.play_batch(jobs), v3.play_batch(jobs)
    for r2, r3 in zip(old, new):  # every field, and every descriptor version 2 has
        assert r2[:5] == r3[:5] and r2.desc_a == r3.desc_a[:10] and r2.desc_b == r3.desc_b[:10]
        assert r3.desc_a[10] <= r3.desc_a[7] and r3.desc_b[10] <= r3.desc_b[7]  # spikes within points
    held = v3.play_batch([(s, a, b, 9, 1) for s, a, b in jobs[:220]])
    # holding changes the match unless a genome's action never changes (a near-idle one can be constant)
    assert sum(h.final_hash != r.final_hash for h, r in zip(held, new[:220])) >= 209


@needs_kernel
def test_holds_follow_the_rule_tick_by_tick():
    """A side at hold h acts at ticks 0, h, 2h, ... and repeats in between: the page's
    loop (held_action) matches play() at every hold tried."""
    v3 = Kernel(KERNEL)
    v3.load_genomes(np.stack([pad_params(row) for row in v2_genomes()[:4]]))
    for ha, hb in [(1, 1), (2, 1), (9, 1), (9, 9), (1, 30), (3000, 1)]:
        r = v3.play(77, 0, 1, hold_a=ha, hold_b=hb)
        assert v3.replay(77, 0, 1, hold_a=ha, hold_b=hb) == (r.ticks, r.final_hash), (ha, hb)


def test_fake_simulator_takes_holds():
    sim = FakeSimulator()
    sim.load_genomes(np.stack([random_params(np.random.default_rng(i)) for i in range(3)]))
    plain = sim.play(5, 0, 1)
    assert sim.play(5, 0, 1, hold_a=1, hold_b=1) == plain
    assert sim.play(5, 0, 1, hold_a=9) != plain
    assert sim.play_batch([(5, 0, 1), (5, 0, 1, 9, 1)]) == [plain, sim.play(5, 0, 1, hold_a=9)]
    assert sim.play(5, 0, 1, hold_a=0).flags == 4
