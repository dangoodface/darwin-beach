"""Scripted bots by name (oracle exploration, plan B4): every simulator lists the bots it plays as
{id: slot}, the coach as scripted-v1 and an oracle build's rungs as oracle-v1-rung0 and up, so ladders
and stores can name a member that has no genome."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arena.kernel import Kernel
from arena.sim import ORACLE_SLOT_BASE, SCRIPTED_ID, SCRIPTED_SLOT, FakeSimulator, oracle_id
from arena.workers import Workers

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts/kernel/0f673ff4b2e4ccb5ff52e63b4fd68f5407526db002eaf638a6996c445e3f07fc.wasm"
ORACLE = ROOT / "artifacts/kernel/5c7bcacca241c6454e281db1395d6fab9e4082c4aa39a7b9f5397bbca3e3afe2.wasm"


def test_the_fake_lists_the_coach_and_a_rung_that_plays_its_own_game():
    sim = FakeSimulator()
    assert sim.bots == {SCRIPTED_ID: SCRIPTED_SLOT, "oracle-v1-rung0": ORACLE_SLOT_BASE}
    sim.load_genomes(np.zeros((1, sim.genome_len), dtype=np.float32))
    coach, rung = sim.bots[SCRIPTED_ID], sim.bots[oracle_id(0)]
    for seed in range(8):
        a, b = sim.play(seed, rung, 0), sim.play(seed, coach, 0)
        assert not a.flags & 4 and a.final_hash != b.final_hash and a.desc_a != b.desc_a
        assert sim.play(seed, rung, 0) == a  # a pure function of its inputs
    assert not sim.play(3, rung, coach).flags & 4


@pytest.mark.parametrize("count", [1, 2])
def test_workers_list_the_bots_without_a_kernel_object(count):
    with Workers("fake", "", count) as w:
        assert w.bots == FakeSimulator().bots
        assert w.bots[SCRIPTED_ID] == w.simulator_info.scripted_slot


def test_a_build_without_the_oracle_lists_only_the_coach():
    assert Kernel(V3).bots == {SCRIPTED_ID: SCRIPTED_SLOT}
    with Workers("kernel", str(V3), 1) as w:
        assert w.bots == {SCRIPTED_ID: w.simulator_info.scripted_slot}


def test_the_oracle_build_lists_eleven_rungs_and_rung_0_beats_the_coach():
    k = Kernel(ORACLE)
    assert k.bots == {SCRIPTED_ID: SCRIPTED_SLOT} | {f"oracle-v1-rung{i}": 0xFFFFFF00 + i for i in range(11)}
    assert [k.oracle_rung(i)[0] for i in range(11)] == [0, 1, 2, 3, 4, 6, 8, 12, 16, 2, 2]
    k.load_genomes(np.zeros((1, k.genome_len), dtype=np.float32))
    r = k.play(20261028, k.bots[oracle_id(0)], k.bots[SCRIPTED_ID], 6000)
    assert not r.flags & 4 and r.points_a == 5 > r.points_b
    assert k.play(20261028, k.bots[oracle_id(0)], k.bots[SCRIPTED_ID], 6000) == r  # deterministic
