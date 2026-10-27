"""The PPO species' fake-free parts, against the real kernel: the observation mirror, the
rollout harness and the genome export. Skipped when no kernel is built."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena.genome import genome_id, random_params  # noqa: E402
from arena.kernel import Kernel, find_kernel  # noqa: E402

KERNEL = find_kernel()
pytestmark = pytest.mark.skipif(KERNEL is None, reason="no kernel: build it (README) or set ARENA_KERNEL")

from species.ppo import policy as P  # noqa: E402
from species.ppo.rollout import COACH, MatchRunner  # noqa: E402

RNG = np.random.default_rng(11)
GENOMES = [random_params(RNG) * s for s in (1.0, 2.0, 0.5)]


@pytest.fixture(scope="module")
def runner():
    return MatchRunner(KERNEL)


def test_unpack_pack_round_trip():
    g = GENOMES[0]
    assert np.array_equal(P.pack(*P.unpack(g)), g)


def test_mirror_and_exact_policy_equal_the_kernels_own_policy(runner):
    """Along random play, the policy's mode on the mirrored observation is the kernel's policy_action."""
    k = runner.k
    k.load_genomes(np.stack(GENOMES))
    net = P.unpack(GENOMES[0])
    ex = k._exports
    rng, checked = np.random.default_rng(3), 0
    for seed in range(8):
        runner._reset(runner.store, seed)
        feed = P.ObsFeed()
        while runner.state[P.LIVES_A] > 0 and runner.state[P.LIVES_B] > 0 and checked < 4000:
            for side in (0, 1):
                got = P.bits_to_action(P.mode_bits(P.logits_exact(feed.get(runner.state, side), net)))
                assert got == ex["policy_action"](runner.store, side, 0), (seed, side, runner.state[P.TICK])
                checked += 1
            runner._set(runner.store, int(rng.integers(8)), int(rng.integers(8)))
            feed.after_step(runner._step(runner.store))
            if runner.state[P.TICK] > 1200:
                break
    assert checked > 1500


def test_a_greedy_rollout_replays_kernel_play_tick_for_tick(runner):
    """Greedy + exact arithmetic is the kernel's deterministic match, so hash, ticks and points agree,
    on either side, against an elite in a slot and against the coach."""
    k = runner.k
    for seed in (5, 99):
        for side in (0, 1):
            ep = runner.play(GENOMES[0], GENOMES[1], seed, side, greedy=True, exact=True)
            k.load_genomes(np.stack([GENOMES[0], GENOMES[1]]))
            a, b = (0, 1) if side == 0 else (1, 0)
            r = k.play(seed, a, b)
            mine, theirs = (r.points_a, r.points_b) if side == 0 else (r.points_b, r.points_a)
            assert (ep.ticks, ep.final_hash, ep.points_for, ep.points_against) == (r.ticks, r.final_hash, mine, theirs)
            assert ep.reward.sum() == ep.points_for - ep.points_against
        ep = runner.play(GENOMES[2], COACH, seed, 0, greedy=True, exact=True)
        k.load_genomes(np.stack([GENOMES[2]]))
        r = k.play(seed, 0, k.scripted_slot)
        assert (ep.ticks, ep.final_hash, ep.points_for, ep.points_against) == (r.ticks, r.final_hash, r.points_a, r.points_b)


def test_sampling_records_are_consistent(runner):
    ep = runner.play(GENOMES[0], COACH, 7, 0, rng_seed=1)
    assert ep.obs.shape == (ep.ticks, 12) and ep.bits.shape == (ep.ticks, 3) and ep.ticks > 100
    net = P.unpack(GENOMES[0])
    for t in (0, ep.ticks // 2, ep.ticks - 1):
        lg = P.logits(ep.obs[t].astype(np.float64), net)
        assert abs(P.log_prob(lg, ep.bits[t]) - ep.logp[t]) < 1e-3
    again = runner.play(GENOMES[0], COACH, 7, 0, rng_seed=1)
    assert np.array_equal(again.bits, ep.bits) and again.final_hash == ep.final_hash  # reproducible
    other = runner.play(GENOMES[0], COACH, 7, 0, rng_seed=2)
    assert not np.array_equal(other.bits[:200], ep.bits[:200])


def test_log_prob_matches_the_bernoulli_formula():
    lg, bits = np.array([0.3, -1.2, 4.0]), np.array([1, 0, 0], dtype=np.uint8)
    p = 1 / (1 + np.exp(-lg))
    assert abs(P.log_prob(lg, bits) - (np.log(p[0]) + np.log(1 - p[1]) + np.log(1 - p[2]))) < 1e-12


def test_export_writes_a_genome_the_kernel_agrees_with(tmp_path):
    from species.ppo.export import export_genome, verify_against_kernel
    out = export_genome(GENOMES[1], tmp_path, KERNEL)
    k = Kernel(KERNEL)
    assert out.name == genome_id(GENOMES[1], k.kernel_sha) + ".bin"
    assert np.array_equal(np.fromfile(out, dtype="<f4"), GENOMES[1]) and out.stat().st_size == 515 * 4
    assert verify_against_kernel(out, KERNEL, n=1000) == 1000
    bad = tmp_path / "bad.bin"
    bad.write_bytes(out.read_bytes()[:-4])
    with pytest.raises(ValueError, match="515"):
        verify_against_kernel(bad, KERNEL)
