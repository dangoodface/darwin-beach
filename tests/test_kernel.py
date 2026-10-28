"""The Kernel class over the real wasm: the Simulator protocol as the trainer
uses it, the bad-job rules, the 64-bit crossing, and the interactive replay
the page relies on."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from arena.kernel import FLAG_BAD_JOB, FLAG_MAX_TICKS, Kernel, find_kernel
from arena.sim import SCRIPTED_SLOT, MatchResult

KERNEL = find_kernel()
pytestmark = pytest.mark.skipif(KERNEL is None, reason="no kernel: build it (README) or set ARENA_KERNEL")

GLEN = Kernel(KERNEL).genome_len if KERNEL else 1027
# Sixteen genomes from near idle to wild.
TABLE = (np.random.default_rng(7).standard_normal((16, GLEN)) * np.linspace(0.2, 2.0, 16)[:, None]).astype(np.float32)


@pytest.fixture(scope="module")
def kernel() -> Kernel:
    k = Kernel(KERNEL)
    k.load_genomes(TABLE)
    return k


def test_protocol_attributes(kernel):
    assert kernel.genome_len == 1027
    assert kernel.kernel_version == 3 and kernel.desc_count == 11 and kernel.holds
    assert kernel.max_ticks_cap == 12_000
    assert kernel.scripted_slot == SCRIPTED_SLOT == 0xFFFFFFFF
    assert kernel.kernel_sha == hashlib.sha256(KERNEL.read_bytes()).hexdigest()


def test_a_match_is_a_pure_function_of_its_inputs(kernel):
    other = Kernel(KERNEL)
    other.load_genomes(np.zeros((1, GLEN), dtype=np.float32))
    first = kernel.play(123, 3, 4)
    assert kernel.play(123, 3, 4) == first
    other.load_genomes(TABLE)
    assert other.play(123, 3, 4) == first
    assert kernel.play(124, 3, 4) != first
    assert isinstance(first, MatchResult) and first.tick_hashes is None
    assert len(first.desc_a) == len(first.desc_b) == kernel.desc_count
    assert first.desc_a[7] == first.points_a and first.desc_b[7] == first.points_b


def test_recorded_hashes_end_on_the_final_hash(kernel):
    r = kernel.play(5, 0, SCRIPTED_SLOT, record_hashes=True)
    assert len(r.tick_hashes) == r.ticks and r.tick_hashes[-1] == r.final_hash
    assert len(set(r.tick_hashes)) == r.ticks  # a running hash never repeats in practice


def test_seeds_cross_2_to_the_63(kernel):
    high = kernel.play(2**64 - 1, 1, 2)
    assert high.flags & FLAG_BAD_JOB == 0
    assert kernel.play(2**63, 1, 2) != kernel.play(2**63 - 1, 1, 2)
    assert kernel.play_batch([(2**64 - 1, 1, 2)])[0] == high
    assert 0 <= high.final_hash < 2**64


def test_batch_equals_single_plays_across_chunks_and_bad_jobs(kernel):
    jobs = [(s, s % 16, (s * 7 + 3) % 16) for s in range(300)]
    jobs[10] = (10, 0, 16)  # past the loaded rows: stale genomes, so bad
    jobs[20] = (20, -1, 0)
    jobs[30] = (30, SCRIPTED_SLOT, SCRIPTED_SLOT)
    jobs[40] = (40, 2000, SCRIPTED_SLOT)
    batch = kernel.play_batch(jobs)
    assert len(batch) == len(jobs)
    assert batch == [kernel.play(*job) for job in jobs]
    assert [i for i, r in enumerate(batch) if r.flags & FLAG_BAD_JOB] == [10, 20, 40]
    assert batch[10] == kernel.bad_job


def test_max_ticks_rules(kernel):
    short = kernel.play(9, 1, 2, max_ticks=10)
    assert short.ticks == 10 and short.flags == FLAG_MAX_TICKS
    assert kernel.play(9, 1, 2, max_ticks=6000).flags & FLAG_BAD_JOB == 0  # evaluations may run long
    for bad in (0, kernel.max_ticks_cap + 1):
        assert kernel.play(9, 1, 2, max_ticks=bad).flags == FLAG_BAD_JOB
        assert kernel.play_batch([(9, 1, 2), (9, 2, 1)], max_ticks=bad) == [kernel.bad_job, kernel.bad_job]


def test_load_genomes_checks_shape_and_capacity(kernel):
    with pytest.raises(ValueError):
        kernel.load_genomes(np.zeros((4, GLEN - 1), dtype=np.float32))
    with pytest.raises(ValueError):
        kernel.load_genomes(np.zeros((1025, GLEN), dtype=np.float32))
    kernel.load_genomes(TABLE)


def test_interactive_replay_matches_play(kernel):
    for seed, a, b in [(77, 5, 6), (2**64 - 9, SCRIPTED_SLOT, 3), (3, SCRIPTED_SLOT, SCRIPTED_SLOT)]:
        r = kernel.play(seed, a, b)
        assert kernel.replay(seed, a, b) == (r.ticks, r.final_hash)


def test_holds_agree_across_play_batch_and_replay(kernel):
    for ha, hb in [(1, 1), (9, 1), (1, 9), (4, 6)]:
        r = kernel.play(31, 3, SCRIPTED_SLOT, hold_a=ha, hold_b=hb)
        assert r.flags & FLAG_BAD_JOB == 0
        assert kernel.play_batch([(31, 3, SCRIPTED_SLOT, ha, hb)])[0] == r
        assert kernel.replay(31, 3, SCRIPTED_SLOT, hold_a=ha, hold_b=hb) == (r.ticks, r.final_hash)
    assert kernel.play(31, 3, SCRIPTED_SLOT, hold_a=9) != kernel.play(31, 3, SCRIPTED_SLOT)
    assert kernel.play_batch([(31, 3, SCRIPTED_SLOT)])[0] == kernel.play_batch([(31, 3, SCRIPTED_SLOT, 1, 1)])[0]
    assert kernel.play(31, 3, 4, hold_a=0) == kernel.bad_job
    assert kernel.play_batch([(31, 3, 4, 0, 1), (31, 3, 4)])[0] == kernel.bad_job

