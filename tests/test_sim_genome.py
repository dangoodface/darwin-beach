import hashlib

import numpy as np

from arena import genome as G
from arena.sim import FakeSimulator, MatchResult


def _table(n=6, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([G.random_params(rng) for _ in range(n)])


def test_genome_len_matches_abi():
    assert G.GENOME_LEN == 515


def test_genome_id_is_sha256_of_f32_le_plus_kernel_sha():
    p = _table(1)[0]
    expect = hashlib.sha256(p.astype("<f4").tobytes() + b"abc").hexdigest()
    assert G.genome_id(p, "abc") == expect
    assert G.genome_id(p, "abd") != expect


def test_same_rng_same_children():
    p = _table(2)
    a = G.mutate(p[0], np.random.default_rng(5))
    b = G.mutate(p[0], np.random.default_rng(5))
    assert a.dtype == np.float32 and np.array_equal(a, b)
    c = G.iso_line(p[0], p[1], np.random.default_rng(9))
    d = G.iso_line(p[0], p[1], np.random.default_rng(9))
    assert c.dtype == np.float32 and np.array_equal(c, d)
    assert not np.array_equal(c, p[0])


def test_mutation_step_size():
    p = G.random_params(np.random.default_rng(0))
    child = G.mutate(p, np.random.default_rng(1), sigma=0.1)
    assert 0.08 < float(np.std(child - p)) < 0.12


def test_make_genome_is_frozen_and_hashed():
    p = _table(1)[0]
    g = G.make_genome(p, "k", (), "random", 0)
    assert g.id == G.genome_id(p, "k")
    assert not g.params.flags.writeable


def test_fake_is_deterministic_and_well_formed():
    sim = FakeSimulator()
    sim.load_genomes(_table())
    r1 = sim.play(123, 0, 1)
    r2 = sim.play(123, 0, 1)
    assert isinstance(r1, MatchResult) and r1 == r2
    assert 300 <= r1.ticks <= 3000
    assert max(r1.points_a, r1.points_b) == 5
    assert len(r1.desc_a) == 8 and len(r1.desc_b) == 8
    assert 0.0 <= r1.desc_a[0] <= 1.0 and 0.0 <= r1.desc_a[1] <= 1.0
    assert r1.desc_a[7] == r1.points_a
    assert sim.play(124, 0, 1).final_hash != r1.final_hash


def test_fake_batch_equals_singles_and_record_hashes():
    sim = FakeSimulator()
    sim.load_genomes(_table())
    jobs = [(s, s % 6, (s + 2) % 6) for s in range(10)]
    assert sim.play_batch(jobs) == [sim.play(*j) for j in jobs]
    r = sim.play(7, 0, 1, record_hashes=True)
    assert r.tick_hashes is not None and len(r.tick_hashes) == r.ticks
    assert sim.play(7, 0, 1).tick_hashes is None


def test_fake_bad_slot_sets_flag_bit_2():
    sim = FakeSimulator()
    sim.load_genomes(_table(2))
    assert sim.play(1, 0, 5).flags & 4


def test_fake_strength_is_learnable():
    # a genome nearer the hidden target should win more often: there is a gradient to climb
    sim = FakeSimulator()
    rng = np.random.default_rng(3)
    good = sim._target.astype(np.float32)
    near = good + 0.05 * rng.standard_normal(G.GENOME_LEN).astype(np.float32)
    far = G.random_params(rng)
    sim.load_genomes(np.stack([near, far]))
    wins = sum(sim.play(s, 0, 1).points_a == 5 for s in range(100))
    assert wins > 80
