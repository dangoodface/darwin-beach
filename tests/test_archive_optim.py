import numpy as np

from arena import genome as G
from arena.archive import NEW, REJECTED, REPLACED, Archive
from arena.optim import EvalResult, MapElites

LEN = 8


def _g(params, op="random", parents=()):
    return G.make_genome(np.asarray(params, dtype=np.float32), "toy", parents, op, 0)


def test_add_new_replaced_rejected_and_ties():
    a = Archive(((0, 1), (0, 1)), (4, 4))
    g1, g2, g3 = _g([1] * LEN), _g([2] * LEN), _g([3] * LEN)
    assert a.add(g1, (0.1, 0.1), 1.0) == NEW
    assert a.add(g2, (0.2, 0.2), 0.5) == REJECTED  # same cell, worse
    assert a.add(g2, (0.2, 0.2), 1.0) == REJECTED  # tie keeps the elder
    assert a.add(g3, (0.2, 0.2), 2.0) == REPLACED
    assert a.get((0, 0)).genome.id == g3.id
    assert a.filled() == 1


def test_cells_clamp_and_edges():
    a = Archive(((0, 1), (0, 1)), (20, 20))
    assert a.cell_of((0.0, 1.0)) == (0, 19)
    assert a.cell_of((-3.0, 7.0)) == (0, 19)
    assert a.cell_of((0.5, 0.049)) == (10, 0)


def test_qd_score_and_order():
    a = Archive(((0, 1), (0, 1)), (2, 2), qd_offset=5.0)
    a.add(_g([1] * LEN), (0.9, 0.9), -1.0)
    a.add(_g([2] * LEN), (0.1, 0.1), 2.0)
    assert a.qd_score() == (-1.0 + 5.0) + (2.0 + 5.0)
    assert list(a.cells()) == [(0, 0), (1, 1)]
    assert a.best().fitness == 2.0 and a.capacity() == 4


def _toy_run(seed, batches=60):
    """Fitness is closeness to the origin; behaviour is two squashed coordinates."""
    rng = np.random.default_rng(seed)
    arch = Archive(((0, 1), (0, 1)), (10, 10))
    opt = MapElites(arch, rng, "toy", LEN, batch_size=32, init_size=20)
    qd = []
    for _ in range(batches):
        cands = opt.ask()
        res = []
        for g in cands:
            p = g.params.astype(float)
            bd = tuple(0.5 + 0.5 * p[:2] / (1 + np.abs(p[:2])))
            res.append(EvalResult(g, bd, -float(np.mean(p**2))))
        opt.tell(res)
        qd.append(arch.qd_score())
    return arch, qd


def test_map_elites_fills_and_never_regresses():
    arch, qd = _toy_run(1)
    assert arch.filled() > 20  # grew past the random start
    assert all(b >= a - 1e-12 for a, b in zip(qd, qd[1:]))
    assert qd[-1] > qd[0]


def test_map_elites_reproducible_and_seed_sensitive():
    a1, _ = _toy_run(2)
    a2, _ = _toy_run(2)
    a3, _ = _toy_run(3)
    ids = lambda a: [(k, e.genome.id) for k, e in a.cells().items()]
    assert ids(a1) == ids(a2)
    assert ids(a1) != ids(a3)


def test_children_record_parents_and_ops():
    rng = np.random.default_rng(0)
    arch = Archive(((0, 1), (0, 1)), (10, 10))
    opt = MapElites(arch, rng, "toy", LEN, batch_size=40, init_size=5)
    first = opt.ask()
    assert all(g.op == "random" and g.parents == () for g in first)
    opt.tell([EvalResult(g, (float(i % 10) / 10, 0.5), 0.0) for i, g in enumerate(first)])
    kids = opt.ask()
    ops = {g.op for g in kids}
    assert ops == {"mutate", "iso_line"}
    for g in kids:
        assert len(g.parents) == (2 if g.op == "iso_line" else 1)
