"""CMA-ME through pyribs (optim.kind = "cma_me"): reproducible, fills the map, feeds its
emitters what happened in our archive, and leaves every MAP-Elites config unchanged."""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("ribs")

from arena.config import Config, load_config  # noqa: E402
from arena.optim import CmaMe  # noqa: E402
from arena.run import run  # noqa: E402


def _cfg(**kw):
    base = dict(seed=13, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10), optim_kind="cma_me", cma_emitters=4)
    base.update(kw)
    return Config(**base)


def _fingerprint(ctx):
    return [(k, e.genome.id, e.fitness, e.bd) for k, e in ctx.archive.cells().items()]


def test_cma_me_run_reproduces_bit_for_bit():
    a, b, c = run(_cfg(), generations=30), run(_cfg(), generations=30), run(_cfg(seed=14), generations=30)
    assert _fingerprint(a) == _fingerprint(b)
    assert _fingerprint(a) != _fingerprint(c)


def test_both_optimisers_fill_cells_on_the_fake():
    me = run(_cfg(optim_kind="map_elites"), generations=40)
    cma = run(_cfg(), generations=40)
    assert me.archive.filled() > 30 and cma.archive.filled() > 30
    ops = {e.genome.op for e in cma.archive.elites()}
    assert "cma_me" in ops and "iso_line" not in ops and "mutate" not in ops


def test_emitters_learn_from_the_archive_and_children_name_a_parent():
    import numpy as np

    from arena.archive import Archive
    from arena.optim import EvalResult

    archive = Archive(((0.0, 1.0), (0.0, 1.0)), (10, 10))
    rng = np.random.default_rng(0)
    cma = CmaMe(archive, rng, "sha", 515, batch_size=8, init_size=4, emitters=2)
    for g in range(2):
        cands = cma.ask()
        for i, c in enumerate(cands):  # fitness and descriptors from the parameters, so cells differ
            bd = (float(abs(c.params[0])) % 1.0, float(abs(c.params[1])) % 1.0)
            cma.tell([EvalResult(c, bd, float(c.params[:10].sum()))])
    assert all(c.op == "cma_me" for c in cands) and len(cands) == 8
    ids = {e.genome.id for e in archive.elites()} | {c.id for c in cands}
    assert all(len(c.parents) == 1 for c in cands)
    assert [em.itrs for em in cma._emitters] == [1, 1]  # each emitter was told its batch once
    assert set(cma._anchors) <= ids | {p for c in cands for p in c.parents}


def test_status_and_value_follow_cma_me():
    import numpy as np

    from arena.archive import Archive
    from arena.genome import make_genome, random_params
    from arena.optim import EvalResult

    archive = Archive(((0.0, 1.0),), (4,))
    cma = CmaMe(archive, np.random.default_rng(1), "sha", 515, batch_size=4, init_size=1, emitters=1)
    g = [make_genome(random_params(np.random.default_rng(i)), "sha", (), "random", 0) for i in range(4)]
    cma._told = []
    cma.tell([EvalResult(g[0], (0.1,), 1.0)])  # new cell
    cma.tell([EvalResult(g[1], (0.1,), 1.5)])  # improves it by 0.5
    cma.tell([EvalResult(g[2], (0.1,), 1.2)])  # rejected, 0.3 short
    cma.tell([EvalResult(g[3], (0.9,), -2.0)])  # new cell with a negative fitness
    assert [(s, round(v, 9)) for s, v, _, _ in cma._told] == [(2, 1.0), (1, 0.5), (0, -0.3), (2, -2.0)]


def test_configs_and_defaults():
    assert Config().optim_kind == "map_elites"
    assert load_config("configs/slime-v0.toml").optim_kind == "map_elites"
    r3, r4 = load_config("configs/night/r3.toml"), load_config("configs/night/r4-cma.toml")
    changed = {f.name for f in dataclasses.fields(r3) if getattr(r3, f.name) != getattr(r4, f.name)}
    assert changed == {"slug", "seed", "optim_kind"}
    assert r4.optim_kind == "cma_me" and r4.batch_size % r4.cma_emitters == 0
    with pytest.raises(ValueError):
        _cfg(batch_size=10, cma_emitters=4).validate()
    with pytest.raises(ValueError):
        _cfg(optim_kind="ga").validate()


def test_cma_me_survives_a_ladder_promotion():
    ctx = run(_cfg(ladder_enabled=True, ladder_check_every=5, ladder_seeds=10), generations=60)
    assert ctx.ladder.era >= 1
    assert ctx.archive.filled() > 30
