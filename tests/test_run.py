import numpy as np

from arena.config import Config, load_config
from arena.run import run
from arena.workers import Workers


def _cfg(**kw):
    base = dict(seed=7, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10))
    base.update(kw)
    return Config(**base)


class Rec:
    def __init__(self):
        self.gens, self.started, self.ended = [], 0, 0

    def on_start(self, ctx): self.started += 1
    def on_generation(self, ctx, info): self.gens.append(info)
    def on_end(self, ctx, info): self.ended += 1


def _fingerprint(ctx):
    return [(k, e.genome.id, e.fitness, e.bd) for k, e in ctx.archive.cells().items()]


def test_config_files_load():
    real = load_config("configs/slime-v0.toml")
    assert real.resolution == (20, 20) and real.workers == 8 and real.sim_kind == "kernel"
    assert real.kernel_path.startswith("artifacts/kernel/") and real.scripted_probe
    practice = load_config("configs/practice-fake.toml")
    assert practice.resolution == (20, 20) and practice.sim_kind == "fake"


def test_run_fills_and_notifies():
    rec = Rec()
    ctx = run(_cfg(), [rec], generations=30)
    assert rec.started == 1 and rec.ended == 1 and len(rec.gens) == 30
    assert ctx.archive.filled() > 10
    assert rec.gens[-1].matches_total == 30 * 16 * 3 * 2 * 2  # gens * cands * (scripted + 2 probes) * seeds * sides
    assert all(len(g.results) == len(g.outcomes) == len(g.evicted) for g in rec.gens)
    assert all(e is None for g in rec.gens for o, e in zip(g.outcomes, g.evicted) if o != "replaced")


def test_run_improves():
    rec = Rec()
    ctx = run(_cfg(), [rec], generations=60)
    first = max(r.fitness for r in rec.gens[0].results)
    assert ctx.archive.best().fitness >= first
    assert ctx.archive.qd_score() > 0


def test_same_config_same_archive_bit_for_bit():
    a = run(_cfg(), generations=25)
    b = run(_cfg(), generations=25)
    c = run(_cfg(seed=8), generations=25)
    assert _fingerprint(a) == _fingerprint(b)
    assert _fingerprint(a) != _fingerprint(c)


def test_pool_matches_inline():
    # results in task order: a spawned pool must reproduce the in-process run exactly
    a = run(_cfg(workers=1), generations=6)
    b = run(_cfg(workers=3, jobs_per_task=7), generations=6)
    assert _fingerprint(a) == _fingerprint(b)


def test_resolved_config_records_probes_and_seeds():
    ctx = run(_cfg(), generations=1)
    assert len(ctx.cfg.eval_seeds) == 2 and len(ctx.probes) == 2
    again = run(_cfg(eval_seeds=ctx.cfg.eval_seeds), generations=1)
    assert again.cfg.eval_seeds == ctx.cfg.eval_seeds
    assert [p.id for p in again.probes] == [p.id for p in ctx.probes]


def test_workers_results_in_task_order():
    from arena.genome import random_params
    rng = np.random.default_rng(0)
    table = np.stack([random_params(rng) for _ in range(5)])
    jobs = [(s, s % 5, (s + 1) % 5) for s in range(40)]
    with Workers("fake", count=1) as w1, Workers("fake", count=3, jobs_per_task=6) as w3:
        assert w1.play(table, jobs) == w3.play(table, jobs)
