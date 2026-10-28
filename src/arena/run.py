"""The training loop: ask, evaluate, tell, notify observers.

One SeedSequence from the config seed spawns every random stream, in a fixed
order, so the same config reproduces the same run bit for bit. Wall-clock time
only decides when to stop, never what happens.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from .archive import NEW, Archive
from .config import Config, load_config
from .evaluate import Evaluator
from .fresh import FRESH_SEEDS, FreshCheck
from .genome import Genome, make_genome, random_params
from .ladder import Challenge, Ladder
from .optim import CmaMe, EvalResult, MapElites
from .pool import load_pool, load_warm
from .workers import Workers, make_simulator

# Spawn order of the root SeedSequence. Appending is safe; reordering changes every run.
STREAM_PROBES, STREAM_OPTIM, STREAM_SEEDS, STREAM_LADDER, STREAM_FRESH = 0, 1, 2, 3, 4
RESCORE_BATCH = 256  # elites per evaluation when an era re-scores the archive; fits the kernel table


@dataclass
class RunContext:
    run_id: str
    cfg: Config  # resolved: eval_seeds filled in
    kernel_sha: str
    kernel_version: int
    genome_len: int
    probes: list[Genome]  # frozen generation-zero genomes; the scripted bot has no genome
    probe_ids: list[str]  # the whole probe set in order, "scripted-v1" first when present
    archive: Archive
    started_at: datetime
    started_mono: float = field(default_factory=time.monotonic)
    ladder: Ladder | None = None  # v1
    fresh: FreshCheck | None = None  # the fresh-serve check; never used for selection


@dataclass
class GenerationInfo:
    generation: int
    results: list[EvalResult]
    outcomes: list[str]  # "new" | "replaced" | "rejected", aligned with results
    evicted: list[str | None]  # genome id displaced by each "replaced", else None
    matches_total: int
    elapsed_s: float
    era: int = 0  # the era this generation's candidates were evaluated in
    matches_per_candidate: int = 0  # matches each of this generation's candidates played
    challenge: Challenge | None = None  # v1: the promotion check played after this generation, if any
    rescored: list[EvalResult] | None = None  # v1: the live elites re-evaluated when the check promoted


class Observer(Protocol):
    def on_start(self, ctx: RunContext) -> None: ...
    def on_generation(self, ctx: RunContext, info: GenerationInfo) -> None: ...
    def on_end(self, ctx: RunContext, info: GenerationInfo | None) -> None: ...


def make_run_id(slug: str, now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{slug}"


def rescore(archive: Archive, evaluator: Evaluator) -> tuple[Archive, list[EvalResult]]:
    """A new era: every live elite is re-evaluated against the grown ladder and
    re-inserted into a fresh archive; only fitness and rating change.

    Descriptors come from the fixed probe set, the same matches as before, so
    each elite is placed by the descriptors it already has. The fresh ones must
    agree; the kernel repeats them exactly, while FakeSimulator's matrix
    product over a bigger table can move the last bit, which must not move a cell.
    """
    elites = archive.elites()
    results: list[EvalResult] = []
    for i in range(0, len(elites), RESCORE_BATCH):
        results += evaluator.evaluate([e.genome for e in elites[i : i + RESCORE_BATCH]])
    fresh = Archive(archive.bounds, archive.resolution, archive.qd_offset)
    placed: list[EvalResult] = []
    for e, r in zip(elites, results):
        if not np.allclose(r.bd, e.bd, rtol=1e-9, atol=1e-12):
            raise RuntimeError(f"re-scoring changed {r.genome.id}'s descriptors: they must not depend on the ladder")
        r = r._replace(bd=e.bd)
        if fresh.add(r.genome, r.bd, r.fitness) != NEW:
            raise RuntimeError(f"re-scoring put two elites in cell {fresh.cell_of(r.bd)}")
        placed.append(r)
    return fresh, placed


def tell_each(opt, archive: Archive, results: list[EvalResult]) -> tuple[list[str], list[str | None]]:
    """Tells one result at a time so each eviction is attributed to the right newcomer."""
    outcomes: list[str] = []
    evicted: list[str | None] = []
    for r in results:
        resident = archive.get(archive.cell_of(r.bd))
        outcomes.extend(opt.tell([r]))
        evicted.append(resident.genome.id if resident and outcomes[-1] == "replaced" else None)
    return outcomes, evicted


def run(
    cfg: Config,
    observers: list[Observer] | tuple[()] = (),
    generations: int | None = None,
    max_seconds: float | None = None,
    now: datetime | None = None,
) -> RunContext:
    root = np.random.SeedSequence(cfg.seed)
    probe_ss, optim_ss, seed_ss, ladder_ss, fresh_ss = root.spawn(5)

    with Workers(cfg.sim_kind, cfg.kernel_path, cfg.workers, cfg.jobs_per_task) as workers:
        kernel_sha, kernel_version, genome_len, sim_scripted = workers.simulator_info
        if cfg.scripted_probe and sim_scripted is None:
            raise ValueError("eval.scripted_probe is set but the simulator has no scripted bot")
        scripted_slot = sim_scripted if cfg.scripted_probe else None

        ladder = None
        if cfg.ladder_enabled:
            if sim_scripted is None:
                raise ValueError("ladder.enabled needs the simulator's scripted bot as its anchor")
            ladder_seeds = cfg.ladder_seed_list or tuple(
                int(s) for s in ladder_ss.generate_state(cfg.ladder_seeds, dtype=np.uint64)
            )
            pool = load_pool(cfg.ladder_pool, kernel_sha, genome_len)
            ladder = Ladder(sim_scripted, ladder_seeds, cfg.ladder_threshold, cfg.ladder_cycle_window,
                            pool, cfg.ladder_hold_anchor, cfg.ladder_weight_scale)
            if pool:  # the round robin: 100 blocks of ladder seeds for a pool of 23, a few seconds
                ladder.rate_pool(workers, cfg.max_ticks)

        probe_rng = np.random.default_rng(probe_ss)
        probes = [
            make_genome(random_params(probe_rng, genome_len), kernel_sha, (), "probe", -1)
            for _ in range(cfg.probe_count)
        ]
        seeds = cfg.eval_seeds or tuple(
            int(s) for s in seed_ss.generate_state(cfg.seeds_per_probe, dtype=np.uint64)
        )
        resolved = Config.from_dict({**cfg.to_dict(), "seed": cfg.seed, "eval": {
            "scripted_probe": cfg.scripted_probe, "probe_count": cfg.probe_count, "seeds_per_probe": len(seeds), "eval_seeds": list(seeds)},
            "ladder": {**cfg.to_dict()["ladder"], "seed_list": list(ladder.seeds) if ladder else []}})

        archive = Archive(cfg.bounds, cfg.resolution, cfg.qd_offset)
        optim_rng = np.random.default_rng(optim_ss)
        if cfg.optim_kind == "cma_me":
            opt = CmaMe(archive, optim_rng, kernel_sha, genome_len, cfg.batch_size, cfg.init_size,
                        cfg.cma_emitters, cfg.cma_sigma0, cfg.cma_es)
        else:
            opt = MapElites(archive, optim_rng, kernel_sha, genome_len,
                            cfg.batch_size, cfg.init_size, cfg.sigma, cfg.crossover_frac)
        evaluator = Evaluator(workers, probes, scripted_slot, seeds, cfg.indices, cfg.max_ticks, ladder)

        fresh = None
        if sim_scripted is not None:
            fresh_seeds = tuple(int(x) for x in fresh_ss.generate_state(FRESH_SEEDS, dtype=np.uint64))
            fresh = FreshCheck(make_simulator(cfg.sim_kind, cfg.kernel_path), fresh_seeds, cfg.max_ticks)

        started = now or datetime.now(timezone.utc)
        ctx = RunContext(make_run_id(cfg.slug, started), resolved, kernel_sha, kernel_version,
                         genome_len, probes, [o for o, _ in evaluator.opponents], archive, started,
                         ladder=ladder, fresh=fresh)
        for o in observers:
            o.on_start(ctx)

        info: GenerationInfo | None = None
        gen = 0
        warm = load_warm(cfg.warm_runs, kernel_sha, genome_len, ladder.member_ids() if ladder else set())
        if warm:  # generation 0 is the warm start: earlier elites scored under this run's rules
            era = ladder.era if ladder else 0
            per_candidate = evaluator.matches_per_candidate
            results = []
            for i in range(0, len(warm), RESCORE_BATCH):
                results += evaluator.evaluate(warm[i : i + RESCORE_BATCH])
            outcomes, evicted = tell_each(opt, archive, results)
            matches = evaluator.matches_total + (ladder.matches_total if ladder else 0)
            info = GenerationInfo(0, results, outcomes, evicted, matches, time.monotonic() - ctx.started_mono,
                                  era, per_candidate)
            for o in observers:
                o.on_generation(ctx, info)
            opt.generation = gen = 1
        while (generations is None or gen < generations) and (
            max_seconds is None or time.monotonic() - ctx.started_mono < max_seconds
        ):
            era = ladder.era if ladder else 0
            per_candidate = evaluator.matches_per_candidate
            cands = opt.ask()
            results = evaluator.evaluate(cands)
            outcomes, evicted = tell_each(opt, archive, results)
            challenge = rescored = None
            if ladder is not None and (gen + 1) % cfg.ladder_check_every == 0 and archive.best() is not None:
                challenge = ladder.challenge(workers, archive.best().genome, gen, cfg.max_ticks)
                if challenge is not None and challenge.promoted:
                    archive, rescored = rescore(archive, evaluator)
                    opt.archive = ctx.archive = archive
            matches = evaluator.matches_total + (ladder.matches_total if ladder else 0)
            info = GenerationInfo(gen, results, outcomes, evicted, matches,
                                  time.monotonic() - ctx.started_mono, era, per_candidate, challenge, rescored)
            for o in observers:
                o.on_generation(ctx, info)
            gen += 1

        for o in reversed(observers):  # like nested contexts: the exporter ends before the store closes (observers list the store first)
            o.on_end(ctx, info)
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the slime arena trainer.")
    ap.add_argument("--config", default="configs/slime-v0.toml")
    ap.add_argument("--generations", type=int)
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--no-store", action="store_true", help="skip the store and the exporter")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    observers: list[Observer] = []
    if not args.no_store:
        from .export import ExportObserver
        from .store import StoreObserver

        so = StoreObserver()
        observers = [so, ExportObserver(so, cfg.export_every_s, tree_full_every_s=cfg.tree_full_every_s)]
    ctx = run(cfg, observers, generations=args.generations, max_seconds=args.max_seconds)
    print(f"{ctx.run_id}: filled {ctx.archive.filled()}/{ctx.archive.capacity()}, "
          f"qd {ctx.archive.qd_score():.1f}")


if __name__ == "__main__":
    main()
