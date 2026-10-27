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

from .archive import Archive
from .config import Config, load_config
from .evaluate import Evaluator
from .genome import Genome, make_genome, random_params
from .optim import EvalResult, MapElites
from .workers import Workers

# Spawn order of the root SeedSequence. Appending is safe; reordering changes every run.
STREAM_PROBES, STREAM_OPTIM, STREAM_SEEDS = 0, 1, 2


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


@dataclass
class GenerationInfo:
    generation: int
    results: list[EvalResult]
    outcomes: list[str]  # "new" | "replaced" | "rejected", aligned with results
    evicted: list[str | None]  # genome id displaced by each "replaced", else None
    matches_total: int
    elapsed_s: float


class Observer(Protocol):
    def on_start(self, ctx: RunContext) -> None: ...
    def on_generation(self, ctx: RunContext, info: GenerationInfo) -> None: ...
    def on_end(self, ctx: RunContext, info: GenerationInfo | None) -> None: ...


def make_run_id(slug: str, now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{slug}"


def run(
    cfg: Config,
    observers: list[Observer] | tuple[()] = (),
    generations: int | None = None,
    max_seconds: float | None = None,
    now: datetime | None = None,
) -> RunContext:
    root = np.random.SeedSequence(cfg.seed)
    probe_ss, optim_ss, seed_ss = root.spawn(3)

    with Workers(cfg.sim_kind, cfg.kernel_path, cfg.workers, cfg.jobs_per_task) as workers:
        kernel_sha, kernel_version, genome_len, scripted_slot = workers.simulator_info
        if cfg.scripted_probe and scripted_slot is None:
            raise ValueError("eval.scripted_probe is set but the simulator has no scripted bot")
        scripted_slot = scripted_slot if cfg.scripted_probe else None

        probe_rng = np.random.default_rng(probe_ss)
        probes = [
            make_genome(random_params(probe_rng, genome_len), kernel_sha, (), "probe", -1)
            for _ in range(cfg.probe_count)
        ]
        seeds = cfg.eval_seeds or tuple(
            int(s) for s in seed_ss.generate_state(cfg.seeds_per_probe, dtype=np.uint64)
        )
        resolved = Config.from_dict({**cfg.to_dict(), "seed": cfg.seed, "eval": {
            "scripted_probe": cfg.scripted_probe, "probe_count": cfg.probe_count, "seeds_per_probe": len(seeds), "eval_seeds": list(seeds)}})

        archive = Archive(cfg.bounds, cfg.resolution, cfg.qd_offset)
        opt = MapElites(archive, np.random.default_rng(optim_ss), kernel_sha, genome_len,
                        cfg.batch_size, cfg.init_size, cfg.sigma, cfg.crossover_frac)
        evaluator = Evaluator(workers, probes, scripted_slot, seeds, cfg.indices, cfg.max_ticks)

        started = now or datetime.now(timezone.utc)
        ctx = RunContext(make_run_id(cfg.slug, started), resolved, kernel_sha, kernel_version,
                         genome_len, probes, [o for o, _ in evaluator.opponents], archive, started)
        for o in observers:
            o.on_start(ctx)

        info: GenerationInfo | None = None
        gen = 0
        while (generations is None or gen < generations) and (
            max_seconds is None or time.monotonic() - ctx.started_mono < max_seconds
        ):
            cands = opt.ask()
            results = evaluator.evaluate(cands)
            outcomes = []
            evicted: list[str | None] = []
            # tell one result at a time so each eviction is attributed to the right newcomer
            for r in results:
                cell = archive.cell_of(r.bd)
                resident = archive.get(cell)
                outcomes.extend(opt.tell([r]))
                evicted.append(resident.genome.id if resident and outcomes[-1] == "replaced" else None)
            info = GenerationInfo(gen, results, outcomes, evicted, evaluator.matches_total,
                                  time.monotonic() - ctx.started_mono)
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
        observers = [so, ExportObserver(so, cfg.export_every_s)]
    ctx = run(cfg, observers, generations=args.generations, max_seconds=args.max_seconds)
    print(f"{ctx.run_id}: filled {ctx.archive.filled()}/{ctx.archive.capacity()}, "
          f"qd {ctx.archive.qd_score():.1f}")


if __name__ == "__main__":
    main()
