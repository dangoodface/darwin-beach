"""One TOML file describes a run. The resolved config (with the probe set and the
evaluation seeds written out) is what the store keeps, so a later re-score can
replay exactly the same matches."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    slug: str = "slime-v0"
    seed: int = 0
    sim_kind: str = "fake"
    kernel_path: str = ""
    max_ticks: int = 3000
    descriptors: tuple[str, ...] = ("court_position", "airtime")
    indices: tuple[int, ...] = (0, 1)
    bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    resolution: tuple[int, ...] = (20, 20)
    qd_offset: float = 5.0
    lit_floor: float = 0.0  # a cell is lit on the page when its elite's fitness reaches this
    batch_size: int = 64
    init_size: int = 100
    sigma: float = 0.1
    crossover_frac: float = 0.5
    optim_kind: str = "map_elites"  # or "cma_me" (pyribs, the qd extra)
    cma_emitters: int = 4  # cma_me: emitters, each proposing batch_size / emitters a generation
    cma_sigma0: float = 0.1  # cma_me: each emitter's starting step size
    cma_es: str = "cma_es"  # cma_me: pyribs' evolution strategy ("cma_es", "sep_cma_es", "lm_ma_es")
    workers: int = 15
    jobs_per_task: int = 128
    scripted_probe: bool = True  # the kernel's scripted bot is probe 0
    probe_count: int = 3
    seeds_per_probe: int = 4
    eval_seeds: tuple[int, ...] = field(default_factory=tuple)  # empty: drawn from the run seed
    export_every_s: float = 60.0
    tree_full_every_s: float = 3600.0  # tree-full.json grows without bound: at the end, and at most this often
    # v1, the honest ladder (src/arena/ladder.py); off keeps the v0 behaviour
    ladder_enabled: bool = False
    ladder_check_every: int = 10  # generations between promotion checks
    ladder_seeds: int = 50  # promotion and champion ratings: this many seeds, each on both sides
    ladder_threshold: float = 0.55  # share of the head-to-head a challenger must win
    ladder_cycle_window: int = 5  # a new champion's results against this many earlier members are kept as the cycle check
    ladder_seed_list: tuple[int, ...] = field(default_factory=tuple)  # empty: drawn from the run seed
    # run 6: a pool of members present from era 0 ({path, label, id} each), rated once by a round robin
    ladder_pool: tuple[dict, ...] = field(default_factory=tuple)
    ladder_hold_anchor: bool = False  # a promotion must also win the threshold against the anchor
    ladder_weight_scale: float = 0.0  # fitness weights a member by 10 ** ((rating - top) / scale); 0: all equal
    ladder_fitness: str = "weighted"  # 28 October: "weighted" (the mean above) or "softmin" (the worst matchup improves)
    ladder_softmin_top: int = 8  # softmin: over the mean margins against this many highest-rated members
    ladder_softmin_temp: float = 0.5  # softmin: temperature in points of margin
    ladder_coach_gate: bool = False  # a candidate that loses to the coach on average scores that margin minus ten
    ladder_max_ticks: int = 0  # ladder matches (pool rating, promotion checks, newcomers) at this length; 0: max_ticks
    inbox_every: int = 0  # every this many generations, genomes dropped in runs/<id>/inbox join the ladder; 0: off
    warm_runs: tuple[str, ...] = field(default_factory=tuple)  # exports (run ids or paths) whose elites seed the archive

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        sim, grid, opt = d.get("sim", {}), d.get("grid", {}), d.get("optim", {})
        wk, ev, ex = d.get("workers", {}), d.get("eval", {}), d.get("export", {})
        ld = d.get("ladder", {})
        base = cls()
        cfg = cls(
            slug=d.get("slug", base.slug),
            seed=int(d.get("seed", base.seed)),
            sim_kind=sim.get("kind", base.sim_kind),
            kernel_path=sim.get("kernel_path", base.kernel_path),
            max_ticks=int(sim.get("max_ticks", base.max_ticks)),
            descriptors=tuple(grid.get("descriptors", base.descriptors)),
            indices=tuple(int(i) for i in grid.get("indices", base.indices)),
            bounds=tuple((float(lo), float(hi)) for lo, hi in grid.get("bounds", base.bounds)),
            resolution=tuple(int(r) for r in grid.get("resolution", base.resolution)),
            qd_offset=float(grid.get("qd_offset", base.qd_offset)),
            lit_floor=float(grid.get("lit_floor", base.lit_floor)),
            batch_size=int(opt.get("batch_size", base.batch_size)),
            init_size=int(opt.get("init_size", base.init_size)),
            sigma=float(opt.get("sigma", base.sigma)),
            crossover_frac=float(opt.get("crossover_frac", base.crossover_frac)),
            optim_kind=opt.get("kind", base.optim_kind),
            cma_emitters=int(opt.get("emitters", base.cma_emitters)),
            cma_sigma0=float(opt.get("sigma0", base.cma_sigma0)),
            cma_es=opt.get("es", base.cma_es),
            workers=int(wk.get("count", base.workers)),
            jobs_per_task=int(wk.get("jobs_per_task", base.jobs_per_task)),
            scripted_probe=bool(ev.get("scripted_probe", base.scripted_probe)),
            probe_count=int(ev.get("probe_count", base.probe_count)),
            seeds_per_probe=int(ev.get("seeds_per_probe", base.seeds_per_probe)),
            eval_seeds=tuple(int(s) for s in ev.get("eval_seeds", ())),
            export_every_s=float(ex.get("every_s", base.export_every_s)),
            tree_full_every_s=float(ex.get("tree_full_every_s", base.tree_full_every_s)),
            ladder_enabled=bool(ld.get("enabled", base.ladder_enabled)),
            ladder_check_every=int(ld.get("check_every", base.ladder_check_every)),
            ladder_seeds=int(ld.get("seeds", base.ladder_seeds)),
            ladder_threshold=float(ld.get("threshold", base.ladder_threshold)),
            ladder_cycle_window=int(ld.get("cycle_window", base.ladder_cycle_window)),
            ladder_seed_list=tuple(int(s) for s in ld.get("seed_list", ())),
            ladder_pool=tuple(dict(p) for p in ld.get("pool", ())),
            ladder_hold_anchor=bool(ld.get("hold_anchor", base.ladder_hold_anchor)),
            ladder_weight_scale=float(ld.get("weight_scale", base.ladder_weight_scale)),
            ladder_fitness=str(ld.get("fitness", base.ladder_fitness)),
            ladder_softmin_top=int(ld.get("softmin_top", base.ladder_softmin_top)),
            ladder_softmin_temp=float(ld.get("softmin_temp", base.ladder_softmin_temp)),
            ladder_coach_gate=bool(ld.get("coach_gate", base.ladder_coach_gate)),
            ladder_max_ticks=int(ld.get("max_ticks", base.ladder_max_ticks)),
            inbox_every=int(ld.get("inbox_every", base.inbox_every)),
            warm_runs=tuple(str(r) for r in d.get("warm", {}).get("runs", ())),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        n = len(self.indices)
        if not (len(self.bounds) == len(self.resolution) == len(self.descriptors) == n):
            raise ValueError("descriptors, indices, bounds and resolution must have the same length")
        if self.sim_kind not in ("fake", "kernel"):
            raise ValueError(f"sim.kind must be 'fake' or 'kernel', got {self.sim_kind!r}")
        if self.sim_kind == "kernel" and not self.kernel_path:
            raise ValueError("sim.kernel_path is required when sim.kind = 'kernel'")
        if not 1 <= self.jobs_per_task <= 256:
            raise ValueError("workers.jobs_per_task must be 1..256 (the kernel's batch_capacity)")
        if self.probe_count + int(self.scripted_probe) < 1 or self.seeds_per_probe < 1 or self.batch_size < 1:
            raise ValueError("the probe set must be non-empty, and seeds_per_probe and batch_size at least 1")
        if self.ladder_check_every < 1 or self.ladder_seeds < 1 or self.ladder_cycle_window < 1:
            raise ValueError("ladder.check_every, ladder.seeds and ladder.cycle_window must be at least 1")
        if self.optim_kind not in ("map_elites", "cma_me"):
            raise ValueError(f"optim.kind must be 'map_elites' or 'cma_me', got {self.optim_kind!r}")
        if self.optim_kind == "cma_me" and (self.cma_emitters < 1 or self.batch_size % self.cma_emitters):
            raise ValueError("optim.batch_size must split evenly over optim.emitters")
        if not 0.5 <= self.ladder_threshold <= 1.0:
            raise ValueError("ladder.threshold must be between 0.5 and 1")
        if self.ladder_fitness not in ("weighted", "softmin"):
            raise ValueError("ladder.fitness must be weighted or softmin")
        if self.ladder_softmin_top < 1 or self.ladder_softmin_temp <= 0 or self.ladder_max_ticks < 0 or self.inbox_every < 0:
            raise ValueError("ladder.softmin_top, softmin_temp, max_ticks and inbox_every must be positive (max_ticks and inbox_every may be 0)")
        if self.ladder_weight_scale < 0:
            raise ValueError("ladder.weight_scale must be 0 (equal weights) or positive")
        if self.ladder_pool and not self.ladder_enabled:
            raise ValueError("ladder.pool needs ladder.enabled")
        if any("path" not in p for p in self.ladder_pool):
            raise ValueError("every ladder.pool entry needs a path")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the store; u64 seeds travel as decimal strings."""
        return {
            "slug": self.slug, "seed": str(self.seed),
            "sim": {"kind": self.sim_kind, "kernel_path": self.kernel_path, "max_ticks": self.max_ticks},
            "grid": {"descriptors": list(self.descriptors), "indices": list(self.indices),
                     "bounds": [list(b) for b in self.bounds], "resolution": list(self.resolution),
                     "qd_offset": self.qd_offset, "lit_floor": self.lit_floor},
            "optim": {"batch_size": self.batch_size, "init_size": self.init_size,
                      "sigma": self.sigma, "crossover_frac": self.crossover_frac, "kind": self.optim_kind,
                      "emitters": self.cma_emitters, "sigma0": self.cma_sigma0, "es": self.cma_es},
            "workers": {"count": self.workers, "jobs_per_task": self.jobs_per_task},
            "eval": {"scripted_probe": self.scripted_probe, "probe_count": self.probe_count, "seeds_per_probe": self.seeds_per_probe,
                     "eval_seeds": [str(s) for s in self.eval_seeds]},
            "export": {"every_s": self.export_every_s, "tree_full_every_s": self.tree_full_every_s},
            "ladder": {"enabled": self.ladder_enabled, "check_every": self.ladder_check_every,
                       "fitness": self.ladder_fitness, "softmin_top": self.ladder_softmin_top,
                       "softmin_temp": self.ladder_softmin_temp, "coach_gate": self.ladder_coach_gate,
                       "max_ticks": self.ladder_max_ticks, "inbox_every": self.inbox_every,
                       "seeds": self.ladder_seeds, "threshold": self.ladder_threshold,
                       "cycle_window": self.ladder_cycle_window,
                       "seed_list": [str(s) for s in self.ladder_seed_list],
                       "pool": [dict(p) for p in self.ladder_pool], "hold_anchor": self.ladder_hold_anchor,
                       "weight_scale": self.ladder_weight_scale},
            "warm": {"runs": list(self.warm_runs)},
        }


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        return Config.from_dict(tomllib.load(f))
