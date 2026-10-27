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
    batch_size: int = 64
    init_size: int = 100
    sigma: float = 0.1
    crossover_frac: float = 0.5
    workers: int = 15
    jobs_per_task: int = 128
    scripted_probe: bool = True  # the kernel's scripted bot is probe 0
    probe_count: int = 3
    seeds_per_probe: int = 4
    eval_seeds: tuple[int, ...] = field(default_factory=tuple)  # empty: drawn from the run seed
    export_every_s: float = 60.0
    tree_full_every_s: float = 3600.0  # tree-full.json grows without bound: at the end, and at most this often

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        sim, grid, opt = d.get("sim", {}), d.get("grid", {}), d.get("optim", {})
        wk, ev, ex = d.get("workers", {}), d.get("eval", {}), d.get("export", {})
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
            batch_size=int(opt.get("batch_size", base.batch_size)),
            init_size=int(opt.get("init_size", base.init_size)),
            sigma=float(opt.get("sigma", base.sigma)),
            crossover_frac=float(opt.get("crossover_frac", base.crossover_frac)),
            workers=int(wk.get("count", base.workers)),
            jobs_per_task=int(wk.get("jobs_per_task", base.jobs_per_task)),
            scripted_probe=bool(ev.get("scripted_probe", base.scripted_probe)),
            probe_count=int(ev.get("probe_count", base.probe_count)),
            seeds_per_probe=int(ev.get("seeds_per_probe", base.seeds_per_probe)),
            eval_seeds=tuple(int(s) for s in ev.get("eval_seeds", ())),
            export_every_s=float(ex.get("every_s", base.export_every_s)),
            tree_full_every_s=float(ex.get("tree_full_every_s", base.tree_full_every_s)),
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

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the store; u64 seeds travel as decimal strings."""
        return {
            "slug": self.slug, "seed": str(self.seed),
            "sim": {"kind": self.sim_kind, "kernel_path": self.kernel_path, "max_ticks": self.max_ticks},
            "grid": {"descriptors": list(self.descriptors), "indices": list(self.indices),
                     "bounds": [list(b) for b in self.bounds], "resolution": list(self.resolution),
                     "qd_offset": self.qd_offset},
            "optim": {"batch_size": self.batch_size, "init_size": self.init_size,
                      "sigma": self.sigma, "crossover_frac": self.crossover_frac},
            "workers": {"count": self.workers, "jobs_per_task": self.jobs_per_task},
            "eval": {"scripted_probe": self.scripted_probe, "probe_count": self.probe_count, "seeds_per_probe": self.seeds_per_probe,
                     "eval_seeds": [str(s) for s in self.eval_seeds]},
            "export": {"every_s": self.export_every_s, "tree_full_every_s": self.tree_full_every_s},
        }


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        return Config.from_dict(tomllib.load(f))
