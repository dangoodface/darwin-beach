"""Rollout throughput: ticks a second for 1, 8 and 12 processes, against the coach and an elite.

    python -m species.ppo.bench --kernel K.wasm [--elite GENOME.bin] [--matches 6]

Run it under systemd-run with the CPU quota you will train with: the numbers are
only as parallel as the quota allows.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from arena.genome import random_params
from arena.kernel import find_kernel

from .rollout import COACH, RolloutPool


def measure(kernel: Path, procs: int, opponent, matches: int, params: np.ndarray) -> tuple[int, float]:
    with RolloutPool(kernel, procs) as pool:
        pool.run([(params, opponent, 1, 0, 3000, 0)] * procs)  # warm-up: spawn, import, first match
        jobs = [(params, opponent, 100 + i, i % 2, 3000, i) for i in range(matches * procs)]
        t0 = time.perf_counter()
        eps = pool.run(jobs)
        wall = time.perf_counter() - t0
    return sum(e.ticks for e in eps), wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", type=Path, default=find_kernel())
    ap.add_argument("--elite", type=Path, help="a genome .bin to use as the opponent")
    ap.add_argument("--matches", type=int, default=4, help="matches per process")
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 8, 12])
    args = ap.parse_args()
    params = random_params(np.random.default_rng(0))
    opponents = {"coach": COACH}
    if args.elite:
        opponents["elite"] = np.fromfile(args.elite, dtype="<f4")
    print(f"cpus visible {os.cpu_count()}, loadavg {os.getloadavg()[0]:.1f}")
    for name, opp in opponents.items():
        for n in args.procs:
            ticks, wall = measure(args.kernel, n, opp, args.matches, params)
            print(f"vs {name:<5} {n:>2} processes: {ticks / wall:>9,.0f} ticks/s total, "
                  f"{ticks / wall / n:>8,.0f} per process ({ticks:,} ticks in {wall:.1f}s)")


if __name__ == "__main__":
    main()
