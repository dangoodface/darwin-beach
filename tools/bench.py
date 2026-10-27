"""Benchmark the kernel: full matches a second through Kernel.play_batch.

    uv run python tools/bench.py [--kernel PATH] [--seconds 10] [--procs 1,8,15]

Each process count runs two workloads, in batches of 100 jobs:

- training: fresh genomes (arena.genome.random_params) against the probe set
  the trainer uses, scripted-v1 and two frozen genomes, on both sides;
- two nets: genome against genome, the dearest tick there is (two MLPs, the
  scripted bot costs almost nothing). Fresh genomes lose points fast, so the
  last column divides ticks a second by 3,000: matches a second if every
  match ran to the cap, a floor for any population, however good it gets.

The gate (PLAN.md, K3) is 1,000 full matches a second on 15 processes. Run it
under the limits training uses:

    systemd-run --user --scope -p MemoryMax=4G -p CPUQuota=1400% \\
        uv run python tools/bench.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from arena.genome import random_params
from arena.kernel import Kernel, find_kernel

BATCH = 100
N_GENOMES = 256
GATE = 1000.0


def make_batches(kernel: Kernel, workload: str, rng: np.random.Generator) -> tuple[np.ndarray, list[list]]:
    table = np.stack([random_params(rng) for _ in range(N_GENOMES + 2)])
    bot = kernel.scripted_slot
    probes = [bot, N_GENOMES, N_GENOMES + 1]
    batches = []
    for _ in range(64):
        seeds = rng.integers(0, 2**64, size=BATCH, dtype=np.uint64).tolist()
        if workload == "two nets":
            jobs = [(s, int(rng.integers(N_GENOMES)), int(rng.integers(N_GENOMES))) for s in seeds]
        else:
            jobs = []
            for j, s in enumerate(seeds):
                g, p = int(rng.integers(N_GENOMES)), probes[j % 3]
                jobs.append((s, g, p) if j % 2 == 0 else (s, p, g))
        batches.append(jobs)
    return table, batches


def worker(path: str, workload: str, seconds: float, seed: int, barrier, queue) -> None:
    kernel = Kernel(path)
    table, batches = make_batches(kernel, workload, np.random.default_rng(seed))
    kernel.load_genomes(table)
    kernel.play_batch(batches[0])  # warm up
    barrier.wait()
    start, matches, ticks, i = time.perf_counter(), 0, 0, 0
    while (elapsed := time.perf_counter() - start) < seconds:
        results = kernel.play_batch(batches[i % len(batches)])
        assert not any(r.flags & 5 for r in results), "NaN or bad job in the benchmark"
        matches += len(results)
        ticks += sum(r.ticks for r in results)
        i += 1
    queue.put((matches, ticks, elapsed))


def measure(path: str, workload: str, procs: int, seconds: float) -> tuple[float, float, float]:
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(procs), ctx.Queue()
    workers = [ctx.Process(target=worker, args=(path, workload, seconds, 1000 + i, barrier, queue)) for i in range(procs)]
    for w in workers:
        w.start()
    parts = [queue.get() for _ in workers]
    for w in workers:
        w.join()
    matches_per_s = sum(m / e for m, _, e in parts)
    ticks_per_s = sum(t / e for _, t, e in parts)
    return matches_per_s, ticks_per_s, sum(t for _, t, _ in parts) / sum(m for m, _, _ in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kernel", default=None, help="wasm to benchmark (default: $ARENA_KERNEL or the cargo build)")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--procs", default="1,8,15")
    args = parser.parse_args()
    path = args.kernel or find_kernel()
    if path is None:
        raise SystemExit("no kernel: build it or pass --kernel")
    kernel = Kernel(path)
    print(f"kernel {kernel.kernel_sha} (version {kernel.kernel_version}), batches of {BATCH}, {args.seconds:g} s each")
    print(f"{'workload':10} {'procs':>5} {'matches/s':>10} {'ticks/s':>12} {'ticks/match':>11} {'at 3,000 ticks':>14}")
    at15 = {}
    for workload in ("training", "two nets"):
        for procs in (int(p) for p in args.procs.split(",")):
            m, t, mean = measure(str(path), workload, procs, args.seconds)
            print(f"{workload:10} {procs:5d} {m:10.0f} {t:12.0f} {mean:11.0f} {t / 3000:14.0f}", flush=True)
            if procs == 15:
                at15[workload] = (m, t / 3000)
    if "training" in at15:
        m = at15["training"][0]
        print(f"gate, 1,000 full matches a second on 15 processes: {'pass' if m >= GATE else 'FAIL'} ({m:.0f})")
    if "two nets" in at15:
        floor = at15["two nets"][1]
        print(f"floor, two nets for all 3,000 ticks on 15 processes: {'pass' if floor >= GATE else 'FAIL'} ({floor:.0f})")


if __name__ == "__main__":
    main()
