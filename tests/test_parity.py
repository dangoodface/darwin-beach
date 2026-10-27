"""Parity: the same wasm bytes play the same match in wasmtime and in Node's V8.

1,000 seeded matches, every per-tick hash compared, which is the v0
acceptance check behind `make parity`. Genomes come from a fixed numpy seed
at three scales (near idle, twitchy, wild), the seeds cover the whole u64
range (so the signed crossing at 2^63 is exercised on both sides), and a
tenth of the matches are the scripted bot against itself, whose long rallies
run to the 3,000-tick cap.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from arena.kernel import FLAG_BAD_JOB, FLAG_MAX_TICKS, FLAG_NAN, JOB_DTYPE, ROOT, Kernel, find_kernel

N_MATCHES = 1000
MAX_TICKS = 3000
SEED = 20261026
PARITY_JS = ROOT / "tools" / "parity.mjs"

KERNEL = find_kernel()
pytestmark = [
    pytest.mark.skipif(KERNEL is None, reason="no kernel: build it (README) or set ARENA_KERNEL"),
    pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH"),
]


def make_matches(kernel: Kernel) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    rng = np.random.default_rng(SEED)
    scales = np.repeat([0.1, 1.0, 3.0], [200, 400, 400])
    genomes = (rng.standard_normal((len(scales), kernel.genome_len)) * scales[:, None]).astype(np.float32)
    seeds = rng.integers(0, 2**64, size=N_MATCHES, dtype=np.uint64, endpoint=False).tolist()
    n, bot = len(genomes), kernel.scripted_slot
    jobs = []
    for i, seed in enumerate(seeds):
        kind = i % 10
        if kind < 7:
            a, b = (2 * i) % n, (2 * i + 1) % n
        elif kind == 7:
            a, b = i % n, bot
        elif kind == 8:
            a, b = bot, i % n
        else:
            a, b = bot, bot
        jobs.append((seed, a, b))
    return genomes, jobs


def run_node(genomes: np.ndarray, jobs: list[tuple[int, int, int]]) -> list[tuple[int, int, int, np.ndarray]]:
    head = np.array([len(genomes), len(jobs), MAX_TICKS], dtype="<u4")
    packed = np.empty(len(jobs), dtype=JOB_DTYPE)
    packed["seed"] = [s for s, _, _ in jobs]
    packed["a"] = [a for _, a, _ in jobs]
    packed["b"] = [b for _, _, b in jobs]
    payload = head.tobytes() + np.ascontiguousarray(genomes, dtype="<f4").tobytes() + packed.tobytes()
    proc = subprocess.run(["node", str(PARITY_JS), str(KERNEL)], input=payload, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr.decode()
    out, at, matches = proc.stdout, 0, []
    for _ in jobs:
        ticks, flags = np.frombuffer(out, dtype="<u4", count=2, offset=at).tolist()
        final = int(np.frombuffer(out, dtype="<u8", count=1, offset=at + 8)[0])
        hashes = np.frombuffer(out, dtype="<u8", count=ticks, offset=at + 16)
        matches.append((ticks, flags, final, hashes))
        at += 16 + 8 * ticks
    assert at == len(out), "node wrote more than the matches"
    return matches


def test_wasmtime_and_node_agree_tick_by_tick():
    kernel = Kernel(KERNEL)
    genomes, jobs = make_matches(kernel)
    kernel.load_genomes(genomes)
    node = run_node(genomes, jobs)

    mismatches, ended_on_cap, ended_on_lives = [], 0, 0
    for j, ((seed, a, b), (n_ticks, n_flags, n_final, n_hashes)) in enumerate(zip(jobs, node)):
        r = kernel.play(seed, a, b, MAX_TICKS, record_hashes=True)
        assert r.flags & (FLAG_NAN | FLAG_BAD_JOB) == 0, f"match {j}: flags {r.flags}"
        ended_on_cap += bool(r.flags & FLAG_MAX_TICKS)
        ended_on_lives += 5 in (r.points_a, r.points_b)
        ours = np.array(r.tick_hashes, dtype="<u8")
        if (r.ticks, r.flags, r.final_hash) == (n_ticks, n_flags, n_final) and np.array_equal(ours, n_hashes):
            continue
        common = min(len(ours), len(n_hashes))
        differ = np.nonzero(ours[:common] != n_hashes[:common])[0]
        first = int(differ[0]) + 1 if len(differ) else common + 1
        mismatches.append(
            f"match {j} (seed {seed}, slots {a} v {b}): first differing tick {first}; "
            f"wasmtime {r.ticks} ticks flags {r.flags}, node {n_ticks} ticks flags {n_flags}"
        )
    assert not mismatches, "\n".join(mismatches[:10])
    assert ended_on_cap > 0 and ended_on_lives > 0, (ended_on_cap, ended_on_lives)
