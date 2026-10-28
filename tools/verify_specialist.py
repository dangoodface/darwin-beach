"""Replay a specialist's fresh record from its own files and check it, match for match.

    uv run python tools/verify_specialist.py <specialist run dir> [--hold N]

The run's result.json names the final genome, the rivals and the fresh seeds; the run's
folder keeps the genomes and the kernel it was played with. Every fresh match is played
again, both sides, the specialist at its hold (result.json's "hold", 1 when absent, or
--hold) and the rivals at 1, and won, n and margin per rival must equal result.json's
final.fresh exactly. Exit 0 when they all do, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from arena.genome import pad_params
from arena.kernel import Kernel


def fresh_record(run: Path, hold: int | None = None, max_ticks: int = 3000) -> tuple[dict, dict, int]:
    """(recomputed, stored, hold used): per rival {won, n, margin}."""
    r = json.loads((run / "result.json").read_text())
    hold = hold if hold is not None else int(r.get("hold", 1))
    kernel = Kernel(run / "kernel" / f"{r['kernel_sha']}.wasm")

    def genome(gid: str) -> np.ndarray:
        p = np.fromfile(run / "genomes" / f"{gid}.bin", dtype="<f4")
        return pad_params(p) if len(p) < kernel.genome_len else p

    rivals = [x["id"] for x in r["rivals"]]
    table = [genome(r["final"]["genome_id"])] + [genome(rid) for rid in rivals if rid != "scripted-v1"]
    kernel.load_genomes(np.stack(table))
    slots, k = [], 1
    for rid in rivals:
        if rid == "scripted-v1":
            slots.append(kernel.scripted_slot)
        else:
            slots.append(k)
            k += 1
    seeds = [int(s) for s in r["seeds"]["fresh"]]
    out = {}
    for rid, opp in zip(rivals, slots):
        jobs = []
        for s in seeds:  # the specialist on the left, then on the right
            jobs.append((s, 0, opp, hold, 1))
            jobs.append((s, opp, 0, 1, hold))
        res = kernel.play_batch(jobs, max_ticks)
        mine = np.array([m.points_a if i % 2 == 0 else m.points_b for i, m in enumerate(res)], dtype=float)
        theirs = np.array([m.points_b if i % 2 == 0 else m.points_a for i, m in enumerate(res)], dtype=float)
        out[rid] = {"won": int((mine > theirs).sum()), "n": int(mine.size), "margin": float((mine - theirs).mean())}
    return out, r["final"]["fresh"], hold


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a specialist's fresh record and compare it with result.json.")
    ap.add_argument("run", type=Path)
    ap.add_argument("--hold", type=int, help="the specialist's hold (default: result.json's, else 1)")
    ap.add_argument("--max-ticks", type=int, default=3000)
    args = ap.parse_args(argv)
    got, stored, hold = fresh_record(args.run, args.hold, args.max_ticks)
    same = got == stored
    print(f"{args.run.name}: specialist at hold {hold}, rivals at 1, {len(got)} rivals")
    for rid in stored:
        mark = "same" if got.get(rid) == stored[rid] else "DIFFERENT"
        print(f"  {rid[:16]:16}  stored {stored[rid]}  replayed {got.get(rid)}  {mark}")
    print("fresh record reproduced" if same else "fresh record NOT reproduced")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
