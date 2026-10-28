"""Rate the oracle's rungs (exploration, plan B4) against the official board's frozen members.

    uv run python tools/rate_oracle.py <oracle build .wasm> [--rungs 0,2,4] [--workers 6] [--seeds 64]

Each rung plays every member and the coach on the official serves, both sides, at the board's
max_ticks, through the oracle build: kernel version 3 plus the oracle's slots, whose members play
exactly as on the official kernel (checked first on a sample of member matches, hash for hash).
A rung's rating is the Elo that best explains its record with the members' frozen ratings, the
way `arena.official.rate` rates a genome. Results go to $ARENA_DATA/official/oracle/ (the build,
one file per rung, ratings.json), never to board.jsonl: the oracle is not a member.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from arena.export import _write_json
from arena.kernel import Kernel
from arena.ladder import elo_mle
from arena.official import MEASURABLE_BELOW, MEASURABLE_MIN, load_ladder, official_dir
from arena.pool import read_params
from arena.workers import Workers


def same_play(oracle: Kernel, official: Kernel, table: np.ndarray, seeds: list[int], n: int = 48) -> int:
    """Member and coach matches on both builds: how many of `n` differ in hash, ticks or points."""
    oracle.load_genomes(table)
    official.load_genomes(table)
    rng = np.random.default_rng(20261028)
    jobs = []
    for _ in range(n):
        a, b = (int(x) for x in rng.integers(-1, len(table), size=2))
        jobs.append((seeds[int(rng.integers(len(seeds)))], official.scripted_slot if a < 0 else a,
                     official.scripted_slot if b < 0 else b))
    mine, theirs = oracle.play_batch(jobs, 6000), official.play_batch(jobs, 6000)
    return sum(m[:4] != t[:4] for m, t in zip(mine, theirs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rate the oracle's rungs against the official board.")
    ap.add_argument("wasm", type=Path, help="an oracle build (kernel/target/.../slime_kernel.wasm)")
    ap.add_argument("--rungs", help="comma-separated rung numbers (default: all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=64, help="the first N official serves (64: all)")
    args = ap.parse_args(argv)

    doc = load_ladder()
    seeds = [int(s) for s in doc["seeds"]][: args.seeds]
    out = official_dir() / "oracle"
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(args.wasm.read_bytes()).hexdigest()
    build = out / f"{sha}.wasm"
    if not build.exists():
        shutil.copyfile(args.wasm, build)

    oracle = Kernel(build)
    if not oracle.oracle_rungs:
        sys.exit(f"{args.wasm} has no oracle slots")
    table = np.stack([read_params(Path(m["path"]), oracle.genome_len) for m in doc["members"]]).astype(np.float32)
    differ = same_play(oracle, Kernel(doc["kernel_path"]), table, seeds)
    print(f"oracle build {sha[:12]} (kernel version {oracle.kernel_version}, {oracle.oracle_rungs} rungs); "
          f"members play as on the official kernel {doc['kernel_sha'][:12]}: {48 - differ} of 48 sample matches identical")
    if differ:
        sys.exit("the oracle build does not play the members like the official kernel")

    rungs = [int(r) for r in args.rungs.split(",")] if args.rungs else list(range(oracle.oracle_rungs))
    opponents = [(m["label"], float(m["rating"]), i) for i, m in enumerate(doc["members"])]
    opponents.insert(0, ("the coach", float(doc["anchor_rating"]), oracle.scripted_slot))
    jobs = []
    for r in rungs:
        me = oracle.oracle_slot_base + r
        for _, _, slot in opponents:
            jobs += [job for s in seeds for job in ((s, me, slot), (s, slot, me))]
    with Workers("kernel", str(build), args.workers, 32) as workers:
        results = workers.play(table, jobs, doc["max_ticks"])

    summary = json.loads((out / "ratings.json").read_text()) if (out / "ratings.json").exists() else {}
    k = 0
    for r in rungs:
        rec, records = {}, []
        for label, rating, _ in opponents:
            block = results[k : k + 2 * len(seeds)]
            k += 2 * len(seeds)
            mine = np.array([m.points_a if i % 2 == 0 else m.points_b for i, m in enumerate(block)])
            theirs = np.array([m.points_b if i % 2 == 0 else m.points_a for i, m in enumerate(block)])
            won, lost = int((mine > theirs).sum()), int((mine < theirs).sum())
            tied = len(block) - won - lost
            rec[label] = {"rating": rating, "won": won, "lost": lost, "tied": tied, "n": len(block),
                          "score": (won + 0.5 * tied) / len(block), "margin": float((mine - theirs).mean())}
            records.append((rating, won + 0.5 * tied, len(block)))
        under = sum(1 for x in rec.values() if x["score"] < MEASURABLE_BELOW)
        delay, noise = oracle.oracle_rung(r)
        res = {"rung": r, "delay": delay, "noise": noise, "slot": oracle.oracle_slot_base + r, "kernel_sha": sha,
               "rating": elo_mle(records), "measurable": under >= MEASURABLE_MIN, "members_under_90": under,
               "beaten": sum(1 for x in rec.values() if x["won"] > x["lost"]), "of": len(rec),
               "seeds": len(seeds), "max_ticks": doc["max_ticks"], "record": rec}
        _write_json(out / f"rung-{r}-{sha[:8]}.json", res)
        summary[f"{sha[:8]}/{r}"] = {key: v for key, v in res.items() if key != "record"}
        worst = min(rec.items(), key=lambda kv: kv[1]["score"])
        print(f"rung {r}: {res['rating']:.0f} {'measurable' if res['measurable'] else 'ABOVE THE LADDER'}, "
              f"beats {res['beaten']} of {res['of']}, {under} under 90%, worst {worst[0]} {worst[1]['score']:.2f}")
    _write_json(out / "ratings.json", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
