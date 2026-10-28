"""The definitive verdict (run 6): a run's members, old and new, in a round robin
on held-out serves.

Usage: python -m arena.definitive <run_id> [--seeds 16] [--workers 15]

Writes out/<run_id>/definitive.json: every member of the run's ladder rated
jointly on serves drawn from a stream the run never used (the run spawns
streams 0 to 4 of its seed; this is SeedSequence([seed, 6])), ranked; the
champion's record against each other member on those serves; and the full
win matrix. A bot that beats this pool beats this pool, not every possible bot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .export import _write_json
from .genome import make_genome
from .ladder import Ladder, Member, Pair
from .pool import read_params
from .sim import SCRIPTED_ID
from .store import Store, data_root, store_path
from .workers import Workers

VERDICT_STREAM = 6


def load_members(out: Path, genome_len: int, kernel_sha: str) -> tuple[list[Member], dict]:
    lad = json.loads((out / "ladder.json").read_text())
    members = []
    for m in lad["members"]:
        if m["kind"] == "scripted":
            continue
        g = make_genome(read_params(out / "genomes" / f"{m['id']}.bin", genome_len), kernel_sha, (), m["kind"], -1)
        if g.id != m["id"]:
            raise ValueError(f"{m['id']} does not hash to itself under kernel {kernel_sha[:8]}")
        members.append(Member(g.id, m["kind"], m["rating"], m["since_era"], g, label=m.get("label")))
    return members, lad


def matrix(ids: list[str], pairs: tuple[Pair, ...]) -> list[list[float | None]]:
    index = {mid: k for k, mid in enumerate(ids)}
    win: list[list[float | None]] = [[None] * len(ids) for _ in ids]
    for p in pairs:
        a, b = index[p.a], index[p.b]
        win[a][b], win[b][a] = p.winrate_a, 1.0 - p.winrate_a
    return win


def verdict(run_id: str, n_seeds: int = 16, workers_n: int = 15) -> dict:
    store = Store(store_path(run_id))
    run = store.run_row(run_id)
    store.close()
    cfg = json.loads(run["config_json"])
    out = data_root() / "out" / run_id
    members, lad = load_members(out, run["genome_len"], run["kernel_sha"])
    kernel = out / "kernel" / f"{run['kernel_sha']}.wasm"
    kernel_path = str(kernel) if kernel.exists() else cfg["sim"]["kernel_path"]
    seeds = tuple(int(s) for s in np.random.SeedSequence([int(cfg["seed"]), VERDICT_STREAM])
                  .generate_state(n_seeds, dtype=np.uint64))
    with Workers(cfg["sim"]["kind"], kernel_path, workers_n, cfg["workers"]["jobs_per_task"]) as workers:
        kernel_sha, _, _, scripted = workers.simulator_info
        if kernel_sha != run["kernel_sha"]:
            raise ValueError(f"kernel {kernel_sha[:8]} is not the run's {run['kernel_sha'][:8]}")
        ladder = Ladder(scripted, seeds, pool=members)
        pairs = ladder.rate_pool(workers, cfg["sim"]["max_ticks"])
    rated = ladder.members
    champion = lad.get("champion") or max(rated, key=lambda m: m.rating).id
    run_rating = {m["id"]: m["rating"] for m in lad["members"]}
    label = {m.id: m.label or (m.kind if m.kind != "scripted" else "the coach") for m in rated}
    fresh = {}
    for p in pairs:
        if champion not in (p.a, p.b):
            continue
        other = p.b if p.a == champion else p.a
        won = p.wins_a if p.a == champion else p.wins_b
        margin = p.margin_a if p.a == champion else -p.margin_a
        fresh[other] = {"label": label[other], "kind": next(m.kind for m in rated if m.id == other),
                        "won": won, "n": p.n, "margin": margin}
    board = sorted(rated, key=lambda m: -m.rating)
    result = {
        "run_id": run_id, "kernel_sha": run["kernel_sha"], "seeds": [str(s) for s in seeds],
        "matches_per_pair": 2 * len(seeds), "champion": champion,
        "members_beaten": sum(1 for f in fresh.values() if f["won"] > f["n"] / 2), "of": len(fresh),
        "fresh": fresh,
        "leaderboard": [{"rank": k + 1, "id": m.id, "label": label[m.id], "kind": m.kind, "rating": m.rating,
                         "run_rating": run_rating.get(m.id)} for k, m in enumerate(board)],
        "matrix": {"ids": [m.id for m in rated], "winrate": matrix([m.id for m in rated], pairs)},
    }
    _write_json(out / "definitive.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Rate a run's ladder on held-out serves.")
    ap.add_argument("run_id")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--workers", type=int, default=15)
    args = ap.parse_args()
    r = verdict(args.run_id, args.seeds, args.workers)
    print(f"{r['run_id']}: champion {r['champion'][:8]} beats {r['members_beaten']} of {r['of']} on {len(r['seeds'])} held-out serves")
    for e in r["leaderboard"]:
        print(f"  {e['rank']:>2} {e['rating']:7.1f} {e['label']}")


if __name__ == "__main__":
    main()
