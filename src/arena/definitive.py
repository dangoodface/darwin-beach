"""The definitive verdict (run 6): a run's members, old and new, in a round robin
on held-out serves.

Usage: python -m arena.definitive <run_id> [--seeds 16] [--elites 6] [--workers 15]

Writes out/<run_id>/definitive.json: every member of the run's ladder, plus
the run's top `--elites` live cells by archive rating (kind "elite", so a run
that promoted nobody still gets its verdict), rated jointly on serves drawn
from a stream the run never used (the run spawns streams 0 to 4 of its seed;
this is SeedSequence([seed, 6])), ranked; the run champion's and the held-out
leader's records against every other entrant on those serves; and the full
win matrix. A bot that beats this pool beats this pool, not every possible bot.
"""

from __future__ import annotations

import argparse
import json
import re
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


def load_members(out: Path, genome_len: int, kernel_sha: str, bots: dict[str, int] | None = None) -> tuple[list[Member], dict]:
    lad = json.loads((out / "ladder.json").read_text())
    members = []
    for m in lad["members"]:
        if m["kind"] == "scripted":
            continue
        if m["kind"] == "oracle":  # a scripted rung: no genome, played from the build's slot
            if not bots or m["id"] not in bots:
                raise ValueError(f"the kernel has no scripted bot {m['id']!r}")
            members.append(Member(m["id"], "oracle", m["rating"], m["since_era"], None, label=m.get("label"), slot=bots[m["id"]]))
            continue
        g = make_genome(read_params(out / "genomes" / f"{m['id']}.bin", genome_len), kernel_sha, (), m["kind"], -1)
        if g.id != m["id"]:
            raise ValueError(f"{m['id']} does not hash to itself under kernel {kernel_sha[:8]}")
        members.append(Member(g.id, m["kind"], m["rating"], m["since_era"], g, label=m.get("label")))
    return members, lad


def load_elites(out: Path, n: int, genome_len: int, kernel_sha: str, prefix: str = "") -> list[Member]:
    """The run's top `n` live cells by archive rating (ties by fitness), as members of kind "elite"."""
    if n <= 0:
        return []
    lat = json.loads((out / "latest.json").read_text())
    cells = sorted(lat["cells"], key=lambda c: (-(c.get("rating") or 0.0), -(c.get("fitness") or 0.0)))
    members = []
    for k, c in enumerate(cells[:n], 1):
        g = make_genome(read_params(out / "genomes" / f"{c['genome_id']}.bin", genome_len), kernel_sha, (), "elite", c["gen"])
        if g.id != c["genome_id"]:
            raise ValueError(f"{c['genome_id']} does not hash to itself under kernel {kernel_sha[:8]}")
        members.append(Member(g.id, "elite", float(c.get("rating") or 0.0), int(c.get("era", 0)), g,
                              label=f"{prefix}elite #{k} (gen {c['gen']})"))
    return members


def beats(record: dict) -> int:
    """How many entrants `record`'s owner beat: more wins than losses, the ladder's own definition (a tie counts half)."""
    return sum(1 for f in record.values() if f["won"] > f["lost"])


def matrix(ids: list[str], pairs: tuple[Pair, ...]) -> list[list[float | None]]:
    index = {mid: k for k, mid in enumerate(ids)}
    win: list[list[float | None]] = [[None] * len(ids) for _ in ids]
    for p in pairs:
        a, b = index[p.a], index[p.b]
        win[a][b], win[b][a] = p.winrate_a, 1.0 - p.winrate_a
    return win


def verdict(run_id: str, n_seeds: int = 16, workers_n: int = 15, n_elites: int = 6) -> dict:
    store = Store(store_path(run_id))
    run = store.run_row(run_id)
    store.close()
    cfg = json.loads(run["config_json"])
    out = data_root() / "out" / run_id
    m = re.match(r"run(\d+)", cfg["slug"])
    elites = load_elites(out, n_elites, run["genome_len"], run["kernel_sha"], f"run {m.group(1)} " if m else "")
    kernel = out / "kernel" / f"{run['kernel_sha']}.wasm"
    kernel_path = str(kernel) if kernel.exists() else cfg["sim"]["kernel_path"]
    seeds = tuple(int(s) for s in np.random.SeedSequence([int(cfg["seed"]), VERDICT_STREAM])
                  .generate_state(n_seeds, dtype=np.uint64))
    with Workers(cfg["sim"]["kind"], kernel_path, workers_n, cfg["workers"]["jobs_per_task"]) as workers:
        kernel_sha, _, _, scripted = workers.simulator_info
        if kernel_sha != run["kernel_sha"]:
            raise ValueError(f"kernel {kernel_sha[:8]} is not the run's {run['kernel_sha'][:8]}")
        bots = getattr(workers, "bots", None) or {SCRIPTED_ID: scripted}
        members, lad = load_members(out, run["genome_len"], run["kernel_sha"], bots)
        ladder = Ladder(scripted, seeds, pool=members + elites)
        pairs = ladder.rate_pool(workers, cfg["sim"]["max_ticks"])
    rated = ladder.members
    champion = lad.get("champion") or max(rated, key=lambda m: m.rating).id
    run_rating = {m["id"]: m["rating"] for m in lad["members"]} | {e.id: e.rating for e in elites}
    label = {m.id: m.label or (m.kind if m.kind != "scripted" else "the coach") for m in rated}
    kind = {m.id: m.kind for m in rated}

    def record(who: str) -> dict:
        """`who`'s result against every other entrant on the held-out serves."""
        rec = {}
        for p in pairs:
            if who not in (p.a, p.b):
                continue
            other = p.b if p.a == who else p.a
            won, lost = (p.wins_a, p.wins_b) if p.a == who else (p.wins_b, p.wins_a)
            margin = p.margin_a if p.a == who else -p.margin_a
            rec[other] = {"label": label[other], "kind": kind[other], "won": won, "lost": lost, "tied": p.draws,
                          "n": p.n, "margin": margin}
        return rec

    fresh = record(champion)
    board = sorted(rated, key=lambda m: -m.rating)
    top = record(board[0].id)
    result = {
        "run_id": run_id, "kernel_sha": run["kernel_sha"], "seeds": [str(s) for s in seeds],
        "matches_per_pair": 2 * len(seeds), "elites": len(elites), "champion": champion,
        "members_beaten": beats(fresh), "of": len(fresh),
        "fresh": fresh,
        "top": {"id": board[0].id, "label": label[board[0].id], "kind": kind[board[0].id], "rating": board[0].rating,
                "beaten": beats(top), "of": len(top), "record": top},
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
    ap.add_argument("--elites", type=int, default=6, help="top live cells by archive rating to enter")
    args = ap.parse_args()
    r = verdict(args.run_id, args.seeds, args.workers, args.elites)
    print(f"{r['run_id']}: run champion {r['champion'][:8]} beats {r['members_beaten']} of {r['of']} on {len(r['seeds'])} held-out serves")
    print(f"  held-out leader: {r['top']['label']} ({r['top']['kind']}) beats {r['top']['beaten']} of {r['top']['of']}")
    for e in r["leaderboard"]:
        print(f"  {e['rank']:>2} {e['rating']:7.1f} {e['kind']:<8} {e['label']}")


if __name__ == "__main__":
    main()
