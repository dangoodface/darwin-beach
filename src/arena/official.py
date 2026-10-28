"""The official board (28 October 2026): a frozen ladder, official serves, and ratings that count.

    python -m arena.official init --from-run <run_id>          the ladder from a run's members and ratings
    python -m arena.official rate PATH[:label] ... [--workers]  rate genome files against the frozen members
    python -m arena.official add PATH --label L --rating R      a new member at its measured rating (a promotion)

Members never move: each keeps the rating it had when it joined. A candidate's official rating is the Elo that
best explains its record against the members on the official serves (64 seeds no trainer draws, both sides,
6,000 ticks), the coach fixed at 1000. It counts as measurable only while the candidate scores under 90 per cent
against at least three members; above that it is "above the ladder" and the ladder must grow before a bigger
number means anything. Files under $ARENA_DATA/official/: ladder.json, ratings/<label>-<id>.json, board.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .export import _write_json
from .genome import make_genome
from .ladder import Ladder, Member, elo_mle
from .pool import read_params
from .sim import SCRIPTED_ID
from .store import data_root
from .workers import Workers

OFFICIAL_SEED = (20261028, 7)  # SeedSequence([20261028, 7]): no run or specialist draws from it
N_SEEDS, MAX_TICKS = 64, 6000
MEASURABLE_BELOW, MEASURABLE_MIN = 0.9, 3


def official_dir() -> Path:
    return data_root() / "official"


def ladder_path() -> Path:
    return official_dir() / "ladder.json"


def init_from_run(run_id: str) -> dict:
    out = data_root() / "out" / run_id
    lad = json.loads((out / "ladder.json").read_text())
    kernel = sorted((out / "kernel").glob("*.wasm"))[0]
    members = [{"id": m["id"], "label": m.get("label") or m["kind"], "kind": m["kind"], "rating": m["rating"],
                "path": str(out / "genomes" / f"{m['id']}.bin"), "since": run_id}
               for m in lad["members"] if m["kind"] != "scripted"]
    seeds = [str(int(s)) for s in np.random.SeedSequence(list(OFFICIAL_SEED)).generate_state(N_SEEDS, dtype=np.uint64)]
    doc = {"anchor": SCRIPTED_ID, "anchor_rating": 1000.0, "kernel_path": str(kernel), "kernel_sha": kernel.stem,
           "seeds": seeds, "max_ticks": MAX_TICKS, "members": members}
    _write_json(ladder_path(), doc)
    return doc


def load_ladder() -> dict:
    return json.loads(ladder_path().read_text())


def add_member(path: Path, label: str, rating: float, kind: str = "champion", since: str = "") -> dict:
    doc = load_ladder()
    params = read_params(path, params_len(path))
    gid = make_genome(params, doc["kernel_sha"], (), kind, -1).id
    if any(m["id"] == gid for m in doc["members"]):
        raise ValueError(f"{gid[:8]} is already a member")
    doc["members"].append({"id": gid, "label": label, "kind": kind, "rating": float(rating), "path": str(path), "since": since})
    _write_json(ladder_path(), doc)
    return doc


def params_len(path: Path) -> int:
    return path.stat().st_size // 4


def add_bot(name: str, label: str, rating: float, since: str = "") -> dict:
    """A scripted bot the build exports (an oracle rung) as a member at a measured rating, by name."""
    doc = load_ladder()
    if any(m["id"] == name for m in doc["members"]):
        raise ValueError(f"{name} is already a member")
    doc["members"].append({"id": name, "label": label, "kind": "oracle" if name.startswith("oracle") else "scripted",
                           "rating": float(rating), "since": since})
    _write_json(ladder_path(), doc)
    return doc


def rate(cands: list[tuple[Path, str]], workers_n: int = 15, kernel_path: str | None = None) -> list[dict]:
    doc = load_ladder()
    seeds = tuple(int(s) for s in doc["seeds"])
    with Workers("kernel", kernel_path or doc["kernel_path"], workers_n, 128) as workers:
        sha, _, glen, scripted = workers.simulator_info
        bots = getattr(workers, "bots", None) or {SCRIPTED_ID: scripted}
        members = []
        for m in doc["members"]:
            if "path" not in m:  # a scripted member, played from the build's slot
                if m["id"] not in bots:
                    raise ValueError(f"the kernel has no scripted bot {m['id']!r}")
                members.append(Member(m["id"], m["kind"], float(m["rating"]), 0, None, label=m["label"], slot=bots[m["id"]]))
                continue
            g = make_genome(read_params(Path(m["path"]), glen), sha, (), m["kind"], -1)
            members.append(Member(g.id, m["kind"], float(m["rating"]), 0, g, label=m["label"]))
        ladder = Ladder(scripted, seeds, pool=members)
        results = []
        for path, label in cands:
            g = make_genome(read_params(path, glen), sha, (), "candidate", -1)
            rec, records = {}, []
            for m in ladder.members:
                if m.id == g.id:
                    continue
                p = ladder.play(workers, g, m, doc["max_ticks"])
                rec[m.id] = {"label": m.label or ("the coach" if m.kind == "scripted" else m.kind), "rating": m.rating,
                             "won": p.wins_a, "lost": p.wins_b, "tied": p.draws, "n": p.n,
                             "score": p.winrate_a, "margin": p.margin_a}
                records.append((m.rating, p.score_a, p.n))
            under = sum(1 for r in rec.values() if r["score"] < MEASURABLE_BELOW)
            res = {"id": g.id, "label": label, "path": str(path), "kernel_sha": sha, "rating": elo_mle(records),
                   "measurable": under >= MEASURABLE_MIN, "members_under_90": under,
                   "beaten": sum(1 for r in rec.values() if r["won"] > r["lost"]), "of": len(rec),
                   "seeds": len(seeds), "max_ticks": doc["max_ticks"], "record": rec}
            results.append(res)
            _write_json(official_dir() / "ratings" / f"{label.replace(' ', '-')}-{g.id[:8]}.json", res)
            with (official_dir() / "board.jsonl").open("a") as f:
                f.write(json.dumps({k: v for k, v in res.items() if k != "record"}) + "\n")
    return results


def describe(res: dict) -> str:
    worst = min(res["record"].values(), key=lambda r: r["score"])
    tag = "measurable" if res["measurable"] else "ABOVE THE LADDER"
    return (f"{res['label']} ({res['id'][:8]}): {res['rating']:.0f} {tag}, beats {res['beaten']} of {res['of']}, "
            f"{res['members_under_90']} members under 90%, worst {worst['label']} {worst['score']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="The official board: frozen ladder, official serves.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init"); i.add_argument("--from-run", required=True)
    r = sub.add_parser("rate"); r.add_argument("genomes", nargs="+", help="PATH[:label]")
    r.add_argument("--workers", type=int, default=15); r.add_argument("--kernel")
    a = sub.add_parser("add"); a.add_argument("genome"); a.add_argument("--label", required=True)
    a.add_argument("--rating", type=float, required=True); a.add_argument("--since", default="")
    b = sub.add_parser("add-bot"); b.add_argument("name", help="a scripted bot's name, e.g. oracle-v1-rung2")
    b.add_argument("--label", required=True); b.add_argument("--rating", type=float, required=True)
    b.add_argument("--since", default="")
    args = ap.parse_args()
    if args.cmd == "init":
        doc = init_from_run(args.from_run)
        print(f"official ladder: {len(doc['members'])} members, {len(doc['seeds'])} serves, {doc['max_ticks']} ticks")
    elif args.cmd == "rate":
        cands = [(Path(s.split(":")[0]), s.split(":")[1] if ":" in s else Path(s).stem[:8]) for s in args.genomes]
        for res in rate(cands, args.workers, args.kernel):
            print(describe(res))
    elif args.cmd == "add-bot":
        doc = add_bot(args.name, args.label, args.rating, since=args.since)
        print(f"added {args.label} ({args.name}) at {args.rating:.0f}; {len(doc['members'])} members")
    else:
        doc = add_member(Path(args.genome), args.label, args.rating, since=args.since)
        print(f"added {args.label} at {args.rating:.0f}; {len(doc['members'])} members")


if __name__ == "__main__":
    main()
