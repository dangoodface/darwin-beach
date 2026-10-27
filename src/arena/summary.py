"""Measurements of a run, for the campaign log: what filled when, what lit, who won.

Everything comes from the store. Two things are recomputed rather than read:
the champion's win counts (the store keeps mean margins, not match results, so its
evaluation matches are played again) and the candidates by operator (candidates
that were not admitted are not stored, so the optimiser's random draws are replayed
from the config; the loop is deterministic, so this gives the real numbers).
This module reports measurements only; what they mean is for the reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .export import ancestors, champion_of
from .genome import make_genome
from .optim import MapElites
from .sim import SCRIPTED_ID
from .store import Store, data_root, store_path

FILL_MARKS = (100, 200, 300)
LIT_GENS = (0, 10, 50, 100, 200, 400, 800, 1600, 3200)
BANDS_X = ("back", "mid", "net")  # court position: 0 is the back wall, 1 the net
BANDS_Y = ("grounded", "hops", "kangaroo")  # airtime


def _third(centre: float) -> int:
    return 0 if centre < 1 / 3 else 1 if centre < 2 / 3 else 2


def _short(gid: str | None) -> str | None:
    return gid[:12] if gid else None


# ---------------------------------------------------------------------- facts

def fill_timing(gens: list[dict], events: list[dict], capacity: int) -> dict:
    marks = []  # the fixed marks below capacity, then "all cells"
    for n in (*[m for m in FILL_MARKS if m < capacity], capacity):
        hit = next((g for g in gens if g["filled"] >= n), None)
        marks.append({"cells": n, "all": n == capacity,
                      "gen": hit["gen"] if hit else None, "t_s": hit["t_s"] if hit else None})
    news = [e for e in events if e["event"] == "new"]
    last = news[-1] if news else None
    t_of = {g["gen"]: g["t_s"] for g in gens}
    return {"capacity": capacity, "filled_now": gens[-1]["filled"], "marks": marks,
            "last_cell": ({"cell": list(last["cell"]), "gen": last["gen"], "t_s": t_of[last["gen"]]} if last else None)}


def lit_curve(gens: list[dict], events: list[dict], floor: float) -> dict:
    fit: dict[tuple, float] = {}
    lit = 0
    last_lit = None
    by_gen: dict[int, list[dict]] = {}
    for e in events:
        by_gen.setdefault(e["gen"], []).append(e)
    t_of = {g["gen"]: g["t_s"] for g in gens}
    lit_at: dict[int, int] = {}
    for g in gens:
        for e in by_gen.get(g["gen"], []):
            cell, before = e["cell"], fit.get(e["cell"])
            fit[cell] = e["fitness"]
            if e["fitness"] >= floor and (before is None or before < floor):
                lit += 1
                last_lit = {"cell": list(cell), "gen": g["gen"], "t_s": g["t_s"]}
        lit_at[g["gen"]] = lit
    last_gen = gens[-1]["gen"]
    marks = [g for g in LIT_GENS if g <= last_gen]
    if last_gen not in marks:
        marks.append(last_gen)
    return {"floor": floor, "points": [{"gen": g, "t_s": t_of[g], "lit": lit_at[g]} for g in marks],
            "last_lit": last_lit}


def regions(cells: dict[tuple, dict], resolution: tuple[int, ...], floor: float) -> dict:
    stats = {}
    for bx in range(3):
        for by in range(3):
            stats[(bx, by)] = {"capacity": 0, "fits": []}
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            stats[(_third((i + 0.5) / resolution[0]), _third((j + 0.5) / resolution[1]))]["capacity"] += 1
    for (i, j), e in cells.items():
        stats[(_third((i + 0.5) / resolution[0]), _third((j + 0.5) / resolution[1]))]["fits"].append(e["fitness"])
    out = []
    for (bx, by), s in stats.items():
        f = s["fits"]
        out.append({"court": BANDS_X[bx], "air": BANDS_Y[by], "capacity": s["capacity"], "filled": len(f),
                    "lit": sum(x >= floor for x in f),
                    "mean": float(np.mean(f)) if f else None, "best": max(f) if f else None})
    ranked = [r for r in out if r["filled"]]
    return {"regions": out,
            "strongest": max(ranked, key=lambda r: r["mean"]) if ranked else None,
            "weakest": min(ranked, key=lambda r: r["mean"]) if ranked else None}


def lineage(entrants: list[dict], champion: str) -> dict:
    by_id = {e["genome_id"]: e for e in entrants}
    depth: dict[str, int] = {}
    up: dict[str, str | None] = {}
    for e in entrants:  # insertion order: parents come before children
        ps = [p for p in e["parents"] if p in depth]
        best = max(ps, key=lambda p: depth[p]) if ps else None
        depth[e["genome_id"]] = 1 + (depth[best] if best else 0)
        up[e["genome_id"]] = best
    line, g = [], champion
    while g is not None:
        line.append(g)
        g = up[g]
    line.reverse()  # oldest first
    ops: dict[str, int] = {}
    for g in line:
        ops[by_id[g]["op"]] = ops.get(by_id[g]["op"], 0) + 1
    fits = [by_id[g]["fitness"] for g in line]
    short = fits if len(fits) <= 12 else fits[:6] + fits[-6:]
    return {"ancestors": len(ancestors(champion, by_id)) - 1, "line_length": len(line), "operators": ops,
            "oldest_op": by_id[line[0]]["op"], "fitness_line": short, "fitness_line_shortened": len(fits) > 12,
            "line": line}


def champion_rivals(store: Store, run_id: str, genome_id: str, ent: dict, sim) -> dict | None:
    """Margin and wins per probe, replaying the champion's evaluation matches."""
    run = store.run_row(run_id)
    cfg = json.loads(run["config_json"])
    probe_ids = json.loads(run["probe_ids_json"])
    seeds = [int(s) for s in json.loads(run["eval_seeds_json"])]
    genome_probes = [p for p in probe_ids if p != SCRIPTED_ID]
    rows = [np.frombuffer(store.params(genome_id), dtype="<f4")] + [
        np.frombuffer(store.params(p), dtype="<f4") for p in genome_probes]
    sim.load_genomes(np.stack(rows).astype(np.float32))
    slot = {p: 1 + i for i, p in enumerate(genome_probes)}
    if SCRIPTED_ID in probe_ids:
        if sim.scripted_slot is None:
            return None
        slot[SCRIPTED_ID] = sim.scripted_slot
    out = []
    for pid in probe_ids:
        margins = []
        for s in seeds:
            for left in (True, False):
                a, b = (0, slot[pid]) if left else (slot[pid], 0)
                r = sim.play(s, a, b, cfg["sim"]["max_ticks"])
                mine, theirs = (r.points_a, r.points_b) if left else (r.points_b, r.points_a)
                margins.append(mine - theirs)
        out.append({"probe": pid if pid == SCRIPTED_ID else _short(pid), "matches": len(margins),
                    "margin": float(np.mean(margins)), "wins": sum(m > 0 for m in margins)})
    return {"probes": out,
            "beats_the_coach": (out[0]["margin"] > 0) if probe_ids and probe_ids[0] == SCRIPTED_ID else None}


def candidate_ops(store: Store, run_id: str) -> dict[str, int]:
    """Candidates by operator, by replaying the optimiser's draws (see the module note)."""
    from .run import STREAM_OPTIM  # imported here: run imports store, not summary

    run = store.run_row(run_id)
    cfg = json.loads(run["config_json"])
    gens = store.generations(run_id)
    root = np.random.SeedSequence(int(cfg["seed"]))
    rng = np.random.default_rng(root.spawn(3)[STREAM_OPTIM])
    state = SimpleNamespace(n=0)
    dummy = SimpleNamespace(genome=make_genome(np.zeros(run["genome_len"], np.float32), "x", (), "random", 0))
    stub = SimpleNamespace(filled=lambda: state.n, elites=lambda: [dummy] * state.n)
    o = cfg["optim"]
    opt = MapElites(stub, rng, run["kernel_sha"], run["genome_len"], o["batch_size"], o["init_size"],
                    o["sigma"], o["crossover_frac"])
    counts = {"random": 0, "mutate": 0, "iso_line": 0}
    for g in gens:
        for child in opt.ask():
            counts[child.op] += 1
        state.n = g["filled"]  # the archive after this generation is what the next ask sees
    return counts


def jumps(entrants: list[dict], top: int = 3) -> list[dict]:
    by_id = {e["genome_id"]: e for e in entrants}
    out = []
    for e in entrants:
        ps = [by_id[p] for p in e["parents"] if p in by_id]
        if not ps:
            continue
        parent = max(ps, key=lambda p: p["fitness"])  # the fittest parent: the smaller, honest jump
        out.append({"genome": _short(e["genome_id"]), "gen": e["gen"], "op": e["op"], "parent": _short(parent["genome_id"]),
                    "from": parent["fitness"], "to": e["fitness"], "jump": e["fitness"] - parent["fitness"]})
    out.sort(key=lambda j: -j["jump"])
    return out[:top]


def extremes(cells: dict[tuple, dict], floor: float) -> dict:
    def pick(key, reverse):
        cell, e = sorted(cells.items(), key=lambda kv: (key(kv[1]), kv[0]), reverse=reverse)[0]
        return {"cell": list(cell), "court_position": e["bd"][0], "airtime": e["bd"][1],
                "fitness": e["fitness"], "lit": e["fitness"] >= floor}
    if not cells:
        return {}
    return {"most_airborne": pick(lambda e: e["bd"][1], True), "most_grounded": pick(lambda e: e["bd"][1], False),
            "closest_to_net": pick(lambda e: e["bd"][0], True), "deepest_back": pick(lambda e: e["bd"][0], False)}


# ------------------------------------------------------------------ the report

def summarize(store: Store, run_id: str, lit_floor: float | None = None, sim=None,
              with_candidates: bool = True) -> dict:
    run = store.run_row(run_id)
    cfg = json.loads(run["config_json"])
    gens = store.generations(run_id)
    if not gens:
        raise ValueError(f"run {run_id} has no generations yet")
    floor = lit_floor if lit_floor is not None else float(cfg["grid"].get("lit_floor", 0.0))
    resolution = tuple(cfg["grid"]["resolution"])
    capacity = int(np.prod(resolution))
    entrants = store.entrants(run_id)
    by_id = {e["genome_id"]: e for e in entrants}
    occ = store.occupants(run_id)
    cells = {c: by_id[g] for c, g in occ.items()}
    cur = store.db.execute("SELECT gen, event, cell_json, genome_id, fitness FROM archive_events"
                           " WHERE run_id=? ORDER BY id", (run_id,))
    events = [{"gen": g, "event": ev, "cell": tuple(json.loads(c)), "genome_id": gid, "fitness": f}
              for g, ev, c, gid, f in cur.fetchall()]

    champ = champion_of(entrants, set(occ.values()))
    ce = by_id[champ]
    cell_of = {g: c for c, g in occ.items()}
    rivals, rivals_note = None, None
    if sim is not None:
        rivals = champion_rivals(store, run_id, champ, ce, sim)
    else:
        rivals_note = "simulator not opened: win counts not recomputed"
    last = gens[-1]
    by_op: dict[str, int] = {}
    for e in entrants:
        by_op[e["op"]] = by_op.get(e["op"], 0) + 1
    cand = candidate_ops(store, run_id) if with_candidates else None

    def share(d):
        n = sum(d.values())
        return {k: {"count": v, "share": v / n if n else 0.0} for k, v in d.items()}

    return {
        "run_id": run_id, "lit_floor": floor,
        "fill": fill_timing(gens, events, capacity),
        "lit": lit_curve(gens, events, floor),
        "regions": regions({k: v for k, v in cells.items()}, resolution, floor),
        "champion": {
            "cell": list(cell_of[champ]), "id": _short(champ), "court_position": ce["bd"][0],
            "airtime": ce["bd"][1], "fitness": ce["fitness"], "gen": ce["gen"],
            "stored_margins": ce["margins"], "probe_ids": [p if p == SCRIPTED_ID else _short(p)
                                                          for p in json.loads(run["probe_ids_json"])],
            "rivals": rivals, "rivals_note": rivals_note, "lineage": lineage(entrants, champ)},
        "operators": {"entrants": share(by_op), "candidates": share(cand) if cand else None},
        "jumps": jumps(entrants),
        "extremes": extremes(cells, floor),
        "throughput": {"matches_total": last["matches_total"],
                       "matches_per_s": last["matches_total"] / last["t_s"] if last["t_s"] > 0 else 0.0,
                       "generations": last["gen"] + 1, "elapsed_s": last["t_s"]},
    }


def _t(x) -> str:
    return f"{x:,.0f}s" if x >= 10 else f"{x:.1f}s"


def render(d: dict) -> str:
    L: list[str] = [f"Run {d['run_id']}", ""]
    f = d["fill"]
    L.append(f"1. Fill timing ({f['filled_now']} of {f['capacity']} cells have a bot)")
    for m in f["marks"]:
        what = f"all {m['cells']}" if m["all"] else str(m["cells"])
        L.append(f"   {what} cells: " + (f"generation {m['gen']}, {_t(m['t_s'])}" if m["gen"] is not None else "not yet"))
    if f["last_cell"]:
        c = f["last_cell"]
        L.append(f"   last cell to get a bot: {c['cell']} at generation {c['gen']}, {_t(c['t_s'])}")
    L += ["", f"2. Lit cells (fitness at or above {d['lit_floor']:g})"]
    for p in d["lit"]["points"]:
        L.append(f"   generation {p['gen']:>5}: {p['lit']:>3} lit, {_t(p['t_s'])}")
    ll = d["lit"]["last_lit"]
    L.append("   last cell to light: " + (f"{ll['cell']} at generation {ll['gen']}, {_t(ll['t_s'])}" if ll else "none"))
    L += ["", "3. Regions (court position by airtime): filled/capacity, lit, mean and best fitness"]
    for r in d["regions"]["regions"]:
        stat = f"mean {r['mean']:+.2f}, best {r['best']:+.2f}" if r["filled"] else "empty"
        L.append(f"   {r['court']:<5} x {r['air']:<9}: {r['filled']:>2}/{r['capacity']:<2}, {r['lit']:>2} lit, {stat}")
    for k in ("strongest", "weakest"):
        r = d["regions"][k]
        if r:
            L.append(f"   {k} by mean: {r['court']} x {r['air']} ({r['mean']:+.2f})")
    c = d["champion"]
    L += ["", f"4. The champion: cell {c['cell']}, court position {c['court_position']:.2f}, "
              f"airtime {c['airtime']:.2f}, fitness {c['fitness']:+.2f}, born generation {c['gen']}"]
    if c["rivals"]:
        for p in c["rivals"]["probes"]:
            L.append(f"   against {p['probe']}: margin {p['margin']:+.2f}, won {p['wins']} of {p['matches']}")
        bc = c["rivals"]["beats_the_coach"]
        L.append("   beats the coach: " + ("yes" if bc else "no" if bc is not None else "n/a (no scripted probe)"))
    elif c["stored_margins"]:
        for pid, m in zip(c["probe_ids"], c["stored_margins"]):
            L.append(f"   against {pid}: margin {m:+.2f}")
        L.append(f"   ({c['rivals_note']})")
    else:
        L.append("   no per-rival margins on record" + (f" ({c['rivals_note']})" if c["rivals_note"] else ""))
    ex = d.get("exported")
    if ex:
        L.append(f"   graded serves: {ex['beats_coach']} cells beat the coach")
        fr = ex.get("fresh")
        if fr:
            ch = fr["champion"]
            L.append(f"   fresh serves ({fr['seeds']} new seeds, {fr['matches']} matches each): {fr['beats_coach']} of "
                     f"{fr['checked']} checked elites win more than half"
                     + (f"; the best elite won {ch['won']} of {fr['matches']}, margin {ch['margin']:+.2f}" if ch else ""))
    ln = c["lineage"]
    ops = ", ".join(f"{k} {v}" for k, v in sorted(ln["operators"].items()))
    L.append(f"   ancestors on record: {ln['ancestors']}; longest single line of descent: {ln['line_length']} genomes "
             f"(oldest is {ln['oldest_op']}); operators along it: {ops}")
    fl = ", ".join(f"{x:+.2f}" for x in ln["fitness_line"])
    L.append("   fitness along that line, oldest first" + (" (first and last six)" if ln["fitness_line_shortened"] else "")
             + f": {fl}")
    L += ["", "5. Operators: share of entrants against share of candidates"]
    ent, cand = d["operators"]["entrants"], d["operators"]["candidates"]
    for op in ("random", "mutate", "iso_line"):
        e = ent.get(op, {"count": 0, "share": 0.0})
        cc = cand[op] if cand else None
        L.append(f"   {op:<9} entrants {e['count']:>6} ({e['share']:.1%})"
                 + (f"   candidates {cc['count']:>8} ({cc['share']:.1%})" if cc else "   candidates n/a"))
    L += ["", "6. Biggest parent-to-child jumps (against the fitter parent)"]
    for j in d["jumps"]:
        L.append(f"   {j['jump']:+.2f}: {j['from']:+.2f} to {j['to']:+.2f}, generation {j['gen']}, {j['op']}")
    L += ["", "7. Extremes"]
    for k, e in d["extremes"].items():
        L.append(f"   {k.replace('_', ' '):<15}: cell {e['cell']}, court {e['court_position']:.2f}, airtime "
                 f"{e['airtime']:.2f}, fitness {e['fitness']:+.2f}, {'lit' if e['lit'] else 'not lit'}")
    t = d["throughput"]
    L += ["", f"8. Throughput: {t['matches_total']:,} matches, {t['matches_per_s']:,.0f} a second, "
              f"{t['generations']:,} generations, {_t(t['elapsed_s'])} elapsed"]
    return "\n".join(L)


def exported_stats(run_id: str, root: Path | None = None) -> dict | None:
    """The graded and fresh beats-the-coach figures the exporter last wrote, if it has."""
    try:
        stats = json.loads(((root or data_root()) / "out" / run_id / "latest.json").read_text())["stats"]
    except (OSError, ValueError, KeyError):
        return None
    return {"beats_coach": stats.get("beats_coach"), "fresh": stats.get("fresh")}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Measurements of a run, in plain words.")
    ap.add_argument("run_id")
    ap.add_argument("--root", type=Path, help="data root (default $ARENA_DATA)")
    ap.add_argument("--json", action="store_true", help="the same facts as one JSON object")
    ap.add_argument("--lit-floor", type=float, help="override the run's grid.lit_floor")
    ap.add_argument("--no-rematch", action="store_true", help="skip replaying the champion's matches")
    ap.add_argument("--no-candidates", action="store_true", help="skip replaying the optimiser's draws")
    args = ap.parse_args(argv)
    path = store_path(args.run_id, args.root)
    if not path.is_file():
        print(f"no store at {path}")
        return 2
    store = Store(path)
    sim = None
    if not args.no_rematch:
        from .replay import open_simulator

        try:
            sim = open_simulator(store, args.run_id)
        except (ValueError, OSError, ImportError, FileNotFoundError) as e:
            print(f"note: simulator not available ({e}); the champion's win counts are skipped")
    d = summarize(store, args.run_id, args.lit_floor, sim, not args.no_candidates)
    d["exported"] = exported_stats(args.run_id, args.root)
    print(json.dumps(d, indent=2) if args.json else render(d))
    return 0
