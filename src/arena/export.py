"""Writes the files the page reads, exactly as docs/DATA.md describes them.

Works from the store alone, so it can run beside the loop or after it. Content-
addressed files (genomes, kernels) are written once and never touched again; the
indexes are replaced atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .sim import SCRIPTED_ID, descriptor_names
from .store import Store, data_root

MAX_CURVE_POINTS = 2000
TREE_RECENT = 2000
JS_SAFE = 2**53


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_seed(seed: str | int | None):
    """u64 seeds travel as decimal strings when the page's Number could not hold them."""
    if seed is None:
        return None
    return int(seed) if int(seed) <= JS_SAFE else str(int(seed))


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")  # same directory: on the SSD, never /tmp
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, obj) -> None:
    _write_atomic(path, json.dumps(obj, separators=(",", ":")).encode())


def _write_once(path: Path, data: bytes) -> None:
    if not path.exists():
        _write_atomic(path, data)


def thin(points: list[dict], limit: int = MAX_CURVE_POINTS) -> list[dict]:
    """Every k-th point, but always the last one."""
    n = len(points)
    if n <= limit:
        return points
    k = -(-n // (limit - 1))
    kept = points[::k]
    if kept[-1] is not points[-1]:
        kept.append(points[-1])
    return kept


def layout(entrants: list[dict]) -> dict[str, tuple[float, float]]:
    """x is generation (time runs left to right); y spreads each generation's entrants.

    Within a generation, entrants are ordered by the mean y of their parents, which
    keeps families together and cuts crossings; roots fall back to insertion order.
    """
    by_gen: dict[int, list[dict]] = {}
    for e in entrants:
        by_gen.setdefault(e["gen"], []).append(e)
    max_gen = max(by_gen) if by_gen else 0
    pos: dict[str, tuple[float, float]] = {}
    for gen in sorted(by_gen):
        layer = by_gen[gen]
        n = len(layer)
        want = []
        for idx, e in enumerate(layer):
            ys = [pos[p][1] for p in e["parents"] if p in pos]
            want.append((sum(ys) / len(ys) if ys else (idx + 0.5) / n, idx))
        x = gen / max_gen if max_gen else 0.0
        for rank, (_, idx) in enumerate(sorted(want)):
            pos[layer[idx]["genome_id"]] = (x, (rank + 0.5) / n)
    return pos


def lit_curve(events: list[tuple[int, tuple[int, ...], str, float]], floor: float) -> dict[int, int]:
    """Lit cells at the end of each generation that had an archive event, by replaying the events.

    Every cell gets some bot within seconds; a cell is lit only once its elite's fitness reaches the
    floor, which is the fill that takes time and the one the page shows. Generations without an
    event keep the previous count (the caller reads the last known value).
    """
    cells: dict[tuple[int, ...], float] = {}
    out: dict[int, int] = {}
    last = 0
    for gen, cell, _gid, fit in events:
        cells[cell] = fit
        last = sum(1 for f in cells.values() if f >= floor)
        out[gen] = last
    return out


def coach_curve(events: list[tuple[int, tuple[int, ...], str, float]], by_id: dict[str, dict],
                probe_ids: list[str]) -> dict[int, int] | None:
    """Cells whose elite beats the scripted bot (margin above zero), at the end of each generation
    that had an archive event. None when the run has no scripted probe or no margins were stored.
    """
    if not probe_ids or probe_ids[0] != SCRIPTED_ID or not any(e["margins"] for e in by_id.values()):
        return None
    beats: dict[tuple[int, ...], bool] = {}
    count = 0
    out: dict[int, int] = {}
    for gen, cell, gid, _fit in events:
        m = by_id[gid]["margins"]
        now = bool(m and m[0] > 0)
        count += now - beats.get(cell, False)  # swap this cell's old verdict for the new one
        beats[cell] = now
        out[gen] = count
    return out


def ancestors(champion: str | None, by_id: dict[str, dict]) -> list[str]:
    """The champion and every entrant it descends from, oldest first."""
    if champion is None:
        return []
    seen, stack = set(), [champion]
    while stack:
        g = stack.pop()
        if g in seen or g not in by_id:
            continue
        seen.add(g)
        stack.extend(by_id[g]["parents"])
    return sorted(seen, key=lambda g: (by_id[g]["gen"], g))


def current_fitness(entrants: list[dict], scores: dict[str, dict]) -> dict[str, float]:
    """Each entrant's fitness now: its latest score in v1 (a new era re-scores the live elites),
    else the fitness it entered with."""
    return {e["genome_id"]: scores[e["genome_id"]]["fitness"] if e["genome_id"] in scores else e["fitness"]
            for e in entrants}


def champion_of(entrants: list[dict], live: set[str], fitness: dict[str, float] | None = None) -> str | None:
    """The fittest live elite; ties go to the elder (insertion order)."""
    order = {e["genome_id"]: i for i, e in enumerate(entrants)}
    fit = fitness or {e["genome_id"]: e["fitness"] for e in entrants}
    return max(live, key=lambda g: (fit[g], -order[g]), default=None)


def ladder_json(members: list[dict], pairs: list[dict], era: int, cycle_window: int) -> dict:
    """docs/DATA.md ladder.json: the members in joining order, and who beats whom.

    matrix.winrate[i][j] is member i's share of the matches against member j
    (a win 1, a tie 1/2), from the block each newcomer played against every
    earlier member when it was promoted; null on the diagonal or if unplayed.
    """
    ids = [m["member_id"] for m in members]
    index = {mid: k for k, mid in enumerate(ids)}
    win: list[list[float | None]] = [[None] * len(ids) for _ in ids]
    for p in pairs:
        if p["a_id"] in index and p["b_id"] in index and p["a_id"] != p["b_id"]:
            a, b = index[p["a_id"]], index[p["b_id"]]
            share = (p["wins_a"] + 0.5 * p["draws"]) / p["n"]
            win[a][b], win[b][a] = share, 1.0 - share
    out = []
    for m in members:
        entry = {"id": m["member_id"], "kind": m["kind"], "rating": m["rating"], "since_era": m["era"]}
        if m["kind"] != "scripted":
            entry["promoted_at_gen"] = m["gen"]
            entry["winrate_vs_previous"] = m["winrate_vs_previous"]
        out.append(entry)
    return {"era": era, "members": out, "matrix": {"ids": ids, "winrate": win}, "cycle_window": cycle_window}


def export_run(
    store: Store,
    run_id: str,
    root: Path | None = None,
    now: datetime | None = None,
    write_full: bool = True,
    fresh=None,
    fresh_history: dict[int, int] | None = None,
) -> Path:
    root = root or data_root()
    now = now or datetime.now(timezone.utc)
    out = root / "out" / run_id
    run = store.run_row(run_id)
    cfg = json.loads(run["config_json"])
    gens = store.generations(run_id)
    if not gens:
        raise ValueError(f"run {run_id} has no generations yet")
    entrants = store.entrants(run_id)
    by_id = {e["genome_id"]: e for e in entrants}
    occupants = store.occupants(run_id)
    cell_of = store.cell_of_entrants(run_id)
    live = set(occupants.values())
    last = gens[-1]

    # v1 runs have a ladder: the champion is its latest member (scripted-v1 until the
    # first promotion) and ratings are filled in; v0 runs keep the fittest elite.
    members = store.ladder_members(run_id)
    scores = store.latest_scores(run_id)
    fitness = current_fitness(entrants, scores)
    rating = {gid: s["rating"] for gid, s in scores.items()}
    best = champion_of(entrants, live, fitness)
    champion = members[-1]["member_id"] if members else best
    champion_rating = members[-1]["rating"] if members else None

    # genomes the page can fetch: live elites, replay opponents (probes), the champion's ancestry, the ladder
    lineage = ancestors(champion, by_id)
    want = (set(live) | set(lineage) | {p for p in json.loads(run["probe_ids_json"]) if p != SCRIPTED_ID}
            | {m["member_id"] for m in members if m["kind"] != "scripted"})  # the scripted bot has no genome
    glen = run["genome_len"]
    for gid in sorted(want):
        data = store.params(gid)
        assert len(data) == glen * 4, f"genome {gid} has {len(data)} bytes, expected {glen * 4}"
        _write_once(out / "genomes" / f"{gid}.bin", data)

    kernel_sha = run["kernel_sha"]
    kpath = cfg["sim"].get("kernel_path")
    if cfg["sim"]["kind"] == "kernel" and kpath:
        blob = Path(kpath).read_bytes()
        if hashlib.sha256(blob).hexdigest() != kernel_sha:
            raise ValueError("kernel file does not match the sha recorded for this run")
        _write_once(out / "kernel" / f"{kernel_sha}.wasm", blob)

    # fresh-serve check: the best elite and every elite that beats the scripted bot on the graded serves
    probe_ids = json.loads(run["probe_ids_json"])
    fresh_res: dict = {}
    if fresh is not None:
        pick = {gid for gid in live if probe_ids and probe_ids[0] == SCRIPTED_ID
                and by_id[gid]["margins"] and by_id[gid]["margins"][0] > 0}
        if best is not None:
            pick.add(best)
        fresh_res = fresh.check({gid: np.frombuffer(store.params(gid), dtype=np.float32) for gid in sorted(pick)})
    fresh_stats = None
    if fresh is not None:
        fresh_stats = {"seeds": len(fresh.seeds), "matches": fresh.matches, "checked": len(fresh_res),
                       "beats_coach": sum(1 for r in fresh_res.values() if r.won > fresh.matches // 2),
                       "champion": ({"won": fresh_res[best].won, "margin": fresh_res[best].margin}
                                    if best in fresh_res else None)}
        if fresh_history is not None and last["gen"] not in fresh_history:
            fresh_history[last["gen"]] = fresh_stats["beats_coach"]

    grid = cfg["grid"]
    floor = float(grid.get("lit_floor", 0.0))
    cells = []
    for cell, gid in sorted(occupants.items()):
        e = by_id[gid]
        cells.append({
            "i": cell[0], "j": cell[1], "genome_id": gid, "fitness": fitness[gid], "rating": rating.get(gid),
            "lit": fitness[gid] >= floor,
            "bd": e["bd"], "gen": e["gen"], "era": e["era"], "margins": e["margins"], "desc": e["desc"],
            "fresh": ({"won": fresh_res[gid].won, "margin": fresh_res[gid].margin} if gid in fresh_res else None),
            "replay": {"seed": _json_seed(e["replay_seed"]), "opponent_id": e["replay_opponent_id"],
                       "side": e["replay_side"], "final_hash": e["replay_final_hash"],
                       "ticks": e["replay_ticks"],
                       "points": [e["replay_points_for"], e["replay_points_against"]]},
        })
    events = store.events(run_id)
    coach = coach_curve(events, by_id, json.loads(run["probe_ids_json"]))
    coach_by_gen: dict[int, int | None] = {}
    carry_c = 0 if coach is not None else None
    for g in gens:
        carry_c = coach.get(g["gen"], carry_c) if coach is not None else None
        coach_by_gen[g["gen"]] = carry_c
    elapsed = last["t_s"]
    _write_json(out / "latest.json", {
        "run_id": run_id, "kernel_sha": kernel_sha, "kernel_version": run["kernel_version"],
        "generated_at": _iso(now), "generation": last["gen"], "era": last["era"],
        "probes": json.loads(run["probe_ids_json"]),
        "descriptor_names": list(descriptor_names(run["kernel_version"])),
        "grid": {"descriptors": grid["descriptors"], "indices": grid["indices"],
                 "bounds": grid["bounds"], "resolution": grid["resolution"], "lit_floor": floor},
        "cells": cells,
        "stats": {"filled": last["filled"], "lit": sum(1 for c in cells if c["lit"]),
                  "beats_coach": coach_by_gen[last["gen"]],
                  "qd_score": last["qd_score"], "champion_id": champion, "best_id": best,
                  "fresh": fresh_stats,
                  "champion_rating": champion_rating, "matches_total": last["matches_total"],
                  "matches_per_s": last["matches_total"] / elapsed if elapsed > 0 else 0.0,
                  "elapsed_s": elapsed},
    })

    lit_by_gen = lit_curve(events, floor)
    carry = 0
    for g in gens:
        carry = lit_by_gen.get(g["gen"], carry)
        lit_by_gen[g["gen"]] = carry

    def champion_rating_at(gen: int) -> float | None:
        """The rating of the champion standing at the end of `gen`: a step that only rises."""
        held = [m["rating"] for m in members if m["gen"] <= gen]
        return held[-1] if held else None

    def fresh_at(gen: int) -> int | None:
        """beats_coach_fresh at the last export at or before `gen`; null before the first one."""
        held = [v for g, v in sorted((fresh_history or {}).items()) if g <= gen]
        return held[-1] if held else None

    _write_json(out / "curve.json", {"run_id": run_id, "points": thin([
        {"gen": g["gen"], "t": g["t_s"], "filled": g["filled"], "lit": lit_by_gen.get(g["gen"], 0),
         "beats_coach": coach_by_gen[g["gen"]], "beats_coach_fresh": fresh_at(g["gen"]),
         "qd_score": g["qd_score"], "champion_rating": champion_rating_at(g["gen"]), "best_fitness": g["best_fitness"],
         "era": g["era"], "matches_total": g["matches_total"]}
        for g in gens])})

    pos = layout(entrants)
    nodes = []
    for e in entrants:
        gid = e["genome_id"]
        x, y = pos[gid]
        c = cell_of.get(gid)
        nodes.append({"id": gid, "parents": e["parents"], "op": e["op"], "gen": e["gen"], "era": e["era"],
                      "cell": list(c) if c else None, "fitness": fitness[gid], "rating": rating.get(gid),
                      "alive": gid in live, "champion": gid == champion, "x": x, "y": y})

    def tree(sel: list[dict]) -> dict:
        ids = {n["id"] for n in sel}
        return {"run_id": run_id, "nodes": sel,
                "edges": [[p, n["id"]] for n in sel for p in n["parents"] if p in ids],
                "champion_lineage": lineage}

    keep = set(lineage) | live | {n["id"] for n in nodes[-TREE_RECENT:]}
    _write_json(out / "tree.json", tree([n for n in nodes if n["id"] in keep]))
    if write_full:
        _write_json(out / "tree-full.json", tree(nodes))

    if members:
        ladder_cfg = cfg.get("ladder", {})
        _write_json(out / "ladder.json", ladder_json(members, store.ladder_pairs(run_id, "member"), last["era"],
                                                     int(ladder_cfg.get("cycle_window", 5))))

    _update_index(root, run_id, cfg["slug"], run["started_at"], last["gen"])
    return out


def _update_index(root: Path, run_id: str, title: str, started_at: str, latest_generation: int) -> None:
    path = root / "out" / "index.json"
    try:
        runs = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        runs = []
    runs = [r for r in runs if r["run_id"] != run_id]
    runs.append({"run_id": run_id, "title": title, "started_at": started_at,
                 "latest_generation": latest_generation})
    runs.sort(key=lambda r: r["run_id"])
    _write_json(path, runs)


class ExportObserver:
    """Exports every `every_s` seconds while the loop runs, and once at the end.

    List it after the StoreObserver: generations reach the store first, and the loop
    calls on_end in reverse, so the final export runs while the store is still open.
    """

    def __init__(self, store_observer, every_s: float = 60.0, root: Path | None = None,
                 tree_full_every_s: float = 3600.0) -> None:
        self.so, self.every_s, self.root = store_observer, every_s, root
        self.tree_full_every_s = tree_full_every_s
        self._last = self._last_full = 0.0
        self._fresh_history: dict[int, int] = {}  # generation of an export -> elites that beat the coach on fresh serves

    def on_start(self, ctx) -> None:
        self._last = self._last_full = time.monotonic()

    def on_generation(self, ctx, info) -> None:
        now = time.monotonic()
        if now - self._last >= self.every_s:
            self._export(ctx, full=now - self._last_full >= self.tree_full_every_s)

    def on_end(self, ctx, info) -> None:
        if info is not None:
            self._export(ctx, full=True)

    def _export(self, ctx, full: bool) -> None:
        export_run(self.so.store, ctx.run_id, self.root, write_full=full, fresh=getattr(ctx, "fresh", None),
                   fresh_history=self._fresh_history)
        self._last = time.monotonic()
        if full:
            self._last_full = self._last
