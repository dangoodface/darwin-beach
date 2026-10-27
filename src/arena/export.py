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

from .sim import SCRIPTED_ID
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


def export_run(
    store: Store,
    run_id: str,
    root: Path | None = None,
    now: datetime | None = None,
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

    # the champion is the fittest live elite; ties go to the elder (insertion order)
    order = {e["genome_id"]: i for i, e in enumerate(entrants)}
    champion = max(live, key=lambda g: (by_id[g]["fitness"], -order[g]), default=None)

    # genomes the page can fetch: live elites, replay opponents (probes), the champion's ancestry
    lineage = ancestors(champion, by_id)
    want = set(live) | set(lineage) | {p for p in json.loads(run["probe_ids_json"]) if p != SCRIPTED_ID}  # the scripted bot has no genome
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

    grid = cfg["grid"]
    cells = []
    for cell, gid in sorted(occupants.items()):
        e = by_id[gid]
        cells.append({
            "i": cell[0], "j": cell[1], "genome_id": gid, "fitness": e["fitness"], "rating": None,
            "bd": e["bd"], "gen": e["gen"], "era": e["era"],
            "replay": {"seed": _json_seed(e["replay_seed"]), "opponent_id": e["replay_opponent_id"],
                       "side": e["replay_side"], "final_hash": e["replay_final_hash"],
                       "ticks": e["replay_ticks"],
                       "points": [e["replay_points_for"], e["replay_points_against"]]},
        })
    elapsed = last["t_s"]
    _write_json(out / "latest.json", {
        "run_id": run_id, "kernel_sha": kernel_sha, "kernel_version": run["kernel_version"],
        "generated_at": _iso(now), "generation": last["gen"], "era": last["era"],
        "grid": {"descriptors": grid["descriptors"], "indices": grid["indices"],
                 "bounds": grid["bounds"], "resolution": grid["resolution"]},
        "cells": cells,
        "stats": {"filled": last["filled"], "qd_score": last["qd_score"], "champion_id": champion,
                  "champion_rating": None, "matches_total": last["matches_total"],
                  "matches_per_s": last["matches_total"] / elapsed if elapsed > 0 else 0.0,
                  "elapsed_s": elapsed},
    })

    _write_json(out / "curve.json", {"run_id": run_id, "points": thin([
        {"gen": g["gen"], "t": g["t_s"], "filled": g["filled"], "qd_score": g["qd_score"],
         "champion_rating": None, "best_fitness": g["best_fitness"], "era": g["era"],
         "matches_total": g["matches_total"]} for g in gens])})

    pos = layout(entrants)
    nodes = []
    for e in entrants:
        gid = e["genome_id"]
        x, y = pos[gid]
        c = cell_of.get(gid)
        nodes.append({"id": gid, "parents": e["parents"], "op": e["op"], "gen": e["gen"], "era": e["era"],
                      "cell": list(c) if c else None, "fitness": e["fitness"], "rating": None,
                      "alive": gid in live, "champion": gid == champion, "x": x, "y": y})

    def tree(sel: list[dict]) -> dict:
        ids = {n["id"] for n in sel}
        return {"run_id": run_id, "nodes": sel,
                "edges": [[p, n["id"]] for n in sel for p in n["parents"] if p in ids],
                "champion_lineage": lineage}

    keep = set(lineage) | live | {n["id"] for n in nodes[-TREE_RECENT:]}
    _write_json(out / "tree.json", tree([n for n in nodes if n["id"] in keep]))
    _write_json(out / "tree-full.json", tree(nodes))

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

    def __init__(self, store_observer, every_s: float = 60.0, root: Path | None = None) -> None:
        self.so, self.every_s, self.root = store_observer, every_s, root
        self._last = 0.0

    def on_start(self, ctx) -> None:
        self._last = time.monotonic()

    def on_generation(self, ctx, info) -> None:
        if time.monotonic() - self._last >= self.every_s:
            self._export(ctx)

    def on_end(self, ctx, info) -> None:
        if info is not None:
            self._export(ctx)

    def _export(self, ctx) -> None:
        export_run(self.so.store, ctx.run_id, self.root)
        self._last = time.monotonic()
