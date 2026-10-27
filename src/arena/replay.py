"""Python side of the page's verified tick: re-run a stored replay and compare hashes.

A replay is (kernel, seed, this genome on its side, the opponent by id). The store
keeps the hash the trainer saw; this plays the match again on the run's simulator
and says whether it comes out the same. With the kernel it is the integration
check after a swap, and the thing to point at when the page shows a red cross.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .export import champion_of
from .sim import SCRIPTED_ID, Simulator
from .store import Store, store_path
from .workers import make_simulator


@dataclass(frozen=True)
class Check:
    genome_id: str
    cell: tuple[int, ...] | None
    stored_hash: str
    got_hash: str
    stored_ticks: int
    got_ticks: int
    stored_points: tuple[int, int]
    got_points: tuple[int, int]

    @property
    def ok(self) -> bool:
        return (self.stored_hash, self.stored_ticks, self.stored_points) == (
            self.got_hash, self.got_ticks, self.got_points)


def open_simulator(store: Store, run_id: str) -> Simulator:
    run = store.run_row(run_id)
    sim_cfg = json.loads(run["config_json"])["sim"]
    sim = make_simulator(sim_cfg["kind"], sim_cfg.get("kernel_path", ""))
    if sim.kernel_sha != run["kernel_sha"]:
        raise ValueError(f"simulator kernel {sim.kernel_sha} is not the run's kernel {run['kernel_sha']}")
    return sim


def check_genome(store: Store, sim: Simulator, run_id: str, genome_id: str,
                 cell: tuple[int, ...] | None = None) -> Check:
    ent = next((e for e in store.entrants(run_id) if e["genome_id"] == genome_id), None)
    if ent is None:
        raise KeyError(f"{genome_id} is not an entrant of {run_id}")
    return _check(store, sim, ent, cell, json.loads(store.run_row(run_id)["config_json"])["sim"]["max_ticks"])


def _check(store: Store, sim: Simulator, ent: dict, cell, max_ticks: int) -> Check:
    mine = np.frombuffer(store.params(ent["genome_id"]), dtype="<f4")
    rows = [mine]
    opp = ent["replay_opponent_id"]
    if opp == SCRIPTED_ID:
        if sim.scripted_slot is None:
            raise ValueError("the replay names the scripted bot but the simulator has none")
        opp_slot = sim.scripted_slot
    else:
        rows.append(np.frombuffer(store.params(opp), dtype="<f4"))
        opp_slot = 1
    sim.load_genomes(np.stack(rows).astype(np.float32))
    a, b = (0, opp_slot) if ent["replay_side"] == "A" else (opp_slot, 0)
    r = sim.play(int(ent["replay_seed"]), a, b, max_ticks)
    mine_pts, their_pts = (r.points_a, r.points_b) if ent["replay_side"] == "A" else (r.points_b, r.points_a)
    return Check(ent["genome_id"], cell, ent["replay_final_hash"], f"0x{r.final_hash:016x}",
                 ent["replay_ticks"], r.ticks, (ent["replay_points_for"], ent["replay_points_against"]),
                 (mine_pts, their_pts))


def live_cells(store: Store, run_id: str) -> dict[tuple[int, ...], str]:
    return store.occupants(run_id)


def check_all(store: Store, sim: Simulator, run_id: str) -> list[Check]:
    by_id = {e["genome_id"]: e for e in store.entrants(run_id)}
    max_ticks = json.loads(store.run_row(run_id)["config_json"])["sim"]["max_ticks"]
    return [_check(store, sim, by_id[g], cell, max_ticks) for cell, g in sorted(live_cells(store, run_id).items())]


def champion_id(store: Store, run_id: str) -> str | None:
    return champion_of(store.entrants(run_id), set(live_cells(store, run_id).values()))


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Re-run a stored replay and compare its final hash.")
    ap.add_argument("run_id")
    ap.add_argument("--root", type=Path, help="data root (default $ARENA_DATA)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cell", nargs=2, type=int, metavar=("I", "J"))
    g.add_argument("--genome")
    g.add_argument("--champion", action="store_true")
    g.add_argument("--all", action="store_true", help="every live cell")
    args = ap.parse_args(argv)

    path = store_path(args.run_id, args.root)
    if not path.is_file():
        print(f"no store at {path}")
        return 2
    store = Store(path)
    try:
        sim = open_simulator(store, args.run_id)
        if args.all:
            checks = check_all(store, sim, args.run_id)
        else:
            cells = live_cells(store, args.run_id)
            if args.cell:
                cell = tuple(args.cell)
                if cell not in cells:
                    print(f"cell {cell} is empty")
                    return 2
                gid = cells[cell]
            elif args.champion:
                gid, cell = champion_id(store, args.run_id), None
                if gid is None:
                    print("the archive is empty")
                    return 2
            else:
                gid, cell = args.genome, None
            checks = [check_genome(store, sim, args.run_id, gid, cell)]
    except KeyError as e:
        print(f"not found: {e}")
        return 2
    for c in checks:
        where = f"cell {list(c.cell)} " if c.cell else ""
        status = "verified" if c.ok else "MISMATCH"
        print(f"{status} {where}{c.genome_id[:12]} stored {c.stored_hash} got {c.got_hash} "
              f"ticks {c.stored_ticks}/{c.got_ticks} points {list(c.stored_points)}/{list(c.got_points)}")
    bad = sum(not c.ok for c in checks)
    if args.all or len(checks) > 1:
        print(f"{len(checks) - bad} of {len(checks)} verified")
    return 1 if bad else 0
