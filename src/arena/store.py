"""The run's memory: one SQLite file, append-only.

Nothing is ever updated or deleted (triggers refuse it), every row carries the
era and the kernel sha it was made under, and only archive entrants are stored
with parents. Whether a genome is still alive is derived from archive_events.
Seeds and hashes are u64, so they are stored as text (SQLite integers are signed).
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .run import GenerationInfo, RunContext

ERA = 0  # eras arrive with promotion in v1

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, slug TEXT, started_at TEXT, era INTEGER, kernel_sha TEXT,
  kernel_version INTEGER, genome_len INTEGER, config_json TEXT,
  probe_ids_json TEXT, eval_seeds_json TEXT);
CREATE TABLE IF NOT EXISTS genomes (
  genome_id TEXT PRIMARY KEY, run_id TEXT, era INTEGER, kernel_sha TEXT, gen INTEGER,
  kind TEXT, op TEXT, parents_json TEXT);
CREATE TABLE IF NOT EXISTS genome_params (genome_id TEXT PRIMARY KEY, params BLOB);
CREATE TABLE IF NOT EXISTS evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, era INTEGER, kernel_sha TEXT, gen INTEGER,
  genome_id TEXT, fitness REAL, bd_json TEXT, n_matches INTEGER,
  replay_seed TEXT, replay_opponent_id TEXT, replay_side TEXT, replay_final_hash TEXT,
  replay_ticks INTEGER, replay_points_for INTEGER, replay_points_against INTEGER);
CREATE TABLE IF NOT EXISTS archive_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, era INTEGER, kernel_sha TEXT, gen INTEGER,
  event TEXT, cell_json TEXT, genome_id TEXT, evicted_id TEXT, fitness REAL);
CREATE TABLE IF NOT EXISTS promotions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, era INTEGER, kernel_sha TEXT, gen INTEGER,
  genome_id TEXT, winrate_vs_previous REAL);
CREATE TABLE IF NOT EXISTS generations (
  run_id TEXT, era INTEGER, kernel_sha TEXT, gen INTEGER, t_s REAL, filled INTEGER,
  qd_score REAL, best_fitness REAL, champion_id TEXT, matches_total INTEGER,
  PRIMARY KEY (run_id, gen));
"""
TABLES = ("runs", "genomes", "genome_params", "evaluations", "archive_events", "promotions", "generations")


def data_root() -> Path:
    """$ARENA_DATA, default ~/.local/share/arena. Never /tmp, which is RAM."""
    root = Path(os.environ.get("ARENA_DATA") or Path.home() / ".local/share/arena")
    if str(root.resolve()).startswith("/tmp"):
        raise ValueError("ARENA_DATA must not be under /tmp (it is RAM)")
    return root


def store_path(run_id: str, root: Path | None = None) -> Path:
    return (root or data_root()) / "runs" / run_id / "store.sqlite"


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        # added after the first stores were written: old files get the column, nothing is rewritten
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(evaluations)")}
        if "margins_json" not in cols:
            self.db.execute("ALTER TABLE evaluations ADD COLUMN margins_json TEXT")
        for t in TABLES:
            for op in ("UPDATE", "DELETE"):
                self.db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t} "
                    f"BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # writes -----------------------------------------------------------------

    def insert_run(self, ctx: RunContext) -> None:
        cfg = ctx.cfg
        with self.db:
            self.db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ctx.run_id, cfg.slug, ctx.started_at.strftime("%Y-%m-%dT%H:%M:%SZ"), ERA,
                 ctx.kernel_sha, ctx.kernel_version, ctx.genome_len, json.dumps(cfg.to_dict()),
                 json.dumps(ctx.probe_ids), json.dumps([str(s) for s in cfg.eval_seeds])))
            for p in ctx.probes:
                self._genome(ctx, p, "probe")

    def _genome(self, ctx: RunContext, g, kind: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO genomes VALUES (?,?,?,?,?,?,?,?)",
                        (g.id, ctx.run_id, ERA, ctx.kernel_sha, g.gen, kind, g.op, json.dumps(list(g.parents))))
        self.db.execute("INSERT OR IGNORE INTO genome_params VALUES (?,?)", (g.id, g.params.astype("<f4").tobytes()))

    def record_generation(self, ctx: RunContext, info: GenerationInfo) -> None:
        arch = ctx.archive
        n_matches = len(ctx.cfg.eval_seeds) * len(ctx.probe_ids) * 2
        with self.db:
            for r, outcome, evicted in zip(info.results, info.outcomes, info.evicted):
                if outcome == "rejected":
                    continue
                g = r.genome
                self._genome(ctx, g, "entrant")
                rp = r.replay
                self.db.execute(
                    "INSERT INTO evaluations (run_id, era, kernel_sha, gen, genome_id, fitness, bd_json, n_matches,"
                    " replay_seed, replay_opponent_id, replay_side, replay_final_hash, replay_ticks,"
                    " replay_points_for, replay_points_against, margins_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ctx.run_id, ERA, ctx.kernel_sha, info.generation, g.id, r.fitness, json.dumps(list(r.bd)),
                     n_matches, *((str(rp.seed), rp.opponent_id, rp.side, f"0x{rp.final_hash:016x}",
                                   rp.ticks, rp.points[0], rp.points[1]) if rp else (None,) * 7),
                     json.dumps(list(r.margins)) if r.margins is not None else None))
                self.db.execute(
                    "INSERT INTO archive_events (run_id, era, kernel_sha, gen, event, cell_json, genome_id,"
                    " evicted_id, fitness) VALUES (?,?,?,?,?,?,?,?,?)",
                    (ctx.run_id, ERA, ctx.kernel_sha, info.generation, outcome,
                     json.dumps(list(arch.cell_of(r.bd))), g.id, evicted, r.fitness))
            best = arch.best()
            self.db.execute(
                "INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ctx.run_id, ERA, ctx.kernel_sha, info.generation, info.elapsed_s, arch.filled(),
                 arch.qd_score(), best.fitness if best else None, best.genome.id if best else None,
                 info.matches_total))

    # reads (the exporter works from these, so it can run without the loop) ----

    def run_row(self, run_id: str) -> dict:
        cur = self.db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(zip([c[0] for c in cur.description], row))

    def generations(self, run_id: str) -> list[dict]:
        cur = self.db.execute("SELECT * FROM generations WHERE run_id=? ORDER BY gen", (run_id,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def entrants(self, run_id: str) -> list[dict]:
        """Every genome that ever held a cell, with its parents, in insertion order."""
        cur = self.db.execute(
            "SELECT g.genome_id, g.gen, g.era, g.op, g.parents_json, e.fitness, e.bd_json, e.replay_seed,"
            " e.replay_opponent_id, e.replay_side, e.replay_final_hash, e.replay_ticks,"
            " e.replay_points_for, e.replay_points_against, e.margins_json"
            " FROM genomes g JOIN evaluations e ON e.genome_id = g.genome_id"
            " WHERE g.run_id=? AND g.kind='entrant' ORDER BY e.id", (run_id,))
        cols = [c[0] for c in cur.description]
        out = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            d["parents"] = json.loads(d.pop("parents_json"))
            d["bd"] = json.loads(d.pop("bd_json"))
            m = d.pop("margins_json")
            d["margins"] = json.loads(m) if m else None
            out.append(d)
        return out

    def occupants(self, run_id: str) -> dict[tuple[int, ...], str]:
        """Current genome per cell, by replaying archive_events in order."""
        cells: dict[tuple[int, ...], str] = {}
        cur = self.db.execute("SELECT cell_json, genome_id FROM archive_events WHERE run_id=? ORDER BY id", (run_id,))
        for cell_json, gid in cur.fetchall():
            cells[tuple(json.loads(cell_json))] = gid
        return cells

    def events(self, run_id: str) -> list[tuple[int, tuple[int, ...], str, float]]:
        """Every archive event in order: (gen, cell, genome_id, fitness)."""
        cur = self.db.execute("SELECT gen, cell_json, genome_id, fitness FROM archive_events WHERE run_id=? ORDER BY id", (run_id,))
        return [(gen, tuple(json.loads(cj)), gid, fit) for gen, cj, gid, fit in cur.fetchall()]

    def cell_of_entrants(self, run_id: str) -> dict[str, tuple[int, ...]]:
        """Last cell each genome was written to."""
        out: dict[str, tuple[int, ...]] = {}
        cur = self.db.execute("SELECT genome_id, cell_json FROM archive_events WHERE run_id=? ORDER BY id", (run_id,))
        for gid, cell_json in cur.fetchall():
            out[gid] = tuple(json.loads(cell_json))
        return out

    def params(self, genome_id: str) -> bytes:
        return self.db.execute("SELECT params FROM genome_params WHERE genome_id=?", (genome_id,)).fetchone()[0]


class StoreObserver:
    """Run-loop observer that writes the store under $ARENA_DATA/runs/<run_id>/."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self.store: Store | None = None

    def on_start(self, ctx: RunContext) -> None:
        self.store = Store(store_path(ctx.run_id, self.root))
        self.store.insert_run(ctx)

    def on_generation(self, ctx: RunContext, info: GenerationInfo) -> None:
        self.store.record_generation(ctx, info)

    def on_end(self, ctx: RunContext, info: GenerationInfo | None) -> None:
        if self.store:
            self.store.close()
            self.store = None
