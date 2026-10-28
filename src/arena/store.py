"""The run's memory: one SQLite file, append-only.

Nothing is ever updated or deleted (triggers refuse it), every row carries the
era and the kernel sha it was made under, and only archive entrants are stored
with parents. Whether a genome is still alive is derived from archive_events.
Seeds and hashes are u64, so they are stored as text (SQLite integers are signed).

v1 adds the ladder: its members with their frozen ratings, every block of
ladder matches (challenges, and each new champion against every member), and
a score per genome per era, because a promotion re-scores the live elites
under the grown ladder. A re-score is also an archive event (same cell, same
genome, new fitness), so the occupancy and the lit count replay as before.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from .run import GenerationInfo, RunContext

ERA = 0  # the era a run starts in; promotions open the next ones (v1)

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
CREATE TABLE IF NOT EXISTS ladder_members (
  run_id TEXT, era INTEGER, kernel_sha TEXT, gen INTEGER, member_id TEXT, kind TEXT, rating REAL,
  winrate_vs_previous REAL, label TEXT, PRIMARY KEY (run_id, member_id));
CREATE TABLE IF NOT EXISTS ladder_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, era INTEGER, kernel_sha TEXT, gen INTEGER,
  purpose TEXT, a_id TEXT, b_id TEXT, n INTEGER, wins_a INTEGER, draws INTEGER, wins_b INTEGER);
CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, era INTEGER, kernel_sha TEXT, gen INTEGER,
  genome_id TEXT, fitness REAL, rating REAL, reason TEXT);
"""
TABLES = ("runs", "genomes", "genome_params", "evaluations", "archive_events", "promotions", "generations",
          "ladder_members", "ladder_results", "scores")


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
        if "desc_json" not in cols:
            self.db.execute("ALTER TABLE evaluations ADD COLUMN desc_json TEXT")
        for col in ("replay_hold_a", "replay_hold_b"):  # kernel version 3: the replay's holds
            if col not in cols:
                self.db.execute(f"ALTER TABLE evaluations ADD COLUMN {col} INTEGER")
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
                self._genome(ctx, p, "probe", ERA)
            if ctx.ladder is not None:
                for m in ctx.ladder.members:  # scripted-v1, the anchor, then the pool (run 6)
                    if m.genome is not None:
                        self._genome(ctx, m.genome, "pool", ERA)
                    self.db.execute("INSERT INTO ladder_members VALUES (?,?,?,?,?,?,?,?,?)",
                                    (ctx.run_id, m.since_era, ctx.kernel_sha, -1, m.id, m.kind, m.rating, None, m.label))
                for p in ctx.ladder.pool_pairs:  # the round robin that rated the pool
                    self._pair(ctx, ERA, -1, "pool", p)

    def _genome(self, ctx: RunContext, g, kind: str, era: int) -> None:
        self.db.execute("INSERT OR IGNORE INTO genomes VALUES (?,?,?,?,?,?,?,?)",
                        (g.id, ctx.run_id, era, ctx.kernel_sha, g.gen, kind, g.op, json.dumps(list(g.parents))))
        self.db.execute("INSERT OR IGNORE INTO genome_params VALUES (?,?)", (g.id, g.params.astype("<f4").tobytes()))

    def record_generation(self, ctx: RunContext, info: GenerationInfo) -> None:
        arch = ctx.archive
        era = info.era
        n_matches = info.matches_per_candidate or len(ctx.cfg.eval_seeds) * len(ctx.probe_ids) * 2
        with self.db:
            for r, outcome, evicted in zip(info.results, info.outcomes, info.evicted):
                if outcome == "rejected":
                    continue
                g = r.genome
                self._genome(ctx, g, "entrant", era)
                rp = r.replay
                self.db.execute(
                    "INSERT INTO evaluations (run_id, era, kernel_sha, gen, genome_id, fitness, bd_json, n_matches,"
                    " replay_seed, replay_opponent_id, replay_side, replay_final_hash, replay_ticks,"
                    " replay_points_for, replay_points_against, margins_json, desc_json, replay_hold_a, replay_hold_b)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ctx.run_id, era, ctx.kernel_sha, info.generation, g.id, r.fitness, json.dumps(list(r.bd)),
                     n_matches, *((str(rp.seed), rp.opponent_id, rp.side, f"0x{rp.final_hash:016x}",
                                   rp.ticks, rp.points[0], rp.points[1]) if rp else (None,) * 7),
                     json.dumps(list(r.margins)) if r.margins is not None else None,
                     json.dumps(list(r.desc)) if r.desc is not None else None,
                     *((rp.hold_a, rp.hold_b) if rp else (None, None))))
                self.db.execute(
                    "INSERT INTO archive_events (run_id, era, kernel_sha, gen, event, cell_json, genome_id,"
                    " evicted_id, fitness) VALUES (?,?,?,?,?,?,?,?,?)",
                    (ctx.run_id, era, ctx.kernel_sha, info.generation, outcome,
                     json.dumps(list(arch.cell_of(r.bd))), g.id, evicted, r.fitness))
                if r.rating is not None:
                    self._score(ctx, era, info.generation, r, "eval")
            if info.challenge is not None:
                era = self._record_challenge(ctx, info)
            if info.joined:
                era = self._record_joined(ctx, info)
            best = arch.best()
            self.db.execute(
                "INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ctx.run_id, era, ctx.kernel_sha, info.generation, info.elapsed_s, arch.filled(),
                 arch.qd_score(), best.fitness if best else None, best.genome.id if best else None,
                 info.matches_total))

    def _score(self, ctx: RunContext, era: int, gen: int, r, reason: str) -> None:
        self.db.execute("INSERT INTO scores (run_id, era, kernel_sha, gen, genome_id, fitness, rating, reason)"
                        " VALUES (?,?,?,?,?,?,?,?)", (ctx.run_id, era, ctx.kernel_sha, gen, r.genome.id,
                                                      r.fitness, r.rating, reason))

    def _pair(self, ctx: RunContext, era: int, gen: int, purpose: str, p) -> None:
        self.db.execute("INSERT INTO ladder_results (run_id, era, kernel_sha, gen, purpose, a_id, b_id, n, wins_a,"
                        " draws, wins_b) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (ctx.run_id, era, ctx.kernel_sha, gen, purpose, p.a, p.b, p.n, p.wins_a, p.draws, p.wins_b))

    def _record_joined(self, ctx: RunContext, info: GenerationInfo) -> int:
        """Inbox newcomers (28 October): each a member row, its blocks against every earlier member, and the
        re-scored archive once, unless a promotion this generation already wrote it. Returns the new era."""
        gen = info.generation
        for m, vs in info.joined:
            self.db.execute("INSERT INTO ladder_members VALUES (?,?,?,?,?,?,?,?,?)",
                            (ctx.run_id, m.since_era, ctx.kernel_sha, gen, m.id, m.kind, m.rating, None, m.label))
            for p in vs:
                self._pair(ctx, m.since_era, gen, "member", p)
        era = info.joined[-1][0].since_era
        if not (info.challenge is not None and info.challenge.promoted):
            arch = ctx.archive
            for r in info.rescored or []:
                self._score(ctx, era, gen, r, "rescore")
                self.db.execute(
                    "INSERT INTO archive_events (run_id, era, kernel_sha, gen, event, cell_json, genome_id,"
                    " evicted_id, fitness) VALUES (?,?,?,?,?,?,?,?,?)",
                    (ctx.run_id, era, ctx.kernel_sha, gen, "rescore", json.dumps(list(arch.cell_of(r.bd))),
                     r.genome.id, None, r.fitness))
        return era

    def _record_challenge(self, ctx: RunContext, info: GenerationInfo) -> int:
        """Writes a promotion check; returns the era the generation ends in."""
        c, gen = info.challenge, info.generation
        self._pair(ctx, c.era, gen, "challenge", c.headtohead)
        if not c.promoted:
            return info.era
        m = c.member
        self.db.execute("INSERT INTO promotions (run_id, era, kernel_sha, gen, genome_id, winrate_vs_previous)"
                        " VALUES (?,?,?,?,?,?)", (ctx.run_id, m.since_era, ctx.kernel_sha, gen, m.id,
                                                  m.winrate_vs_previous))
        self.db.execute("INSERT INTO ladder_members VALUES (?,?,?,?,?,?,?,?,?)",
                        (ctx.run_id, m.since_era, ctx.kernel_sha, gen, m.id, m.kind, m.rating,
                         m.winrate_vs_previous, None))
        for p in c.vs_members:  # the new champion against every earlier member: its row of the matrix
            self._pair(ctx, m.since_era, gen, "member", p)
        arch = ctx.archive  # already the fresh archive of the new era
        for r in info.rescored or []:
            self._score(ctx, m.since_era, gen, r, "rescore")
            self.db.execute(
                "INSERT INTO archive_events (run_id, era, kernel_sha, gen, event, cell_json, genome_id,"
                " evicted_id, fitness) VALUES (?,?,?,?,?,?,?,?,?)",
                (ctx.run_id, m.since_era, ctx.kernel_sha, gen, "rescore", json.dumps(list(arch.cell_of(r.bd))),
                 r.genome.id, None, r.fitness))
        return m.since_era

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
            " e.replay_points_for, e.replay_points_against, e.margins_json, e.desc_json, e.replay_hold_a, e.replay_hold_b"
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
            dj = d.pop("desc_json")
            d["desc"] = json.loads(dj) if dj else None
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

    def ladder_members(self, run_id: str) -> list[dict]:
        """The ladder in joining order, scripted-v1 first; empty for a v0 run."""
        cur = self.db.execute("SELECT * FROM ladder_members WHERE run_id=? ORDER BY era, rowid", (run_id,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def ladder_pairs(self, run_id: str, purpose: str) -> list[dict]:
        cur = self.db.execute("SELECT * FROM ladder_results WHERE run_id=? AND purpose=? ORDER BY id", (run_id, purpose))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def latest_scores(self, run_id: str) -> dict[str, dict]:
        """Each genome's most recent score (fitness and rating in the era it was scored)."""
        cur = self.db.execute("SELECT genome_id, era, gen, fitness, rating FROM scores WHERE run_id=? ORDER BY id",
                              (run_id,))
        return {gid: {"era": era, "gen": gen, "fitness": fit, "rating": rating}
                for gid, era, gen, fit, rating in cur.fetchall()}

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
