"""Specialists: one genome trained against the real champions, with its style forced by construction.

OpenAI-ES (antithetic pairs, centred ranks, Adam) walks one genome, warm-started from an existing
elite, against a few rivals: the champions of earlier runs and the scripted bot. A style is never only
a bonus. It is a mask on the output layer (`repair`, applied to every candidate and to the result), a
hard window on the match descriptors (`Window`: a penalty while training, a strict check at the end),
or a different fitness; each has an unconstrained twin with the same start and the same seeds.

    python -m arena.specialist --style flat_earther [--twin] --minutes 20 --workers 15 \
        --config configs/specialists.toml

The output contract is docs/DATA.md, "Specialists".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .export import _json_seed, _write_atomic, _write_json, _write_once
from .genome import GENOME_LEN, N_HID, genome_id
from .sim import DESCRIPTOR_NAMES, SCRIPTED_ID
from .store import data_root
from .pool import fit_params
from .workers import Workers

# Output layer of the 12-32-3 genome (kernel/src/policy.rs): W2[3][32] at 416, b2[3] at 512.
# Output i is button i (bit 0 forward, bit 1 backward, bit 2 jump), pressed when the output is above zero.
W2, B2, JUMP = 416, 512, 2
EXTRA_NAMES = ("juggled_points", "max_streak", "spiked_points")  # kernel v2 adds two, v3 (28 October) a third  # descriptors 8 and 9, kernel version 2
STYLES = ("flat_earther", "kangaroo", "long_ball", "lob_artist", "sprinter", "all_rounder", "joga_bonito",
          "spiker", "dinker", "one_touch", "lazy", "assassin", "clavada")  # 28 October: gap and request styles, assassins
WINDOW_STYLES = ("long_ball", "lob_artist", "spiker", "dinker")  # a window on hit speed and height, from [window.<name>]
NO_TWIN = ("all_rounder", "assassin")  # nothing to remove: they are the plain margin against their rivals


# ------------------------------------------------------------------ styles

def pin_jump(params: np.ndarray, on: bool) -> np.ndarray:
    """The jump output's weights zeroed and its bias set to +1 (always pressed) or -1 (never).
    Offsets follow the genome's own width (12-H-3: 16H + 3 values; H is 32 in kernel v2, 64 in v3)."""
    p = np.array(params, dtype=np.float32, copy=True)
    n_hid = (len(p) - 3) // 16
    if 16 * n_hid + 3 != len(p):
        raise ValueError(f"a 12-H-3 genome has 16H + 3 values, not {len(p)}")
    w2, b2 = 13 * n_hid, 16 * n_hid
    p[w2 + JUMP * n_hid : w2 + (JUMP + 1) * n_hid] = 0.0
    p[b2 + JUMP] = 1.0 if on else -1.0
    return p


@dataclass(frozen=True)
class Bound:
    """lo <= descriptors[name] <= hi, either end optional. `lo_of` = (other, fraction) means lo = fraction * other."""
    name: str
    lo: float | None = None
    hi: float | None = None
    scale: float = 1.0  # a violation of one `scale` costs `weight` points
    lo_of: tuple[str, float] | None = None
    hi_of: tuple[str, float] | None = None  # (other, fraction): hi = fraction * other

    def violation(self, d: dict[str, float]) -> float:
        v = d[self.name]
        lo = self.lo if self.lo_of is None else self.lo_of[1] * d[self.lo_of[0]]
        hi = self.hi if self.hi_of is None else self.hi_of[1] * d[self.hi_of[0]]
        out = 0.0
        if lo is not None:
            out += max(0.0, lo - v)
        if hi is not None:
            out += max(0.0, v - hi)
        return out / self.scale


@dataclass(frozen=True)
class Window:
    bounds: tuple[Bound, ...]

    def violation(self, d: dict[str, float]) -> float:
        return sum(b.violation(d) for b in self.bounds)

    def satisfied(self, d: dict[str, float]) -> bool:
        return all(b.violation(d) == 0.0 for b in self.bounds)

    def detail(self) -> dict:
        return {b.name: {"min": b.lo, "max": b.hi, "min_fraction_of": list(b.lo_of) if b.lo_of else None,
                         "max_fraction_of": list(b.hi_of) if b.hi_of else None}
                for b in self.bounds}


@dataclass(frozen=True)
class Style:
    name: str
    pin: bool | None = None  # True: jump always pressed; False: never; None: free
    window: Window | None = None
    fitness: str = "mean"  # mean | min | sprinter | joga | lazy
    descriptors: int = 8  # how many descriptors the simulator must report
    twin: bool = False  # the unconstrained twin: same start, no repair, window or extra terms

    def repair(self, params: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=np.float32)
        return pin_jump(p, self.pin) if self.pin is not None and not self.twin else p.copy()

    @property
    def constrained(self) -> bool:
        return not self.twin and (self.pin is not None or self.window is not None or self.fitness != "mean")


@dataclass
class SpecialistConfig:
    seed: int = 20261027
    kernel_path: str = ""
    sim_kind: str = "kernel"
    max_ticks: int = 3000
    rivals: list[dict] = field(default_factory=list)  # {id, kind: champion|coach, out_run}
    start_run: str = ""  # the export whose cells give warm starts, then fallback_run
    fallback_run: str = ""
    population: int = 64
    sigma: float = 0.03
    lr: float = 0.01
    train_seeds: int = 8
    graded_seeds: int = 16
    fresh_seeds: int = 8
    graded_every: int = 5
    penalty_weight: float = 2.0
    windows: dict[str, dict] = field(default_factory=dict)
    sprinter_lambda: float = 0.1
    joga_bonus: float = 0.25
    joga_min_fraction: float = 1 / 3
    coach_gate: bool = True  # 28 October: a candidate that loses to the coach scores below every one that does not
    clavada_min_fraction: float = 1 / 3  # clavada: at least this share of points won are spiked at the net
    lazy_fraction: float = 0.5  # lazy: free travel up to this fraction of the start's travel per tick
    lazy_lambda: float = 2.0  # lazy: points of fitness lost per cap-width of travel above the cap
    lazy_cap: float | None = None  # set from the start genome when training begins


def load_config(path: str | Path) -> SpecialistConfig:
    t = tomllib.loads(Path(path).read_text())
    es, pen = t.get("es", {}), t.get("penalty", {})
    c = SpecialistConfig(
        seed=t.get("seed", 20261027), kernel_path=t.get("sim", {}).get("kernel_path", ""),
        sim_kind=t.get("sim", {}).get("kind", "kernel"), max_ticks=t.get("sim", {}).get("max_ticks", 3000),
        rivals=t.get("rival", []), start_run=t.get("start", {}).get("run", ""),
        fallback_run=t.get("start", {}).get("fallback_run", ""),
        population=es.get("population", 64), sigma=es.get("sigma", 0.03), lr=es.get("lr", 0.01),
        train_seeds=es.get("train_seeds", 8), graded_seeds=es.get("graded_seeds", 16),
        fresh_seeds=es.get("fresh_seeds", 8), graded_every=es.get("graded_every", 5),
        penalty_weight=pen.get("weight", 2.0), windows=t.get("window", {}),
        sprinter_lambda=t.get("sprinter", {}).get("lambda", 0.1),
        joga_bonus=t.get("joga", {}).get("bonus", 0.25),
        joga_min_fraction=t.get("joga", {}).get("min_fraction", 1 / 3),
        coach_gate=t.get("gate", {}).get("coach", True),
        clavada_min_fraction=t.get("clavada", {}).get("min_fraction", 1 / 3),
        lazy_fraction=t.get("lazy", {}).get("cap_fraction", 0.5), lazy_lambda=t.get("lazy", {}).get("lambda", 2.0))
    if c.population < 2 or c.population % 2:
        raise ValueError("es.population must be even (antithetic pairs)")
    return c


def make_style(name: str, cfg: SpecialistConfig, twin: bool = False) -> Style:
    if name not in STYLES:
        raise ValueError(f"unknown style {name!r}; choose from {', '.join(STYLES)}")
    w = cfg.windows
    if name == "flat_earther":
        s = Style(name, pin=False)
    elif name == "kangaroo":
        s = Style(name, pin=True)
    elif name in WINDOW_STYLES:
        c = w[name]
        bounds = []
        if "speed_min" in c:
            bounds.append(Bound("mean_speed_after_hit", lo=c["speed_min"], scale=c["speed_scale"]))
        if "speed_max" in c:
            bounds.append(Bound("mean_speed_after_hit", hi=c["speed_max"], scale=c["speed_scale"]))
        if "hit_height_min" in c:
            bounds.append(Bound("mean_hit_height", lo=c["hit_height_min"], scale=c["hit_height_scale"]))
        if "hit_height_max" in c:
            bounds.append(Bound("mean_hit_height", hi=c["hit_height_max"], scale=c["hit_height_scale"]))
        s = Style(name, window=Window(tuple(bounds)))
    elif name == "one_touch":  # every return on the first touch: no point won after a juggle
        s = Style(name, window=Window((Bound("juggled_points", hi=0.0, scale=1.0),)), descriptors=10)
    elif name == "lazy":  # the least travel that still wins: a soft cap on distance per tick
        s = Style(name, fitness="lazy")
    elif name == "clavada":  # Daniel's: carry the ball to the net and spike it; a third of points won that way (kernel v3)
        s = Style(name, window=Window((Bound("spiked_points", lo_of=("points", cfg.clavada_min_fraction),
                                             scale=max(1.0, 5 * cfg.clavada_min_fraction)),)), descriptors=11)
    elif name == "sprinter":
        s = Style(name, fitness="sprinter")
    elif name == "all_rounder":
        s = Style(name, fitness="min")
    elif name == "assassin":  # the plain mean margin; the rivals (one target, the coach as the gate) are the style
        s = Style(name)
    else:  # joga_bonito
        s = Style(name, window=Window((Bound("juggled_points", lo_of=("points", cfg.joga_min_fraction),
                                             scale=max(1.0, 5 * cfg.joga_min_fraction)),)),
                  fitness="joga", descriptors=10)
    if twin:
        if name in NO_TWIN:
            raise ValueError(f"{name} is its own twin: it has no constraint to remove")
        return Style(name, twin=True, pin=s.pin, window=s.window, fitness=s.fitness)
    return s


# ------------------------------------------------------------------ playing

class Matches:
    """Results of `n` candidates against R rivals on S seeds, both sides, in array form.

    All arrays are the candidate's view: mine/theirs points, ticks, and its own side's descriptors.
    Shapes are (n, R, S, 2) with the last axis candidate-on-the-left then candidate-on-the-right.
    """

    def __init__(self, mine, theirs, ticks, desc):
        self.mine, self.theirs, self.ticks, self.desc = mine, theirs, ticks, desc  # desc: (n, R, S, 2, D)

    def margins(self) -> np.ndarray:
        """(n, R): mean points for minus against, per rival."""
        return (self.mine - self.theirs).mean(axis=(2, 3))

    def mean_desc(self) -> np.ndarray:
        """(n, D): mean over rivals, seeds and sides."""
        return self.desc.mean(axis=(1, 2, 3))

    def ticks_per_point(self) -> np.ndarray:
        """(n,): mean over matches of ticks / points played."""
        return (self.ticks / np.maximum(1, self.mine + self.theirs)).mean(axis=(1, 2, 3))


class Arena:
    def __init__(self, workers: Workers, rival_params: list[np.ndarray | None], scripted_slot: int | None,
                 max_ticks: int) -> None:
        """`rival_params[r]` is a genome, or None for the scripted bot."""
        self.workers, self.max_ticks = workers, max_ticks
        self.genomes = [p for p in rival_params if p is not None]
        self.slots, k = [], 0
        for p in rival_params:
            if p is None:
                if scripted_slot is None:
                    raise ValueError("a rival is the scripted bot but the simulator has none")
                self.slots.append(scripted_slot)
            else:
                self.slots.append(k)
                k += 1
        self.matches_total = 0

    def play(self, cands: np.ndarray, seeds: tuple[int, ...]) -> Matches:
        n, R, S = len(cands), len(self.slots), len(seeds)
        base = len(self.genomes)
        table = np.concatenate([np.stack(self.genomes), cands]) if base else np.asarray(cands)
        jobs = []
        for i in range(n):
            for opp in self.slots:
                for s in seeds:
                    jobs.append((s, base + i, opp))
                    jobs.append((s, opp, base + i))
        res = self.workers.play(table.astype(np.float32), jobs, self.max_ticks)
        self.matches_total += len(jobs)
        D = len(res[0].desc_a)
        mine = np.empty((n, R, S, 2)); theirs = np.empty_like(mine); ticks = np.empty_like(mine)
        desc = np.empty((n, R, S, 2, D))
        k = 0
        for i in range(n):
            for r in range(R):
                for s in range(S):
                    for side in (0, 1):
                        m = res[k]; k += 1
                        a, b = (m.points_a, m.points_b) if side == 0 else (m.points_b, m.points_a)
                        mine[i, r, s, side], theirs[i, r, s, side], ticks[i, r, s, side] = a, b, m.ticks
                        desc[i, r, s, side] = m.desc_a if side == 0 else m.desc_b
        return Matches(mine, theirs, ticks, desc)


def desc_names(d: int) -> tuple[str, ...]:
    return DESCRIPTOR_NAMES + EXTRA_NAMES[: max(0, d - len(DESCRIPTOR_NAMES))]


def travel_per_tick(m: Matches, names: tuple[str, ...]) -> np.ndarray:
    """(n,): the candidate's distance per tick, mean over its matches."""
    return (m.desc[..., names.index("distance")] / np.maximum(1, m.ticks)).mean(axis=(1, 2, 3))


def score(style: Style, cfg: SpecialistConfig, m: Matches, names: tuple[str, ...],
          weights: np.ndarray | None = None, coach: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(fitness, penalty) per candidate. A twin, and every style's non-window terms off, is the mean margin
    over the rivals (weighted when the rivals carry weights). The coach gate (28 October) applies to every style
    and twin: a candidate that loses to the coach on average scores that margin minus ten, below any that does not."""
    marg = m.margins()
    w = np.ones(marg.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    wmean = marg @ w / w.sum()
    if style.fitness == "min" and not style.twin:
        fit = marg.min(axis=1)
    else:
        fit = wmean
    if not style.twin:
        if style.fitness == "sprinter":
            fit = fit - cfg.sprinter_lambda * m.ticks_per_point()
        if style.fitness == "joga":
            fit = fit + cfg.joga_bonus * m.mean_desc()[:, names.index("juggled_points")]
        if style.fitness == "lazy" and cfg.lazy_cap:
            fit = fit - cfg.lazy_lambda * np.maximum(0.0, travel_per_tick(m, names) / cfg.lazy_cap - 1.0)
    pen = np.zeros(len(marg))
    if style.window is not None and not style.twin:
        md = m.mean_desc()
        pen = np.array([cfg.penalty_weight * style.window.violation(dict(zip(names, row))) for row in md])
    if cfg.coach_gate and coach is not None:
        cm = marg[:, coach]
        fit = np.where(cm < 0, cm - 10.0, fit)
    return fit - pen, pen


# ------------------------------------------------------------------ evolution strategy

def centred_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


class Adam:
    def __init__(self, n: int, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self.m, self.v, self.t, self.lr, self.b1, self.b2, self.eps = np.zeros(n), np.zeros(n), 0, lr, b1, b2, eps

    def step(self, g: np.ndarray) -> np.ndarray:
        """The ascent step for gradient estimate g."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh, vh = self.m / (1 - self.b1 ** self.t), self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


def es_gradient(eps: np.ndarray, f_pos: np.ndarray, f_neg: np.ndarray, sigma: float) -> np.ndarray:
    """Antithetic estimate from rank-centred fitness; eps is (pairs, n), f_pos/f_neg score theta +- sigma*eps."""
    u = centred_ranks(np.concatenate([f_pos, f_neg]))
    k = len(f_pos)
    return (u[:k] - u[k:]) @ eps / (2 * k * sigma)


# ------------------------------------------------------------------ starts, rivals, verdict

def out_dir(run: str) -> Path:
    return data_root() / "out" / run


def load_cells(run: str) -> dict:
    return json.loads((out_dir(run) / "latest.json").read_text())


def load_genome(run: str, gid: str) -> np.ndarray:
    return np.frombuffer((out_dir(run) / "genomes" / f"{gid}.bin").read_bytes(), dtype=np.float32).copy()


def pick_start(style: Style, cfg: SpecialistConfig) -> tuple[str, str, list[int] | None, str]:
    """(genome id, run, cell or None, reason). The best lit cell that already fits the style, else run 5's champion."""
    champ_run = cfg.start_run
    champ = load_cells(champ_run)["stats"]["champion_id"]
    names = DESCRIPTOR_NAMES

    def fits(c: dict) -> bool:
        if style.name == "flat_earther":
            return c["bd"][1] <= 0.05
        if style.name == "kangaroo":
            return c["bd"][1] >= 0.9
        if style.window is not None and c.get("desc"):
            names = desc_names(len(c["desc"]))
            if any(b.name not in names for b in style.window.bounds):  # the export predates this descriptor
                return False
            return style.window.satisfied(dict(zip(names, c["desc"])))
        return False

    if style.name in ("flat_earther", "kangaroo") or (style.window is not None and style.name != "joga_bonito"):
        for run in (cfg.start_run, cfg.fallback_run):
            if not run:
                continue
            ok = [c for c in load_cells(run)["cells"] if c["lit"] and fits(c)]
            if ok:
                best = max(ok, key=lambda c: c["fitness"])
                return best["genome_id"], run, [best["i"], best["j"]], f"best lit cell that fits, of {len(ok)}"
    return champ, champ_run, None, "the champion of the start run"


def fresh_verdict(style: Style, cfg: SpecialistConfig, m: Matches, names: tuple[str, ...],
                  rival_ids: list[str], seeds: tuple[int, ...]) -> dict:
    mine, theirs = m.mine[0], m.theirs[0]  # (R, S, 2)
    per = {rid: {"won": int((mine[r] > theirs[r]).sum()), "n": int(mine[r].size),
                 "margin": float((mine[r] - theirs[r]).mean())} for r, rid in enumerate(rival_ids)}
    d = dict(zip(names, (float(x) for x in m.mean_desc()[0])))
    proof: dict = {"ticks_per_point": float(m.ticks_per_point()[0])}
    ok = True
    flat = m.desc[0].reshape(-1, m.desc.shape[-1])  # every fresh match, candidate's side
    if style.pin is not None and not style.twin:
        j = flat[:, names.index("jumps")]
        proof.update({"jumps_mean": float(j.mean()), "jumps_min": float(j.min()), "jumps_max": float(j.max()),
                      "airtime": d["airtime"]})
        ok = bool(j.max() == 0) if style.pin is False else bool(j.min() >= 1)
    if style.window is not None and not style.twin:
        for b in style.window.bounds:
            proof[b.name] = {"mean": d[b.name], "min": b.lo, "max": b.hi,
                             "min_fraction_of": list(b.lo_of) if b.lo_of else None}
        ok = ok and style.window.satisfied(d)
    if style.fitness == "min" and not style.twin:
        proof["min_margin"] = min(v["margin"] for v in per.values())
    if style.fitness == "lazy" and not style.twin:
        travel = float(travel_per_tick(m, names)[0])
        proof.update({"travel_per_tick": travel, "travel_cap": cfg.lazy_cap})
        ok = ok and cfg.lazy_cap is not None and travel <= cfg.lazy_cap
    points = float((mine + theirs).mean())  # per match; a stall is a match nobody scores in
    return {"per": per, "desc": d, "proof": proof, "ok": ok,
            "wins_all": all(v["won"] > v["n"] / 2 for v in per.values()), "points_per_match": points,
            "stalled": points < 2.0}


def showcase(m: Matches, names: tuple[str, ...], rival_ids: list[str], seeds: tuple[int, ...], per_rival: int = 2) -> list:
    out = []
    for r, rid in enumerate(rival_ids):
        cand = sorted(((float(m.mine[0, r, s, side] - m.theirs[0, r, s, side]), s, side)
                       for s in range(len(seeds)) for side in (0, 1)), reverse=True)[:per_rival]
        for margin, s, side in cand:
            out.append({"seed": _json_seed(seeds[s]), "side": "AB"[side], "rival_id": rid, "margin": margin,
                        "descriptors": dict(zip(names, (float(x) for x in m.desc[0, r, s, side])))})
    return out


# ------------------------------------------------------------------ the run

def train(style: Style, cfg: SpecialistConfig, workers: Workers, rival_params, rival_ids, scripted_slot,
          start: np.ndarray, minutes: float | None, generations: int | None, log=print,
          weights: np.ndarray | None = None, coach: int | None = None) -> dict:
    root = np.random.SeedSequence(cfg.seed)
    train_ss, graded_ss, fresh_ss = root.spawn(3)
    train_rng = np.random.default_rng(train_ss)
    graded = tuple(int(s) for s in graded_ss.generate_state(cfg.graded_seeds, dtype=np.uint64))
    fresh = tuple(int(s) for s in fresh_ss.generate_state(cfg.fresh_seeds, dtype=np.uint64))
    arena = Arena(workers, rival_params, scripted_slot, cfg.max_ticks)
    rng = np.random.default_rng(np.random.SeedSequence(cfg.seed + 1))  # perturbations; the twin sees the same ones
    names = None

    def play_scores(cands, seeds):
        nonlocal names
        m = arena.play(cands, seeds)
        names = names or desc_names(m.desc.shape[-1])
        if len(names) < style.descriptors:
            raise SystemExit(f"{style.name} needs {style.descriptors} descriptors; the simulator reports {len(names)} "
                             "(kernel version 2 adds juggled_points and max_streak)")
        fit, pen = score(style, cfg, m, names, weights, coach)
        return m, fit, pen

    theta = style.repair(start).astype(np.float64)
    adam = Adam(len(theta), cfg.lr)
    t0 = time.monotonic()
    curve: list[dict] = []
    m0, f0, _ = play_scores(theta[None, :].astype(np.float32), graded)
    if style.fitness == "lazy" and not style.twin and cfg.lazy_cap is None:
        cfg.lazy_cap = float(cfg.lazy_fraction * travel_per_tick(m0, names)[0])
        log(f"lazy cap: {cfg.lazy_cap:.4f} per tick, {cfg.lazy_fraction:.2f} of the start's travel")
        _, f0, _ = play_scores(theta[None, :].astype(np.float32), graded)
    best, best_fit = theta.copy(), float(f0[0])
    gen = 0
    while (generations is None or gen < generations) and (minutes is None or time.monotonic() - t0 < minutes * 60):
        seeds = tuple(int(s) for s in train_rng.integers(0, 2**63, size=cfg.train_seeds, dtype=np.int64))
        eps = rng.standard_normal((cfg.population // 2, len(theta)))
        cands = np.concatenate([theta + cfg.sigma * eps, theta - cfg.sigma * eps])
        cands = np.stack([style.repair(c) for c in cands]).astype(np.float32)
        m, fit, pen = play_scores(cands, seeds)
        k = len(eps)
        theta = style.repair((theta + adam.step(es_gradient(eps, fit[:k], fit[k:], cfg.sigma))).astype(np.float32)
                             ).astype(np.float64)
        top = int(np.argmax(fit))
        curve.append({"gen": gen, "t": round(time.monotonic() - t0, 2), "fitness_best": float(fit[top]),
                      "fitness_mean": float(fit.mean()),
                      "margin_by_rival": {rid: float(v) for rid, v in zip(rival_ids, m.margins()[top])},
                      "constraint_penalty": float(pen[top])})
        gen += 1
        if gen % cfg.graded_every == 0:
            _, f, _ = play_scores(theta[None, :].astype(np.float32), graded)
            if f[0] > best_fit:
                best, best_fit = theta.copy(), float(f[0])
            log(f"gen {gen}: pop best {fit[top]:+.2f} mean {fit.mean():+.2f}, graded {f[0]:+.2f} (best {best_fit:+.2f})")
    final = style.repair(best.astype(np.float32))
    return {"final": final, "curve": curve, "generations": gen, "elapsed_s": time.monotonic() - t0,
            "graded": graded, "fresh": fresh, "arena": arena, "best_graded_fitness": best_fit,
            "play": play_scores, "names": names}


def resolve_genome(path: str, root: Path) -> np.ndarray:
    p = Path(path)
    p = p if p.is_absolute() else root / p
    return np.frombuffer(p.read_bytes(), dtype="<f4").copy()


def run_specialist(style_name: str, twin: bool, cfg: SpecialistConfig, minutes: float | None, workers_n: int,
                   generations: int | None = None, root: Path | None = None, now: datetime | None = None,
                   log=print, name: str | None = None, start_path: str | None = None,
                   rivals: list[dict] | None = None) -> Path:
    """`rivals` (28 October) replaces the config's: dicts with `path` (a genome file, relative to the data root)
    or `kind: coach`, plus optional `weight` (default 1) and `label`. The coach is always a rival: it is the gate."""
    style = make_style(style_name, cfg, twin)
    now = now or datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-spec-{name or style_name}" + ("-twin" if twin else "")
    root = root or data_root()
    out = root / "specialists" / run_id
    if start_path:
        start, start_run, start_cell, why = resolve_genome(start_path, root), f"file:{start_path}", None, "the genome file given"
        start_id = None  # known once the kernel is
    else:
        start_id, start_run, start_cell, why = pick_start(make_style(style_name, cfg, False), cfg)
        start = load_genome(start_run, start_id)
    specs = list(rivals if rivals is not None else cfg.rivals)
    if not any(r.get("kind") == "coach" for r in specs):
        specs.append({"kind": "coach"})
    rival_ids, rival_params, rival_kinds, rival_weights, rival_labels = [], [], [], [], []
    for r in specs:
        if r.get("kind") == "coach":
            rival_ids.append(SCRIPTED_ID); rival_params.append(None); rival_kinds.append("coach")
        else:
            params = resolve_genome(r["path"], root) if "path" in r else load_genome(r["out_run"], r["id"])
            rival_ids.append(r.get("id")); rival_params.append(params); rival_kinds.append(r.get("kind", "rival"))
        rival_weights.append(float(r.get("weight", 1.0))); rival_labels.append(r.get("label"))
    coach = rival_ids.index(SCRIPTED_ID)
    with Workers(cfg.sim_kind, cfg.kernel_path, workers_n, 128) as workers:
        kernel_sha, kernel_version, glen, scripted = workers.simulator_info
        padded = len(start) != glen  # a start from a narrower kernel: padded, and its id is new under this kernel
        start = fit_params(start, glen, "the start genome")
        rival_params = [None if p is None else fit_params(p, glen, "a rival") for p in rival_params]
        start_id = genome_id(start, kernel_sha) if padded or not start_id else start_id
        rival_ids = [rid or genome_id(p, kernel_sha) for rid, p in zip(rival_ids, rival_params)]
        res = train(style, cfg, workers, rival_params, rival_ids, scripted, start, minutes, generations, log,
                    np.array(rival_weights), coach)
        final, arena, names = res["final"], res["arena"], res["names"]
        g_m = arena.play(final[None, :], res["graded"])
        f_m = arena.play(final[None, :], res["fresh"])
    verdict = fresh_verdict(style, cfg, f_m, names, rival_ids, res["fresh"])
    graded = fresh_verdict(style, cfg, g_m, names, rival_ids, res["graded"])["per"]
    fid = genome_id(final, kernel_sha)

    for gid, params in [(start_id, start), (fid, final)] + [
            (rid, p) for rid, p in zip(rival_ids, rival_params) if p is not None]:
        _write_once(out / "genomes" / f"{gid}.bin", np.asarray(params, dtype="<f4").tobytes())
    if cfg.sim_kind == "kernel" and cfg.kernel_path:
        blob = Path(cfg.kernel_path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != kernel_sha:
            raise ValueError("kernel file does not match the kernel in use")
        _write_once(out / "kernel" / f"{kernel_sha}.wasm", blob)

    if twin:
        constraint = {"kind": "none", "detail": "the unconstrained twin: mean margin over the rivals, no repair or window"}
    elif style.pin is not None:
        constraint = {"kind": "pinned_output", "detail": {"output": "jump", "state": "on" if style.pin else "off",
                                                          "weights_zeroed": f"W2[{JUMP}][*]", "bias": 1.0 if style.pin else -1.0}}
    elif style.window is not None:
        constraint = {"kind": "descriptor_window", "detail": style.window.detail()}
    elif style.name == "assassin":
        constraint = {"kind": "rivals", "detail": "the plain mean margin; the target rival is the style"}
    else:
        constraint = {"kind": "fitness", "detail": style.fitness}
    result = {
        "style": style_name, "twin": twin, "constraint": constraint, "name": name or style_name, "seed": cfg.seed,
        "gate": {"coach": cfg.coach_gate}, "lazy_cap": cfg.lazy_cap,
        "start": {"genome_id": start_id, "run_id": start_run, "cell": start_cell, "reason": why},
        "rivals": [{"id": i, "kind": k, "weight": w, "label": l}
                   for i, k, w, l in zip(rival_ids, rival_kinds, rival_weights, rival_labels)],
        "seeds": {"train": {"per_generation": cfg.train_seeds, "resampled": "every generation, from the run's stream"},
                  "graded": [_json_seed(s) for s in res["graded"]], "fresh": [_json_seed(s) for s in res["fresh"]]},
        "generations": res["generations"], "matches_total": arena.matches_total, "elapsed_s": res["elapsed_s"],
        "kernel_sha": kernel_sha, "kernel_version": kernel_version,
        "final": {"genome_id": fid, "graded": graded, "fresh": verdict["per"],
                  "descriptors_fresh_mean": verdict["desc"], "constraint_satisfied": verdict["ok"],
                  "wins_all": verdict["wins_all"], "stalled": verdict["stalled"],
                  "points_per_match": verdict["points_per_match"],
                  "proof": verdict["proof"], "showcase": showcase(f_m, names, rival_ids, res["fresh"])},
        "curve": res["curve"],
    }
    _write_json(out / "result.json", result)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train a style specialist against the real champions.")
    ap.add_argument("--style", required=True, choices=STYLES)
    ap.add_argument("--twin", action="store_true", help="the unconstrained twin: same start and seeds, no style")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--generations", type=int, help="stop after this many (tests, smoke runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="configs/specialists.toml")
    ap.add_argument("--name", help="run id and result name instead of the style's (assassins, the patch test)")
    ap.add_argument("--start-genome", help="start from this genome file instead of the start run's cells")
    ap.add_argument("--rival", action="append", default=[],
                    help="PATH[:WEIGHT[:LABEL]], repeatable; replaces the config's rivals (the coach is always added)")
    ap.add_argument("--seed", type=int, help="override the config seed (a twin keeps its specialist's seed)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    rivals = None
    if args.rival:
        rivals = []
        for spec in args.rival:
            parts = spec.split(":")
            rivals.append({"path": parts[0], "weight": float(parts[1]) if len(parts) > 1 and parts[1] else 1.0,
                           "label": parts[2] if len(parts) > 2 else None, "kind": "rival"})
    out = run_specialist(args.style, args.twin, cfg, args.minutes, args.workers, args.generations,
                         name=args.name, start_path=args.start_genome, rivals=rivals)
    r = json.loads((out / "result.json").read_text())["final"]
    print(f"{out}: fresh " + ", ".join(f"{k[:8]} {v['won']}/{v['n']} {v['margin']:+.2f}" for k, v in r["fresh"].items())
          + f"; constraint satisfied: {r['constraint_satisfied']}; beats all: {r['wins_all']}; stalled: {r['stalled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
