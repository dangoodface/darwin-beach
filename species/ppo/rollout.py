"""Rollouts: one process plays one match through the kernel's interactive exports.

The learner's policy is a numpy copy of the network; each tick it reads the state
block (zero-copy), builds the mirrored observation, samples the three action bits
and records observation, bits, log-probability and reward. The opponent is the
coach (scripted_action) or a genome in table slot 0 (policy_action). Rewards are the
kernel's own: +1 when the learner wins a point and -1 when it loses one.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:  # the torch venv has no `arena` installed; use the checkout
    sys.path.insert(0, str(ROOT / "src"))

from arena.kernel import Kernel  # noqa: E402

from .policy import (  # noqa: E402
    LIVES_A, LIVES_B, OBS_IDX, OBS_SIGN, STATE_LEN, TICK, bits_to_action, log_prob, logits, logits_exact,
    mode_bits, unpack, ObsFeed,
)

COACH = "coach"


@dataclass
class Episode:
    obs: np.ndarray  # (T, 12) float32, as the learner saw it
    bits: np.ndarray  # (T, 3) uint8, the presses
    logp: np.ndarray  # (T,) float32, log-probability of those presses under the sampling policy
    reward: np.ndarray  # (T,) float32, +1 / -1 on the tick a point is decided
    ticks: int
    points_for: int
    points_against: int
    final_hash: int
    side: int
    seed: int
    opponent: str


class MatchRunner:
    """One kernel instance, one match at a time (the interactive game is a single global state)."""

    def __init__(self, kernel_path: str | Path) -> None:
        self.k = Kernel(kernel_path)
        k = self.k
        addr = k._base(k._state_ptr, STATE_LEN * 8)
        self.state = np.frombuffer((ctypes.c_double * STATE_LEN).from_address(addr), dtype=np.float64)
        ex, self.store = k._exports, k._store
        self._reset, self._set, self._step = ex["reset"], ex["set_actions"], ex["step"]
        self._scripted, self._policy = ex["scripted_action"], ex["policy_action"]
        self._hash = ex["state_hash"]

    def play(self, params: np.ndarray, opponent, seed: int, side: int = 0, max_ticks: int = 3000,
             rng_seed: int = 0, greedy: bool = False, exact: bool = False) -> Episode:
        """`opponent` is COACH or an f32[515] genome (loaded into table slot 0).
        `greedy` plays the mode instead of sampling; `exact` uses the kernel's arithmetic
        (slow; for tests). Both must hold for a replay to match `Kernel.play` tick for tick."""
        net = unpack(params)
        if isinstance(opponent, str):
            opp_name = opponent
        else:
            self.k.load_genomes(np.asarray(opponent, dtype=np.float32)[None, :])
            opp_name = "genome"
        rng = np.random.default_rng(rng_seed)
        fwd = logits_exact if exact else logits
        st, opp_side = self.state, 1 - side
        idx, sgn = OBS_IDX[side], OBS_SIGN[side]
        obs_buf = np.empty((max_ticks, 12), np.float32)
        bits_buf = np.empty((max_ticks, 3), np.uint8)
        logp_buf = np.empty(max_ticks, np.float32)
        rew_buf = np.zeros(max_ticks, np.float32)
        pts_for = pts_against = 0
        self._reset(self.store, seed - (1 << 64) if seed >= 1 << 63 else seed)
        feed = ObsFeed()
        t = 0
        while st[LIVES_A] > 0 and st[LIVES_B] > 0 and st[TICK] < max_ticks:
            obs = feed.get(st, side)  # stale for one tick after a point, as in the kernel
            lg = fwd(obs, net)
            if greedy:
                bits = mode_bits(lg)
            else:
                bits = (rng.random(3) < 1.0 / (1.0 + np.exp(-lg))).astype(np.uint8)
            obs_buf[t] = obs
            bits_buf[t] = bits
            logp_buf[t] = log_prob(lg, bits)
            mine = bits_to_action(bits)
            if opponent is COACH or opp_name == COACH:
                theirs = self._scripted(self.store, opp_side)
            else:
                theirs = self._policy(self.store, opp_side, 0)
            a, b = (mine, theirs) if side == 0 else (theirs, mine)
            self._set(self.store, a, b)
            r = self._step(self.store)  # +1 when A wins the point, -1 when B does
            feed.after_step(r)
            r = r if side == 0 else -r
            if r:
                rew_buf[t] = r
                pts_for += r > 0
                pts_against += r < 0
            t += 1
        return Episode(obs_buf[:t].copy(), bits_buf[:t].copy(), logp_buf[:t].copy(), rew_buf[:t].copy(), t,
                       int(pts_for), int(pts_against), self._hash(self.store) & ((1 << 64) - 1), side, seed,
                       opp_name)


# ---- a pool: N matches at once ----------------------------------------------------------

_runner: MatchRunner | None = None


def _init(kernel_path: str) -> None:
    global _runner
    _runner = MatchRunner(kernel_path)


def _play(job: tuple) -> Episode:
    assert _runner is not None
    return _runner.play(**job[0]) if isinstance(job, tuple) and isinstance(job[0], dict) else _runner.play(*job)


class RolloutPool:
    """`n` processes, each with its own kernel. Results come back in job order."""

    def __init__(self, kernel_path: str | Path, n: int) -> None:
        self.n = n
        self._local = MatchRunner(kernel_path) if n <= 1 else None
        self._pool = None if n <= 1 else mp.get_context("spawn").Pool(n, _init, (str(kernel_path),))

    def run(self, jobs: list[tuple]) -> list[Episode]:
        """Each job is the positional arguments of MatchRunner.play."""
        if self._pool is None:
            return [self._local.play(*j) for j in jobs]
        return self._pool.map(_play, jobs, chunksize=1)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
