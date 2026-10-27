"""The kernel's own network, in numpy: the observation mirror and the policy.

Layout and rules are docs/ABI.md: 12 inputs, 32 hidden units with softsign, 3 linear
outputs; parameters W1[32][12], b1[32], W2[3][32], b2[3] as f32, 515 in all. An action
bit is pressed when its output is above zero. Training treats the three outputs as
logits of independent Bernoulli presses; the deterministic game plays their mode,
which is exactly that rule.
"""

from __future__ import annotations

import numpy as np

N_IN, N_HID, N_OUT = 12, 32, 3
GENOME_LEN = N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT  # 515

# State block (f64, ABI): ball x y vx vy | A x y vx vy | B x y vx vy | lives A, B | tick | delay | last point
STATE_LEN = 17
LIVES_A, LIVES_B, TICK = 12, 13, 14

# The observation for side A (dir = -1) and side B (dir = +1), as indexes into the state block and
# signs, from the ABI: x*dir, y, vx*dir, vy, bx*dir, by, bvx*dir, bvy, ox*(-dir), oy, ovx*(-dir), ovy.
_OWN = {0: (4, 5, 6, 7), 1: (8, 9, 10, 11)}
_OPP = {0: (8, 9, 10, 11), 1: (4, 5, 6, 7)}
OBS_IDX = {}
OBS_SIGN = {}
for side, dirn in ((0, -1.0), (1, 1.0)):
    ox, oy, ovx, ovy = _OWN[side]
    px, py, pvx, pvy = _OPP[side]
    OBS_IDX[side] = np.array([ox, oy, ovx, ovy, 0, 1, 2, 3, px, py, pvx, pvy])
    OBS_SIGN[side] = np.array([dirn, 1, dirn, 1, dirn, 1, dirn, 1, -dirn, 1, -dirn, 1])


def mirror_obs(state: np.ndarray, side: int) -> np.ndarray:
    """The 12 inputs a genome sees from `side` (0 = A, 1 = B), from the state block."""
    return state[OBS_IDX[side]] * OBS_SIGN[side] / 10.0  # divided, not multiplied by 0.1, as the kernel does


class ObsFeed:
    """What the kernel's network actually sees tick by tick.

    The kernel refreshes each slime's observation at the end of a tick only when no point
    was decided (the original Python returns before `updateState`), so on the tick after a
    point both sides still see the previous tick's observation. `get` reproduces that:
    call `after_step(result)` with what `step()` returned, and `reset()` at a new match.
    """

    def __init__(self) -> None:
        self.prev: list[np.ndarray | None] = [None, None]
        self.stale = False

    def reset(self) -> None:
        self.prev, self.stale = [None, None], False

    def get(self, state: np.ndarray, side: int) -> np.ndarray:
        if not (self.stale and self.prev[side] is not None):
            self.prev[side] = mirror_obs(state, side)
        return self.prev[side]

    def after_step(self, result: int) -> None:
        self.stale = result != 0


def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (GENOME_LEN,):
        raise ValueError(f"expected {GENOME_LEN} parameters, got {p.shape}")
    a = N_HID * N_IN
    w1 = p[:a].reshape(N_HID, N_IN)
    b1 = p[a : a + N_HID]
    c = a + N_HID
    w2 = p[c : c + N_OUT * N_HID].reshape(N_OUT, N_HID)
    b2 = p[c + N_OUT * N_HID :]
    return w1, b1, w2, b2


def pack(w1, b1, w2, b2) -> np.ndarray:
    return np.concatenate([np.ravel(w1), np.ravel(b1), np.ravel(w2), np.ravel(b2)]).astype(np.float32)


def logits(obs: np.ndarray, net) -> np.ndarray:
    """Fast path (BLAS) for rollouts. Not bit-identical to the kernel's sequential sums."""
    w1, b1, w2, b2 = net
    h = w1 @ obs + b1
    h = h / (1.0 + np.abs(h))
    return w2 @ h + b2


def logits_exact(obs: np.ndarray, net) -> np.ndarray:
    """The kernel's own arithmetic: each unit accumulates `acc = acc + w[j] * x[j]` in order,
    with no fused multiply-add. `obs` may be one row (12,) or a batch (N, 12)."""
    w1, b1, w2, b2 = net
    x = np.atleast_2d(obs)
    acc = np.broadcast_to(b1, (x.shape[0], N_HID)).copy()
    for j in range(N_IN):
        acc = acc + x[:, j : j + 1] * w1[:, j][None, :]
    h = acc / (1.0 + np.abs(acc))
    out = np.broadcast_to(b2, (x.shape[0], N_OUT)).copy()
    for j in range(N_HID):
        out = out + h[:, j : j + 1] * w2[:, j][None, :]
    return out[0] if np.ndim(obs) == 1 else out


def mode_bits(lg: np.ndarray) -> np.ndarray:
    """The deterministic policy: press when the logit is above zero (the kernel's rule)."""
    return (lg > 0.0).astype(np.uint8)


def bits_to_action(bits) -> int:
    return int(bits[0]) | int(bits[1]) << 1 | int(bits[2]) << 2


def log_prob(lg: np.ndarray, bits: np.ndarray) -> float:
    """log P(bits) for independent Bernoulli presses with probability sigmoid(logit)."""
    # log sigmoid(l) = -log(1 + e^-l); log(1 - sigmoid(l)) = log sigmoid(-l)
    return float(np.sum(np.where(bits == 1, -np.logaddexp(0.0, -lg), -np.logaddexp(0.0, lg))))
