"""Genomes: 515 f32 parameters of the 12-32-3 policy, plus how they are born.

All randomness comes from a numpy Generator the caller passes, so a run is a
pure function of its config.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

N_IN, N_HID, N_OUT = 12, 32, 3
GENOME_LEN = N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT  # 515, kernel versions 1 and 2 (docs/ABI.md)
N_HID_V3 = 64  # kernel version 3's wider brain: 12-64-3
GENOME_LEN_V3 = N_HID_V3 * N_IN + N_HID_V3 + N_OUT * N_HID_V3 + N_OUT  # 1027

SIGMA = 0.1  # Gaussian mutation
SIGMA_ISO = 0.01  # iso-line crossover: spread around the first parent
SIGMA_LINE = 0.1  # iso-line crossover: spread along the line to the second parent


@dataclass(frozen=True, eq=False)
class Genome:
    id: str
    params: np.ndarray  # float32, shape (genome_len,)
    parents: tuple[str, ...]  # () for generation zero, one for mutation, two for crossover
    op: str  # "random", "mutate" or "iso_line"
    gen: int


def genome_id(params: np.ndarray, kernel_sha: str) -> str:
    """Hex sha256 of the f32 little-endian bytes followed by the kernel sha as ASCII."""
    data = np.ascontiguousarray(params, dtype="<f4").tobytes()
    return hashlib.sha256(data + kernel_sha.encode("ascii")).hexdigest()


def make_genome(
    params: np.ndarray, kernel_sha: str, parents: tuple[str, ...], op: str, gen: int
) -> Genome:
    p = np.ascontiguousarray(params, dtype=np.float32)
    p.setflags(write=False)
    return Genome(genome_id(p, kernel_sha), p, tuple(parents), op, gen)


def random_params(rng: np.random.Generator, genome_len: int = GENOME_LEN) -> np.ndarray:
    """Fan-in scaled normal weights and zero biases, as a fresh MLP would start."""
    hidden = {GENOME_LEN: N_HID, GENOME_LEN_V3: N_HID_V3}.get(genome_len)
    if hidden is None:
        return rng.standard_normal(genome_len).astype(np.float32) * 0.3
    w1 = rng.standard_normal(hidden * N_IN) / np.sqrt(N_IN)
    w2 = rng.standard_normal(N_OUT * hidden) / np.sqrt(hidden)
    return np.concatenate([w1, np.zeros(hidden), w2, np.zeros(N_OUT)]).astype(np.float32)


def pad_params(v2: np.ndarray) -> np.ndarray:
    """A 515-parameter genome (kernel versions 1 and 2, 12-32-3) in kernel version 3's
    12-64-3 layout: each block keeps its old values first and gains zeros for hidden
    units 32 to 63 (W1's rows, b1's entries, the last 32 columns of each W2 row). The
    zero units output 0, so the padded genome plays exactly the same matches."""
    v2 = np.asarray(v2, dtype=np.float32)
    if v2.shape != (GENOME_LEN,):
        raise ValueError(f"expected {GENOME_LEN} parameters, got {v2.shape}")
    w1, rest = v2[: N_HID * N_IN], v2[N_HID * N_IN :]
    b1, rest = rest[:N_HID], rest[N_HID:]
    w2, b2 = rest[: N_OUT * N_HID].reshape(N_OUT, N_HID), rest[N_OUT * N_HID :]
    pad = N_HID_V3 - N_HID
    out = np.concatenate([
        w1, np.zeros(pad * N_IN, np.float32),
        b1, np.zeros(pad, np.float32),
        np.concatenate([w2, np.zeros((N_OUT, pad), np.float32)], axis=1).ravel(),
        b2,
    ])
    assert out.shape == (GENOME_LEN_V3,)
    return out


def mutate(params: np.ndarray, rng: np.random.Generator, sigma: float = SIGMA) -> np.ndarray:
    z = rng.standard_normal(params.shape[0])
    return (params.astype(np.float64) + sigma * z).astype(np.float32)


def iso_line(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    sigma_iso: float = SIGMA_ISO,
    sigma_line: float = SIGMA_LINE,
) -> np.ndarray:
    """Iso+LineDD crossover (Vassiliades and Mouret 2018).

    child = a + sigma_iso * N(0, I) + sigma_line * N(0, 1) * (b - a). Mostly a small
    isotropic step, plus a long step along the line joining two elites, which is
    how it crosses between niches.
    """
    z = rng.standard_normal(a.shape[0])
    s = rng.standard_normal()
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return (a64 + sigma_iso * z + sigma_line * s * (b64 - a64)).astype(np.float32)
