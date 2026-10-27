"""Genomes: 515 f32 parameters of the 12-32-3 policy, plus how they are born.

All randomness comes from a numpy Generator the caller passes, so a run is a
pure function of its config.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

N_IN, N_HID, N_OUT = 12, 32, 3
GENOME_LEN = N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT  # 515, layout in docs/ABI.md

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
    if genome_len != GENOME_LEN:
        return rng.standard_normal(genome_len).astype(np.float32) * 0.3
    w1 = rng.standard_normal(N_HID * N_IN) / np.sqrt(N_IN)
    w2 = rng.standard_normal(N_OUT * N_HID) / np.sqrt(N_HID)
    return np.concatenate([w1, np.zeros(N_HID), w2, np.zeros(N_OUT)]).astype(np.float32)


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
