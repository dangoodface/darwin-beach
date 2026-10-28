"""The seam between the trainer and a match simulator.

The real simulator is the Rust kernel (src/arena/kernel.py, Track K). Until it
lands, FakeSimulator answers the same questions with deterministic nonsense so
the whole loop, store and exporter can run.
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple, Protocol

import numpy as np

N_DESC = 8
DESCRIPTOR_NAMES = ("court_position", "airtime", "hits", "mean_hit_height", "mean_speed_after_hit",
                    "jumps", "distance", "points")  # docs/ABI.md order, kernel version 1
DESCRIPTOR_NAMES_V2 = DESCRIPTOR_NAMES + ("juggled_points", "max_streak")  # kernel version 2 adds two
DESCRIPTOR_NAMES_V3 = DESCRIPTOR_NAMES_V2 + ("spiked_points",)  # kernel version 3 adds one


def descriptor_names(kernel_version: int) -> tuple[str, ...]:
    """The descriptors a kernel version reports per side; the fake (version 0) reports eight."""
    if kernel_version >= 3:
        return DESCRIPTOR_NAMES_V3
    return DESCRIPTOR_NAMES_V2 if kernel_version == 2 else DESCRIPTOR_NAMES
MASK64 = (1 << 64) - 1
SCRIPTED_SLOT = 0xFFFFFFFF  # ABI: this slot index means the kernel's scripted bot
SCRIPTED_ID = "scripted-v1"  # its id in our files; there is no genome file for it
ORACLE_SLOT_BASE = 0xFFFFFF00  # oracle builds (plan B4): slot ORACLE_SLOT_BASE + i plays rung i


def oracle_id(rung: int) -> str:
    """An oracle rung's id in our files, like SCRIPTED_ID: no genome file, a slot in `bots`."""
    return f"oracle-v1-rung{rung}"


class MatchResult(NamedTuple):
    final_hash: int  # u64
    ticks: int
    points_a: int
    points_b: int
    flags: int
    desc_a: tuple[float, ...]  # ABI order: 8 values, 10 in kernel version 2, 11 from version 3
    desc_b: tuple[float, ...]
    tick_hashes: tuple[int, ...] | None  # only when record_hashes


class Simulator(Protocol):
    genome_len: int
    kernel_sha: str  # hex sha256 of the wasm bytes; FakeSimulator uses "fake-" plus a fixed string
    kernel_version: int
    scripted_slot: int | None  # slot index that plays the scripted bot, or None if the simulator has none
    bots: dict[str, int]  # the scripted bots it plays, by id: SCRIPTED_ID at scripted_slot, an oracle build's rungs

    def load_genomes(self, table: np.ndarray) -> None:
        """Shape (n, genome_len), float32, slot i = row i."""

    def play(
        self, seed: int, a: int, b: int, max_ticks: int = 3000, record_hashes: bool = False,
        hold_a: int = 1, hold_b: int = 1,
    ) -> MatchResult:
        """`hold_a`, `hold_b`: ticks each side holds its action (kernel version 3); 1 acts every tick."""

    def play_batch(self, jobs: list[tuple[int, ...]], max_ticks: int = 3000) -> list[MatchResult]:
        """Jobs are (seed, a, b) or (seed, a, b, hold_a, hold_b), slot indexes into the loaded table."""


def _u64(*parts: bytes) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return int.from_bytes(h.digest()[:8], "little")


def _splitmix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


class FakeSimulator:
    """Deterministic stand-in for the kernel.

    Not physics. Each genome is projected through fixed random vectors into eight
    pseudo-descriptors and a pseudo-strength, so small mutations make small moves
    on the map and there is something for MAP-Elites to climb. The seed and both
    genomes are hashed into the noise, so a match is a pure function of its inputs.
    """

    genome_len = 515
    kernel_version = 0
    scripted_slot = SCRIPTED_SLOT
    kernel_sha = "fake-" + hashlib.sha256(b"arena fake simulator v0").hexdigest()[:16]

    def __init__(self) -> None:
        rng = np.random.default_rng(20261026)
        self._proj = rng.standard_normal((N_DESC, self.genome_len)) / np.sqrt(self.genome_len)
        self._target = rng.standard_normal(self.genome_len) * 0.5
        self._digest: list[int] = []
        self._desc = np.zeros((0, N_DESC))
        self._strength = np.zeros(0)
        # a pretend scripted bot: a fixed digest, middling strength, a net-hugging, grounded style
        self._scripted = (_u64(b"fake scripted-v1"), -0.8, np.array([0.8, 0.2, 30.0, 8.0, 12.0, 6.0, 150.0, 0.0]))
        # a pretend oracle rung: another fixed digest, stronger than the coach, a deeper, airborne style
        self._oracle = (_u64(b"fake oracle-v1-rung0"), -0.2, np.array([0.5, 0.4, 40.0, 10.0, 15.0, 15.0, 250.0, 0.0]))

    @property
    def bots(self) -> dict[str, int]:
        """The coach and one pretend oracle rung, listed as an oracle build lists its bots."""
        return {SCRIPTED_ID: SCRIPTED_SLOT, oracle_id(0): ORACLE_SLOT_BASE}

    def load_genomes(self, table: np.ndarray) -> None:
        table = np.ascontiguousarray(table, dtype=np.float32)
        if table.ndim != 2 or table.shape[1] != self.genome_len:
            raise ValueError(f"expected (n, {self.genome_len}) float32, got {table.shape}")
        g = table.astype(np.float64)
        z = g @ self._proj.T  # roughly N(0, |g|^2 / genome_len) per descriptor
        squash = 0.5 + 0.5 * z * 2.0 / (1.0 + np.abs(z * 2.0))
        desc = np.empty_like(squash)
        desc[:, 0] = squash[:, 0]  # court_position in [0, 1]
        desc[:, 1] = squash[:, 1]  # airtime in [0, 1]
        desc[:, 2] = np.floor(squash[:, 2] * 60.0)  # hits
        desc[:, 3] = squash[:, 3] * 20.0  # mean_hit_height
        desc[:, 4] = squash[:, 4] * 22.5  # mean_speed_after_hit
        desc[:, 5] = np.floor(squash[:, 5] * 40.0)  # jumps
        desc[:, 6] = squash[:, 6] * 400.0  # distance
        desc[:, 7] = 0.0  # points: filled per match
        self._desc = desc
        self._strength = -np.mean((g - self._target) ** 2, axis=1)
        self._digest = [_u64(row.tobytes()) for row in table]

    def _slot(self, i: int):
        if i == SCRIPTED_SLOT:
            return self._scripted
        if i == ORACLE_SLOT_BASE:
            return self._oracle
        if 0 <= i < len(self._digest):
            return self._digest[i], self._strength[i], self._desc[i]
        return None

    def play(
        self, seed: int, a: int, b: int, max_ticks: int = 3000, record_hashes: bool = False,
        hold_a: int = 1, hold_b: int = 1,
    ) -> MatchResult:
        ra, rb = self._slot(a), self._slot(b)
        if ra is None or rb is None or hold_a < 1 or hold_b < 1:
            return MatchResult(0, 0, 0, 0, 4, (0.0,) * N_DESC, (0.0,) * N_DESC, None)
        (dig_a, str_a, desc_a), (dig_b, str_b, desc_b) = ra, rb
        h = _splitmix(seed & MASK64 ^ _splitmix(dig_a) ^ (_splitmix(dig_b) << 1 & MASK64))
        if (hold_a, hold_b) != (1, 1):  # a held side plays a different match; holds of 1 change nothing
            h = _splitmix(h ^ (hold_a << 32 | hold_b))
        ticks = 300 + h % max(1, min(max_ticks, 3000) - 299)
        edge = (str_a - str_b) * 4.0  # roughly +-1 for typical genomes
        noise = ((h >> 20) & 0xFFFF) / 65535.0 - 0.5  # [-0.5, 0.5]
        a_wins = (edge + noise * 1.5) >= 0.0
        lost = int(np.clip(round(4.0 / (1.0 + np.exp(abs(edge) * 3.0)) * 2.0 + (h >> 40) % 2), 0, 4))
        points_a, points_b = (5, lost) if a_wins else (lost, 5)
        desc_a, desc_b = desc_a.copy(), desc_b.copy()
        desc_a[7], desc_b[7] = points_a, points_b
        hashes = None
        if record_hashes:
            hh, out = h, []
            for _ in range(ticks):
                hh = _splitmix(hh)
                out.append(hh)
            hashes = tuple(out)
        return MatchResult(
            final_hash=_splitmix(h ^ ticks),
            ticks=int(ticks),
            points_a=points_a,
            points_b=points_b,
            flags=0 if ticks < max_ticks else 2,
            desc_a=tuple(float(x) for x in desc_a),
            desc_b=tuple(float(x) for x in desc_b),
            tick_hashes=hashes,
        )

    def play_batch(self, jobs: list[tuple[int, ...]], max_ticks: int = 3000) -> list[MatchResult]:
        return [self.play(j[0], j[1], j[2], max_ticks, False, *j[3:5]) for j in jobs]
