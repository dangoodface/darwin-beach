"""The MAP-Elites grid: one elite per cell of the behaviour space."""

from __future__ import annotations

from dataclasses import dataclass

from .genome import Genome

NEW, REPLACED, REJECTED = "new", "replaced", "rejected"


@dataclass(frozen=True)
class Entry:
    genome: Genome
    bd: tuple[float, ...]
    fitness: float


class Archive:
    """Fixed-resolution grid over `bounds`, e.g. 20 by 20 over court position and airtime.

    A behaviour outside the bounds is clamped to the edge cell, so nothing is lost.
    A newcomer replaces the resident only when strictly fitter, so ties keep the
    older elite and a run stays reproducible.

    QD-score is the sum of (fitness + qd_offset) over filled cells. Fitness is a
    score margin that can be negative (about -5 to 5), so the offset keeps every
    filled cell worth at least zero and a bigger map always scores higher.
    """

    def __init__(
        self,
        bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0)),
        resolution: tuple[int, ...] = (20, 20),
        qd_offset: float = 5.0,
    ) -> None:
        if len(bounds) != len(resolution):
            raise ValueError("bounds and resolution need the same length")
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
        self.resolution = tuple(int(r) for r in resolution)
        self.qd_offset = qd_offset
        self._cells: dict[tuple[int, ...], Entry] = {}

    def cell_of(self, bd: tuple[float, ...]) -> tuple[int, ...]:
        idx = []
        for x, (lo, hi), n in zip(bd, self.bounds, self.resolution):
            i = int((x - lo) / (hi - lo) * n)
            idx.append(min(n - 1, max(0, i)))
        return tuple(idx)

    def get(self, cell: tuple[int, ...]) -> Entry | None:
        return self._cells.get(cell)

    def add(self, genome: Genome, bd: tuple[float, ...], fitness: float) -> str:
        cell = self.cell_of(bd)
        old = self._cells.get(cell)
        if old is None:
            self._cells[cell] = Entry(genome, tuple(bd), float(fitness))
            return NEW
        if fitness > old.fitness:
            self._cells[cell] = Entry(genome, tuple(bd), float(fitness))
            return REPLACED
        return REJECTED

    def cells(self) -> dict[tuple[int, ...], Entry]:
        """Filled cells in a fixed order (row-major), so iteration is reproducible."""
        return {k: self._cells[k] for k in sorted(self._cells)}

    def elites(self) -> list[Entry]:
        return list(self.cells().values())

    def filled(self) -> int:
        return len(self._cells)

    def capacity(self) -> int:
        n = 1
        for r in self.resolution:
            n *= r
        return n

    def qd_score(self) -> float:
        return sum(e.fitness + self.qd_offset for e in self._cells.values())

    def best(self) -> Entry | None:
        if not self._cells:
            return None
        return max(self.cells().values(), key=lambda e: e.fitness)
