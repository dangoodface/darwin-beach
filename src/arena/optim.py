"""Optimisers speak one protocol: ask() for candidates, tell() with their results.

The archive is the shared object, so hand-written MAP-Elites, a pyribs CMA-ME
adapter and OpenAI-ES can plug into the same loop.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np

from .archive import Archive
from .genome import Genome, SIGMA, iso_line, make_genome, mutate, random_params


class Replay(NamedTuple):
    """One match worth re-running in the browser: the candidate's first evaluation match."""

    seed: int
    opponent_id: str
    side: str  # "A" (left) or "B"
    final_hash: int
    ticks: int
    points: tuple[int, int]  # (candidate, opponent)


class EvalResult(NamedTuple):
    genome: Genome
    bd: tuple[float, ...]
    fitness: float
    replay: Replay | None = None
    margins: tuple[float, ...] | None = None  # mean margin per opponent, in probe order
    desc: tuple[float, ...] | None = None  # all eight mean descriptors, ABI order; bd is desc at the grid's indices


class Optimizer(Protocol):
    def ask(self) -> list[Genome]: ...

    def tell(self, results: list[EvalResult]) -> list[str]:
        """Feed back results in the order asked; returns each add outcome."""


class MapElites:
    """Random initialisation until `init_size` cells are filled, then variation.

    Each child picks a random elite and either mutates it (Gaussian) or, with
    probability `crossover_frac` and at least two elites, crosses it with a second
    random elite (iso+line). Candidates are generated in a fixed order from the
    one Generator, so the same seed gives the same run.
    """

    def __init__(
        self,
        archive: Archive,
        rng: np.random.Generator,
        kernel_sha: str,
        genome_len: int,
        batch_size: int = 64,
        init_size: int = 100,
        sigma: float = SIGMA,
        crossover_frac: float = 0.5,
    ) -> None:
        self.archive = archive
        self.rng = rng
        self.kernel_sha = kernel_sha
        self.genome_len = genome_len
        self.batch_size = batch_size
        self.init_size = init_size
        self.sigma = sigma
        self.crossover_frac = crossover_frac
        self.generation = 0
        self._random_issued = 0

    def _initialising(self) -> bool:
        # the cap stops a degenerate simulator that maps everything to one cell from looping forever
        return self.archive.filled() < self.init_size and self._random_issued < 20 * self.init_size

    def ask(self) -> list[Genome]:
        gen = self.generation
        self.generation += 1
        out: list[Genome] = []
        if self._initialising():
            for _ in range(self.batch_size):
                p = random_params(self.rng, self.genome_len)
                out.append(make_genome(p, self.kernel_sha, (), "random", gen))
            self._random_issued += self.batch_size
            return out
        elites = self.archive.elites()
        for _ in range(self.batch_size):
            a = elites[int(self.rng.integers(len(elites)))].genome
            if len(elites) >= 2 and self.rng.random() < self.crossover_frac:
                b = elites[int(self.rng.integers(len(elites)))].genome
                p = iso_line(a.params, b.params, self.rng)
                out.append(make_genome(p, self.kernel_sha, (a.id, b.id), "iso_line", gen))
            else:
                p = mutate(a.params, self.rng, self.sigma)
                out.append(make_genome(p, self.kernel_sha, (a.id,), "mutate", gen))
        return out

    def tell(self, results: list[EvalResult]) -> list[str]:
        return [self.archive.add(r.genome, r.bd, r.fitness) for r in results]
