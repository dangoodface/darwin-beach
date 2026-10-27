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
    rating: float | None = None  # v1: Elo against the ladder members; None in v0
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
        if self._initialising():
            return self._random_batch(gen)
        return self._vary(gen)

    def _random_batch(self, gen: int) -> list[Genome]:
        out = []
        for _ in range(self.batch_size):
            p = random_params(self.rng, self.genome_len)
            out.append(make_genome(p, self.kernel_sha, (), "random", gen))
        self._random_issued += self.batch_size
        return out

    def _vary(self, gen: int) -> list[Genome]:
        out: list[Genome] = []
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


class _ArchiveView:
    """The little of a pyribs archive that an EvolutionStrategyEmitter reads, backed by our
    Archive: sizes, dtypes, and an elite to restart from, drawn by CmaMe's Generator."""

    def __init__(self, owner: "CmaMe", solution_dim: int, measure_dim: int) -> None:
        self._owner = owner
        self.solution_dim = solution_dim
        self.measure_dim = measure_dim
        self.dtypes = {"solution": np.float64, "objective": np.float64, "measures": np.float64}

    def sample_elites(self, n: int) -> dict[str, np.ndarray]:
        return {"solution": np.stack([self._owner._restart_point() for _ in range(n)])}


class CmaMe(MapElites):
    """CMA-ME (Fontaine et al. 2020): CMA-ES emitters ranked by what they do to the archive.

    The same random start as MapElites until `init_size` cells are filled (the same draws
    from the same Generator, so an A/B against MapElites shares its first generations).
    Then `emitters` pyribs EvolutionStrategyEmitters each propose batch_size / emitters
    genomes a generation, so a generation costs what a MapElites one does. No crossover:
    the emitters make every proposal.

    The emitters read our Archive through _ArchiveView and are told what happened in it,
    as pyribs' GridArchive reports it for CMA-ME: a new cell (status 2, value the fitness),
    an improvement (1, the fitness minus the evicted elite's) or a rejection (0, the
    fitness minus the resident's). Each emitter's seed, its starting elite and every elite
    it restarts from come from the optimiser's Generator, never from pyribs' defaults.
    A child's one parent is the elite its emitter last started from.
    """

    def __init__(
        self,
        archive: Archive,
        rng: np.random.Generator,
        kernel_sha: str,
        genome_len: int,
        batch_size: int = 64,
        init_size: int = 100,
        emitters: int = 4,
        sigma0: float = 0.1,
        es: str = "cma_es",
    ) -> None:
        if emitters < 1 or batch_size % emitters:
            raise ValueError(f"batch_size {batch_size} must split evenly over {emitters} emitters")
        try:  # fail at the start of a run, not when the emitters start after the random phase
            import ribs  # noqa: F401
        except ImportError as e:
            raise ImportError("optim.kind = 'cma_me' needs pyribs: uv sync --extra qd") from e
        super().__init__(archive, rng, kernel_sha, genome_len, batch_size, init_size)
        self.emitters, self.sigma0, self.es = emitters, sigma0, es
        self._view = _ArchiveView(self, genome_len, len(archive.resolution))
        self._emitters: list = []
        self._anchors: list[str] = []
        self._calling = 0  # the emitter being told, so a restart is credited to it
        self._batches: list[np.ndarray] | None = None  # this generation's float64 proposals, per emitter
        self._told: list[tuple[int, float, float, tuple[float, ...]]] = []  # status, value, fitness, bd

    def _restart_point(self) -> np.ndarray:
        elites = self.archive.elites()
        e = elites[int(self.rng.integers(len(elites)))]
        if self._calling < len(self._anchors):
            self._anchors[self._calling] = e.genome.id
        else:
            self._anchors.append(e.genome.id)
        return e.genome.params.astype(np.float64)

    def _start_emitters(self) -> None:
        from ribs.emitters import EvolutionStrategyEmitter  # the qd extra

        for k in range(self.emitters):
            seed = int(self.rng.integers(2**63))
            self._calling = k
            x0 = self._restart_point()
            self._emitters.append(EvolutionStrategyEmitter(
                self._view, x0=x0, sigma0=self.sigma0, ranker="2imp", es=self.es,
                selection_rule="filter", restart_rule="no_improvement",
                batch_size=self.batch_size // self.emitters, seed=seed))

    def ask(self) -> list[Genome]:
        gen = self.generation
        self.generation += 1
        self._told = []
        if self._initialising():
            self._batches = None
            return self._random_batch(gen)
        if not self._emitters:
            self._start_emitters()
        self._batches = [em.ask() for em in self._emitters]
        return [make_genome(sol.astype(np.float32), self.kernel_sha, (self._anchors[k],), "cma_me", gen)
                for k, batch in enumerate(self._batches) for sol in batch]

    def tell(self, results: list[EvalResult]) -> list[str]:
        outcomes = []
        for r in results:
            resident = self.archive.get(self.archive.cell_of(r.bd))
            outcome = self.archive.add(r.genome, r.bd, r.fitness)
            if resident is None:
                status, value = 2, r.fitness
            else:
                status, value = (1 if outcome == "replaced" else 0), r.fitness - resident.fitness
            self._told.append((status, value, r.fitness, tuple(r.bd)))
            outcomes.append(outcome)
        if self._batches is not None and len(self._told) == self.batch_size:
            self._update_emitters()
        return outcomes

    def _update_emitters(self) -> None:
        per = self.batch_size // self.emitters
        for k, (em, sols) in enumerate(zip(self._emitters, self._batches)):
            told = self._told[k * per : (k + 1) * per]
            self._calling = k
            em.tell(sols, np.array([t[2] for t in told]), np.array([t[3] for t in told]),
                    {"status": np.array([t[0] for t in told], dtype=np.int32),
                     "value": np.array([t[1] for t in told], dtype=np.float64)})
        self._batches = None

