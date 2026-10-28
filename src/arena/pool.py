"""Run 6's pool and warm start: genomes from earlier runs, loaded under this kernel.

A genome's id is a hash of its parameters and the kernel it runs on, so a bot
from an older kernel gets a new id here; `Member.source` keeps the old one.
Paths are absolute or relative to $ARENA_DATA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .genome import Genome, make_genome
from .ladder import ANCHOR_RATING, Member


def resolve(path: str) -> Path:
    from .store import data_root  # lazy: the store imports the run loop

    p = Path(path).expanduser()
    return p if p.is_absolute() else data_root() / p


def read_params(path: Path, genome_len: int) -> np.ndarray:
    data = path.read_bytes()
    if len(data) != genome_len * 4:
        raise ValueError(f"{path}: {len(data)} bytes, expected {genome_len * 4}")
    return np.frombuffer(data, dtype="<f4").astype(np.float32)


def load_pool(entries: Sequence[dict], kernel_sha: str, genome_len: int) -> list[Member]:
    """Each entry is {path, label, id}: the genome file, the name the page shows and
    the id it had where it came from. Ratings are provisional until Ladder.rate_pool."""
    members = []
    for e in entries:
        g = make_genome(read_params(resolve(e["path"]), genome_len), kernel_sha, (), "pool", -1)
        members.append(Member(g.id, "pool", ANCHOR_RATING, 0, g, label=e.get("label"), source=e.get("id")))
    ids = [m.id for m in members]
    if len(set(ids)) != len(ids):
        raise ValueError("the pool lists the same genome twice")
    return members


def load_warm(runs: Sequence[str], kernel_sha: str, genome_len: int, exclude: set[str] = frozenset()) -> list[Genome]:
    """Every cell's elite in each export (a run id under out/, or a path), as
    generation-zero genomes under this kernel; a genome seen twice is loaded once,
    and `exclude` (the pool) is skipped: a member is a rival, not a contestant."""
    out: list[Genome] = []
    seen: set[str] = set(exclude)
    for r in runs:
        d = resolve(r if "/" in r else f"out/{r}")
        cells = json.loads((d / "latest.json").read_text())["cells"]
        for c in cells:
            g = make_genome(read_params(d / "genomes" / f"{c['genome_id']}.bin", genome_len), kernel_sha, (), "warm", 0)
            if g.id not in seen:
                seen.add(g.id)
                out.append(g)
    return out
