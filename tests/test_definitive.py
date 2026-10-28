"""The verdict's elite entrants: top live cells by archive rating, hashed against their genome files."""

import json

import numpy as np
import pytest

from arena.definitive import load_elites
from arena.genome import make_genome

KERNEL = "ab" * 32


def _cell(out, rng, gen, rating, fitness, genome_len=8):
    params = rng.standard_normal(genome_len).astype(np.float32)
    g = make_genome(params, KERNEL, (), "elite", gen)
    (out / "genomes" / f"{g.id}.bin").write_bytes(params.tobytes())
    return {"genome_id": g.id, "gen": gen, "era": 0, "rating": rating, "fitness": fitness}


def test_load_elites_ranks_by_rating_then_fitness_and_labels(tmp_path):
    (tmp_path / "genomes").mkdir()
    rng = np.random.default_rng(1)
    cells = [_cell(tmp_path, rng, 10, 1500.0, 0.1), _cell(tmp_path, rng, 20, 1700.0, 0.2),
             _cell(tmp_path, rng, 30, 1700.0, 0.9), _cell(tmp_path, rng, 40, None, 2.0)]
    (tmp_path / "latest.json").write_text(json.dumps({"cells": cells}))
    elites = load_elites(tmp_path, 3, 8, KERNEL, "run 6 ")
    assert [e.genome.id for e in elites] == [cells[2]["genome_id"], cells[1]["genome_id"], cells[0]["genome_id"]]
    assert [e.label for e in elites] == ["run 6 elite #1 (gen 30)", "run 6 elite #2 (gen 20)", "run 6 elite #3 (gen 10)"]
    assert all(e.kind == "elite" for e in elites) and elites[0].rating == 1700.0
    assert load_elites(tmp_path, 0, 8, KERNEL) == []


def test_load_elites_refuses_a_genome_that_does_not_hash_to_its_id(tmp_path):
    (tmp_path / "genomes").mkdir()
    rng = np.random.default_rng(2)
    c = _cell(tmp_path, rng, 5, 1600.0, 0.3)
    (tmp_path / "genomes" / f"{c['genome_id']}.bin").write_bytes(np.zeros(8, dtype=np.float32).tobytes())
    (tmp_path / "latest.json").write_text(json.dumps({"cells": [c]}))
    with pytest.raises(ValueError, match="does not hash"):
        load_elites(tmp_path, 1, 8, KERNEL)
