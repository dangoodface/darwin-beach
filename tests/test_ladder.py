"""The honest ladder (v1): Elo by maximum likelihood, promotion, eras, the
re-score, and what the exporter writes for it. All but the last test run on
FakeSimulator; the last runs the real kernel."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

from arena.config import Config, load_config
from arena.export import ExportObserver
from arena.kernel import find_kernel
from arena.ladder import ANCHOR_RATING, elo_mle, expected_score
from arena.run import run
from arena.sim import SCRIPTED_ID
from arena.store import StoreObserver

NOW = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)


def _cfg(**kw):
    base = dict(seed=11, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                resolution=(10, 10), ladder_enabled=True, ladder_check_every=5, ladder_seeds=10)
    base.update(kw)
    return Config(**base)


class Rec:
    def __init__(self):
        self.gens = []

    def on_start(self, ctx): pass
    def on_generation(self, ctx, info): self.gens.append(info)
    def on_end(self, ctx, info): pass


def _fingerprint(ctx):
    return [(k, e.genome.id, e.fitness, e.bd) for k, e in ctx.archive.cells().items()]


# ---- Elo ---------------------------------------------------------------------


def test_elo_against_one_opponent_has_a_closed_form():
    for s, n in [(0, 16), (3, 16), (8, 16), (16, 16), (0.5, 1), (75, 100)]:
        want = ANCHOR_RATING + 400 * math.log10((s + 0.5) / (n - s + 0.5))
        assert elo_mle([(ANCHOR_RATING, s, n)]) == pytest.approx(want, abs=1e-6)


def test_elo_solves_the_likelihood_equation_and_rises_with_wins():
    members = [1000.0, 1400.0, 1650.0]
    previous = -math.inf
    for wins in range(0, 21, 4):
        records = [(r, wins, 20) for r in members]
        rating = elo_mle(records)
        expected = sum((n + 1) * expected_score(rating, r) for r, _, n in records)
        assert expected == pytest.approx(sum(s + 0.5 for _, s, _ in records), abs=1e-6)
        assert rating > previous
        previous = rating


def test_a_perfect_or_a_pointless_record_stays_finite():
    assert 1000 < elo_mle([(1000.0, 100, 100)]) < 2000
    assert 0 < elo_mle([(1000.0, 0, 100)]) < 1000


# ---- the loop ----------------------------------------------------------------


def test_promotions_happen_and_ratings_only_rise():
    rec = Rec()
    ctx = run(_cfg(), [rec], generations=120)
    ladder = ctx.ladder
    assert ladder.era >= 2 and ladder.members[0].id == SCRIPTED_ID and ladder.members[0].rating == 1000.0
    ratings = [m.rating for m in ladder.members]
    assert all(b > a for a, b in zip(ratings, ratings[1:]))
    checks = [g.challenge for g in rec.gens if g.challenge is not None]
    promoted = [c for c in checks if c.promoted]
    assert len(promoted) == ladder.era
    for c in promoted:
        assert c.headtohead.winrate_a >= 0.55 and c.rating == c.member.rating
        assert len(c.vs_members) == c.member.since_era  # one block against every earlier member
        assert c.cycle == tuple((p.b, p.winrate_a) for p in c.vs_members[-5:])
    # a challenge that failed lost the head-to-head or could not out-rate the champion of its era
    for c in checks:
        if not c.promoted:
            assert c.headtohead.winrate_a < 0.55 or c.rating <= ladder.members[c.era].rating


def test_rescore_keeps_every_elite_in_its_cell():
    first_bd, rescored_at = {}, []

    class Track:
        def on_start(self, ctx): pass
        def on_end(self, ctx, info): pass

        def on_generation(self, ctx, info):
            for r in info.results:
                first_bd.setdefault(r.genome.id, r.bd)
            if info.rescored:
                cells = ctx.archive.cells()
                assert len(info.rescored) == len(cells)
                for r in info.rescored:
                    assert r.bd == first_bd[r.genome.id]  # the same descriptors, float for float
                    assert cells[ctx.archive.cell_of(r.bd)].genome.id == r.genome.id
                    assert r.rating is not None
                rescored_at.append(info.generation)

    run(_cfg(), [Track()], generations=60)
    assert rescored_at


def test_ladder_runs_reproduce_bit_for_bit():
    a, b = run(_cfg(), generations=60), run(_cfg(), generations=60)
    assert [(m.id, m.rating) for m in a.ladder.members] == [(m.id, m.rating) for m in b.ladder.members]
    assert _fingerprint(a) == _fingerprint(b)
    assert a.cfg.ladder_seed_list == b.cfg.ladder_seed_list and len(a.cfg.ladder_seed_list) == 10


def test_era_zero_descriptors_are_v0_descriptors():
    v0, v1 = Rec(), Rec()
    run(_cfg(ladder_enabled=False), [v0], generations=1)
    run(_cfg(), [v1], generations=1)
    a, b = v0.gens[0].results, v1.gens[0].results
    assert [r.genome.id for r in a] == [r.genome.id for r in b]
    assert [r.bd for r in a] == [r.bd for r in b]
    assert all(r.rating is None for r in a) and all(r.rating is not None for r in b)


def test_configs():
    assert load_config("configs/slime-v1.toml").ladder_enabled
    assert not load_config("configs/slime-v0.toml").ladder_enabled
    with pytest.raises(ValueError):
        _cfg(ladder_threshold=0.4).validate()


# ---- the files the page reads ------------------------------------------------


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("ladder")
    so, rec = StoreObserver(root), Rec()
    ctx = run(_cfg(), [so, ExportObserver(so, 1e9, root), rec], generations=120, now=NOW)
    out = root / "out" / ctx.run_id

    def load(name):
        return json.loads((out / name).read_text())

    return ctx, rec, out, load


def test_ladder_json(exported):
    ctx, _, out, load = exported
    ladder = load("ladder.json")
    assert ladder["era"] == ctx.ladder.era and ladder["cycle_window"] == 5
    assert [m["id"] for m in ladder["members"]] == [m.id for m in ctx.ladder.members]
    assert ladder["members"][0] == {"id": SCRIPTED_ID, "kind": "scripted", "rating": 1000.0, "since_era": 0}
    for k, m in enumerate(ladder["members"][1:], 1):
        assert m["kind"] == "champion" and m["since_era"] == k
        assert m["promoted_at_gen"] >= 0 and m["winrate_vs_previous"] >= 0.55
        assert (out / "genomes" / f"{m['id']}.bin").is_file()
    win = ladder["matrix"]["winrate"]
    assert ladder["matrix"]["ids"] == [m["id"] for m in ladder["members"]]
    for i, row in enumerate(win):
        assert row[i] is None
        for j, w in enumerate(row):
            if i != j:
                assert w + win[j][i] == pytest.approx(1.0)
    assert all(win[k][k - 1] >= 0.55 for k in range(1, len(win)))  # each champion beat the one before


def test_ratings_and_champion_in_latest_curve_and_tree(exported):
    ctx, rec, _, load = exported
    latest = load("latest.json")
    stats = latest["stats"]
    assert latest["era"] == ctx.ladder.era
    assert stats["champion_id"] == ctx.ladder.champion.id
    assert stats["champion_rating"] == ctx.ladder.champion.rating
    by_cell = {(c["i"], c["j"]): c for c in latest["cells"]}
    assert all(c["rating"] is not None for c in latest["cells"])
    for cell, e in ctx.archive.cells().items():  # the current era's fitness, after any re-score
        assert by_cell[cell]["fitness"] == e.fitness and by_cell[cell]["genome_id"] == e.genome.id
    best = by_cell[ctx.archive.cell_of(next(e.bd for e in ctx.archive.elites() if e.genome.id == stats["best_id"]))]
    assert best["fitness"] == ctx.archive.best().fitness

    curve = load("curve.json")["points"]
    ratings = [p["champion_rating"] for p in curve]
    assert ratings[0] == 1000.0 and ratings[-1] == ctx.ladder.champion.rating
    assert all(b >= a for a, b in zip(ratings, ratings[1:]))
    by_gen = {p["gen"]: p for p in curve}
    for info in rec.gens:
        if info.rescored:
            # the lit count replays the re-score: the new era's fitness against the floor
            assert by_gen[info.generation]["lit"] == sum(r.fitness >= 0.0 for r in info.rescored)
            assert by_gen[info.generation]["era"] == info.challenge.member.since_era
    assert curve[-1]["lit"] == stats["lit"]

    tree = load("tree.json")
    assert all(n["rating"] is not None for n in tree["nodes"])
    assert [n["id"] for n in tree["nodes"] if n["champion"]] == [ctx.ladder.champion.id]
    assert tree["champion_lineage"][-1] == ctx.ladder.champion.id


def test_v0_files_have_no_ladder(tmp_path):
    so = StoreObserver(tmp_path)
    ctx = run(_cfg(ladder_enabled=False), [so, ExportObserver(so, 1e9, tmp_path)], generations=10, now=NOW)
    out = tmp_path / "out" / ctx.run_id
    latest = json.loads((out / "latest.json").read_text())
    assert not (out / "ladder.json").exists()
    assert latest["stats"]["champion_id"] == latest["stats"]["best_id"] and latest["stats"]["champion_rating"] is None
    assert all(c["rating"] is None for c in latest["cells"])


# ---- the real kernel ---------------------------------------------------------


@pytest.mark.skipif(find_kernel() is None, reason="no kernel: build it (README) or set ARENA_KERNEL")
def test_era_zero_holds_on_the_kernel():
    """Fresh genomes cannot beat scripted-v1 (the first real run's elites lost 16 of 16), so every
    challenge fails, ratings sit below the anchor, and the records stay finite."""
    cfg = _cfg(sim_kind="kernel", kernel_path=str(find_kernel()), batch_size=8, init_size=8, probe_count=1,
               seeds_per_probe=1, ladder_check_every=1, ladder_seeds=8)
    rec = Rec()
    ctx = run(cfg, [rec], generations=3)
    assert ctx.ladder.era == 0 and ctx.ladder.champion.id == SCRIPTED_ID
    checks = [g.challenge for g in rec.gens if g.challenge is not None]
    assert checks and all(not c.promoted and c.headtohead.winrate_a < 0.55 for c in checks)
    assert all(c.headtohead.n == 16 for c in checks)
    ratings = [r.rating for g in rec.gens for r in g.results]
    assert all(0 < r < ANCHOR_RATING for r in ratings)


@pytest.mark.skipif(find_kernel() is None, reason="no kernel: build it (README) or set ARENA_KERNEL")
def test_kernel_descriptors_do_not_depend_on_the_ladder():
    """On the kernel a match is a pure function of its inputs, so a champion joining the ladder
    leaves every candidate's descriptors and replay exactly as they were; only fitness moves."""
    import numpy as np

    from arena.evaluate import Evaluator
    from arena.genome import make_genome, random_params
    from arena.ladder import Ladder, Member
    from arena.workers import Workers

    rng = np.random.default_rng(5)
    with Workers("kernel", str(find_kernel()), 1) as workers:
        sha, _, genome_len, slot = workers.simulator_info
        probes = [make_genome(random_params(rng, genome_len), sha, (), "probe", -1)]
        cands = [make_genome(random_params(rng, genome_len), sha, (), "random", 0) for _ in range(6)]
        ladder = Ladder(slot, (3,))
        before = Evaluator(workers, probes, slot, (1, 2), (0, 1), 3000, ladder).evaluate(cands)
        ladder.members.append(Member(cands[0].id, "champion", 1200.0, 1, cands[0], 0, 0.6))
        after = Evaluator(workers, probes, slot, (1, 2), (0, 1), 3000, ladder).evaluate(cands)
    assert [r.bd for r in before] == [r.bd for r in after]
    assert [r.replay for r in before] == [r.replay for r in after]
    assert any(a.fitness != b.fitness for a, b in zip(before, after))
