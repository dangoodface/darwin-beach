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


# ---- run 6: the pool, its ratings, holding the anchor, the warm start --------


def test_joint_elo_matches_the_single_fit_against_the_anchor():
    from arena.ladder import Pair, joint_elo

    r = joint_elo([SCRIPTED_ID, "a"], [Pair("a", SCRIPTED_ID, 20, 14, 2, 4)], SCRIPTED_ID)
    assert r[SCRIPTED_ID] == ANCHOR_RATING
    assert abs(r["a"] - elo_mle([(ANCHOR_RATING, 15.0, 20)])) < 1e-6


def test_joint_elo_orders_a_transitive_pool_and_explains_its_records():
    from arena.ladder import Pair, joint_elo

    ids = [SCRIPTED_ID, "a", "b"]
    pairs = [Pair("a", SCRIPTED_ID, 10, 7, 0, 3), Pair("b", SCRIPTED_ID, 10, 9, 0, 1), Pair("b", "a", 10, 7, 0, 3)]
    r = joint_elo(ids, pairs, SCRIPTED_ID)
    assert r["b"] > r["a"] > ANCHOR_RATING
    # at the fit, each member's expected total (with the virtual draws) equals its actual total
    for i, opps in (("a", [(SCRIPTED_ID, 7.0, 10), ("b", 3.0, 10)]), ("b", [(SCRIPTED_ID, 9.0, 10), ("a", 7.0, 10)])):
        expected = sum((n + 1) * expected_score(r[i], r[o]) for o, _, n in opps)
        assert abs(expected - sum(s + 0.5 for _, s, _ in opps)) < 1e-6


def test_hold_anchor_refuses_a_challenger_the_coach_beats(monkeypatch):
    import numpy as np

    from arena.genome import make_genome
    from arena.ladder import Ladder, Member, Pair

    rng = np.random.default_rng(1)
    pool_g = make_genome(rng.standard_normal(515).astype(np.float32), "k", (), "pool", -1)
    challenger = make_genome(rng.standard_normal(515).astype(np.float32), "k", (), "random", 3)

    def fake_play(self, workers, genome, opponent, max_ticks=3000):
        if opponent.id == SCRIPTED_ID:
            return Pair(genome.id, opponent.id, 20, 6, 0, 14, -1.0)  # the coach beats it 14 of 20
        return Pair(genome.id, opponent.id, 20, 18, 0, 2, 3.0)  # it beats the pool's champion 18 of 20

    monkeypatch.setattr(Ladder, "play", fake_play)
    for hold, promoted in ((False, True), (True, False)):
        ladder = Ladder(0, (1, 2), pool=[Member(pool_g.id, "pool", 1050.0, 0, pool_g, label="p")], hold_anchor=hold)
        assert ladder.champion.id == pool_g.id and ladder.anchor.id == SCRIPTED_ID
        c = ladder.challenge(None, challenger, 10)
        assert c is not None and c.promoted == promoted and c.rating > 1050.0
        assert c.vs_members[0].b == SCRIPTED_ID and c.vs_members[0].winrate_a == 0.3
        assert (ladder.era == 1) == promoted


def test_weights_count_the_top_member_as_one_and_everyone_equal_without_a_scale():
    from arena.ladder import Ladder, Member

    pool = [Member("p1", "pool", 1800.0, 0, label="one"), Member("p2", "pool", 1000.0, 0, label="two")]
    assert Ladder(0, (1,), pool=pool).weights() == {SCRIPTED_ID: 1.0, "p1": 1.0, "p2": 1.0}
    w = Ladder(0, (1,), pool=pool, weight_scale=800.0).weights()
    assert w["p1"] == 1.0 and abs(w["p2"] - 0.1) < 1e-12 and w[SCRIPTED_ID] == w["p2"]


def test_pool_run_warm_starts_from_an_export_and_rates_the_pool(tmp_path):
    so = StoreObserver(tmp_path)
    first = run(_cfg(ladder_enabled=False, slug="seed-run"), [so, ExportObserver(so, 0.0, tmp_path)], generations=3, now=NOW)
    out = tmp_path / "out" / first.run_id
    cells = json.loads((out / "latest.json").read_text())["cells"]
    assert len(cells) >= 3
    pool = tuple({"path": str(out / "genomes" / f"{c['genome_id']}.bin"), "label": f"cell {c['i']},{c['j']}",
                  "id": c["genome_id"]} for c in cells[:2])
    rec, so2 = Rec(), StoreObserver(tmp_path)
    ctx = run(_cfg(slug="pool-run", ladder_pool=pool, ladder_hold_anchor=True, ladder_weight_scale=800.0,
                   warm_runs=(str(out),)), [so2, ExportObserver(so2, 0.0, tmp_path), rec], generations=12, now=NOW)
    # generation 0 is the warm start: every exported elite but the two pool members, scored under the pool ladder
    assert rec.gens[0].generation == 0 and len(rec.gens[0].results) == len(cells) - 2
    assert {r.genome.op for r in rec.gens[0].results} == {"warm"}
    assert rec.gens[1].generation == 1 and rec.gens[1].results[0].genome.gen == 1
    assert ctx.archive.filled() >= 1
    members = ctx.ladder.members
    assert members[0].id == SCRIPTED_ID and members[0].rating == ANCHOR_RATING
    assert [m.kind for m in members[1:3]] == ["pool", "pool"] and [m.label for m in members[1:3]] == [p["label"] for p in pool]
    assert len(ctx.ladder.pool_pairs) == 3  # two against the anchor, one against each other
    assert ctx.ladder.champion.id == max(members, key=lambda m: m.rating).id
    assert all(m.rating != ANCHOR_RATING for m in members[1:3])  # rated, not provisional
    # the export carries the pool: labels, the champion by rating, a full matrix among the first three
    out2 = tmp_path / "out" / ctx.run_id
    lad = json.loads((out2 / "ladder.json").read_text())
    assert lad["champion"] == ctx.ladder.champion.id
    assert [m.get("label") for m in lad["members"][:3]] == [None, pool[0]["label"], pool[1]["label"]]
    w = lad["matrix"]["winrate"]
    assert all(w[i][j] is not None for i in range(3) for j in range(3) if i != j)
    for m in members[1:3]:
        assert (out2 / "genomes" / f"{m.id}.bin").exists()
    latest = json.loads((out2 / "latest.json").read_text())
    assert latest["stats"]["champion_id"] == ctx.ladder.champion.id


def test_run6_config_round_trips_its_pool():
    c = load_config("configs/run6-definitive.toml")
    assert len(c.ladder_pool) == 23 and c.ladder_hold_anchor and c.ladder_weight_scale == 800.0 and len(c.warm_runs) == 2
    again = Config.from_dict({**c.to_dict(), "seed": c.seed})
    assert again.ladder_pool == c.ladder_pool and again.warm_runs == c.warm_runs
