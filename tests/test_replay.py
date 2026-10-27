import json
from datetime import datetime, timezone

import pytest

from arena.config import Config
from arena.replay import check_all, main, open_simulator
from arena.run import run
from arena.store import Store, StoreObserver, store_path

NOW = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def done(tmp_path):
    cfg = Config(seed=3, workers=1, batch_size=16, init_size=20, probe_count=2, seeds_per_probe=2,
                 resolution=(10, 10))
    ctx = run(cfg, [StoreObserver(tmp_path)], generations=30, now=NOW)
    return ctx, tmp_path, Store(store_path(ctx.run_id, tmp_path))


def test_every_live_cell_verifies(done):
    ctx, _, st = done
    checks = check_all(st, open_simulator(st, ctx.run_id), ctx.run_id)
    assert len(checks) == ctx.archive.filled() and all(c.ok for c in checks)


def test_cli_modes_and_exit_codes(done, capsys):
    ctx, root, st = done
    cell = sorted(st.occupants(ctx.run_id))[0]
    gid = st.occupants(ctx.run_id)[cell]
    assert main([ctx.run_id, "--root", str(root), "--all"]) == 0
    assert f"{ctx.archive.filled()} of {ctx.archive.filled()} verified" in capsys.readouterr().out
    assert main([ctx.run_id, "--root", str(root), "--cell", str(cell[0]), str(cell[1])]) == 0
    assert main([ctx.run_id, "--root", str(root), "--genome", gid]) == 0
    assert main([ctx.run_id, "--root", str(root), "--champion"]) == 0
    assert "verified" in capsys.readouterr().out
    assert main([ctx.run_id, "--root", str(root), "--cell", "9", "9"] if (9, 9) not in st.occupants(ctx.run_id)
                else [ctx.run_id, "--root", str(root), "--genome", "nope"]) == 2
    assert main(["no-such-run", "--root", str(root), "--all"]) == 2


def test_a_tampered_hash_is_a_mismatch(done, capsys):
    ctx, root, st = done
    cell, gid = sorted(st.occupants(ctx.run_id).items())[0]
    st.db.execute("DROP TRIGGER evaluations_no_update")
    st.db.execute("UPDATE evaluations SET replay_final_hash='0x0000000000000000' WHERE genome_id=?", (gid,))
    st.db.commit()
    assert main([ctx.run_id, "--root", str(root), "--genome", gid]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_run_row_missing_raises_keyerror(done):
    _, _, st = done
    with pytest.raises(KeyError, match="nope"):
        st.run_row("nope")


def test_replay_matches_what_the_exporter_wrote(done, tmp_path):
    from arena.export import export_run
    ctx, root, st = done
    export_run(st, ctx.run_id, root, now=NOW)
    latest = json.loads((root / "out" / ctx.run_id / "latest.json").read_text())
    stored = {(c["i"], c["j"]): c["replay"]["final_hash"] for c in latest["cells"]}
    got = {c.cell: c.got_hash for c in check_all(st, open_simulator(st, ctx.run_id), ctx.run_id)}
    assert {tuple(k): v for k, v in stored.items()} == got
