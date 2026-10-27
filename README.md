# Darwin Beach

Where volleyball evolves.

Two slimes play volleyball. The winners have children with small random
changes. A map of playing styles fills up as evolution finds a bot for every
way to play, and a public page shows that map, the champion's family tree
and replays anyone can watch, or play against. Daniel is building it to learn
how machines learn, by watching it happen.

## How it works, in plain words

- The game itself is written once (in Rust) and packed into one small file
  (WebAssembly). The training computer and your browser run that same file,
  so a replay in the browser is the exact match the computer played.
- A bot is 515 numbers. Evolution copies the best bot of every style, nudges
  the numbers, plays the children against a fixed set of opponents, and keeps
  a newcomer only when it beats the bot already holding its style.
- Every bot that ever held a place on the map is kept, with its parents, in a
  record that is never edited (a small database).
- Every few minutes small files describing the map are written and sent to
  the cloud; the page reads them. The map lights up as the files arrive.

## Where things are

- `PLAN.md`: who builds what and in what order, and how we know a piece is
  done.
- `docs/ABI.md`: the contract for the game file. `docs/DATA.md`: the files
  the page reads. `docs/CONVENTIONS.md`: house rules.
- `kernel/`: the game (Rust). `src/arena/`: evolution, the record, the
  exporter and the uploader (Python). `web/`: the page. `systemd/`: the
  units that run it on Daniel's PC. `configs/`: one file per run.
- `PREFLIGHT.md`: the twenty ideas this was picked from and the checks made
  on the machine, 26 October 2026.

## Commands

Python needs `uv`; the game needs Rust with the `wasm32-unknown-unknown`
target.

```
uv sync --dev                                  # once
uv run pytest -q                               # the Python tests
uv run python -m arena.run --generations 200   # a practice run on the pretend game
uv run python tools/upload.py --to ~/.local/share/arena/mirror --dry-run
```

The game build, the twin test (the same matches in Python and in a browser)
and the speed test arrive with the kernel; this list grows as they land.

## Licence

Apache License 2.0, see `LICENSE` and `NOTICE`. The slime volleyball physics
is ported from David Ha's slimevolleygym, also Apache 2.0.
