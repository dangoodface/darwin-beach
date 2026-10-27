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

The game file:

```
make kernel    # build it, and say whether it is one of the files in artifacts/kernel/
make parity    # the twin test: 1,000 matches in Python and in Node, compared tick by tick
make bench     # the speed test, under the memory and CPU limits training uses
```

`make kernel` runs `cargo +1.95.0 build --release --target
wasm32-unknown-unknown --manifest-path kernel/Cargo.toml`. The `+1.95.0` is
deliberate: a different compiler makes a different file, and a shell that
sets `RUSTUP_TOOLCHAIN` (this machine's sets it to `stable`) quietly
overrides `kernel/rust-toolchain.toml`. Every game file the arena has used
stays in `artifacts/kernel/`, named by its sha256; version 1 is
`5487b2a6…` (the full sha is in that folder's README).

Speed on 27 October 2026 (Ryzen 7 5700X, 8 cores and 16 threads; MemoryMax
4G, CPUQuota 1400%; batches of 100 matches), from `make bench`:

| Full matches a second | 1 process | 8 processes | 15 processes |
|---|---|---|---|
| Training: fresh bots against the probe set | 2,895 | 22,180 | 22,975 |
| Floor: two bots, every match run to the 3,000-tick cap | 539 | 4,166 | 4,258 |

The goal was 1,000 full matches a second on 15 processes. Fresh bots lose
points quickly (about 650 ticks a match), so the floor row is the honest
one: it divides the ticks a second of the dearest matches (two bots, no
scripted side) by 3,000, the speed even if every match went the distance.
Past 8 processes the gain is small: the second thread on each core adds
little to this kind of arithmetic.

## Licence

Apache License 2.0, see `LICENSE` and `NOTICE`. The slime volleyball physics
is ported from David Ha's slimevolleygym, also Apache 2.0.
