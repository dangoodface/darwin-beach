# Plan: step 0 of the arena

Written 26 October 2026 by the lead session (Fable). The design it executes
is the "Decision (26 Oct, later)" section of `PREFLIGHT.md`; the contracts
are `docs/ABI.md` and `docs/DATA.md`. Daniel speaks through the "funsies"
session; gates that need him are listed at the end.

## The cut

| Cut | Contents | Done when |
|---|---|---|
| v0 "the map fills" | kernel with parity and benchmark, workers, hand-written MAP-Elites, store, exporter, uploader, page with the grid and in-browser replays | the v0 acceptance list below passes end to end on this machine |
| v1 "honest and playable" | frozen ladder, eras, promotion, cycle detector, champion tree, play mode, Chrome parity run, CMA-ME tab | its own list, written when v0 is done |
| v2 "learning tabs" | OpenAI-ES specialist, second descriptor grid, write-ups per optimiser | ongoing |

## Tracks and owners

Three tracks run in parallel because the contracts are fixed first. Each
track owns its files; a change to a contract goes through the lead.

| Track | Owner | Branch, worktree | Owns |
|---|---|---|---|
| K, kernel | Opus (`set-default-model-opus`) | `kernel`, `~/projects/arena-kernel` | `kernel/`, `src/arena/kernel.py`, `tests/test_kernel*.py`, `tools/parity.mjs`, `tools/bench.py`, `artifacts/kernel/` |
| T, trainer | Sonnet (`sonnet`) | `trainer`, `~/projects/arena-trainer` | `src/arena/` except `kernel.py`, `tests/` except kernel tests, `systemd/`, `tools/upload.py` |
| W, page | funsies | `web`, `~/projects/arena-web` | `web/` |
| lead | Fable | `main` | `PLAN.md`, `docs/`, `pyproject.toml`, merges, integration, acceptance |

## Track K: kernel (Opus)

K1. Port `slimevolley.py` to `kernel/src/physics.rs`, policy to `policy.rs`,
PCG32 and FNV-1a to `rng.rs` and `hash.rs`, the ABI exports to `lib.rs`,
exactly as `docs/ABI.md`. `cargo build --release --target
wasm32-unknown-unknown` with `opt-level = 3`, `panic = "abort"`, no
wasm-bindgen. `cargo test` covers: one serve seed gives the documented vx;
a ball dropped on a still slime bounces; a full match between two zero
genomes ends by lives or at 3,000 ticks; NaN injection sets flag bit 0.
K2. `src/arena/kernel.py`: a `Kernel` class over wasmtime that loads a wasm
by path, exposes `load_genomes(table: np.ndarray)`, `play(seed, a, b,
max_ticks, record_hashes=False) -> MatchResult` and `play_batch(jobs) ->
list[MatchResult]`, writing memory in bulk (`Memory.write` or a ctypes
memmove on `data_ptr`), never element by element. `tools/parity.mjs`: Node
loads the same wasm, runs the same seeds and genomes, prints per-tick
hashes. `tests/test_parity.py` runs both on 1,000 seeds with random genomes
(a fixed numpy seed) and asserts every tick hash is identical and no flag
bit 0. Any divergence reports the first differing tick.
K3. `tools/bench.py`: matches per second for one process, then 8 and 15
processes with `play_batch` of 100, both printed, and the number lands in
`README.md`. Gate: 1,000 full matches a second on 15 workers.
K4. `scripted_action` (the anchor bot) and the descriptor accumulators, with
a test that a net-hugging scripted variant and a back-wall variant land in
different columns of the grid.
K5. Copy the built module to `artifacts/kernel/<sha256>.wasm` and commit it
with the sha in the commit message.

Hand-off from K to T: `Kernel` in `src/arena/kernel.py` implements the
`Simulator` protocol in `src/arena/sim.py` (T1). Until K2 lands, T works
against `FakeSimulator`.

## Track T: trainer (Sonnet)

T1. `src/arena/sim.py`: the `Simulator` protocol (`load_genomes`, `play`,
`play_batch`, `genome_len`, `kernel_sha`, `kernel_version`) and
`FakeSimulator`, deterministic nonsense physics (hash the seed and both
genomes into points and descriptors) so the whole loop runs before the
kernel exists. `src/arena/genome.py`: id (`docs/DATA.md`), f32 params,
Gaussian mutation with sigma 0.1, iso-line crossover, all from a numpy
`Generator` the caller passes.
T2. `src/arena/archive.py`: `Archive(bounds, resolution)` with `add(genome,
bd, fitness)` returning `new`, `replaced` or `rejected`, `cells()`,
`qd_score()`, `filled()`. `src/arena/optim.py`: the `Optimizer` protocol
(`ask() -> list[Genome]`, `tell(results)`) and `MapElites` (random
initialisation until N entrants, then mutation and crossover from random
elites). Tests on a toy fitness.
T3. `src/arena/workers.py`: a process pool of `workers` processes, one
simulator per process created at start, batches of `batch_size` jobs,
results returned in task order whatever the completion order. `src/arena/
run.py`: the loop. Config is one TOML (`configs/slime-v0.toml`): run slug,
kernel path, grid, population, sigma, workers, batch size, matches per
evaluation, probe set, seed. One `numpy.random.SeedSequence` from the config
seed spawns every stream, so the same config reproduces the same run bit
for bit. Fitness in v0: mean score margin against the probe set (the
scripted bot plus two frozen generation-zero genomes), both sides, K seeds
fixed per run.
T4. `src/arena/store.py`: SQLite at `$ARENA_DATA/runs/<run_id>/store.sqlite`,
WAL, append-only tables `runs`, `genomes`, `evaluations`, `archive_events`,
`promotions`, every row tagged `era` and `kernel_sha`. Only entrants are
stored with parents; `genome_params` holds the bytes.
T5. `src/arena/export.py`: writes `docs/DATA.md` exactly, to
`$ARENA_DATA/out/<run_id>/`, every `export_every_s` seconds from the loop
and at the end; tree layout computed here. A test loads a FakeSimulator run
of 50 generations and validates every file against the contract.
T6. `tools/upload.py`: mirrors `$ARENA_DATA/out/` to a target, uploading
only files that are not there yet (content-addressed) and always the
indexes; `--dry-run` to a local folder for now; the R2 mode reads the
bucket name and credentials from environment variables that a systemd
drop-in sets from `pass` at start (`pass show arena/r2-...` in
`ExecStartPre` is not allowed to write files, so use `EnvironmentFile`
pointing at a `pass`-managed file outside the repo, documented, never
committed). No bucket exists yet; Daniel's gate.
T7. `systemd/`: `arena-train.service` (the loop under `MemoryMax=4G`,
`CPUQuota=1400%`, `WorkingDirectory` the worktree, `ExecStart` through
`uv run`) and `arena-upload.timer` every ten minutes. Not installed until
the lead says so.

## Track W: page (funsies)

W1. `web/`: plain TypeScript, esbuild to one bundle, one `index.html`, a
`npm run dev` that serves `web/` plus a `data/` symlink to
`$ARENA_DATA/out/`; fixtures in `web/fixtures/` hand-written from
`docs/DATA.md` until T5 produces real ones. Tests with `node --test`.
W2. The grid: 20 by 20 canvas, colour by fitness (later rating), empty cells
dark, hover shows descriptors, click selects. Polls `latest.json` every 60
seconds. Works at phone width.
W3. Replay: load `kernel/<sha>.wasm`, the two genomes, `reset(seed)`, run
`policy_action` for both sides each tick at 30 ticks a second with an
accumulator, draw ball, slimes, fence; show the score; at the end compare
`state_hash()` to `replay.final_hash` and show the verified tick or a red
cross. Until K1 lands, a stub kernel module with the same exports drawn from
a fixture.
W4. The curve from `curve.json` and a first tree view from `tree.json`
(nodes at the exporter's x, y; champion lineage highlighted).
W5. Play mode (v1, cheap enough to start): arrow keys for one side,
`policy_action` for the other.

## Lead

- Keep the three contracts stable; a change is a commit on `main` with a note
  to all three tracks.
- Review and merge each track's branch into `main` locally; after K2 lands,
  swap `FakeSimulator` for `Kernel` in a config and run the first real
  generation.
- Integration: the one-hour run under systemd-run, the reproducibility run,
  the benchmark on 15 workers, the acceptance list.
- Daniel's gates through funsies: name, domain, GitHub repo, R2 bucket,
  Pages deploy, any LLM budget (none in step 0).

## v0 acceptance

1. `make parity`: 1,000 seeded full matches hash-identical in wasmtime and
   Node, zero NaN flags, including matches that end at 3,000 ticks.
2. `make bench`: at least 1,000 full matches a second on 15 workers under
   `MemoryMax=4G` and `CPUQuota=1400%`, the number in the README.
3. A one-hour MAP-Elites run from `configs/slime-v0.toml` fills a recorded
   share of cells, logs QD-score and best fitness per generation, and a
   second run from the same config reproduces the archive bit for bit.
4. Indexes, genomes and the kernel reach the upload target within ten
   minutes of a generation (dry-run folder until the bucket exists).
5. The page loads the grid from the run's `latest.json`, a clicked cell
   replays in the browser and shows the verified tick, and it works on a
   phone.
6. `docs/ABI.md` and `docs/DATA.md` still describe what runs; the README has
   the five commands (build kernel, parity, bench, train, dev page).

## Status

| Date | What |
|---|---|
| 26 October 2026 | Repo scaffolded, contracts written, tracks assigned, first tasks handed out |
