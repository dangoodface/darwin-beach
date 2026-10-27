# Preflight: the evolution arena

Written 26 October 2026. A public page where bots play each other, winners have offspring, and
anyone can watch the lineage improve live. The point for Daniel: have fun and learn machine
learning optimization by building it. Twenty ideas came in from four sessions (this one, Sonnet,
Fable, Opus); this is the synthesis, what was checked today, and what to build first.

## The brief, as rules

1. The grind is the show. Agents and optimizers run on Daniel's PC; the page shows the family
   tree, an honest rating curve and replays.
2. Each arena teaches one optimization family. The page is built once; arenas plug in.
3. Honest progress: ratings anchored to a frozen ladder (scripted baselines plus a hall of fame of
   past champions), plus a who-beats-whom matrix to expose cycles (Fable and Opus, independently).
4. Static page. The trainer writes small JSON and gzipped replays to R2 every ten minutes; the
   page polls them. No redeploy per update (Pages free plan: 500 deploys a month), no Durable
   Object rows.
5. Deterministic sims: a replay is (seed, bot ids), re-run in the browser where the sim is small.
6. Heavy runs under `systemd-run --user -p MemoryMax=...`, data and checkpoints on the SSD, never /tmp.
7. Its own repo and Cloudflare project, standalone. No em dashes in copy.

## Verified today

| Check | Result |
|---|---|
| CPU | Ryzen 7 5700X, 16 threads, load 0.1 |
| RAM | 31 GB, about 23 GB available |
| GPU | RTX 5060, 8 GB (compute 12.0), driver reports CUDA 13.4. PyTorch 2.11.0+cu128 installs with uv and sees the card; an 8192x8192 matmul runs. Speed numbers (3.8 TFLOPS fp32, 2.4 fp16) were measured without warm-up, so they are a floor, not a benchmark. Test venv: `~/.cache/arena-preflight` (6.6 GB) |
| Disk | 661 GB free on the SSD |
| Python | 3.13 with uv; gcc and make present (pMARS builds from source) |
| R2 | the token can list and create buckets, so a new bucket is one call |
| Packages | `ribs` 0.12.0 (pyribs, MAP-Elites and CMA-ME), `pymunk` 7.3.1 (2D physics), `cma` installs |
| Core War | corewar.co.uk up with pMARS and the hills pages; koth.org did not answer |

## The twenty ideas, by what they teach

| Method family | Ideas (who) | Converged? |
|---|---|---|
| LLM-driven code evolution (FunSearch, AlphaEvolve) | Tron (me, Fable), Tanques de Barro (Sonnet), **Core War (Opus)** | all four sessions |
| Quality-diversity (MAP-Elites, CMA-ME) | creature zoo (me), MAP-Elites zoo (Sonnet), drift racing (Fable), Core War archive (Opus) | all four |
| Evolution strategies on weights (OpenAI-ES, CMA-ES) | **Slime volleyball (me, Opus)**, ES duel (Sonnet) | three |
| Bodies and brains (NEAT, coevolution) | Sumo creatures (me, Fable, Opus) | three |
| Self-play RL with search (AlphaZero) | Connect Four with a solver (Opus, Sonnet), Hex 9x9 (Fable) | three |
| RL plus population-based training, leagues | Fábrica de Bots (Sonnet), capture the flag with exploiters (Opus) | two |
| Evolutionary game theory | **Dilema repetido, Axelrod's prisoner's dilemma (Fable)** | one |
| Game theory by regret minimization (CFR) | Liar's dice (me) | one |

## The three standouts

**Core War (Opus).** Two assembly programs fight in a shared 8,000-cell memory, drawn as pixels.
LLM agents mutate warriors, a numeric search tunes constants, and a MAP-Elites archive keeps one
slot per strategy, because Core War has built-in rock, paper, scissors (bombers beat scanners,
scanners beat replicators, replicators beat bombers). It also solves the honest-rating problem
for free: four decades of published human warriors are the frozen ladder. Share line: "our
lineage now beats X% of the classics." Weekend, CPU plus LLM budget.

**Slime volleyball (me, Opus, Sonnet's ES duel).** Tiny nets, tournament self-play first (David
Ha showed it alone learns to play), then OpenAI-ES and CMA-ES. Same physics in Python and JS, so
visitors play any ancestor. Share line: "find the generation where humans stop winning." Weekend,
CPU only, NumPy.

**Connect Four AlphaZero (Opus, Sonnet).** The only arena whose referee knows the truth: Connect
Four is solved, so the page plots blunder rate against perfect play. The GPU project. A weekend
to a week.

## Recommended path

| Step | Arena | Teaches | Why here | Size |
|---|---|---|---|---|
| 0 | Dilema repetido | evolutionary game theory, frozen ladder, cycle detector | builds the page plumbing (tree.json, ratings, replays, R2) on a sim that costs nothing; Axelrod's waves of cooperation are a good first post | an evening |
| 1 | Slime volleyball | tournament self-play, OpenAI-ES, CMA-ES | first visual, first "play it" hook | a weekend |
| 2 | Core War | LLM code evolution, quality-diversity, non-transitivity | the showpiece; external benchmark of classic warriors | a weekend, then it runs |
| 3 | Connect Four | AlphaZero: policy and value nets, MCTS, gating, PyTorch on the GPU | truth-referee curve | a weekend to a week |
| later | Sumo, PPO plus PBT league | NEAT, RL, leagues and exploiters | bigger | several weekends |

## Open decisions for Daniel

- Start at step 0 (live page tonight) or jump straight to Slime volleyball.
- Name and address (a subdomain of thegoodface.ai, or its own).
- LLM budget for Core War mutations (a few hundred to a few thousand calls a day).

## Decision (26 Oct, later): Slime, illuminated

Daniel: visual and shareable, slime volleyball plus the map that fills, AlphaZero out, and a step 0
that builds the infrastructure ("overengineering for good, but also efficiency"). This merges
this session's design with Fable's independent one.

**The concept.** The volleyball is the map. A 20 by 20 MAP-Elites grid: X is court position (mean
x in its own half, baseliner to net hugger), Y is airtime (share of ticks off the ground), colour
is strength. Descriptors are measured against a fixed probe set so a cell does not flicker;
fitness is score margin against the frozen ladder. Click a cell to watch that style, or play it.
A second grid later: hit height against ball speed.

**The one big decision (Fable's, adopted): one kernel, both sides.** The simulator is written once
in Rust, compiled to WebAssembly, and the same `.wasm` runs in Python through wasmtime for
training and in the browser for replays and play. WebAssembly doubles are deterministic by spec
(no fused multiply-add, no flush to zero), so parity comes from running the same bytes, not from
keeping two implementations in sync. Inside the kernel: only + - * / sqrt, softsign instead of
tanh, integer ticks, an integer RNG seeded with a u64, fixed update order. Old kernels are kept
forever and every snapshot records the kernel sha, so old replays never rot.

Verified for this: rustc 1.95.0 and cargo present (needs `rustup target add wasm32-unknown-unknown`,
no sudo; raw `extern "C"` exports, no wasm-bindgen or wasm-pack); wasmtime-py 50.0.1 has a
Python 3.13 wheel; Node 24.15.0 for the parity harness.

**Shared interfaces (the good overengineering).**
- Kernel ABI: `play_match(seed, genome_a, genome_b, max_ticks)` returning scores, descriptors,
  ticks and a state hash; plus `reset / set_actions / step / read_state` for the browser. Sumo or
  racing later implements the same ABI and nothing else changes.
- Optimizer protocol: `ask()` / `tell(results)`, with the archive as the shared object. Hand-written
  MAP-Elites, a pyribs CMA-ME adapter and OpenAI-ES all speak it.
- Store: one SQLite file on the SSD, WAL, append-only, every row tagged with era and kernel version;
  genome id = sha256(params + kernel version). Only archive entrants are stored with parents.
- Exporter: does all layout offline; writes content-addressed genomes and kernels plus three small
  indexes (`latest.json` grid, `curve.json`, `tree.json`). A replay is kernel + seed + two ids,
  so no replay files are stored.
- Publishing: a systemd timer uploads only new files to R2 every ten minutes; the page polls;
  Pages redeploys only when the page itself changes.

**Deliberately simple.** One machine, one run at a time, one process pool (15 workers, one wasmtime
instance each, results consumed in task order so runs reproduce), one SQLite file, one HTML page
in plain TypeScript and canvas, polling instead of sockets, no GPU in step 0. Refused: gymnasium,
game engines, job queues, Postgres, frontend frameworks, websockets.

**Honest progress.** Frozen ladder (scripted anchor plus hall of fame with fixed ratings); an elite
is promoted when it beats the champion 55% over 100 seeded matches, which opens a new era and
re-scores the archive; a cycle detector plays each new champion against the last five. QD-score
and champion rating both shown, so neither a pretty map of weak players nor a champion that only
beats a weak ladder can hide.

**Optimizers, in order.** Hand-written MAP-Elites (Gaussian sigma 0.1 plus iso-line crossover), then
CMA-ME through pyribs on the same archive, then OpenAI-ES on one chosen cell. Genome: MLP 12-32-3,
softsign, 515 f32 parameters, mirrored observations. Hardmaru's rules (five lives, 3,000 ticks).

**Size, cut in two.** Fable's full step 0 is three to four weekends. To see the map fill sooner:

| Cut | Contents | Size |
|---|---|---|
| v0 "the map fills" | kernel + Node parity test + benchmark gate, workers, hand-written MAP-Elites, store, exporter, R2, page with the grid and in-browser replays | about a weekend |
| v1 "honest and playable" | frozen ladder, eras, promotion, cycle detector, champion tree, play mode, Chrome parity run, pyribs CMA-ME tab | another weekend |
| v2 "learning tabs" | OpenAI-ES specialist, second descriptor grid, write-ups per optimizer | ongoing |

**Acceptance for v0.** 1,000 seeded full matches hash-identical in wasmtime and Node with zero NaN;
at least 1,000 matches a second on 15 workers under MemoryMax 4G and CPUQuota 1400%; a one-hour
MAP-Elites run fills a recorded share of cells and reproduces bit for bit from the same seed;
indexes reach R2 within ten minutes; a clicked cell replays in the browser and shows a verified
tick (final hash matches the stored one).
