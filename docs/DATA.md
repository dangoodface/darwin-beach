# Data contracts: trainer to page

The trainer writes files; the page reads them. Nothing else crosses. Every
file is JSON unless named `.bin` or `.wasm`. Content-addressed files are
immutable and cached for a year; the three indexes change and are cached for
60 seconds.

Local layout (the exporter writes here; the uploader mirrors it to R2):

```
$ARENA_DATA/out/<run_id>/
  latest.json
  curve.json
  tree.json
  ladder.json            (v1)
  genomes/<genome_id>.bin
  kernel/<kernel_sha>.wasm
$ARENA_DATA/out/index.json   lists runs: [{run_id, title, started_at, latest_generation}]
```

`ARENA_DATA` defaults to `~/.local/share/arena`. Never under `/tmp`.

## Where it is served (decided 26 October 2026)

The page lives at `play.thegoodface.ai/darwin-beach/`: Darwin Beach is its
own Worker with static assets, reached from the `play-thegoodface` Worker
through a service binding at that path. The data files come through the
same Worker, from the R2 bucket, under `darwin-beach/data/`. So the page
uses relative URLs only, never a leading slash, and works the same from a
local dev server at `/`. No subdomain and no r2.dev address.

## Ids

- `genome_id`: hex sha256 of the f32 little-endian parameter bytes followed
  by the kernel sha as ASCII. Same params under a new kernel is a new id.
- `kernel_sha`: hex sha256 of the `.wasm` bytes.
- `run_id`: `YYYYMMDDTHHMMSSZ-<slug>`.

## latest.json

```json
{
  "run_id": "20261101T090000Z-slime-v0",
  "kernel_sha": "…", "kernel_version": 1,
  "generated_at": "2026-11-01T12:00:00Z",
  "generation": 412, "era": 0,
  "probes": ["scripted-v1", "…", "…", "…"],
  "descriptor_names": ["court_position", "airtime", "hits", "mean_hit_height",
                       "mean_speed_after_hit", "jumps", "distance", "points"],
  "grid": {
    "descriptors": ["court_position", "airtime"],
    "indices": [0, 1],
    "bounds": [[0.0, 1.0], [0.0, 1.0]],
    "resolution": [20, 20],
    "lit_floor": 0.0
  },
  "cells": [
    {"i": 3, "j": 17, "genome_id": "…", "fitness": 2.4, "rating": 1180.5, "lit": true,
     "margins": [-1.25, 2.88, 4.25, 3.75],
     "desc": [0.19, 0.86, 3.5, 5.2, 37.1, 14.0, 160.2, 3.0],
     "bd": [0.19, 0.86], "gen": 388, "era": 0,
     "replay": {"seed": 123456789, "opponent_id": "…", "side": "A",
                "final_hash": "0x…", "ticks": 2210, "points": [5, 3]}}
  ],
  "stats": {"filled": 212, "lit": 97, "beats_coach": 3, "qd_score": 311.2, "champion_id": "…",
            "champion_rating": 1234.0, "matches_total": 1880000,
            "matches_per_s": 1430.0, "elapsed_s": 3600}
}
```

Fitness is the mean score margin (points for minus points against) over the
evaluation matches against the probe set (v0) or the ladder (v1). Every cell
gets some bot within seconds of a run starting, so "filled" is not the show;
a cell is `lit` when its elite's fitness reaches `grid.lit_floor` (0.0: it
holds its own against the rivals on average), and `stats.lit` counts them.
The page draws unlit cells dim and lit cells by fitness above the floor.

`probes` is the run's probe set in order, the scripted bot first when the run
has one, then the frozen probe genomes (the runs row's `probe_ids_json`).
`cells[].margins` has one number per probe in that order: the elite's mean
score margin against that opponent over the evaluation seeds on both sides,
so the mean of `margins` equals `fitness` to floating-point error. It is
`null` for elites recorded before margins existed (the page treats null as
"no breakdown"). "Beats the coach" is `margins[0] > 0` when `probes[0]` is
`scripted-v1`. Added 27 October 2026.

`descriptor_names` lists the run kernel's descriptors in ABI order, taken
from the run's recorded kernel version: eight for version 1 (and the pretend
game), ten from version 2, which adds `juggled_points` and `max_streak`.
`cells[].desc` carries the elite's mean of all of them over its evaluation
matches, so `desc` at `grid.indices` equals `bd`. The page draws strategy
on top of the map from it (hits per match, hit height, speed after contact);
it is `null` for elites recorded before 27 October 2026.

`stats.beats_coach` is the number of occupied cells whose elite has a mean
margin above zero against `scripted-v1`, and each `curve.json` point carries
`beats_coach` for the end of that generation (generations with no archive
event repeat the previous value). Both are `null` for a run with no scripted
probe or no stored margins. The first cell to beat the coach is the headline
to watch.

`stats.fresh`, `cells[].fresh` and `beats_coach_fresh` (added 27 October 2026)
grade the same claim on serves no selection ever saw. The run draws 8 seeds
from a stream of its own, never used for fitness, descriptors or the ladder.
At every export the best live elite and every live elite with a mean margin
above zero against `scripted-v1` play those 8 seeds on both sides against the
scripted bot (16 matches each). `cells[].fresh` is `{"won", "margin"}` (matches
won of 16, mean points for minus against) for each elite checked and `null`
for the rest. `stats.fresh` is `{"seeds": 8, "matches": 16, "checked": n,
"beats_coach": elites that won more than 8 of 16, "champion": {"won",
"margin"}}`, where `champion` is the best elite (`stats.best_id`), not the
ladder champion. Each `curve.json` point carries `beats_coach_fresh`, the
count at the latest export at or before that generation, `null` before the
first export. All three are `null` when the export has no checker (a run with
no scripted bot, or an export rebuilt from the store afterwards), and the curve
history is kept in the running process, so a restart begins it again.
Decided 27 October 2026 after the first real run filled 400 of 400 in eight
seconds and took minutes to light them. Rating is
Elo against the frozen anchors (v1; null in v0). `replay` is the match the
page re-runs: kernel, seed, this cell's genome on `side`, the opponent on
the other side; `points` is `[for this cell's genome, against]`.
`opponent_id` is a probe genome id with its own `.bin`, or `scripted-v1`,
which means the kernel's scripted bot (`scripted_slot()`, no genome file).
The page shows a verified tick when `BigInt.asUintN(64, state_hash())`
equals `final_hash`, a hex string of 16 digits with a `0x` prefix.

Seeds are u64. JSON carries a seed as a number when it is at most 2^53 and
as a decimal string above that, so the page always converts with `BigInt`
and passes a BigInt to the kernel.

## curve.json

```json
{"run_id": "…", "points": [
  {"gen": 0, "t": 0, "filled": 14, "lit": 0, "beats_coach": 0, "qd_score": 3.1, "champion_rating": null,
   "best_fitness": -4.2, "era": 0, "matches_total": 2048}
]}
```

One point per generation up to 2,000 points; beyond that the exporter
thins to every k-th generation but always keeps the last one.

## tree.json

```json
{"run_id": "…", "nodes": [
  {"id": "…", "parents": ["…", "…"], "op": "iso_line", "gen": 388, "era": 0,
   "cell": [3, 17], "fitness": 2.4, "rating": null, "alive": true,
   "champion": false, "x": 0.41, "y": 0.77}
 ],
 "edges": [["parent_id", "child_id"]],
 "champion_lineage": ["…", "…"]}
```

Nodes are archive entrants only: everything that ever held a cell, with
parents (one for mutation, two for crossover, none for generation zero).
A cell whose candidates were all rejected has no node. `alive` is false
once evicted. `champion_lineage` is the champion plus every entrant it
descends from, oldest first; with crossover it is a DAG, not a chain.
`x, y` are layout coordinates in [0, 1] computed by the exporter: `x` is
the generation divided by the last generation, and within a generation
entrants are spread in `y` ordered by their parents' mean `y`, so families
stay together. The page draws and never lays out. The exporter caps the
file: the full champion lineage, every live elite, and the last 2,000
entrants; `tree-full.json` holds everything for a "load all" button.
`tree-full.json` is written at the end of a run and at most once an hour
(`export.tree_full_every_s`, default 3600), so for the first hour of a run
it does not exist: the "load all" button handles a 404 by saying so.

## v1 fields (the ladder, decided 27 October 2026)

When `ladder.enabled` is on: `stats.champion_id` is the ladder's champion
(`scripted-v1` in era 0, so the page highlights no lineage until the first
promotion) and `stats.best_id` is the fittest live elite; both exist in v0
files too, equal to each other. `cells[].rating` and `stats.champion_rating`
are Elo against the frozen members; `curve.json` points carry
`champion_rating` (a step that only rises) and `era`; `tree.json` nodes carry
`rating`. Fitness is comparable only within an era, so with the ladder on the
page colours by rating. `genomes/` includes every member's `.bin`, so
champions can be replayed and played against.

## ladder.json (v1)

```json
{"era": 3, "cycle_window": 5, "members": [
  {"id": "scripted-v1", "kind": "scripted", "rating": 1000, "since_era": 0},
  {"id": "…", "kind": "champion", "rating": 1150, "since_era": 1,
   "promoted_at_gen": 240, "winrate_vs_previous": 0.58}
 ],
 "matrix": {"ids": ["scripted-v1", "…"], "winrate": [[null, 0.42], [0.58, null]]}}
```

`cycle_window` is how many previous champions each new champion is played
against for the cycle check. `matrix.winrate[r][c]` is how often row `r` beat column `c` over the
promotion and rating matches, `null` where they never played. The page draws
it as the who-beats-whom table and flags cycles.

### The pool (run 6, 27 October 2026)

A run may start with a pool: members of kind `"pool"` present from era 0,
each with a `label` (the name the page shows). The pool is rated once at the
start by a round robin on the ladder seeds, every member against every other
and the anchor, with the anchor fixed at 1000 (`src/arena/ladder.py`,
`joint_elo`); those blocks are in `matrix` too, so it is full from the start.
`champion` names the top-rated member, which is the one to beat: with a pool
it is not the last member until a promotion happens, so read `champion`, not
`members[-1]`. A promotion under `ladder.hold_anchor` must also win the
threshold against the anchor, and fitness under `ladder.weight_scale` is the
members' mean margins weighted by `10 ** ((rating - top) / scale)`.

## definitive.json (run 6)

Written by `python -m arena.definitive <run_id>` after a run: every member
of its ladder, old and new, in a round robin on held-out serves (a stream
the run never drew from), rated jointly with the anchor at 1000 and ranked.

```json
{"run_id": "…", "seeds": ["…"], "matches_per_pair": 32, "champion": "…",
 "members_beaten": 23, "of": 24,
 "fresh": {"<member id>": {"label": "…", "kind": "pool", "won": 30, "n": 32, "margin": 4.1}},
 "leaderboard": [{"rank": 1, "id": "…", "label": "…", "kind": "champion", "rating": 1900.0, "run_rating": 1880.0}],
 "matrix": {"ids": ["…"], "winrate": [[null, 0.9], [0.1, null]]}}
```

`fresh` is the champion's record against each other member; `run_rating`
is the rating the member held in the run's ladder, for comparison.

## genomes/<id>.bin

Raw f32 little-endian, `genome_len` values, nothing else. Written for the
live elites, the champion's ancestors and the probes (the replay opponents),
once each, never rewritten.

## Replays

Not stored. A replay is `(kernel_sha, seed, genome A, genome B)` re-run in
the browser with the kernel's interactive exports. v0 shows the cell's elite
against the opponent named in `replay`; v1 lets a visitor pick both.

## Specialists

Added 27 October 2026. A specialist is one genome trained against the real champions with its style forced by
construction (`python -m arena.specialist`, `src/arena/specialist.py`, `configs/specialists.toml`). Each style has an
unconstrained twin (`--twin`): same start genome, same seeds, same rivals, no repair, window or extra fitness term.
`all_rounder` is its own twin and has none.

Files, under `$ARENA_DATA/specialists/<run_id>/` with `run_id = <UTC start>-spec-<style>` and `-twin` appended for a twin:

- `result.json`, below.
- `genomes/<id>.bin`: the final genome, the start genome and the champion rivals, content-addressed like `out/`.
- `kernel/<kernel_sha>.wasm`: the kernel the matches were played on. A replay is kernel plus seed plus two ids, as everywhere.

```
{ "style": "flat_earther", "twin": false,
  "constraint": {"kind": "pinned_output" | "descriptor_window" | "fitness" | "none", "detail": ...},
  "start": {"genome_id", "run_id", "cell": [i, j] | null, "reason"},
  "rivals": [{"id", "kind": "champion" | "coach"}],           // the coach's id is "scripted-v1" (no genome file)
  "seeds": {"train": {"per_generation": 8, "resampled": ...}, "graded": [u64 ...], "fresh": [u64 ...]},
  "generations", "matches_total", "elapsed_s", "kernel_sha", "kernel_version",
  "final": {"genome_id",
            "graded": {rival_id: {"won", "n", "margin"}},    // the fixed graded seeds, both sides
            "fresh":  {rival_id: {"won", "n", "margin"}},    // 8 held-out seeds, both sides, n = 16
            "descriptors_fresh_mean": {name: value},         // the genome's own side, over all fresh matches
            "constraint_satisfied": bool,
            "proof": {...},
            "showcase": [{"seed", "side": "A" | "B", "rival_id", "margin", "descriptors": {name: value}}]},
  "curve": [{"gen", "t", "fitness_best", "fitness_mean", "margin_by_rival": {rival_id: m}, "constraint_penalty"}] }
```

Seeds are u64: a decimal string when above 2^53, as in `latest.json`. `won` counts matches with more points for the
genome than against; "beats the coach" is won above half of n, the page's rule. The fresh seeds are never used in training or
for the running best. `margin` is points for minus against, mean over the matches. `showcase` lists the two fresh matches
with the best margin per rival; `side` is the specialist's side, and its `descriptors` are that side's.

`constraint.kind`: `pinned_output` (flat_earther: the jump output's weights are zeroed and its bias is -1, so the button never
fires; kangaroo: bias +1, always pressed), `descriptor_window` (long_ball, lob_artist, joga_bonito: bounds on the mean
descriptors; a penalty of `penalty.weight` points per scale-width of violation while training, a strict check on the fresh
matches at the end), `fitness` (sprinter: margin minus `lambda` times ticks per point; all_rounder: the minimum of the
per-rival margins), `none` (every twin). `constraint_satisfied` is the strict check: for a pinned output, no fresh match has a
jump (flat_earther) or every one has at least one (kangaroo); for a window, every bound holds on the fresh mean. It is true
for styles with no hard constraint. `proof` holds the values behind it: `ticks_per_point` always, `jumps_*` and `airtime`
for a pinned output, each bounded descriptor's mean with its bounds for a window, `min_margin` for all_rounder.
`curve` rows describe the best candidate of that generation. Fitness there includes the window penalty.

`joga_bonito` needs the kernel's descriptors 8 and 9 (`juggled_points`, `max_streak`, kernel version 2); on a kernel that reports
8 the run stops at its start with a message. Its window is juggled points at least a third of points won, and its fitness adds
`joga.bonus` per juggled point to the margin.
