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
  "grid": {
    "descriptors": ["court_position", "airtime"],
    "indices": [0, 1],
    "bounds": [[0.0, 1.0], [0.0, 1.0]],
    "resolution": [20, 20]
  },
  "cells": [
    {"i": 3, "j": 17, "genome_id": "…", "fitness": 2.4, "rating": 1180.5,
     "bd": [0.19, 0.86], "gen": 388, "era": 0,
     "replay": {"seed": 123456789, "opponent_id": "…", "side": "A",
                "final_hash": "0x…", "ticks": 2210, "points": [5, 3]}}
  ],
  "stats": {"filled": 212, "qd_score": 311.2, "champion_id": "…",
            "champion_rating": 1234.0, "matches_total": 1880000,
            "matches_per_s": 1430.0, "elapsed_s": 3600}
}
```

Fitness is the mean score margin (points for minus points against) over the
evaluation matches against the ladder. Rating is Elo against the frozen
anchors (v1; absent or null in v0). `replay` is the match the page re-runs:
kernel, seed, this cell's genome on `side`, the opponent on the other side.
The page shows a verified tick when the kernel's final hash equals
`final_hash`. Seeds are u64; JSON carries them as decimal strings when above
2^53, so always parse with BigInt on the page.

## curve.json

```json
{"run_id": "…", "points": [
  {"gen": 0, "t": 0, "filled": 14, "qd_score": 3.1, "champion_rating": null,
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
`alive` is false once evicted. `x, y` are layout coordinates in [0, 1]
computed by the exporter (layered by generation, siblings spread), so the
page draws and never lays out. The exporter caps the file: the full champion
lineage, every live elite, and the last 2,000 entrants; a `tree-full.json`
holds everything for a "load all" button.

## ladder.json (v1)

```json
{"era": 3, "members": [
  {"id": "scripted-v1", "kind": "scripted", "rating": 1000, "since_era": 0},
  {"id": "…", "kind": "champion", "rating": 1150, "since_era": 1,
   "promoted_at_gen": 240, "winrate_vs_previous": 0.58}
]}
```

## genomes/<id>.bin

Raw f32 little-endian, `genome_len` values, nothing else.

## Replays

Not stored. A replay is `(kernel_sha, seed, genome A, genome B)` re-run in
the browser with the kernel's interactive exports. v0 shows the cell's elite
against the opponent named in `replay`; v1 lets a visitor pick both.
