// Replays stored matches from a real run on the real kernel in Node (V8, the
// browser's engine) and checks they land on the trainer's hash and score.
// Skips when no run is on this machine.

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { Kernel, SCRIPTED_SLOT, replayHeadless } from "../src/kernel.ts";
import { parseSeed, parseHash, SCRIPTED } from "../src/data.ts";

const OUT = process.env.ARENA_OUT ?? join(process.env.HOME, ".local/share/arena/out");
const runs = existsSync(OUT)
  ? readdirSync(OUT).filter((d) => existsSync(join(OUT, d, "latest.json"))).sort()
  : [];

function bytes(path) {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

test("stored replays land on the trainer's hash and score", { skip: runs.length === 0 && "no run on this machine" }, async () => {
  const dir = join(OUT, runs.at(-1));
  const latest = JSON.parse(readFileSync(join(dir, "latest.json"), "utf8"));
  const k = await Kernel.load(readFileSync(join(dir, "kernel", `${latest.kernel_sha}.wasm`)));
  assert.equal(k.version, latest.kernel_version);

  // The champion plus a spread across the map, so different styles and lengths get checked.
  const cells = [...latest.cells].sort((a, b) => b.fitness - a.fitness);
  const pick = [cells[0], ...cells.filter((_, n) => n % 50 === 7)];
  for (const cell of pick) {
    const r = cell.replay;
    const mine = k.loadGenome(cell.genome_id, bytes(join(dir, "genomes", `${cell.genome_id}.bin`)));
    const theirs = r.opponent_id === SCRIPTED
      ? SCRIPTED_SLOT
      : k.loadGenome(r.opponent_id, bytes(join(dir, "genomes", `${r.opponent_id}.bin`)));
    const [left, right] = r.side === "A" ? [mine, theirs] : [theirs, mine];
    const got = replayHeadless(k, parseSeed(r.seed), left, right);
    const [forMine, againstMine] = r.side === "A" ? [got.pointsLeft, got.pointsRight] : [got.pointsRight, got.pointsLeft];
    assert.equal(got.ticks, r.ticks, `cell ${cell.i},${cell.j} ticks`);
    assert.equal(got.hash, parseHash(r.final_hash), `cell ${cell.i},${cell.j} hash`);
    assert.deepEqual([forMine, againstMine], r.points, `cell ${cell.i},${cell.j} points`);
  }
});

test("seed 1 serves as the kernel's test vector says", { skip: runs.length === 0 && "no run on this machine" }, async () => {
  const dir = join(OUT, runs.at(-1));
  const latest = JSON.parse(readFileSync(join(dir, "latest.json"), "utf8"));
  const k = await Kernel.load(readFileSync(join(dir, "kernel", `${latest.kernel_sha}.wasm`)));
  k.reset(1n);
  const s = k.snapshot();
  assert.equal(s.ball[2], 17.560949325561523);
  assert.equal(s.ball[3], 20.37936121225357);
});
