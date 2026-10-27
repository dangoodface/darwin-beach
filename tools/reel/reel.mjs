#!/usr/bin/env node
// Highlight reels: short vertical MP4 clips of a run, made on the kernel and
// encoded with ffmpeg, for posting. A clip is just a seed and two genomes, so
// every frame here is the same match the trainer played.
//
//   node tools/reel/reel.mjs [--run <run_dir>] [--out <dir>]
//
// Makes three clips in <out> (default ~/.local/share/arena/reels/<run_id>/), each named
// <what>__<run>__s<seed>__<left id>-<right id>.mp4 so anyone can re-run it:
//   then-and-now.mp4    a generation-0 slime and the champion, same serve, both against the coach
//   longest-rally.mp4   the longest rally among every style's stored match
//   champion.mp4        the champion's stored match against the coach

import { readFileSync, readdirSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { spawn } from "node:child_process";
import { homedir } from "node:os";
import { Kernel, SCRIPTED_SLOT } from "../../web/src/kernel.ts";
import { parseSeed, SCRIPTED } from "../../web/src/data.ts";

const HERE = dirname(new URL(import.meta.url).pathname);
const FONT = join(HERE, "fonts", "PressStart2P-Regular.ttf");
const W = 1080;
const H = 1920;
const FPS = 30;
const C = {
  night: [11, 16, 38], sea: [16, 26, 68], star: [154, 166, 207], sand: [217, 178, 106], foam: [207, 227, 255],
  slime: [57, 255, 159], rival: [255, 79, 200], sun: [255, 201, 60], white: [255, 255, 255], pupil: [11, 16, 38], dim: [38, 48, 94],
};

// ---------- arguments and data ----------
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const OUT_ROOT = join(process.env.ARENA_DATA ?? join(homedir(), ".local/share/arena"), "out");
const runDir = args.run ?? join(OUT_ROOT, readdirSync(OUT_ROOT).filter((d) => existsSync(join(OUT_ROOT, d, "latest.json"))).sort().at(-1));
const latest = JSON.parse(readFileSync(join(runDir, "latest.json"), "utf8"));
const tree = JSON.parse(readFileSync(join(runDir, "tree.json"), "utf8"));
const outDir = args.out ?? join(process.env.ARENA_DATA ?? join(homedir(), ".local/share/arena"), "reels", latest.run_id);
mkdirSync(outDir, { recursive: true });

const k = await Kernel.load(readFileSync(join(runDir, "kernel", `${latest.kernel_sha}.wasm`)));
const genomeBytes = (id) => {
  const b = readFileSync(join(runDir, "genomes", `${id}.bin`));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const slot = (id) => (id === SCRIPTED ? SCRIPTED_SLOT : k.loadGenome(id, genomeBytes(id)));
const champ = latest.cells.find((c) => c.genome_id === latest.stats.champion_id) ?? [...latest.cells].sort((a, b) => b.fitness - a.fitness)[0];

// ---------- a tiny rasteriser ----------
class Frame {
  constructor() { this.px = Buffer.alloc(W * H * 3); }
  fill(rgb) { for (let i = 0; i < this.px.length; i += 3) { this.px[i] = rgb[0]; this.px[i + 1] = rgb[1]; this.px[i + 2] = rgb[2]; } }
  set(x, y, rgb, a = 1) {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const i = (y * W + x) * 3;
    if (a >= 1) { this.px[i] = rgb[0]; this.px[i + 1] = rgb[1]; this.px[i + 2] = rgb[2]; return; }
    this.px[i] += (rgb[0] - this.px[i]) * a; this.px[i + 1] += (rgb[1] - this.px[i + 1]) * a; this.px[i + 2] += (rgb[2] - this.px[i + 2]) * a;
  }
  rect(x, y, w, h, rgb, a = 1) {
    const x0 = Math.max(0, Math.round(x)), y0 = Math.max(0, Math.round(y));
    const x1 = Math.min(W, Math.round(x + w)), y1 = Math.min(H, Math.round(y + h));
    for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) this.set(xx, yy, rgb, a);
  }
  // Filled circle with a one-pixel soft edge; `upper` keeps only the top half (a slime).
  circle(cx, cy, r, rgb, a = 1, upper = false) {
    const y1 = upper ? cy : cy + r;
    for (let y = Math.floor(cy - r - 1); y <= Math.ceil(y1); y++) {
      for (let x = Math.floor(cx - r - 1); x <= Math.ceil(cx + r + 1); x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
        const cover = Math.max(0, Math.min(1, r - d + 0.5));
        if (cover > 0) this.set(x, y, rgb, a * cover);
      }
    }
  }
}

// The court in world units (slimevolleygym): x from -24 to 24, ground top at 1.5,
// fence at x = 0 up to 3.5 with a round cap, slimes radius 1.5, ball radius 0.5.
const VIEW = { x0: -24, x1: 24, y0: 0, y1: 18 };
function drawCourt(f, top, s, trail, leftColour) {
  const scale = W / (VIEW.x1 - VIEW.x0);
  const ch = (VIEW.y1 - VIEW.y0) * scale;
  const X = (x) => (x - VIEW.x0) * scale;
  const Y = (y) => top + ch - (y - VIEW.y0) * scale;
  f.rect(0, top, W, ch, C.sea);
  for (let n = 0; n < 40; n++) f.rect((n * 263) % W, top + ((n * 151) % Math.round(ch * 0.6)), 3, 3, C.star, 0.6);
  f.rect(0, Y(1.5), W, ch - (Y(1.5) - top), C.sand);
  f.rect(X(-0.5), Y(3.5), scale, Y(1.5) - Y(3.5), C.foam);
  f.circle(X(0), Y(3.5), scale * 0.5, C.foam);
  trail.forEach(([bx, by], n) => f.circle(X(bx), Y(by), scale * 0.5 * (0.4 + n / trail.length * 0.5), C.sun, 0.06 + 0.25 * (n / trail.length)));
  for (const [p, colour] of [[s.a, leftColour], [s.b, C.rival]]) {
    const px = X(p[0]), py = Y(p[1]);
    f.circle(px, py, scale * 1.5, colour, 1, true);
    const ex = px + (p[0] < 0 ? 1 : -1) * scale * 0.6, ey = py - scale * 0.85;
    f.circle(ex, ey, scale * 0.33, C.white);
    const dx = X(s.ball[0]) - ex, dy = Y(s.ball[1]) - ey, d = Math.hypot(dx, dy) || 1;
    f.circle(ex + (dx / d) * scale * 0.14, ey + (dy / d) * scale * 0.14, scale * 0.15, C.pupil);
  }
  const by = Y(s.ball[1]);
  if (by < top) f.circle(X(s.ball[0]), top + 10, 8, C.sun);
  else f.circle(X(s.ball[0]), by, scale * 0.5, C.sun);
  // lives as pips under the sand: five each, a point lost empties one
  for (let n = 0; n < 5; n++) {
    f.circle(60 + n * 34, top + ch + 30, 11, n < s.livesA ? leftColour : C.dim);
    f.circle(W - 60 - n * 34, top + ch + 30, 11, n < s.livesB ? C.rival : C.dim);
  }
  return ch;
}

// ---------- matches ----------
function runMatch(seed, left, right, maxTicks = 3000) {
  k.reset(seed);
  const states = [k.snapshot()];
  const points = [];
  let rallyStart = 0;
  while (!k.done(maxTicks)) {
    const s0 = k.snapshot();
    if (s0.delay > 0) rallyStart = s0.tick + 1;
    const won = k.step(k.policy(0, left), k.policy(1, right));
    const s = k.snapshot();
    states.push(s);
    if (won !== 0) points.push({ tick: s.tick, won, rally: s.tick - rallyStart });
  }
  return { states, points, hash: k.hash() };
}

function storedMatch(cell) {
  const r = cell.replay;
  const mine = slot(cell.genome_id), theirs = slot(r.opponent_id);
  const [left, right] = r.side === "A" ? [mine, theirs] : [theirs, mine];
  return { ...runMatch(parseSeed(r.seed), left, right), left, right, mineOnLeft: r.side === "A" };
}

// ---------- encoding ----------
function esc(t) { return t.replace(/\\/g, "\\\\").replace(/:/g, "\\:").replace(/'/g, "\u2019").replace(/%/g, "\\%"); }
const short = (id) => (id === SCRIPTED ? "coach" : id.slice(0, 8));
const clipName = (slug, seed, a, b) => `${slug}__${latest.run_id}__s${seed}__${short(a)}-${short(b)}.mp4`;
function encode(name, frames, texts) {
  const filters = texts.map(([t, x, y, size, rgb]) =>
    `drawtext=fontfile='${FONT}':text='${esc(t)}':x=${x}:y=${y}:fontsize=${size}:fontcolor=0x${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`).join(",");
  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", `${W}x${H}`, "-r", String(FPS), "-i", "-",
    "-vf", filters || "null", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", join(outDir, name)], { stdio: ["pipe", "inherit", "inherit"] });
  return new Promise((resolve, reject) => {
    ff.on("exit", (code) => (code === 0 ? resolve(join(outDir, name)) : reject(new Error(`ffmpeg ${code}`))));
    (async () => {
      for (const f of frames()) if (!ff.stdin.write(f.px)) await new Promise((r) => ff.stdin.once("drain", r));
      ff.stdin.end();
    })();
  });
}

const centre = (size) => `(w-text_w)/2`;
// "DARWIN" is 6 glyphs and "BEACH" 5, each glyph as wide as the font size; centred as one line.
const title = [["DARWIN", 126, 120, 72, C.slime], ["BEACH", 594, 120, 72, C.rival]];

// 1. Then and now: same serve, generation 0 on top, the champion below, both against the coach.
async function thenAndNow() {
  const seed = parseSeed(champ.replay.seed);
  const gen0 = tree.nodes.filter((n) => n.gen === 0 && existsSync(join(runDir, "genomes", `${n.id}.bin`))).sort((a, b) => a.fitness - b.fitness)[Math.floor(tree.nodes.filter((n) => n.gen === 0).length / 2)];
  const a = runMatch(seed, slot(gen0.id), SCRIPTED_SLOT);
  const b = runMatch(seed, slot(champ.genome_id), SCRIPTED_SLOT);
  const n = Math.min(Math.max(a.states.length, b.states.length), FPS * 25);
  const trailA = [], trailB = [];
  function* frames() {
    for (let t = 0; t < n; t++) {
      const f = new Frame();
      f.fill(C.night);
      const sa = a.states[Math.min(t, a.states.length - 1)], sb = b.states[Math.min(t, b.states.length - 1)];
      trailA.push([sa.ball[0], sa.ball[1]]); if (trailA.length > 8) trailA.shift();
      trailB.push([sb.ball[0], sb.ball[1]]); if (trailB.length > 8) trailB.shift();
      drawCourt(f, 470, sa, trailA, C.slime);
      drawCourt(f, 1120, sb, trailB, C.slime);
      yield f;
    }
  }
  const gens = champ.gen.toLocaleString("en");
  return encode(clipName("then-and-now", seed, gen0.id, champ.genome_id), frames, [
    ...title,
    ["SAME SERVE.", centre(), 250, 40, C.foam],
    [`${gens} GENERATIONS APART.`, centre(), 310, 34, C.foam],
    ["GENERATION 0", 40, 420, 30, C.slime], ["COACH", W - 230, 420, 30, C.rival],
    [`GENERATION ${champ.gen.toLocaleString("en")}`, 40, 1070, 30, C.slime], ["COACH", W - 230, 1070, 30, C.rival],
    ["NOBODY PROGRAMMED", centre(), 1700, 34, C.foam], ["EITHER ONE TO PLAY.", centre(), 1760, 34, C.foam],
  ]);
}

// 2. The longest rally among every style's stored match.
async function longestRally() {
  let best = null;
  for (const cell of latest.cells) {
    const m = storedMatch(cell);
    for (const p of m.points) if (!best || p.rally > best.p.rally) best = { cell, m, p };
  }
  const { m, p, cell } = best;
  const from = Math.max(0, p.tick - p.rally - FPS);
  const to = Math.min(m.states.length - 1, p.tick + FPS);
  const trail = [];
  function* frames() {
    for (let t = from; t <= to; t++) {
      const f = new Frame();
      f.fill(C.night);
      const s = m.states[t];
      trail.push([s.ball[0], s.ball[1]]); if (trail.length > 8) trail.shift();
      drawCourt(f, 760, s, trail, C.slime);
      yield f;
    }
  }
  const secs = (p.rally / FPS).toFixed(1);
  const rival = cell.replay.opponent_id === SCRIPTED ? "COACH" : "SPARRING";
  return [await encode(clipName("longest-rally", cell.replay.seed, cell.genome_id, cell.replay.opponent_id), frames, [
    ...title,
    ["THE LONGEST RALLY", centre(), 330, 44, C.sun],
    [`${secs} SECONDS WITHOUT A POINT`, centre(), 400, 30, C.foam],
    [m.mineOnLeft ? "EVOLVED" : rival, 40, 700, 30, C.slime], [m.mineOnLeft ? rival : "EVOLVED", W - 290, 700, 30, C.rival],
    ["OUT OF 400 STYLES,", centre(), 1500, 32, C.foam], ["THIS ONE KEPT IT ALIVE.", centre(), 1560, 32, C.foam],
  ]), { rally: p.rally, cell: [cell.i, cell.j] }];
}

// 3. The champion's own stored match against the coach.
async function championMatch() {
  const m = storedMatch(champ);
  const ok = m.hash === BigInt(champ.replay.final_hash);
  const n = Math.min(m.states.length, FPS * 45);
  const trail = [];
  function* frames() {
    for (let t = 0; t < n; t++) {
      const f = new Frame();
      f.fill(C.night);
      const s = m.states[t];
      trail.push([s.ball[0], s.ball[1]]); if (trail.length > 8) trail.shift();
      drawCourt(f, 760, s, trail, C.slime);
      yield f;
    }
  }
  const [pf, pa] = champ.replay.points;
  return [await encode(clipName("champion", champ.replay.seed, champ.genome_id, champ.replay.opponent_id), frames, [
    ...title,
    ["THE CHAMPION", centre(), 330, 44, C.sun],
    ["VS THE COACH", centre(), 400, 30, C.foam],
    [m.mineOnLeft ? "CHAMPION" : "COACH", 40, 700, 30, C.slime], [m.mineOnLeft ? "COACH" : "CHAMPION", W - 300, 700, 30, C.rival],
    [`FINAL ${pf} TO ${pa}`, centre(), 1500, 36, C.foam],
    [ok ? "REPLAYED ON THE REAL GAME" : "REPLAY MISMATCH", centre(), 1570, 26, ok ? C.slime : C.rival],
  ]), { ok, points: champ.replay.points }];
}

const made = [];
made.push({ clip: await thenAndNow(), seed: String(champ.replay.seed), right: SCRIPTED, note: "left is a generation-0 slime (median fitness), then the champion, same seed" });
const [rallyPath, rally] = await longestRally();
made.push({ clip: rallyPath, ...rally });
const [champPath, champInfo] = await championMatch();
made.push({ clip: champPath, ...champInfo });
writeFileSync(join(outDir, "reels.json"), JSON.stringify({ run_id: latest.run_id, generation: latest.generation, made }, null, 2));
console.log(JSON.stringify(made, null, 2));
