// The replay gallery: a handful of the night's matches, each stored as a serve (seed)
// and two brains, played live on the training kernel and checked against its hash.
import { Kernel, SCRIPTED_SLOT } from "./kernel.ts";
import { drawCourt, type CourtColours } from "./court.ts";

type Side = { id: string; label: string; kind: "bot" | "coach" };
type Match = {
  act: 1 | 2 | 3 | 4; title: string; caption: string; left: Side; right: Side;
  seed: string; ticks: number; hash: string; points: [number, number]; from?: number;
};
type Gallery = { kernel: string; matches: Match[]; brains: Record<string, string> };

const TICK_MS = 1000 / 30;
const MAX_TICKS = 3000;
const ACTS = { 1: "LEARNING", 2: "FUNSIES", 3: "THE BEST", 4: "THE SPECIALISTS" } as const;

const $ = (id: string) => document.getElementById(id)!;
const bytes = (b64: string) => Uint8Array.from(atob(b64.trim()), (c) => c.charCodeAt(0));
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const data: Gallery = JSON.parse($("gallery-data").textContent!);
const canvas = $("court") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const colours: CourtColours = {
  sky: css("--sea"), sand: css("--sand"), line: css("--foam"), left: css("--slime"), right: css("--rival"),
  ball: css("--sun"), eye: "#ffffff", pupil: css("--night"), text: css("--ink"),
};

let k: Kernel;
let cur = 0;
let pts: [number, number] = [0, 0];
let left = 0, right = 0;
let playing = true, finished = false, speed = 1, acc = 0, last = 0;
let advanceAt = 0;
const slot = (s: Side) => (s.kind === "coach" ? SCRIPTED_SLOT : k.loadGenome(s.id, bytes(data.brains[s.id]).buffer));

function tick() {
  const won = k.step(k.policy(0, left), k.policy(1, right));
  if (won > 0) pts[0]++;
  else if (won < 0) pts[1]++;
}

function draw() {
  const m = data.matches[cur];
  drawCourt(ctx, k.snapshot(), colours, { left: m.left.label, right: m.right.label, score: `${pts[0]} - ${pts[1]}` });
}

function load(i: number) {
  cur = i;
  const m = data.matches[i];
  left = slot(m.left);
  right = slot(m.right);
  k.reset(BigInt(m.seed));
  pts = [0, 0];
  // A long match can start just before its moment; the skipped ticks still run, so the hash still checks.
  for (let t = 0; t < (m.from ?? 0) && !k.done(MAX_TICKS); t++) tick();
  finished = false; playing = true; acc = 0; advanceAt = 0;
  $("m-act").textContent = `ACT ${m.act} · ${ACTS[m.act]} · MATCH ${i + 1} OF ${data.matches.length}`;
  $("m-title").textContent = m.title;
  $("m-caption").textContent = m.caption;
  verdict("playing", m.from ? `Live on the real engine. Skipped to second ${Math.round(m.from / 30)}, the moment that matters.` : "Live on the real engine, the same one that trained these bots.");
  $("pause").textContent = "Pause";
  document.querySelectorAll<HTMLButtonElement>("#list button").forEach((b, j) => b.setAttribute("aria-current", String(j === i)));
  draw();
}

function verdict(kind: string, text: string) {
  const v = $("verdict");
  v.dataset.kind = kind;
  v.textContent = text;
}

function finish() {
  finished = true;
  const m = data.matches[cur];
  const hash = "0x" + k.hash().toString(16).padStart(16, "0");
  const ok = hash === m.hash && pts[0] === m.points[0] && pts[1] === m.points[1];
  const time = `${Math.round(k.snapshot().tick / 30)} seconds`;
  verdict(ok ? "ok" : "bad", ok
    ? `Final ${pts[0]} - ${pts[1]} after ${time}. Your browser just replayed it bit for bit: fingerprint ${hash} matches the night's record.`
    : `Final ${pts[0]} - ${pts[1]}, but the fingerprint ${hash} does not match the record ${m.hash}. Something is off.`);
  if ((document.getElementById("auto") as HTMLInputElement).checked && cur + 1 < data.matches.length) advanceAt = performance.now() + 3000;
}

function frame(now: number) {
  const dt = Math.min(250, now - (last || now));
  last = now;
  if (playing && !finished) {
    acc += dt * speed;
    let n = 0;
    while (acc >= TICK_MS && n++ < 400) {
      tick();
      acc -= TICK_MS;
      if (k.done(MAX_TICKS)) { finish(); break; }
    }
    draw();
  }
  if (advanceAt && now >= advanceAt) load(cur + 1);
  requestAnimationFrame(frame);
}

function list() {
  const ol = $("list");
  let act = 0;
  data.matches.forEach((m, i) => {
    if (m.act !== act) {
      act = m.act;
      const h = document.createElement("li");
      h.className = "act";
      h.textContent = `ACT ${act} · ${ACTS[m.act]}`;
      ol.append(h);
    }
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = `<span class="n">${String(i + 1).padStart(2, "0")}</span><span class="t"></span><span class="who"></span>`;
    b.querySelector(".t")!.textContent = m.title;
    b.querySelector(".who")!.textContent = `${m.left.label} vs ${m.right.label}`;
    b.addEventListener("click", () => { load(i); canvas.scrollIntoView({ behavior: "smooth", block: "nearest" }); });
    li.append(b);
    ol.append(li);
  });
}

async function main() {
  list();
  try {
    k = await Kernel.load(bytes($("kernel-b64").textContent!));
  } catch (e) {
    verdict("bad", `This browser would not start the game engine (${(e as Error).message}).`);
    return;
  }
  $("pause").addEventListener("click", () => {
    if (finished) return load(cur);
    playing = !playing;
    $("pause").textContent = playing ? "Pause" : "Play";
  });
  $("restart").addEventListener("click", () => load(cur));
  $("next").addEventListener("click", () => load((cur + 1) % data.matches.length));
  $("speed").addEventListener("click", () => {
    speed = speed === 1 ? 2 : speed === 2 ? 4 : 1;
    $("speed").textContent = `${speed}x`;
  });
  load(0);
  requestAnimationFrame(frame);
}
main();
