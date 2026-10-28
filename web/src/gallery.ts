// The replay gallery: a handful of the night's matches, each stored as a serve (seed)
// and two brains, played live on the training kernel and checked against its hash.
import { Kernel, SCRIPTED_SLOT, FORWARD, BACKWARD, JUMP } from "./kernel.ts";
import { drawCourt, type CourtColours } from "./court.ts";

type Side = { id: string; label: string; kind: "bot" | "coach" };
type Match = {
  act: 1 | 2 | 3 | 4 | 5 | 6; title: string; caption: string; left: Side; right: Side;
  seed: string; ticks: number; hash: string; points: [number, number]; from?: number;
  holds?: [number, number];
  board?: { rank: number; label: string; rating: number; kind?: string }[];
};
// The arcade: the viewer plays a rated bot, and every result moves the viewer's own Elo.
type Rival = { id: string; label: string; kind: "bot" | "coach"; rating: number; note: string };
type Gallery = { kernel: string; matches: Match[]; brains: Record<string, string>; arcade?: Rival[] };

const TICK_MS = 1000 / 30;
const MAX_TICKS = 3000;
const ACTS = { 1: "LEARNING", 2: "FUNSIES", 3: "THE BEST", 4: "THE SPECIALISTS", 5: "THE DEFINITIVE PLAYER?", 6: "THE NIGHT SHIFT" } as const;

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
let human: Rival | null = null;
const keys = new Set<string>();
const pad = { left: false, right: false, jump: false };
const slot = (s: Side) => (s.kind === "coach" ? SCRIPTED_SLOT : k.loadGenome(s.id, bytes(data.brains[s.id]).buffer));

// The viewer plays the left slime, so forward (towards the net) is to the right.
function humanAction(): number {
  let a = 0;
  if (keys.has("ArrowRight") || keys.has("d") || pad.right) a |= FORWARD;
  if (keys.has("ArrowLeft") || keys.has("a") || pad.left) a |= BACKWARD;
  if (keys.has("ArrowUp") || keys.has("w") || keys.has(" ") || pad.jump) a |= JUMP;
  return a;
}

function tick() {
  const won = k.step(human ? humanAction() : k.policy(0, left), k.policy(1, right));
  if (won > 0) pts[0]++;
  else if (won < 0) pts[1]++;
}

function draw() {
  const m = data.matches[cur];
  const names = human ? { left: "YOU", right: human.label } : { left: m.left.label, right: m.right.label };
  drawCourt(ctx, k.snapshot(), colours, { ...names, score: `${pts[0]} - ${pts[1]}` });
}

function load(i: number) {
  cur = i;
  human = null;
  $("pad").hidden = true;
  const m = data.matches[i];
  left = slot(m.left);
  right = slot(m.right);
  k.reset(BigInt(m.seed), ...(m.holds ?? [1, 1]));
  pts = [0, 0];
  // A long match can start just before its moment; the skipped ticks still run, so the hash still checks.
  for (let t = 0; t < (m.from ?? 0) && !k.done(MAX_TICKS); t++) tick();
  finished = false; playing = true; acc = 0; advanceAt = 0;
  $("m-act").textContent = `ACT ${m.act} · ${ACTS[m.act]} · MATCH ${i + 1} OF ${data.matches.length}`;
  $("m-title").textContent = m.title;
  $("m-caption").textContent = m.caption;
  const b = $("m-board");
  b.hidden = !m.board;
  b.replaceChildren(...(m.board ?? []).map((e) => {
    const li = document.createElement("li");
    li.dataset.kind = e.kind ?? (e.label === "the coach" ? "coach" : e.label.startsWith("run 6") ? "new" : "");
    li.append(Object.assign(document.createElement("span"), { textContent: String(e.rank) }),
      Object.assign(document.createElement("span"), { textContent: e.label }),
      Object.assign(document.createElement("span"), { textContent: String(e.rating) }));
    return li;
  }));
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
  if (human) return finishArcade(human);
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
    acc += dt * (human ? 1 : speed);
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

// ---------- the arcade ----------
type Result = { id: string; r: number; s: number };
const STORE = "darwin-beach-arcade";
let results: Result[] = [];
try { results = JSON.parse(localStorage.getItem(STORE) ?? "[]"); } catch { results = []; }
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(results)); } catch { /* private window: this visit only */ } };

// The rating that makes the viewer's expected score equal the real one, the bots' ratings held fixed.
function yourElo(rs: Result[]) {
  const got = rs.reduce((a, x) => a + x.s, 0);
  if (!rs.length) return "";
  if (got === 0) return `below ${Math.min(...rs.map((x) => x.r))}`;
  if (got === rs.length) return `above ${Math.max(...rs.map((x) => x.r))}`;
  let lo = -1000, hi = 4000;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    const e = rs.reduce((a, x) => a + 1 / (1 + 10 ** ((x.r - mid) / 400)), 0);
    if (e < got) lo = mid; else hi = mid;
  }
  return `about ${Math.round((lo + hi) / 2)}`;
}

function showYou() {
  const w = results.filter((x) => x.s === 1).length, t = results.filter((x) => x.s === 0.5).length;
  $("you").textContent = results.length
    ? `Your Elo: ${yourElo(results)}, after ${results.length} match${results.length === 1 ? "" : "es"} (${w} won${t ? `, ${t} tied` : ""}). The coach is 1000.`
    : "Your Elo: unrated. Play anyone to get a number.";
  $("you-reset").hidden = !results.length;
}

function startArcade(r: Rival) {
  human = r;
  left = 0;
  right = slot(r);
  const seed = crypto.getRandomValues(new BigUint64Array(1))[0];
  k.reset(seed);
  pts = [0, 0];
  finished = false; playing = true; acc = 0; advanceAt = 0;
  $("m-act").textContent = `ARCADE · YOU VS ${r.label} · ELO ${r.rating}`;
  $("m-title").textContent = `You vs ${r.label}`;
  $("m-caption").textContent = r.note;
  $("m-board").hidden = true;
  $("pad").hidden = false;
  $("pause").textContent = "Pause";
  document.querySelectorAll<HTMLButtonElement>("#list button").forEach((b) => b.setAttribute("aria-current", "false"));
  verdict("playing", "You are green, on the left. Arrow keys or A and D to move, Up, W or Space to jump; on a phone, the buttons under the court. First to 5, or the most points when the clock runs out.");
  canvas.scrollIntoView({ behavior: "smooth", block: "center" });
  draw();
}

function finishArcade(r: Rival) {
  const s = pts[0] > pts[1] ? 1 : pts[0] < pts[1] ? 0 : 0.5;
  results.push({ id: r.id, r: r.rating, s });
  save();
  showYou();
  const line = s === 1 ? `You beat ${r.label} ${pts[0]} - ${pts[1]}. Evolution has work to do.` : s === 0 ? `${r.label} wins ${pts[1]} - ${pts[0]}.` : `A ${pts[0]} - ${pts[1]} draw on the clock.`;
  verdict(s === 1 ? "ok" : "bad", `${line} Your Elo is now ${yourElo(results)}.`);
}

function arcade() {
  const box = $("rivals");
  for (const r of data.arcade ?? []) {
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = `<span class="who"></span><span class="elo"></span>`;
    b.querySelector(".who")!.textContent = r.label;
    b.querySelector(".elo")!.textContent = String(r.rating);
    b.title = r.note;
    b.addEventListener("click", () => startArcade(r));
    box.append(b);
  }
  $("you-reset").addEventListener("click", () => { results = []; save(); showYou(); });
  showYou();
  const GAME_KEYS = ["ArrowLeft", "ArrowRight", "ArrowUp", " "];
  window.addEventListener("keydown", (e) => {
    if (human && GAME_KEYS.includes(e.key)) e.preventDefault();
    keys.add(e.key);
  });
  window.addEventListener("keyup", (e) => keys.delete(e.key));
  window.addEventListener("blur", () => keys.clear());
  for (const [id, key] of [["p-left", "left"], ["p-right", "right"], ["p-jump", "jump"]] as const) {
    const b = $(id);
    const on = (e: Event) => { e.preventDefault(); pad[key] = true; };
    const off = (e: Event) => { e.preventDefault(); pad[key] = false; };
    b.addEventListener("pointerdown", on);
    for (const ev of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(ev, off);
  }
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
  arcade();
  load(0);
  requestAnimationFrame(frame);
}
main();
