// Darwin Beach's public page: the map of styles, a court that replays any
// style's stored match on the real kernel (and checks it lands on the
// trainer's hash), the curve, the family tree, and a mode to play a bot yourself.
import { getJSON, getBytes, newestRun, parseSeed, parseHash, litCount, isLit, SCRIPTED, type Latest, type CurvePoint, type Tree, type RunIndexEntry, type Cell } from "./data.ts";
import { Kernel, SCRIPTED_SLOT, FORWARD, BACKWARD, JUMP } from "./kernel.ts";
import { drawCourt, type CourtColours } from "./court.ts";
import { Grid } from "./grid.ts";
import { drawCurve, drawTree } from "./charts.ts";
import { styleSentence, strengthWords, margin, coachLine } from "./describe.ts";

const DATA = "data/";
const TICK_MS = 1000 / 30;
const POLL_MS = 60_000;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const rgb = (hex: string): [number, number, number] => {
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
};

let runId = "";
let latest: Latest;
let kernel: Kernel;
let kernelSha = "";
let grid: Grid;
let courtColours: CourtColours;

// What the court is doing: replaying a stored match, or a human playing a bot.
type Mode =
  | { kind: "replay"; cell: Cell; left: number; right: number; leftName: string; rightName: string; pl: number; pr: number; expect: bigint; ticks: number }
  | { kind: "play"; cell: Cell; bot: number; pl: number; pr: number }
  | { kind: "idle" };
let mode: Mode = { kind: "idle" };
let paused = false;
let speed = 1;
let acc = 0;
let lastTime = 0;
const keys = new Set<string>();
const touch = { left: false, right: false, jump: false };

async function boot() {
  courtColours = {
    sky: css("--sea"), sand: css("--sand"), line: css("--foam"), left: css("--slime"), right: css("--rival"),
    ball: css("--sun"), eye: "#ffffff", pupil: "#0b1026", text: css("--ink"),
  };
  grid = new Grid($("map") as HTMLCanvasElement, {
    empty: css("--night-2"), dim: css("--dim-cell"), lit: rgb(css("--slime")), frame: css("--ink"), select: css("--sun"), champ: css("--rival"),
  });
  wireInput();
  try {
    await refresh(true);
  } catch (e) {
    $("status").textContent = `No run to show yet (${(e as Error).message}).`;
    return;
  }
  setInterval(() => refresh(false).catch(() => {}), POLL_MS);
  requestAnimationFrame(frame);
}

async function refresh(first: boolean) {
  const index = await getJSON<RunIndexEntry[] | { runs: RunIndexEntry[] }>(`${DATA}index.json`);
  runId = newestRun(Array.isArray(index) ? index : index.runs);
  const base = `${DATA}${runId}/`;
  const [l, curve, tree] = await Promise.all([
    getJSON<Latest>(`${base}latest.json`),
    getJSON<{ points: CurvePoint[] }>(`${base}curve.json`),
    getJSON<Tree>(`${base}tree.json`),
  ]);
  latest = l;
  if (kernelSha !== l.kernel_sha) {
    kernel = await Kernel.load(await getBytes(`${base}kernel/${l.kernel_sha}.wasm`));
    kernelSha = l.kernel_sha;
  }
  grid.setData(l);
  grid.draw();
  renderStats(l);
  const what = drawCurve($("curve") as HTMLCanvasElement, curve.points, l.cells.length, chartColours());
  $("curve-label").textContent = what === "lit" ? "STYLES LIT, GENERATION BY GENERATION" : "MAP SCORE, GENERATION BY GENERATION";
  drawTree($("tree") as HTMLCanvasElement, tree, chartColours());
  if (first) {
    const champ = grid.championCell() ?? l.cells[0];
    await select(champ);
  }
}

function chartColours() {
  return { line: css("--slime"), area: css("--slime-wash"), grid: css("--rule"), text: css("--muted"), node: css("--node"), lineage: css("--sun"), lineageFaint: css("--sun-faint"), champ: css("--rival") };
}

function renderStats(l: Latest) {
  const lit = litCount(l);
  $("stat-lit").textContent = `${lit} / ${l.cells.length}`;
  $("stat-gen").textContent = l.generation.toLocaleString("en");
  $("stat-matches").textContent = compact(l.stats.matches_total);
  $("stat-speed").textContent = `${compact(Math.round(l.stats.matches_per_s))} / s`;
  $("status").textContent = `Run ${l.run_id}. Updated ${new Date(l.generated_at).toLocaleString()}.`;
}

function compact(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}k` : n.toLocaleString("en");
}

async function genomeSlot(id: string): Promise<number> {
  if (id === SCRIPTED) return SCRIPTED_SLOT;
  return kernel.loadGenome(id, await getBytes(`${DATA}${runId}/genomes/${id}.bin`));
}

async function select(cell: Cell) {
  grid.selected = cell;
  grid.draw();
  renderCell(cell);
  await startReplay(cell);
}

function renderCell(cell: Cell) {
  const floor = latest.grid.lit_floor ?? 0;
  $("cell-title").textContent = grid.isChampion(cell) ? "The champion" : `Style ${cell.i + 1}, ${cell.j + 1}`;
  $("cell-style").textContent = styleSentence(cell);
  $("cell-strength").textContent = `${strengthWords(cell.fitness)} (${margin(cell.fitness)} against the four rivals).`;
  const coach = coachLine(cell, latest.probes, SCRIPTED);
  $("cell-coach").hidden = coach === null;
  $("cell-coach").textContent = coach ?? "";
  $("cell-born").textContent = `Born in generation ${cell.gen.toLocaleString("en")}.${isLit(cell, floor) ? "" : " This style has not lit up yet."}`;
  const r = cell.replay;
  const rival = r.opponent_id === SCRIPTED ? "the coach (the hand-written bot every style is measured against)" : "a sparring partner";
  $("cell-match").textContent = `The match below is against ${rival}. Final score ${r.points[0]} to ${r.points[1]}${r.points[0] > r.points[1] ? ", a win" : ""}.`;
}

async function startReplay(cell: Cell) {
  const r = cell.replay;
  const mine = await genomeSlot(cell.genome_id);
  const theirs = await genomeSlot(r.opponent_id);
  const rivalName = r.opponent_id === SCRIPTED ? "COACH" : "SPARRING";
  const [left, right, leftName, rightName] = r.side === "A" ? [mine, theirs, "THIS STYLE", rivalName] : [theirs, mine, rivalName, "THIS STYLE"];
  kernel.reset(parseSeed(r.seed));
  mode = { kind: "replay", cell, left, right, leftName, rightName, pl: 0, pr: 0, expect: parseHash(r.final_hash), ticks: r.ticks };
  setVerdict("playing", "Replaying on the real game…");
  paused = false;
  $("pause").textContent = "Pause";
  $("play-mode").hidden = true;
  $("controls").hidden = false;
}

async function startPlay() {
  const cell = grid.selected;
  if (!cell) return;
  const bot = await genomeSlot(cell.genome_id);
  kernel.reset(BigInt(Date.now()));
  mode = { kind: "play", cell, bot, pl: 0, pr: 0 };
  setVerdict("you", "You are green, on the left. Arrow keys or the buttons: move, and up to jump.");
  $("play-mode").hidden = false;
  paused = false;
}

function setVerdict(kind: "playing" | "ok" | "bad" | "you", text: string) {
  const v = $("verdict");
  v.dataset.kind = kind;
  v.textContent = text;
}

function humanAction(): number {
  // You play the left slime, so forward (towards the net) is to the right.
  let a = 0;
  if (keys.has("ArrowRight") || keys.has("d") || touch.right) a |= FORWARD;
  if (keys.has("ArrowLeft") || keys.has("a") || touch.left) a |= BACKWARD;
  if (keys.has("ArrowUp") || keys.has("w") || keys.has(" ") || touch.jump) a |= JUMP;
  return a;
}

function tick() {
  if (mode.kind === "replay") {
    if (kernel.done()) return finishReplay(mode);
    const won = kernel.step(kernel.policy(0, mode.left), kernel.policy(1, mode.right));
    if (won > 0) mode.pl++;
    else if (won < 0) mode.pr++;
  } else if (mode.kind === "play") {
    if (kernel.done()) {
      setVerdict(mode.pl > mode.pr ? "ok" : "bad", mode.pl > mode.pr ? `You won ${mode.pl} to ${mode.pr}. Evolution has work to do.` : `The bot won ${mode.pr} to ${mode.pl}.`);
      mode = { kind: "idle" };
      return;
    }
    const won = kernel.step(humanAction(), kernel.policy(1, mode.bot));
    if (won > 0) mode.pl++;
    else if (won < 0) mode.pr++;
  }
}

function finishReplay(m: Extract<Mode, { kind: "replay" }>) {
  const got = kernel.hash();
  const ticks = kernel.snapshot().tick;
  if (got === m.expect && ticks === m.ticks) setVerdict("ok", `Verified: this browser played the match exactly as the trainer did, all ${ticks.toLocaleString("en")} moves.`);
  else setVerdict("bad", `Mismatch: this browser ended on a different result than the trainer (${ticks} moves). Please tell us.`);
  mode = { kind: "idle" };
}

// tick() can end the match, so the loop asks again each time.
const active = () => (mode as Mode).kind !== "idle";

function frame(t: number) {
  const dt = lastTime ? Math.min(250, t - lastTime) : 0;
  lastTime = t;
  if (!paused && mode.kind !== "idle") {
    acc += dt * (mode.kind === "replay" ? speed : 1);
    let n = 0;
    while (acc >= TICK_MS && active() && n++ < 200) {
      tick();
      acc -= TICK_MS;
    }
  } else acc = 0;
  if (kernel) {
    const s = kernel.snapshot();
    const labels =
      mode.kind === "replay" ? { left: mode.leftName, right: mode.rightName, score: `${mode.pl} : ${mode.pr}` }
      : mode.kind === "play" ? { left: "YOU", right: "BOT", score: `${mode.pl} : ${mode.pr}` }
      : undefined;
    drawCourt(($("court") as HTMLCanvasElement).getContext("2d")!, s, courtColours, labels);
  }
  requestAnimationFrame(frame);
}

function wireInput() {
  const map = $("map") as HTMLCanvasElement;
  map.addEventListener("mousemove", (e) => {
    grid.hover = grid.cellAt(e.clientX, e.clientY);
    grid.draw();
    const c = grid.hover;
    $("hover").textContent = c ? styleSentence(c) : "Tap a square to watch that style play.";
  });
  map.addEventListener("mouseleave", () => {
    grid.hover = null;
    grid.draw();
  });
  map.addEventListener("click", (e) => {
    const c = grid.cellAt(e.clientX, e.clientY);
    if (!c) return;
    select(c);
    // On a phone the court is below the map; bring it into view.
    if (window.matchMedia("(max-width: 820px)").matches) $("court").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  $("pause").addEventListener("click", () => {
    paused = !paused;
    $("pause").textContent = paused ? "Resume" : "Pause";
  });
  $("restart").addEventListener("click", () => grid.selected && startReplay(grid.selected));
  $("speed").addEventListener("click", () => {
    speed = speed === 1 ? 2 : speed === 2 ? 4 : 1;
    $("speed").textContent = `${speed}x`;
  });
  $("champion").addEventListener("click", () => {
    const c = grid.championCell();
    if (c) select(c);
  });
  $("play").addEventListener("click", () => startPlay());
  $("back").addEventListener("click", () => grid.selected && startReplay(grid.selected));
  window.addEventListener("keydown", (e) => {
    if (mode.kind === "play" && ["ArrowLeft", "ArrowRight", "ArrowUp", " "].includes(e.key)) e.preventDefault();
    keys.add(e.key);
  });
  window.addEventListener("keyup", (e) => keys.delete(e.key));
  for (const [id, k] of [["t-left", "left"], ["t-right", "right"], ["t-jump", "jump"]] as const) {
    const b = $(id);
    const on = (e: Event) => { e.preventDefault(); touch[k] = true; };
    const off = (e: Event) => { e.preventDefault(); touch[k] = false; };
    b.addEventListener("pointerdown", on);
    b.addEventListener("pointerup", off);
    b.addEventListener("pointerleave", off);
    b.addEventListener("pointercancel", off);
  }
}

boot();
