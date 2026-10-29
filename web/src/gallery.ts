// Darwin Beach: play the evolved bots, then watch the story, on the same engine that trained them.
// Two engines run the same bytes: one for the viewer's games, one for the stored replays.
import { Kernel, SCRIPTED_SLOT, FORWARD, BACKWARD, JUMP } from "./kernel.ts";
import { drawCourt, type CourtColours } from "./court.ts";
import { initLab, type LabData } from "./lab.ts";

type Side = { id: string; label: string; kind: "bot" | "coach" };
type Match = {
  act: 1 | 2 | 3 | 4 | 5 | 6 | 7; title: string; caption: string; left: Side; right: Side;
  seed: string; ticks: number; hash: string; points: [number, number]; from?: number;
  holds?: [number, number];
  board?: { rank: number; label: string; rating: number; kind?: string }[];
};
// A rated bot the viewer can play; every result moves the viewer's own Elo on the official scale.
type Rival = { id: string; label: string; kind: "bot" | "coach"; rating: number; note: string };
type Facts = { matches: number; boss?: number };
type Gallery = { kernel: string; matches: Match[]; brains: Record<string, string>; arcade: Rival[]; facts?: Facts; lab?: LabData };

const TICK_MS = 1000 / 30;
const MAX_TICKS = 3000;
const SITE = "https://play.thegoodface.ai/beach/";
const ACTS = { 1: "LEARNING", 2: "FUNSIES", 3: "THE BEST", 4: "THE SPECIALISTS", 5: "THE DEFINITIVE PLAYER?", 6: "THE NIGHT SHIFT", 7: "THE LEAGUE" } as const;

const $ = (id: string) => document.getElementById(id)!;
const bytes = (b64: string) => Uint8Array.from(atob(b64.trim()), (c) => c.charCodeAt(0));
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const freshSeed = () => crypto.getRandomValues(new BigUint64Array(1))[0];

const data: Gallery = JSON.parse($("gallery-data").textContent!);
const colours: CourtColours = {
  sky: css("--sea"), sand: css("--sand"), line: css("--foam"), left: css("--slime"), right: css("--rival"),
  ball: css("--sun"), eye: "#ffffff", pupil: css("--night"), text: css("--ink"),
};
const courtCtx = ($("court") as HTMLCanvasElement).getContext("2d")!;
const filmCtx = ($("film") as HTMLCanvasElement).getContext("2d")!;

let game: Kernel;
let theatre: Kernel;
let labApi: { setYou(elo: number | null): void } | null = null;
const slotIn = (k: Kernel, s: { id: string; kind: string }) => (s.kind === "coach" ? SCRIPTED_SLOT : k.loadGenome(s.id, bytes(data.brains[s.id]).buffer));
let last = 0;
// Names and score stay readable when a phone shrinks the court.
const labelScale = (c: CanvasRenderingContext2D) => Math.min(2, Math.max(1, 560 / (c.canvas.clientWidth || 960)));

// ================= the game =================
// preview: the chosen rival plays the coach behind the PLAY button; countdown, play, over: the viewer's match.
type Mode = "preview" | "countdown" | "play" | "over";
let mode: Mode = "preview";
let rival: Rival;
let gLeft = 0, gRight = 0, gTick = 0, gAcc = 0, countEnd = 0, countShown = 0;
let gPts: [number, number] = [0, 0];
let courtVisible = true;
const live = () => mode === "play" || mode === "countdown";

// Anything but the preview is full screen, so the page under it must not scroll.
function setMode(m: Mode) {
  mode = m;
  $("play").dataset.mode = m;
  document.documentElement.style.overflow = m === "preview" ? "" : "hidden";
}

// ---- controls: arrow keys on a keyboard; on a phone, the pad under the court: move on the left, jump on the right
const keys = new Set<string>();
type PadKey = "l" | "r" | "j";
const fingers = new Map<number, PadKey>();

// The pad is a quarter for left, a quarter for right and the right half for jump; a finger can slide between them.
function padKey(x: number): PadKey {
  const r = document.querySelector(".pad")!.getBoundingClientRect();
  const f = (x - r.left) / r.width;
  return f < 0.25 ? "l" : f < 0.5 ? "r" : "j";
}
const holding = (k: PadKey) => [...fingers.values()].includes(k);

// The viewer plays the left slime, so forward (towards the net) is to the right.
function humanAction(): number {
  let a = 0;
  if (keys.has("ArrowRight") || keys.has("d") || keys.has("D") || holding("r")) a |= FORWARD;
  if (keys.has("ArrowLeft") || keys.has("a") || keys.has("A") || holding("l")) a |= BACKWARD;
  if (keys.has("ArrowUp") || keys.has("w") || keys.has("W") || keys.has(" ") || holding("j")) a |= JUMP;
  return a;
}

function padLights() {
  for (const k of ["l", "r", "j"] as const) $(`pad-${k}`).dataset.on = holding(k) ? "1" : "";
}

function controls() {
  const zone = $("zone");
  const pad = document.querySelector<HTMLElement>(".pad")!;
  const onButton = (e: Event) => !!(e.target as Element).closest?.("button");
  pad.addEventListener("pointerdown", (e) => {
    if (!live()) return;
    e.preventDefault();
    try { pad.setPointerCapture(e.pointerId); } catch { /* the finger is still tracked without capture */ }
    fingers.set(e.pointerId, padKey(e.clientX));
  });
  pad.addEventListener("pointermove", (e) => {
    if (!fingers.has(e.pointerId)) return;
    e.preventDefault();
    fingers.set(e.pointerId, padKey(e.clientX));
  });
  for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) pad.addEventListener(ev, (e) => fingers.delete((e as PointerEvent).pointerId));
  zone.addEventListener("contextmenu", (e) => { if (mode !== "preview") e.preventDefault(); });
  // Safari would otherwise scroll, zoom or select under a finger that is playing.
  for (const ev of ["touchstart", "touchmove"]) zone.addEventListener(ev, (e) => { if (mode !== "preview" && !onButton(e)) e.preventDefault(); }, { passive: false });

  const GAME_KEYS = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " "];
  window.addEventListener("keydown", (e) => {
    if (live() && GAME_KEYS.includes(e.key)) e.preventDefault();
    if (mode !== "preview" && e.key === "Escape") return preview(rival);
    if (mode === "over" && e.key === "Enter") return start();
    keys.add(e.key);
  });
  window.addEventListener("keyup", (e) => keys.delete(e.key));
  window.addEventListener("blur", () => { keys.clear(); fingers.clear(); });
  $("quit").addEventListener("click", () => preview(rival));
  $("pick-another").addEventListener("click", () => preview(rival));
  $("share").addEventListener("click", share);
}

// ---- the overlay on the court
function overlay(kind: "go" | "count" | "over" | "none", text = "") {
  const ov = $("ov");
  ov.replaceChildren();
  ov.dataset.dim = kind === "go" || kind === "over" ? "1" : "";
  if (kind === "count") ov.append(Object.assign(document.createElement("span"), { className: "big", textContent: text }));
  if (kind === "go" || kind === "over") {
    const box = document.createElement("div");
    box.className = "over";
    if (kind === "over") box.append(Object.assign(document.createElement("span"), { className: "big", textContent: text }));
    const b = document.createElement("button");
    b.type = "button";
    b.id = "go";
    b.innerHTML = `<span></span><small></small>`;
    b.querySelector("span")!.textContent = kind === "over" ? "▶ AGAIN" : "▶ PLAY";
    b.querySelector("small")!.textContent = `vs ${rival.label} · Elo ${rival.rating}`;
    b.addEventListener("click", start);
    box.append(b);
    ov.append(box);
  }
}

function preview(r: Rival) {
  rival = r;
  setMode("preview");
  fingers.clear();
  gLeft = SCRIPTED_SLOT;
  gRight = slotIn(game, r);
  game.reset(freshSeed());
  gPts = [0, 0]; gTick = 0; gAcc = 0;
  overlay("go");
  $("preview-tag").textContent = r.kind === "coach" ? "preview: the coach plays itself" : `preview: ${r.label} vs the coach`;
  $("note").textContent = r.note;
  document.querySelectorAll<HTMLButtonElement>("#rivals button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === r.id)));
  drawGame();
}

function start() {
  setMode("countdown");
  fingers.clear(); keys.clear();
  gRight = slotIn(game, rival);
  game.reset(freshSeed());
  gPts = [0, 0]; gTick = 0; gAcc = 0;
  countEnd = performance.now() + 3000; countShown = 0;
  $("preview-tag").textContent = "";
  $("share").textContent = "Share my score";
  $("bar-title").textContent = `YOU VS ${rival.label} \u00B7 ELO ${rival.rating}`;
  drawGame();
}

function stepGame() {
  const a = mode === "play" ? humanAction() : game.policy(0, gLeft);
  const won = game.step(a, game.policy(1, gRight));
  gTick++;
  if (won > 0) gPts[0]++;
  else if (won < 0) gPts[1]++;
}

function drawGame() {
  const left = mode === "preview" ? "COACH" : "YOU";
  drawCourt(courtCtx, game.snapshot(), colours, { left, right: rival.label, score: `${gPts[0]} - ${gPts[1]}`, scale: labelScale(courtCtx) });
}

function over() {
  setMode("over");
  const [a, b] = gPts;
  const s = a > b ? 1 : a < b ? 0 : 0.5;
  results.push({ id: rival.id, r: rival.rating, s });
  save();
  showYou();
  overlay("over", s === 1 ? "YOU WIN" : s === 0 ? "YOU LOSE" : "DRAW");
  $("result").textContent = (s === 1 ? `You beat ${rival.label} ${a} - ${b}. Evolution has work to do.` : s === 0 ? `${rival.label} wins ${b} - ${a}.` : `${a} - ${b} when the clock ran out.`) + ` Your Elo: ${yourElo(results)}.`;
}

async function share() {
  const [a, b] = gPts;
  const text = `${a > b ? "I beat" : a < b ? "I lost to" : "I drew with"} ${rival.label} (Elo ${rival.rating}) ${a}-${b} at Darwin Beach, where slimes taught themselves volleyball. My Elo: ${yourElo(results)}. Your turn:`;
  if (navigator.share) {
    try { await navigator.share({ title: "Darwin Beach", text, url: SITE }); return; }
    catch (e) { if ((e as Error).name === "AbortError") return; }
  }
  try { await navigator.clipboard.writeText(`${text} ${SITE}`); $("share").textContent = "Copied! Paste it anywhere"; }
  catch { $("share").textContent = SITE; }
}

// ---- the viewer's Elo
type Result = { id: string; r: number; s: number };
const STORE = "darwin-beach-arcade-v2";
let results: Result[] = [];
try { results = JSON.parse(localStorage.getItem(STORE) ?? "[]"); } catch { results = []; }
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(results)); } catch { /* private window: this visit only */ } };

// The rating that makes the viewer's expected score equal the real one, the bots' ratings held fixed.
// With no loss or no win yet the number is only a bound; the lab's ladder then places the viewer at it.
function eloNumber(rs: Result[]): number | null {
  const got = rs.reduce((a, x) => a + x.s, 0);
  if (!rs.length) return null;
  if (got === 0) return Math.min(...rs.map((x) => x.r)) - 100;
  if (got === rs.length) return Math.max(...rs.map((x) => x.r)) + 100;
  let lo = -1000, hi = 4000;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    const e = rs.reduce((a, x) => a + 1 / (1 + 10 ** ((x.r - mid) / 400)), 0);
    if (e < got) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}
function yourElo(rs: Result[]) {
  const got = rs.reduce((a, x) => a + x.s, 0);
  if (!rs.length) return "unrated";
  if (got === 0) return `below ${Math.min(...rs.map((x) => x.r))}`;
  if (got === rs.length) return `above ${Math.max(...rs.map((x) => x.r))}`;
  return `about ${Math.round(eloNumber(rs)!)}`;
}

function showYou() {
  const w = results.filter((x) => x.s === 1).length;
  $("you").textContent = results.length
    ? `Your Elo: ${yourElo(results)} after ${results.length} match${results.length === 1 ? "" : "es"}, ${w} won. The coach is 1000.`
    : "Your Elo: unrated. Play anyone to get a number. The coach is 1000.";
  $("you-reset").hidden = !results.length;
  labApi?.setYou(eloNumber(results));
}

function rivals() {
  const box = $("rivals");
  for (const r of data.arcade) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.id = r.id;
    b.innerHTML = `<span class="who"></span><span class="elo"></span><span class="tier"></span>`;
    b.dataset.tier = r.rating < 1000 ? "easy" : r.rating < 1900 ? "medium" : "hard";
    b.querySelector(".who")!.textContent = r.label;
    b.querySelector(".elo")!.textContent = `Elo ${r.rating}`;
    b.querySelector(".tier")!.textContent = r.rating < 1000 ? "WARM-UP" : r.rating < 1900 ? "A REAL GAME" : "GOOD LUCK";
    b.addEventListener("click", () => preview(r));
    box.append(b);
  }
  $("you-reset").addEventListener("click", () => { results = []; save(); showYou(); });
  showYou();
}

// ================= the story: stored matches, replayed =================
let cur = 0, fLeft = 0, fRight = 0, fAcc = 0, speed = 1, advanceAt = 0;
let fPts: [number, number] = [0, 0];
let filmPlaying = false, filmFinished = false, filmVisible = false, filmStarted = false;

function filmTick() {
  const won = theatre.step(theatre.policy(0, fLeft), theatre.policy(1, fRight));
  if (won > 0) fPts[0]++;
  else if (won < 0) fPts[1]++;
}

function drawFilm() {
  const m = data.matches[cur];
  drawCourt(filmCtx, theatre.snapshot(), colours, { left: m.left.label, right: m.right.label, score: `${fPts[0]} - ${fPts[1]}`, scale: labelScale(filmCtx) });
}

// One tick per match; a match's tick turns green once this browser has replayed and verified it.
const seen: ("ok" | "bad" | undefined)[] = [];
function progress() {
  const box = $("progress");
  box.replaceChildren(...data.matches.flatMap((m, i) => {
    const t = document.createElement("i");
    t.dataset.s = i === cur ? "now" : seen[i] ?? "";
    return i > 0 && data.matches[i - 1].act !== m.act ? [Object.assign(document.createElement("i"), { className: "gap" }), t] : [t];
  }));
}

function loadMatch(i: number) {
  cur = i;
  filmStarted = true;
  const m = data.matches[i];
  fLeft = slotIn(theatre, m.left);
  fRight = slotIn(theatre, m.right);
  theatre.reset(BigInt(m.seed), ...(m.holds ?? [1, 1]));
  fPts = [0, 0];
  // A long match can start just before its moment; the skipped ticks still run, so the hash still checks.
  for (let t = 0; t < (m.from ?? 0) && !theatre.done(MAX_TICKS); t++) filmTick();
  filmFinished = false; filmPlaying = true; fAcc = 0; advanceAt = 0;
  $("next").dataset.ready = "";
  progress();
  $("m-act").textContent = `ACT ${m.act} · ${ACTS[m.act]} · MATCH ${i + 1} OF ${data.matches.length}`;
  $("m-title").textContent = m.title;
  $("m-caption").textContent = m.caption;
  const b = $("m-board"), more = $("m-board-more");
  const board = m.board ?? [];
  b.hidden = !board.length;
  more.hidden = board.length <= 10;
  more.textContent = `Show all ${board.length}`;
  more.onclick = () => { b.querySelectorAll("li").forEach((li) => { li.hidden = false; }); more.hidden = true; };
  b.replaceChildren(...board.map((e, n) => {
    const li = document.createElement("li");
    li.dataset.kind = e.kind ?? (e.label === "the coach" ? "coach" : e.label.startsWith("run 6") ? "new" : "");
    li.append(Object.assign(document.createElement("span"), { textContent: String(e.rank) }),
      Object.assign(document.createElement("span"), { textContent: e.label }),
      Object.assign(document.createElement("span"), { textContent: String(e.rating) }));
    li.hidden = n >= 10;
    return li;
  }));
  verdict("playing", m.from ? `Skipped to second ${Math.round(m.from / 30)}, the moment that matters.` : "");
  $("pause").textContent = "Pause";
  showAct(m.act);
  drawFilm();
}

function verdict(kind: string, text: string) {
  const v = $("verdict");
  v.dataset.kind = kind;
  v.textContent = text;
}

function finishFilm() {
  filmFinished = true;
  const m = data.matches[cur];
  const hash = "0x" + theatre.hash().toString(16).padStart(16, "0");
  const ok = hash === m.hash && fPts[0] === m.points[0] && fPts[1] === m.points[1];
  const time = `${Math.round(theatre.snapshot().tick / 30)} seconds`;
  verdict(ok ? "ok" : "bad", ok
    ? `Final ${fPts[0]} - ${fPts[1]} after ${time}. Your browser just replayed it bit for bit: fingerprint ${hash} matches the record.`
    : `Final ${fPts[0]} - ${fPts[1]}, but the fingerprint ${hash} does not match the record ${m.hash}. Something is off.`);
  seen[cur] = ok ? "ok" : "bad";
  progress();
  const nextUp = data.matches[cur + 1];
  if (!nextUp) { verdict(ok ? "ok" : "bad", `${$("verdict").textContent} That was the last match: the whole story, verified in your browser.`); return; }
  if (($("auto") as HTMLInputElement).checked) {
    advanceAt = performance.now() + 3000;
    $("verdict").textContent += ` Next: ${nextUp.title}.`;
  } else $("next").dataset.ready = "1";
}

// The story's index: one chip per act, and the chosen act's matches as a short list.
let shownAct = 0;
function acts() {
  const box = $("acts");
  for (const a of [...new Set(data.matches.map((m) => m.act))]) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.act = String(a);
    b.textContent = String(a);
    b.title = ACTS[a];
    b.addEventListener("click", () => loadMatch(data.matches.findIndex((m) => m.act === a)));
    box.append(b);
  }
}

function showAct(a: Match["act"]) {
  document.querySelectorAll<HTMLButtonElement>("#acts button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.act === String(a))));
  if (a !== shownAct) {
    shownAct = a;
    $("act-name").textContent = `ACT ${a} \u00B7 ${ACTS[a]}`;
    $("list").replaceChildren(...data.matches.map((m, i) => ({ m, i })).filter(({ m }) => m.act === a).map(({ m, i }) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.i = String(i);
      b.innerHTML = `<span class="n">${String(i + 1).padStart(2, "0")}</span><span class="t"></span><span class="who"></span>`;
      b.querySelector(".t")!.textContent = m.title;
      b.querySelector(".who")!.textContent = `${m.left.label} vs ${m.right.label}`;
      b.addEventListener("click", () => loadMatch(i));
      li.append(b);
      return li;
    }));
  }
  document.querySelectorAll<HTMLButtonElement>("#list button").forEach((x) => x.setAttribute("aria-current", String(Number(x.dataset.i) === cur)));
}

// ================= tabs =================
const TABS = ["play", "story", "nerds"] as const;
type Tab = (typeof TABS)[number];
function openTab(t: Tab) {
  for (const x of TABS) $(`tab-${x}`).hidden = x !== t;
  document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
  try { history.replaceState(null, "", t === "play" ? location.pathname + location.search : `#${t}`); } catch { /* sandboxed: the tab still opens */ }
  window.scrollTo(0, 0);
}
function tabs() {
  // The story's court sticks just under the tab bar, whatever height the bar wraps to.
  const bar = document.querySelector<HTMLElement>(".tabs")!;
  const measure = () => document.documentElement.style.setProperty("--tabs-h", `${bar.offsetHeight}px`);
  measure();
  window.addEventListener("resize", measure);
  document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((b) => b.addEventListener("click", () => openTab(b.dataset.tab as Tab)));
  document.querySelectorAll<HTMLButtonElement>("[data-go]").forEach((b) => b.addEventListener("click", () => openTab(b.dataset.go as Tab)));
  const h = location.hash.slice(1) as Tab;
  if (TABS.includes(h)) openTab(h);
}

// ================= one clock for both courts =================
function frame(now: number) {
  const dt = Math.min(250, now - (last || now));
  last = now;

  if (mode === "countdown") {
    const n = Math.ceil((countEnd - now) / 1000);
    if (n <= 0) { setMode("play"); overlay("none"); }
    else if (n !== countShown) { countShown = n; overlay("count", String(n)); }
  }
  if (mode === "play" || (mode === "preview" && courtVisible)) {
    gAcc += dt;
    let n = 0;
    while (gAcc >= TICK_MS && n++ < 30) {
      gAcc -= TICK_MS;
      stepGame();
      if (game.done(MAX_TICKS)) {
        if (mode === "play") { over(); break; }
        game.reset(freshSeed()); gPts = [0, 0]; gTick = 0;
      }
    }
    if (gAcc > TICK_MS * 4) gAcc = 0;
    drawGame();
    if (mode === "play") padLights();
  }

  // The replays rest while the viewer plays.
  if (filmStarted && filmPlaying && !filmFinished && filmVisible && !live()) {
    fAcc += dt * speed;
    let n = 0;
    while (fAcc >= TICK_MS && n++ < 400) {
      filmTick();
      fAcc -= TICK_MS;
      if (theatre.done(MAX_TICKS)) { finishFilm(); break; }
    }
    drawFilm();
  }
  if (advanceAt && now >= advanceAt && filmVisible) loadMatch(cur + 1);
  requestAnimationFrame(frame);
}

function facts() {
  const f = data.facts;
  const set = (k: string, v: string) => document.querySelectorAll(`[data-fact="${k}"]`).forEach((e) => { e.textContent = v; });
  set("matchCount", String(data.matches.length));
  if (!f) return;
  set("matchesWords", `${Math.round(f.matches / 1e6)} million`);
  set("matchesShort", `${Math.round(f.matches / 1e6)}M`);
  if (f.boss) set("boss", String(f.boss));
}

async function main() {
  facts();
  acts();
  tabs();
  const wasm = bytes($("kernel-b64").textContent!);
  try {
    [game, theatre] = await Promise.all([Kernel.load(wasm), Kernel.load(wasm)]);
  } catch (e) {
    $("ov").textContent = `This browser would not start the game engine (${(e as Error).message}). The story and the lab need it too; a current Chrome, Safari or Firefox runs it.`;
    return;
  }
  controls();
  rivals();
  preview(data.arcade.find((r) => r.kind === "coach") ?? data.arcade[0]);
  if (data.lab) {
    try {
      labApi = await initLab({ lab: data.lab, bots: data.arcade.filter((r) => r.kind === "bot"), brains: data.brains, wasm, colours, tabOpen: () => !$("tab-nerds").hidden });
      labApi.setYou(eloNumber(results));
    } catch (e) { $("lab-check").textContent = `The lab could not start (${(e as Error).message}).`; }
  }

  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.target.id === "court") courtVisible = e.isIntersecting;
      if (e.target.id === "film") {
        filmVisible = e.isIntersecting;
        if (filmVisible && !filmStarted) loadMatch(0);
      }
    }
  });
  io.observe($("court"));
  io.observe($("film"));

  $("pause").addEventListener("click", () => {
    if (!filmStarted || filmFinished) return loadMatch(cur);
    filmPlaying = !filmPlaying;
    $("pause").textContent = filmPlaying ? "Pause" : "Play";
  });
  $("restart").addEventListener("click", () => loadMatch(cur));
  $("next").addEventListener("click", () => loadMatch((cur + 1) % data.matches.length));
  $("speed").addEventListener("click", () => {
    speed = speed === 1 ? 2 : speed === 2 ? 4 : 1;
    $("speed").textContent = `${speed}x`;
  });
  requestAnimationFrame(frame);
}
main();
