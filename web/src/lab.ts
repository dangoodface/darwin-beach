// The lab (FOR NERDS): a brain thinking live beside its court, the league's map of 400 styles, the
// official ladder with the viewer on it, the league's day in ten seconds, and its curve. Everything
// here is computed from gallery.json and the same engine the trainer ran; nothing is video or a mock.
import { Kernel, SCRIPTED_SLOT, GENOME_LEN, padGenome, type Snapshot } from "./kernel.ts";
import { drawCourt, type CourtColours } from "./court.ts";

export type LabCell = [i: number, j: number, rating: number, fitness: number, gen: number, lit: 0 | 1];
export type LabData = {
  run: string; grid: [number, number]; cells: LabCell[]; playable: Record<string, string>;
  ladder: { label: string; rating: number; kind: string }[];
  curve: [gen: number, filled: number, lit: number, beatsCoach: number][];
  stats: { rounds: number; matches: number; filled: number; lit: number; beatsCoach: number; perSecond: number };
};
export type LabBot = { id: string; label: string; rating?: number };
type Opts = { lab: LabData; bots: LabBot[]; brains: Record<string, string>; wasm: Uint8Array<ArrayBuffer>; colours: CourtColours; tabOpen: () => boolean };

const N_IN = 12, N_HID = 64, N_OUT = 3;
const TICK_MS = 1000 / 30, MAX_TICKS = 3000;
const INPUTS = ["me x", "me y", "me vx", "me vy", "ball x", "ball y", "ball vx", "ball vy", "foe x", "foe y", "foe vx", "foe vy"];
const OUTPUTS = ["FORWARD", "BACK", "JUMP"];
const $ = (id: string) => document.getElementById(id)!;
const bytes = (b64: string) => Uint8Array.from(atob(b64.trim()), (c) => c.charCodeAt(0));
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const freshSeed = () => crypto.getRandomValues(new BigUint64Array(1))[0];

// ---------- the brain, exactly as the kernel computes it (docs/ABI.md, "Policy") ----------
export type Brain = { w1: Float32Array; b1: Float32Array; w2: Float32Array; b2: Float32Array };

export function brainOf(raw: ArrayBuffer): Brain {
  let f: Float32Array = new Float32Array(raw.slice(0));
  if (f.length === GENOME_LEN) f = padGenome(f);
  const w1 = N_HID * N_IN, b1 = w1 + N_HID, w2 = b1 + N_OUT * N_HID;
  return { w1: f.subarray(0, w1), b1: f.subarray(w1, b1), w2: f.subarray(b1, w2), b2: f.subarray(w2, w2 + N_OUT) };
}

// The 12 inputs a side sees, mirrored so one brain plays either side: side 0 is the left slime.
export function observe(s: Snapshot, side: 0 | 1): number[] {
  const dir = side === 0 ? -1 : 1;
  const me = side === 0 ? s.a : s.b, op = side === 0 ? s.b : s.a, b = s.ball;
  return [me[0] * dir, me[1], me[2] * dir, me[3], b[0] * dir, b[1], b[2] * dir, b[3], op[0] * -dir, op[1], op[2] * -dir, op[3]].map((v) => v / 10);
}

// Each unit accumulates sequentially, no fused multiply-add, softsign on the hidden layer, linear outputs.
export function think(br: Brain, x: number[]): { h: Float64Array; out: Float64Array } {
  const h = new Float64Array(N_HID), out = new Float64Array(N_OUT);
  for (let i = 0; i < N_HID; i++) {
    let acc = br.b1[i];
    for (let j = 0; j < N_IN; j++) acc = acc + br.w1[i * N_IN + j] * x[j];
    h[i] = acc / (1 + Math.abs(acc));
  }
  for (let k = 0; k < N_OUT; k++) {
    let acc = br.b2[k];
    for (let i = 0; i < N_HID; i++) acc = acc + br.w2[k * N_HID + i] * h[i];
    out[k] = acc;
  }
  return { h, out };
}

export const actionOf = (out: Float64Array) => (out[0] > 0 ? 1 : 0) | (out[1] > 0 ? 2 : 0) | (out[2] > 0 ? 4 : 0);

// ---------- colours ----------
function mix(a: string, b: string, t: number): string {
  const p = (c: string) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",")})`;
}
// A rating from cold to hot: night, sea, slime, sun.
function heat(t: number): string {
  const stops = ["#1a2350", "#2a4a8a", "#39ff9f", "#ffc93c"];
  const u = Math.max(0, Math.min(0.999, t)) * (stops.length - 1);
  const i = Math.floor(u);
  return mix(stops[i], stops[i + 1], u - i);
}

export async function initLab(o: Opts): Promise<{ setYou(elo: number | null): void }> {
  const lab = o.lab;
  const k = await Kernel.load(o.wasm);
  const slime = css("--slime"), rival = css("--rival"), sun = css("--sun"), muted = css("--muted"), ink = css("--ink"), night = css("--night"), rule = css("--rule");

  // ================= the brain, live =================
  const courtC = $("lab-court") as HTMLCanvasElement, netC = $("lab-net") as HTMLCanvasElement;
  const courtCtx = courtC.getContext("2d")!, netCtx = netC.getContext("2d")!;
  let bot: LabBot = o.bots[o.bots.length - 1];
  let slot = 0, brain: Brain | null = null;
  let agree = 0, ticks = 0, acc = 0, last = 0, paused = false, visible = false;
  let pts: [number, number] = [0, 0];
  let thought: { x: number[]; h: Float64Array; out: Float64Array; a: number } | null = null;

  function pick(b: LabBot) {
    bot = b;
    const raw = bytes(o.brains[b.id]).buffer;
    slot = k.loadGenome(b.id, raw);
    brain = brainOf(raw);
    k.reset(freshSeed());
    pts = [0, 0]; agree = ticks = 0; thought = null;
    document.querySelectorAll<HTMLButtonElement>("#lab-bots button").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.id === b.id)));
    $("lab-who").textContent = `${b.label}${b.rating ? ` · Elo ${b.rating}` : ""} vs the coach`;
    draw();
  }

  function tick() {
    if (!brain) return;
    const s = k.snapshot();
    const x = observe(s, 0);
    const { h, out } = think(brain, x);
    const a = k.policy(0, slot);
    if (a === actionOf(out)) agree++;
    ticks++;
    thought = { x, h, out, a };
    const won = k.step(a, k.policy(1, SCRIPTED_SLOT));
    if (won > 0) pts[0]++;
    else if (won < 0) pts[1]++;
    if (k.done(MAX_TICKS)) { k.reset(freshSeed()); pts = [0, 0]; }
  }

  function draw() {
    drawCourt(courtCtx, k.snapshot(), o.colours, { left: bot.label, right: "COACH", score: `${pts[0]} - ${pts[1]}`, scale: 1.4 });
    drawNet();
    $("lab-check").textContent = ticks
      ? (agree === ticks
        ? `This page recomputes the brain every tick in plain JavaScript and compares it with the engine: every action matched, ${ticks.toLocaleString()} ticks so far.`
        : `This page recomputes the brain every tick in plain JavaScript and compares it with the engine: ${agree.toLocaleString()} of ${ticks.toLocaleString()} actions matched.`)
      : "";
  }

  function node(ctx: CanvasRenderingContext2D, x: number, y: number, v: number, base: number, gain: number) {
    const r = base + gain * Math.min(1, Math.abs(v));
    ctx.fillStyle = Math.abs(v) < 0.02 ? rule : v > 0 ? slime : rival;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  }

  // The canvas's logical size follows its CSS size, so text stays readable: 4:3 on a phone, 16:9 on a desktop.
  function sizeNet() {
    const cw = netC.clientWidth || 960;
    const w = cw < 600 ? 480 : 960, h = cw < 600 ? 360 : 540;
    if (netC.width !== w || netC.height !== h) { netC.width = w; netC.height = h; }
  }
  function drawNet() {
    sizeNet();
    const ctx = netCtx, W = netC.width, H = netC.height, f = W / 960;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = css("--night-2");
    ctx.fillRect(0, 0, W, H);
    const inX = 0.155 * W, hidX0 = 0.34 * W, hidX1 = 0.665 * W, outX = 0.885 * W;
    const inY = (j: number) => 0.095 * H + j * 0.076 * H;
    const hid = (i: number): [number, number] => [hidX0 + (i % 8) * ((hidX1 - hidX0) / 7), 0.13 * H + Math.floor(i / 8) * 0.107 * H];
    const outY = (kk: number) => 0.26 * H + kk * 0.24 * H;
    const small = Math.max(10, Math.round(11 * Math.max(f, 0.9)));
    ctx.font = `${small}px ${css("--body")}`;
    ctx.fillStyle = muted; ctx.textAlign = "center";
    ctx.fillText("12 IN", inX, 0.045 * H); ctx.fillText("64 HIDDEN (softsign)", (hidX0 + hidX1) / 2, 0.045 * H); ctx.fillText("3 OUT", outX, 0.045 * H);
    if (!thought || !brain) {
      ctx.fillStyle = muted; ctx.fillText("waiting for the first serve", W / 2, H / 2);
      return;
    }
    const { x, h, out } = thought;
    // the signal on each wire: weight times input; only the strong ones are drawn, so the eye can follow
    ctx.lineWidth = 1;
    for (let i = 0; i < N_HID; i++) {
      const [hx, hy] = hid(i);
      for (let j = 0; j < N_IN; j++) {
        const s = brain.w1[i * N_IN + j] * x[j];
        const m = Math.abs(s);
        if (m < 0.1) continue;
        ctx.strokeStyle = s > 0 ? slime : rival;
        ctx.globalAlpha = Math.min(0.85, m * 1.2);
        ctx.beginPath(); ctx.moveTo(inX + 8 * f, inY(j)); ctx.lineTo(hx, hy); ctx.stroke();
      }
    }
    for (let kk = 0; kk < N_OUT; kk++) {
      for (let i = 0; i < N_HID; i++) {
        const s = brain.w2[kk * N_HID + i] * h[i];
        const m = Math.abs(s);
        if (m < 0.08) continue;
        const [hx, hy] = hid(i);
        ctx.strokeStyle = s > 0 ? slime : rival;
        ctx.globalAlpha = Math.min(0.85, m * 1.5);
        ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(outX - 0.065 * W, outY(kk)); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    ctx.textAlign = "right"; ctx.fillStyle = ink;
    for (let j = 0; j < N_IN; j++) { ctx.fillText(INPUTS[j], inX - 14 * f, inY(j) + 4); node(ctx, inX, inY(j), x[j], 3 * f, 9 * f); }
    for (let i = 0; i < N_HID; i++) { const [hx, hy] = hid(i); node(ctx, hx, hy, h[i], 2.5 * f, 8 * f); }
    const bw = 0.13 * W, bh = 0.097 * H;
    const label = Math.max(9, Math.round(12 * Math.max(f, 0.75)));
    ctx.textAlign = "center";
    for (let kk = 0; kk < N_OUT; kk++) {
      const on = out[kk] > 0, y = outY(kk);
      ctx.fillStyle = on ? (kk === 2 ? sun : slime) : css("--night-3");
      ctx.strokeStyle = on ? (kk === 2 ? sun : slime) : rule;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.rect(outX - bw / 2, y - bh / 2, bw, bh); ctx.fill(); ctx.stroke();
      ctx.font = `${label}px ${css("--display")}`;
      ctx.fillStyle = on ? night : muted;
      ctx.fillText(OUTPUTS[kk], outX, y + label * 0.35);
      ctx.font = `${small}px ${css("--body")}`; ctx.fillStyle = muted;
      ctx.fillText(out[kk].toFixed(2), outX, y + bh / 2 + small + 2);
    }
  }

  function frame(now: number) {
    const dt = Math.min(250, now - (last || now));
    last = now;
    if (visible && !paused && o.tabOpen() && brain) {
      acc += dt;
      let n = 0;
      while (acc >= TICK_MS && n++ < 4) { acc -= TICK_MS; tick(); }
      if (acc > TICK_MS * 4) acc = 0;
      draw();
    }
    requestAnimationFrame(frame);
  }

  const bots = $("lab-bots");
  for (const b of o.bots) {
    const el = document.createElement("button");
    el.type = "button"; el.dataset.id = b.id; el.textContent = b.label;
    el.addEventListener("click", () => pick(b));
    bots.append(el);
  }
  $("lab-pause").addEventListener("click", () => { paused = !paused; $("lab-pause").textContent = paused ? "Play" : "Pause"; });
  new IntersectionObserver((es) => { for (const e of es) if (e.target === courtC) visible = e.isIntersecting; }).observe(courtC);
  window.addEventListener("resize", () => draw());
  pick(bot);
  requestAnimationFrame(frame);

  // ================= the map: 400 styles =================
  const mapC = $("lab-map") as HTMLCanvasElement, mapCtx = mapC.getContext("2d")!;
  const [GX, GY] = lab.grid;
  const cw = mapC.width / GX, ch = mapC.height / GY;
  const ratings = lab.cells.map((c) => c[2]).filter((r) => r > 0);
  const rMin = Math.min(...ratings), rMax = Math.max(...ratings);
  const byKey = new Map(lab.cells.map((c) => [`${c[0]},${c[1]}`, c]));
  const cellRect = (i: number, j: number): [number, number] => [i * cw, (GY - 1 - j) * ch];
  function drawMap(sel?: [number, number]) {
    const ctx = mapCtx;
    ctx.fillStyle = css("--night"); ctx.fillRect(0, 0, mapC.width, mapC.height);
    for (const [i, j, r, , , lit] of lab.cells) {
      const [x, y] = cellRect(i, j);
      ctx.fillStyle = r > 0 ? heat((r - rMin) / Math.max(1, rMax - rMin)) : css("--night-3");
      ctx.fillRect(x + 1, y + 1, cw - 2, ch - 2);
      if (lit) { ctx.strokeStyle = sun; ctx.lineWidth = 2; ctx.strokeRect(x + 2, y + 2, cw - 4, ch - 4); }
      if (lab.playable[`${i},${j}`]) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x + cw / 2, y + ch / 2, cw * 0.3, 0, Math.PI * 2); ctx.stroke(); }
    }
    if (sel) { const [x, y] = cellRect(sel[0], sel[1]); ctx.strokeStyle = rival; ctx.lineWidth = 3; ctx.strokeRect(x - 1, y - 1, cw + 2, ch + 2); }
  }
  function describe(i: number, j: number) {
    const c = byKey.get(`${i},${j}`);
    const info = $("lab-cell"), watch = $("lab-watch") as HTMLButtonElement;
    drawMap([i, j]);
    if (!c) { info.textContent = `Cell ${i},${j}: empty.`; watch.hidden = true; return; }
    const pos = i < GX / 3 ? "deep on its side" : i < (2 * GX) / 3 ? "mid court" : "at the net";
    const air = j < GY / 3 ? "mostly grounded" : j < (2 * GY) / 3 ? "half the time in the air" : "almost always airborne";
    info.textContent = `Cell ${i},${j}: ${pos}, ${air}. Rating ${c[2]} on the league's scale, found at round ${c[4].toLocaleString()}${c[5] ? ", lit (beats the worst of its eight hardest matchups on average)" : ""}.`;
    const id = lab.playable[`${i},${j}`];
    watch.hidden = !id;
    if (id) watch.onclick = () => { pick({ id, label: `CELL ${i},${j}`, rating: c[2] }); $("lab-brain").scrollIntoView({ behavior: "smooth", block: "nearest" }); };
  }
  mapC.addEventListener("click", (e) => {
    const r = mapC.getBoundingClientRect();
    const i = Math.min(GX - 1, Math.floor(((e.clientX - r.left) / r.width) * GX));
    const j = Math.min(GY - 1, GY - 1 - Math.floor(((e.clientY - r.top) / r.height) * GY));
    describe(i, j);
  });
  drawMap();
  const best = [...lab.cells].sort((a, b) => b[2] - a[2])[0];
  if (best) describe(best[0], best[1]);
  $("lab-map-legend").textContent = `Colour is rating on the league's own scale, ${rMin} (dark) to ${rMax} (bright). Gold border: lit. White ring: one of ${Object.keys(lab.playable).length} cells you can watch think.`;

  // ================= the ladder =================
  const LO = 300, HI = 2300;
  let you: number | null = null, all = false;
  function ladder() {
    const box = $("lab-ladder");
    const rows = [...lab.ladder].sort((a, b) => b.rating - a.rating);
    const shown = all ? rows : rows.filter((m, n) => n < 12 || m.kind === "coach" || m.kind === "oracle" || m.kind === "league");
    const items: { label: string; rating: number; kind: string }[] = [...shown];
    if (you !== null) items.push({ label: "YOU", rating: you, kind: "you" });
    items.sort((a, b) => b.rating - a.rating);
    box.replaceChildren(...items.map((m) => {
      const row = document.createElement("div");
      row.className = "lrow"; row.dataset.kind = m.kind;
      const w = Math.max(2, Math.min(100, ((m.rating - LO) / (HI - LO)) * 100));
      row.innerHTML = `<span class="lname"></span><span class="lbar"><i style="width:${w.toFixed(1)}%"></i></span><span class="lnum"></span>`;
      row.querySelector(".lname")!.textContent = m.label.replace(/^oracle, no delay$/, "oracle, 0").replace(/^oracle, (\d+)-tick delay$/, "oracle, $1-tick").replace(/^league, best cell$/, "league best");
      row.querySelector(".lnum")!.textContent = String(Math.round(m.rating));
      return row;
    }));
    const more = $("lab-ladder-more");
    more.textContent = all ? "Show fewer" : `Show all ${rows.length}`;
    more.onclick = () => { all = !all; ladder(); };
    $("lab-you").textContent = you === null ? "Play a match on the PLAY tab and you appear on this ladder." : `You are the pink bar: about ${Math.round(you)}.`;
  }
  ladder();

  // ================= the day in ten seconds =================
  const evoC = $("lab-evo") as HTMLCanvasElement, evoCtx = evoC.getContext("2d")!;
  const maxGen = Math.max(1, ...lab.cells.map((c) => c[4]));
  let evoStart = 0, evoRun = false, evoDone = false;
  function drawEvo(t: number) {
    const ctx = evoCtx;
    ctx.fillStyle = css("--night"); ctx.fillRect(0, 0, evoC.width, evoC.height);
    const ew = evoC.width / GX, eh = evoC.height / GY;
    let n = 0;
    for (const [i, j, r, , gen, lit] of lab.cells) {
      if (gen > t) continue;
      n++;
      ctx.fillStyle = r > 0 ? heat((r - rMin) / Math.max(1, rMax - rMin)) : css("--night-3");
      ctx.fillRect(i * ew + 1, (GY - 1 - j) * eh + 1, ew - 2, eh - 2);
      if (lit && gen <= t) { ctx.strokeStyle = sun; ctx.lineWidth = 1.5; ctx.strokeRect(i * ew + 2, (GY - 1 - j) * eh + 2, ew - 4, eh - 4); }
    }
    $("lab-evo-cap").textContent = `Round ${Math.round(t).toLocaleString()} of ${maxGen.toLocaleString()}: ${n} of ${lab.cells.length} cells hold the brain that still rules them today.`;
  }
  function evoFrame(now: number) {
    if (!evoRun) return;
    const f = Math.min(1, (now - evoStart) / 10000);
    drawEvo(f * maxGen);
    if (f < 1) requestAnimationFrame(evoFrame);
    else { evoRun = false; evoDone = true; $("lab-evo-play").textContent = "Replay"; }
  }
  function evoPlay() { evoStart = performance.now(); evoRun = true; $("lab-evo-play").textContent = "Playing…"; requestAnimationFrame(evoFrame); }
  $("lab-evo-play").addEventListener("click", evoPlay);
  drawEvo(0);
  new IntersectionObserver((es) => { for (const e of es) if (e.isIntersecting && !evoRun && !evoDone) evoPlay(); }).observe(evoC);

  // ================= the curve =================
  const svg = $("lab-curve");
  const cv = lab.curve, gMax = Math.max(1, ...cv.map((p) => p[0])), X0 = 36, X1 = 560, Y0 = 16, Y1 = 196, CELLS = lab.cells.length || 400;
  const px = (g: number) => X0 + (g / gMax) * (X1 - X0), py = (v: number) => Y1 - (v / CELLS) * (Y1 - Y0);
  const line = (idx: 1 | 2 | 3, colour: string, label: string) => {
    const d = cv.map((p, n) => `${n ? "L" : "M"}${px(p[0]).toFixed(1)},${py(p[idx]).toFixed(1)}`).join(" ");
    const lastP = cv[cv.length - 1];
    return `<path d="${d}" fill="none" stroke="${colour}" stroke-width="2"/><text x="${(X1 + 6).toFixed(0)}" y="${(py(lastP[idx]) + 4).toFixed(0)}" fill="${colour}" font-size="11">${lastP[idx]} ${label}</text>`;
  };
  svg.innerHTML = `<line x1="${X0}" y1="${Y1}" x2="${X1}" y2="${Y1}" stroke="${rule}"/><line x1="${X0}" y1="${Y0}" x2="${X0}" y2="${Y1}" stroke="${rule}"/>`
    + `<text x="${X0}" y="${Y1 + 14}" fill="${muted}" font-size="10">round 0</text><text x="${X1}" y="${Y1 + 14}" fill="${muted}" font-size="10" text-anchor="end">round ${gMax.toLocaleString()}</text>`
    + `<text x="${X0 - 4}" y="${Y0 + 4}" fill="${muted}" font-size="10" text-anchor="end">${CELLS}</text><text x="${X0 - 4}" y="${Y1}" fill="${muted}" font-size="10" text-anchor="end">0</text>`
    + line(1, muted, "filled") + line(3, ink, "beat the coach") + line(2, slime, "lit");

  return { setYou(elo) { you = elo; ladder(); } };
}
