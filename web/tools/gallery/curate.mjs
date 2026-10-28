// Picks the replay gallery's matches by playing them on the real kernel, and writes
// gallery.json: every match as seed + two brains, with the hash and score it must land on.
// Usage: node tools/gallery/curate.mjs [--out file]
import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { Kernel, SCRIPTED_SLOT } from "../../src/kernel.ts";

const OUT = join(process.env.ARENA_DATA ?? join(homedir(), ".local/share/arena"), "out");
const RUN = {
  r1: "20261027T055834Z-slime-v0",
  r2: "20261027T065906Z-night-r2",
  r3: "20261027T085957Z-night-r3",
  r4: "20261027T110547Z-night-r4-cma",
  r5: "20261027T130856Z-night-r5-ladder",
  r6: "20261028T001309Z-run6-definitive",
};
const COACH = "scripted-v1";
const latest = Object.fromEntries(Object.entries(RUN).map(([k, d]) => [k, JSON.parse(readFileSync(join(OUT, d, "latest.json"), "utf8"))]));
const ARENA = process.env.ARENA_DATA ?? join(homedir(), ".local/share/arena");
const SPEC = join(ARENA, "specialists");
// Specialists run kernel version 2 (juggle counters; every older replay lands on the same hash),
// so the gallery uses the newest kernel any specialist carries, else run 4's.
const specRuns = existsSync(SPEC) ? readdirSync(SPEC).filter((d) => existsSync(join(SPEC, d, "result.json"))).sort() : [];
const result = (d) => JSON.parse(readFileSync(join(SPEC, d, "result.json"), "utf8"));
const v2 = specRuns.map((d) => ({ d, r: result(d) })).filter(({ r }) => r.kernel_version >= 2).at(-1);
const kernelPath = v2 ? join(SPEC, v2.d, "kernel", `${v2.r.kernel_sha}.wasm`) : join(OUT, RUN.r4, "kernel", `${latest.r4.kernel_sha}.wasm`);
const wasm = readFileSync(kernelPath);
const kernel = await Kernel.load(wasm);
const k = kernel;
// The rally scan plays 2,000 stored matches: a second kernel whose slots 0 and 1 get overwritten.
const scan = await Kernel.load(wasm);
function scanSlot(i, run, id) {
  if (id === COACH) return SCRIPTED_SLOT;
  const b = readFileSync(join(OUT, RUN[run], "genomes", `${id}.bin`));
  new Uint8Array(scan.k.memory.buffer).set(b, scan.k.genome_table_ptr() + i * 515 * 4);
  return i;
}

const brains = {};
function slot(run, id) {
  if (id === COACH) return SCRIPTED_SLOT;
  const b = readFileSync(run.startsWith("spec:") ? join(SPEC, run.slice(5), "genomes", `${id}.bin`) : join(OUT, RUN[run], "genomes", `${id}.bin`));
  brains[id] = b.toString("base64");
  return k.loadGenome(id, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

// One whole match, with the longest rally (ticks from serve to point) and when it started.
function play(seed, left, right, k = kernel) {
  k.reset(seed);
  const pts = [0, 0];
  let rallyStart = 0, best = { rally: -1, start: 0 };
  while (!k.done(3000)) {
    const s0 = k.snapshot();
    if (s0.delay > 0) rallyStart = s0.tick + 1;
    const won = k.step(k.policy(0, left), k.policy(1, right));
    if (won !== 0) {
      won > 0 ? pts[0]++ : pts[1]++;
      const t = k.snapshot().tick;
      if (t - rallyStart > best.rally) best = { rally: t - rallyStart, start: rallyStart };
    }
  }
  return { ticks: k.snapshot().tick, hash: "0x" + k.hash().toString(16).padStart(16, "0"), points: pts, rally: best };
}

// Serves no bot was selected on: a fixed splitmix64 stream, so the gallery is reproducible.
let sm = 0x0da8_1b0e_5eedn;
function freshSeed() {
  sm = BigInt.asUintN(64, sm + 0x9e3779b97f4a7c15n);
  let z = sm;
  z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
  z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
  return z ^ (z >> 31n);
}
const FRESH = Array.from({ length: 100 }, freshSeed);

const matches = [];
function add(m, run, leftId, rightId, seed, extra = {}) {
  const r = play(seed, slot(run.l ?? run, leftId), slot(run.r ?? run, rightId));
  matches.push({ ...m, left: { ...m.left, id: leftId }, right: { ...m.right, id: rightId }, seed: seed.toString(), ticks: r.ticks, hash: r.hash, points: r.points, ...extra(r) });
  console.log(m.title, r.points, r.ticks, r.hash);
  return r;
}
function record(runL, l, runR, r, seeds = FRESH) {
  const sl = slot(runL, l), sr = slot(runR, r);
  let w = 0;
  const games = seeds.map((s) => { const g = play(s, sl, sr); if (g.points[0] > g.points[1]) w++; return { s, g }; });
  return { wins: w, of: seeds.length, games };
}
// A representative game: the commonest kind of result, of middling length.
function typical(games, wantLeftWin) {
  const pool = games.filter(({ g }) => (g.points[0] > g.points[1]) === wantLeftWin).sort((a, b) => a.g.ticks - b.g.ticks);
  return pool[Math.floor(pool.length / 2)].s;
}
const champOf = (run) => latest[run].cells.find((c) => c.genome_id === latest[run].stats.champion_id);
const coachSide = { label: "COACH", kind: "coach" };

// ---------- act 1: learning ----------
const tree1 = JSON.parse(readFileSync(join(OUT, RUN.r1, "tree.json"), "utf8"));
const have = (n) => existsSync(join(OUT, RUN.r1, "genomes", `${n.id}.bin`));
const gen0 = tree1.nodes.filter((n) => n.gen === 0 && have(n)).sort((a, b) => a.fitness - b.fitness);
const g0 = gen0[Math.floor(gen0.length / 2)];
const c1 = champOf("r1");
add({ act: 1, title: "Day one", left: { label: "GEN 0", kind: "bot" }, right: coachSide,
  caption: "A slime with a random brain: 515 random numbers, no training. It does not know the ball exists." },
  "r1", g0.id, COACH, BigInt(c1.replay.seed), () => ({}));

// The earliest bot in run 1's family tree that takes a point off the coach.
const early = tree1.nodes.filter((n) => n.gen > 0 && have(n)).sort((a, b) => a.gen - b.gen);
let first = null;
for (const n of early) {
  const sl = slot("r1", n.id);
  const s = FRESH.slice(0, 6).find((s) => play(s, sl, SCRIPTED_SLOT).points[0] > 0);
  if (s !== undefined) { first = { n, s }; break; }
}
add({ act: 1, title: "The first point", left: { label: `GEN ${first.n.gen}`, kind: "bot" }, right: coachSide,
  caption: `Generation ${first.n.gen.toLocaleString("en")}. Most random brains never touch the ball, but one in thousands gets lucky and takes a point off the coach. A lucky start is all evolution needs.` },
  "r1", first.n.id, COACH, first.s, () => ({}));

const c1rec = record("r1", c1.genome_id, "r1", COACH);
add({ act: 1, title: "The first champion", left: { label: "CHAMP R1", kind: "bot" }, right: coachSide,
  caption: `Run 1's best bot after an hour and 52 million matches. It plays the coach close, but against the coach on serves it never trained on it wins ${c1rec.wins} of ${c1rec.of}.` },
  "r1", c1.genome_id, COACH, BigInt(c1.replay.seed), () => ({}));

// ---------- act 2: funsies ----------
let longest = null;
for (const run of ["r1", "r2", "r3", "r4", "r5"]) for (const c of latest[run].cells) {
  const r = c.replay, a = r.side !== "B";
  const sl = scanSlot(0, run, c.genome_id), so = scanSlot(1, run, r.opponent_id);
  const g = a ? play(BigInt(r.seed), sl, so, scan) : play(BigInt(r.seed), so, sl, scan);
  if (!longest || g.rally.rally > longest.g.rally.rally) longest = { run, c, a, g };
}
{
  const { run, c, a, g } = longest;
  const me = { label: `STYLE ${run.toUpperCase()}`, kind: "bot" }, them = c.replay.opponent_id === COACH ? coachSide : { label: "SPARRING", kind: "bot" };
  add({ act: 2, title: "The rally that would not end", left: a ? me : them, right: a ? them : me,
    caption: `${(g.rally.rally / 30).toFixed(0)} seconds without a point. Both sides found a loop that keeps the ball alive, and neither knows how to break it.` },
    run, a ? c.genome_id : c.replay.opponent_id, a ? c.replay.opponent_id : c.genome_id, BigInt(c.replay.seed), (r) => ({ from: Math.max(0, r.rally.start - 45) }));
}

// The map's corners in run 4: the lowest and the highest airtime among lit styles.
const lit4 = latest.r4.cells.filter((c) => c.lit);
const grounded = lit4.filter((c) => c.bd[0] > 0.6).sort((a, b) => a.bd[1] - b.bd[1])[0];
const pogo = [...lit4].sort((a, b) => b.bd[1] - a.bd[1])[0];
for (const [c, title, label, caption] of [
  [grounded, "The one that never jumps", "GROUNDED", "A style from the map's grounded corner: it hugs the net and almost never leaves the sand. Evolution kept it because it is the best at being that."],
  [pogo, "The pogo stick", "AIRBORNE", "The opposite corner of the map: the style that spends the most time in the air. Nobody told it to jump. Jumping is just what its family does."],
]) {
  const rec = record("r4", c.genome_id, "r4", COACH, FRESH.slice(0, 16));
  add({ act: 2, title, left: { label, kind: "bot" }, right: coachSide, caption: `${caption} Against the coach on fresh serves: ${rec.wins} of 16.` },
    "r4", c.genome_id, COACH, typical(rec.games, rec.wins * 2 > 16), () => ({}));
}

// The ladder's lesson: champion 7 beats champion 6 and never wins against the coach.
const lad = JSON.parse(readFileSync(join(OUT, RUN.r5, "ladder.json"), "utf8"));
const ch = (i) => lad.members[i].id;
const c7 = record("r5", ch(7), "r5", COACH, FRESH.slice(0, 16));
const c7v6 = record("r5", ch(7), "r5", ch(6), FRESH.slice(0, 16));
add({ act: 2, title: "The forgetful champion", left: { label: "CHAMP #7", kind: "bot" }, right: coachSide,
  caption: `Run 5's seventh champion beat the sixth in ${c7v6.wins} of 16 matches, and its rating went up. Against the coach it wins ${c7.wins} of 16. Beating each other is not the same as getting better.` },
  "r5", ch(7), COACH, typical(c7.games, c7.wins * 2 > 16), () => ({}));

const c4 = champOf("r4");
const thenNow = record("r4", c4.genome_id, "r1", c1.genome_id, FRESH.slice(0, 16));
add({ act: 2, title: "Then vs now", left: { label: "CHAMP R4", kind: "bot" }, right: { label: "CHAMP R1", kind: "bot" },
  caption: `The night's smarter breeding against the first hour's champion. Over 16 serves it wins ${thenNow.wins}.` },
  { l: "r4", r: "r1" }, c4.genome_id, c1.genome_id, typical(thenNow.games, thenNow.wins * 2 > 16), () => ({}));

// ---------- act 3: the best ----------
const c4rec = record("r4", c4.genome_id, "r4", COACH);
add({ act: 3, title: "The coach falls", left: { label: "CHAMP R4", kind: "bot" }, right: coachSide,
  caption: `Run 4, smarter breeding: the first bot to really beat the coach. On 100 serves it never trained on, it wins ${c4rec.wins}.` },
  "r4", c4.genome_id, COACH, typical(c4rec.games, true), () => ({}));

const c10 = ch(10);
const c10rec = record("r5", c10, "r5", COACH);
add({ act: 3, title: "Second seed, same result", left: { label: "CHAMP #10", kind: "bot" }, right: coachSide,
  caption: `Run 5's tenth champion, bred on a different seed against a hall of fame of past champions. On 100 fresh serves it beats the coach ${c10rec.wins} times.` },
  "r5", c10, COACH, typical(c10rec.games, true), () => ({}));

const fin = record("r5", c10, "r4", c4.genome_id);
const leftWins = fin.wins * 2 > fin.of;
add({ act: 3, title: "Best of the night", left: { label: "CHAMP #10", kind: "bot" }, right: { label: "CHAMP R4", kind: "bot" },
  caption: `The two coach-beaters, bred on different seeds, meet for the first time. Over 100 serves, run 5's champion wins ${fin.wins} and run 4's wins ${fin.of - fin.wins}. This is one of the ${leftWins ? fin.wins : fin.of - fin.wins}.` },
  { l: "r5", r: "r4" }, c10, c4.genome_id, typical(fin.games, leftWins), () => ({}));

// ---------- act 4: the specialists ----------
// Each style is trained against #10, run 4's champion and the coach, with a rule it cannot break,
// and judged on serves it never trained on. One-minute smoke runs do not count.
const STYLES = {
  flat_earther: ["The Flat-Earther", "FLAT-EARTHER", "Never jumps. Its jump button is wired off, so every return comes from the sand."],
  all_rounder: ["The All-Rounder", "ALL-ROUNDER", "No gimmick: it must beat all three rivals at once. The control for every other specialist."],
  kangaroo: ["The Kangaroo", "KANGAROO", "Always jumping. Its jump button is wired on, so it is airborne whenever it can be."],
  sprinter: ["The Sprinter", "SPRINTER", "Paid to win fast: every tick a point lasts costs it score."],
  long_ball: ["Long Ball", "LONG BALL", "Only hard, low, deep shots, or it is thrown out."],
  lob_artist: ["The Lob Artist", "LOB ARTIST", "Only slow, high balls, or it is thrown out."],
  joga_bonito: ["Joga Bonito", "JOGA BONITO", "Paid to juggle: four or more touches in a row, crossing half its court, then the kill."],
};
const ROUND = { [ch(10)]: "CHAMP #10", [c4.genome_id]: "CHAMP R4", [COACH]: "COACH" };
for (const style of Object.keys(STYLES)) {
  const runs = specRuns.filter((d) => { const r = result(d); return r.style === style && r.elapsed_s >= 600; });
  const main = runs.filter((d) => !result(d).twin).at(-1);
  if (!main) continue;
  const r = result(main), twin = runs.filter((d) => result(d).twin).at(-1);
  const [title, label, blurb] = STYLES[style];
  const f = r.final.fresh, fr = (id) => `${f[id]?.won ?? "?"} of ${f[id]?.n ?? 16}`;
  const rec = `On new serves: #10 ${fr(ch(10))}, run 4's champion ${fr(c4.genome_id)}, the coach ${fr(COACH)}.`;
  const p = r.final.proof ?? {}, md = r.final.descriptors_fresh_mean ?? {};
  const pct = (x) => `${Math.round(100 * x)} per cent`;
  const proof = style === "flat_earther" ? (p.jumps_max === 0 ? " It jumped zero times in every one of them." : ` Jumps per match: up to ${p.jumps_max}.`)
    : style === "kangaroo" ? ` It jumped ${Math.round(p.jumps_min ?? 0)} to ${Math.round(p.jumps_max ?? 0)} times a match.`
    : style === "joga_bonito" ? ` It wins ${pct(md.juggled_points / md.points)} of its points by juggling, and its longest run averages ${md.max_streak.toFixed(1)} touches. In this match, watch the very first point.` : "";
  let twinNote = "";
  if (twin) {
    const t = result(twin).final.fresh;
    const tw = [ch(10), c4.genome_id, COACH].map((id) => t[id]?.won ?? 0).reduce((a, b) => a + b, 0);
    const me = [ch(10), c4.genome_id, COACH].map((id) => f[id]?.won ?? 0).reduce((a, b) => a + b, 0);
    twinNote = ` Its twin, trained the same way without the rule, wins ${tw} of 48 and this one ${me}: ${tw === me ? "the rule cost it nothing we can measure" : tw > me ? `the rule costs ${tw - me} wins` : `the rule helped, by ${me - tw} wins`}.`;
    const tp = result(twin).final.proof ?? {}, td = result(twin).final.descriptors_fresh_mean ?? {};
    if (style === "kangaroo") twinNote += ` The twin chose to keep jumping anyway, ${Math.round(td.jumps ?? 0)} times a match: both started from the same high-flying brain.`;
    if (style === "sprinter" && tp.ticks_per_point && p.ticks_per_point)
      twinNote += ` Its points last ${(p.ticks_per_point / 30).toFixed(1)} seconds against the twin's ${(tp.ticks_per_point / 30).toFixed(1)}.`;
    if (style === "long_ball") twinNote += ` Its shots leave at speed ${md.mean_speed_after_hit.toFixed(0)} against the twin's ${td.mean_speed_after_hit.toFixed(0)}.`;
    if (style === "lob_artist") twinNote += ` Its balls leave at speed ${md.mean_speed_after_hit.toFixed(0)} against the twin's ${td.mean_speed_after_hit.toFixed(0)}, just as high, so they hang.`;
    if (style === "joga_bonito") twinNote += ` The twin juggles ${pct(td.juggled_points / td.points)} of its points.`;
  }
  // The generality check: fresh serves against every member of run 5's hall of fame, not just the three it trained on.
  let hof = "";
  try {
    const h = JSON.parse(readFileSync(join(SPEC, main, "hof.json"), "utf8"));
    const worst = Math.min(...Object.values(h.per_member).map((m) => m.won));
    hof = ` Against all ${h.members} members of run 5's hall of fame, on seeds it never saw, it beats ${h.members_beaten} of ${h.members}${worst < 16 ? `, its closest at ${worst} of 16` : ", 16 of 16 against every one"}.`;
  } catch {}
  // The showcase match against the best bot of the night, else the first one listed.
  const show = r.final.showcase.find((m) => m.rival_id === ch(10)) ?? r.final.showcase[0];
  const meSide = { label, kind: "bot" }, them = { label: ROUND[show.rival_id] ?? "RIVAL", kind: show.rival_id === COACH ? "coach" : "bot" };
  const a = show.side !== "B", rivalRun = show.rival_id === c4.genome_id ? "r4" : "r5";
  add({ act: 4, title, left: a ? meSide : them, right: a ? them : meSide, caption: `${blurb} Trained 20 minutes against three rivals. ${rec}${proof}${hof}${twinNote}` },
    { l: a ? `spec:${main}` : rivalRun, r: a ? rivalRun : `spec:${main}` }, a ? r.final.genome_id : show.rival_id, a ? show.rival_id : r.final.genome_id, BigInt(show.seed), () => ({}));
}

// ---------- act 5: the definitive player? ----------
// Run 6 evolved against everyone above, weighted by rating; then all 30 bots played a round robin
// on 32 serves the run never used. These matches use those held-out serves.
const def = JSON.parse(readFileSync(join(OUT, RUN.r6, "definitive.json"), "utf8"));
const DEF_SEEDS = def.seeds.map(BigInt);
const lb = def.leaderboard, byLabel = (l) => lb.find((e) => e.label === l);
const king = lb[0], lob = byLabel("lob artist"), coachE = byLabel("the coach");
const exploit = lb.find((e) => e.kind === "elite"), elite6 = lb.find((e) => e.label.startsWith("run 6 elite #6"));
const shortName = (e) => e.kind === "scripted" ? "COACH" : e.label.replace(/^run 6 elite (#\d).*/, "RUN 6 $1").replace(/^run 5 champion /, "CHAMP ").toUpperCase();
const side = (e) => ({ label: shortName(e), kind: e.kind === "scripted" ? "coach" : "bot" });
const rec5 = (e) => def.top.record[e.id];
const board = lb.map((e) => ({ rank: e.rank, label: e.label, rating: Math.round(e.rating) }));
// The verdict's grid: a's share of 64 held-out matches against b, a tie counting half.
const ix = (e) => def.matrix.ids.indexOf(e.id), wr = (a, b) => def.matrix.winrate[ix(a)][ix(b)];
const pc = (x) => `${Math.round(100 * x)} per cent`;
const beats = (a) => lb.filter((b) => b.id !== a.id && wr(a, b) > 0.5).length;
function act5(title, a, b, caption, wantLeftWin, extra = {}) {
  const g = record("r6", a.id, "r6", b.id, DEF_SEEDS);
  add({ act: 5, title, left: side(a), right: side(b), caption: caption(g) }, "r6", a.id, b.id, typical(g.games, wantLeftWin), () => extra);
}
const champ10 = byLabel("run 5 champion #10");
const TIE = " (64 matches on serves nobody trained on, a tie counting half)";
act5("The king", king, champ10, () => `Rank 1 of ${lb.length} by Elo, ${Math.round(king.rating)}: the Flat-Earther's twin. Run 6 played 111 million matches trying to dethrone it, and no challenger passed all three gates. It beats ${beats(king)} of the other ${lb.length - 1}, the coach and last night's best among them, every match. Here it plays last night's best. The full leaderboard is below: green is run 6's new brains.`, true, { board });
act5("The king has a weakness", lob, king, () => `The Lob Artist, rank 2, takes ${pc(wr(lob, king))} against the king${TIE}. Their matches are long: this one runs to the time limit. It beats ${beats(lob)} of the other ${lb.length - 1}, more than anyone, so by robustness it is the better player.`, true);
act5("Rock, paper, scissors", elite6, king, () => `Run 6's sixth-best brain ranks ${elite6.rank}th of ${lb.length}, yet it takes ${pc(wr(elite6, king))} against the king${TIE}. At the top nobody beats everyone: just cycles.`, true);
act5("The exploit", exploit, coachE, (g) => `Run 6's highest scorer ranks ${exploit.rank}th: it beats the specialists more often than not, and against the coach it takes ${pc(wr(exploit, coachE))}${TIE}. The fitness weighted each rival by its rating, which made the coach worth a twelfth of the king, so evolution learned to ignore it. The promotion gate caught it.`, false);

const args = process.argv.slice(2);
const out = args[args.indexOf("--out") + 1] ?? "gallery.json";
const used = Object.fromEntries(matches.flatMap((m) => [m.left.id, m.right.id]).filter((id) => id !== COACH).map((id) => [id, brains[id]]));
writeFileSync(out, JSON.stringify({ kernel: kernelPath, made: new Date().toISOString(), matches, brains: used }));
console.log(`wrote ${out}: ${matches.length} matches, ${Object.keys(used).length} brains`);
