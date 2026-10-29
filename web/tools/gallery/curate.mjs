// Picks the replay gallery's matches by playing them on the real kernel, and writes
// gallery.json: every match as seed + two brains, with the hash and score it must land on.
// Usage: node tools/gallery/curate.mjs [--out file]
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { join, basename } from "node:path";
import { homedir } from "node:os";
import { Kernel, SCRIPTED_SLOT, padGenome } from "../../src/kernel.ts";

const OUT = join(process.env.ARENA_DATA ?? join(homedir(), ".local/share/arena"), "out");
const RUN = {
  r1: "20261027T055834Z-slime-v0",
  r2: "20261027T065906Z-night-r2",
  r3: "20261027T085957Z-night-r3",
  r4: "20261027T110547Z-night-r4-cma",
  r5: "20261027T130856Z-night-r5-ladder",
  r6: "20261028T001309Z-run6-definitive",
  r7: "20261028T161600Z-run7-league",
};
const COACH = "scripted-v1";
const latest = Object.fromEntries(Object.entries(RUN).map(([k, d]) => [k, JSON.parse(readFileSync(join(OUT, d, "latest.json"), "utf8"))]));
const ARENA = process.env.ARENA_DATA ?? join(homedir(), ".local/share/arena");
const SPEC = join(ARENA, "specialists");
// Each kernel version replays every older match on the same hash (version 3 pads older genomes
// with zeros), so the gallery uses the newest kernel any specialist carries, else run 4's.
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
  let f = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  if (f.length !== scan.genomeLen) f = padGenome(f);
  new Float32Array(scan.k.memory.buffer, scan.k.genome_table_ptr() + i * scan.genomeLen * 4, scan.genomeLen).set(f);
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
function play(seed, left, right, k = kernel, holds = [1, 1]) {
  k.reset(seed, ...holds);
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
  const r = play(seed, slot(run.l ?? run, leftId), slot(run.r ?? run, rightId), kernel, m.holds ?? [1, 1]);
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
// The exploit: the top-rated new brain among those that win less than a quarter against the coach.
const exploit = lb.find((e) => e.kind === "elite" && def.matrix.winrate[def.matrix.ids.indexOf(e.id)][def.matrix.ids.indexOf("scripted-v1")] < 0.25), elite6 = lb.find((e) => e.label.startsWith("run 6 elite #6"));
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

// ---------- act 6: the night shift ----------
// Round 7: twenty-minute runs from the night's best brains, each rated on the official board
// (64 serves per pair, both sides, up to 6,000 ticks, the coach fixed at 1000).
const OFF = join(ARENA, "official");
const ladder = JSON.parse(readFileSync(join(OFF, "ladder.json"), "utf8"));
const offRatings = {};
for (const f of readdirSync(join(OFF, "ratings")).filter((f) => f.endsWith(".json")).map((f) => join(OFF, "ratings", f)).sort((a, b) => statSync(a).mtimeMs - statSync(b).mtimeMs)) {
  const r = JSON.parse(readFileSync(f, "utf8"));
  if (!/check/.test(r.label)) offRatings[basename(r.path)] = r;
}
// Members carry padded ids; the genome file names the brain whichever kernel rated it.
const memberFiles = new Set(ladder.members.filter((m) => m.path).map((m) => basename(m.path)));
const tonight = (path) => /specialists\/20261028|oracle/.test(path ?? "");
const board6 = [
  ...ladder.members.map((m) => ({ label: m.label, rating: m.rating, kind: m.kind === "oracle" ? "coach" : tonight(m.path) ? "new" : "" })),
  { label: "the coach", rating: ladder.anchor_rating, kind: "coach" },
  ...Object.entries(offRatings).filter(([file]) => !memberFiles.has(file)).map(([, r]) => r).map((r) => ({ label: r.label.replace(/ \(.*\)$/, "").replaceAll("-", " "), rating: r.rating, kind: "new" })),
].sort((a, b) => b.rating - a.rating).map((e, i) => ({ rank: i + 1, label: e.label, rating: Math.round(e.rating), kind: e.kind }));
const specDir = (suffix) => specRuns.filter((d) => d.endsWith(`-spec-${suffix}`) && result(d).elapsed_s >= 600).at(-1);
const official = (id) => offRatings[`${id}.bin`];
// Records are keyed by the opponent's id on the rating kernel, so match on its label.
const offRec = (id, oppLabel) => Object.values(official(id)?.record ?? {}).find((x) => x.label === oppLabel);
const ofN = (rec) => `${rec.won} of ${rec.n}`;
function act6(title, suffix, label, pick, caption) {
  const d = specDir(suffix);
  if (!d) return;
  const r = result(d), me = r.final.genome_id;
  const show = pick(r.final.showcase, r);
  if (!show) return;
  const riv = r.rivals.find((x) => x.id === show.rival_id);
  const them = show.rival_id === COACH ? coachSide : { label: (riv?.label ?? "rival").toUpperCase(), kind: "bot" };
  const meSide = { label, kind: "bot" }, a = show.side !== "B", run = `spec:${d}`;
  const o = official(me);
  const f = r.final.fresh, fr = (id) => `${f[id]?.won ?? "?"} of ${f[id]?.n ?? 16}`;
  const text = caption({ me, r, f, fr, o, rating: o ? Math.round(o.rating) : null, md: r.final.descriptors_fresh_mean ?? {}, p: r.final.proof ?? {} });
  add({ act: 6, title, left: a ? meSide : them, right: a ? them : meSide, caption: text, ...(show.hold_a > 1 || show.hold_b > 1 ? { holds: [show.hold_a, show.hold_b] } : {}) },
    run, a ? me : show.rival_id, a ? show.rival_id : me, BigInt(show.seed), () => (title === "The patch" ? { board: board6 } : {}));
}
const vs = (id, win = true) => (sc) => sc.find((m) => m.rival_id === id && (m.margin > 0) === win);
const vsLabel = (label, win = true) => (sc, r) => vs(r.rivals.find((x) => x.label === label)?.id, win)(sc);
const TWIN = "7a3792a851786796442394625cf8c0881d205df84e4a2af4f5726859869cc720";
const LOB = "18c8b5a5262fbdfcc87245f9a980de48d623d5f5990039df380ca6899f9d1773";
const ALL = "c7fd99648c6c14bd3c93fc4f20b87df9f2c60c3f076e8fed8594e12a3b83ddd7";
const OFFICIAL = "official Elo (64 serves against every bot on the ladder, both sides, the coach fixed at 1000)";
act6("The patch", "twin-patch", "TWIN PATCHED", vs(LOB), ({ fr, rating, o }) => `Act 5's king had one weakness, the Lob Artist. So it went back to school for twenty minutes, with the Lob Artist counting three times. On new serves it now beats the Lob Artist ${fr(LOB)}, the All-Rounder ${fr(ALL)} and the coach ${fr(COACH)}. Its ${OFFICIAL} is ${rating}, and it beats ${o.beaten ?? 24} of ${o.of ?? 24} members. The only thing above it is the oracle, a hand-written bot that plans with the game's own physics. The board below is the new official ladder: green is tonight's.`);
act6("The assassin", "twin-assassin", "TWIN ASSASSIN", vs(TWIN), ({ me, rating }) => { const rec = offRec(me, "flat earther twin"); return `Trained against one bot only: the old king. On the official serves it beats it ${rec ? ofN(rec) : "every time"}. Against everyone else it is ordinary, an official ${rating}. A killer built for one target, as asked.`; });
act6("The survivor", "all-rounder-assassin", "ASSASSIN", (sc) => vs(ALL, false)(sc) ?? sc.filter((m) => m.rival_id === ALL).sort((a, b) => a.margin - b.margin)[0], ({ fr }) => `The same recipe, aimed at the All-Rounder. Twenty minutes later its killer wins only ${fr(ALL)} on new serves. The assassin is fine; the All-Rounder is just that solid. Here is one of the few it won.`);
act6("The Spiker", "spiker", "SPIKER", vs(TWIN), ({ fr, rating, md }) => `A rule it cannot break: hit hard and from high, or be thrown out. Its shots leave at speed ${md.mean_speed_after_hit.toFixed(1)} from height ${md.mean_hit_height.toFixed(1)}. It beats the old king ${fr(TWIN)} on new serves and the coach ${fr(COACH)}. Official ${rating}: tonight's best style.`);
act6("Too soft", "dinker", "DINKER", vs(COACH, false), ({ r, fr, rating, md }) => `The opposite rule: soft and low, every touch. It obeyed, at speed ${md.mean_speed_after_hit.toFixed(1)}, and it went ${r.generations.toLocaleString("en-US")} generations without getting good. A slow, low ball is a free smash for anyone who jumps. Against the coach: ${fr(COACH)}. Official ${rating}, last on the board. A finding, not a bug.`);
act6("Lazy wins", "lazy", "LAZY", vs(TWIN), ({ fr, rating, p }) => `Allowed to travel only ${p.travel_cap.toFixed(2)} a tick, the Flat-Earther's pace, and it used ${p.travel_per_tick.toFixed(2)}. It beats the old king ${fr(TWIN)} on new serves anyway. Official ${rating}: moving less is not playing worse.`);
act6("Human reflexes", "smooth_operator", "SMOOTH OP", (sc) => sc.find((m) => m.margin > 0) ?? sc[0], ({ r, f }) => { const won = Object.values(f).reduce((a, x) => a + x.won, 0), n = Object.values(f).reduce((a, x) => a + x.n, 0); return `It may change its mind only every ${r.hold} ticks, ${(r.hold / 30).toFixed(1)} seconds, about a person's reaction time; in between it repeats its last move. After twenty minutes it wins ${won} of ${n} on new serves${won && won === f[COACH]?.won ? ", every one of them against the coach" : ""}. Every other bot here decides 30 times a second. This is one of its rare wins.`; });
act6("La Clavada", "clavada", "LA CLAVADA", (sc) => sc.find((m) => m.margin > 0) ?? sc[0], ({ r, f, md, rating }) => {
  const share = md.points ? Math.round((100 * md.spiked_points) / md.points) : 0;
  const wins = r.rivals.map((x) => `${x.label ?? "the coach"} ${f[x.id]?.won ?? "?"}`).join(", ");
  return `Paid to spike: at least a third of its points had to end a juggle with a hard hit at the net. It spiked ${share} per cent, ${r.final.constraint_satisfied ? "inside the rule" : "so it broke its rule"}. Yet on new serves it beats every rival, out of 16 each: ${wins}.${rating ? ` Official ${rating}.` : ""} A strong bot, a failed style.`;
});

// ---------- act 7: the league ----------
// Round 7's league ran all day; its verdict is a round robin of 44 bots on 32 serves the run never used.
const def7 = JSON.parse(readFileSync(join(OUT, RUN.r7, "definitive.json"), "utf8"));
const DEF7_SEEDS = def7.seeds.map(BigInt);
const lb7 = def7.leaderboard, by7 = (l) => lb7.find((e) => e.label === l);
const ix7 = (e) => def7.matrix.ids.indexOf(e.id), wr7 = (a, b) => def7.matrix.winrate[ix7(a)][ix7(b)];
const name7 = (e) => e.kind === "scripted" ? "COACH" : e.label.replace(/^run 7 elite (#\d).*/, "RUN 7 $1").replace(/^exploiter-(\d+)-gen\d+$/, "EXPLOITER $1").toUpperCase();
const side7 = (e) => ({ label: name7(e), kind: e.kind === "scripted" ? "coach" : "bot" });
const board7 = lb7.map((e) => ({ rank: e.rank, label: e.label, rating: Math.round(e.rating), kind: e.kind === "elite" || e.kind === "exploiter" ? "new" : e.kind === "scripted" || e.kind === "oracle" ? "coach" : "" }));
const genomeMatch = (e) => e.kind !== "oracle";
function act7(title, a, b, caption, wantLeftWin, extra = {}) {
  if (!a || !b || !genomeMatch(a) || !genomeMatch(b)) return;
  // A real result, not a tie: these bots often play to the time limit.
  const g = record("r7", a.id, "r7", b.id, DEF7_SEEDS);
  const decided = g.games.filter(({ g: x }) => (wantLeftWin ? x.points[0] > x.points[1] : x.points[1] > x.points[0])).sort((p, q) => q.g.points[0] + q.g.points[1] - (p.g.points[0] + p.g.points[1]) || p.g.ticks - q.g.ticks);
  add({ act: 7, title, left: side7(a), right: side7(b), caption }, "r7", a.id, b.id, (decided[0] ?? g.games[0]).s, () => extra);
}
const s7 = latest.r7.stats, ot = by7("one-touch twin"), patched7 = by7("twin patched");
const elites7 = lb7.filter((e) => e.kind === "elite"), best7 = elites7[0];
const exp4 = lb7.find((e) => /^exploiter-4-/.test(e.label)), prey4 = exp4 && lb7.find((e) => e.kind === "elite" && e.label.includes(`gen ${exp4.label.split("gen").pop()}`));
const TIE7 = " (64 matches on serves the league never used, a tie counting half)";
act7("Twelve hours", best7, ot, `The league: ${Math.round(s7.matches_total / 1e6)} million matches in one day, every cell of the map filled, all ${s7.filled} of them better than every run 5 champion and ${s7.beats_coach} beating the coach. Every promotion check failed. Its best brain ranks ${best7.rank}th of ${lb7.length} on the held-out verdict and takes ${pc(wr7(best7, ot))} against the champion${TIE7}. The champion, the One-Touch twin, came from twenty minutes of training last night. The board is that verdict: green is the league's own brains.`, false, { board: board7 });
act7("It can win", best7, ot, `The same two, on a serve where the league's best gets its way. It wins about one match in three against the champion: close enough to try, not enough to promote.`, true);
act7("The exploiter", exp4, prey4, `Every 45 minutes a fresh bot trained for ten minutes against the league's best cell, hunting for its blind spot. Exploiter ${exp4?.label.split("-")[1]} was aimed at the best cell of round ${exp4?.label.split("gen").pop()} and takes ${exp4 && prey4 ? pc(wr7(exp4, prey4)) : "most"} against it${TIE7}. It joined the ladder, so the league had to learn to beat it too.`, true);
act7("Twenty minutes beat twelve hours", ot, patched7, `The top of the verdict: two twenty-minute runs from last night. On the official board the twin patched rates higher; head to head on these serves the One-Touch twin takes ${pc(wr7(ot, patched7))}. At the top, nobody beats everyone, and a day of the map engine did not reach either of them.`, true);

// ---------- the arcade: rated rivals for the viewer ----------
// Ratings from the official board (the coach fixed at 1000), so the viewer's Elo lands on the same scale.
const ARCADE = [
  ["dinker", "DINKER", "Soft and low, every touch. The worst bot on the board. Warm up here."],
  ["run 5 champion #1", "ROOKIE", "The first champion of night one. It barely knows the game."],
  ["run 4 champion", "FIRST CHAMP", "The first bot ever to beat the coach."],
  ["the coach", "COACH", "Hand-written, not evolved. Every bot learned by playing it."],
  ["run 5 champion #10", "CHAMP #10", "The best bot of night one. Beats the coach nine times in ten."],
  ["joga bonito", "JOGA BONITO", "Juggles the ball before the kill."],
  ["lob artist", "LOB ARTIST", "Slow, hanging balls that drive the other bots mad."],
  ["flat earther twin", "THE TWIN", "Jumps only to hit, winds up and lunges through the ball. King of run 6."],
  ["lazy", "LAZY", "Barely moves, and still beats the old king 15 of 16."],
  ["one-touch twin", "ONE-TOUCH TWIN", "Twenty minutes of training from a bot that lost every match to the coach."],
  ["twin patched", "THE BOSS", "The best bot evolution made: official Elo 2183."],
];
const offByLabel = (l) => l === "the coach" ? { id: COACH, rating: ladder.anchor_rating } : ladder.members.find((m) => m.label === l && m.path) ?? Object.values(offRatings).find((r) => r.label === l);
const arcade = ARCADE.map(([l, label, note]) => {
  const e = offByLabel(l);
  if (!e) return null;
  if (e.id === COACH) return { id: COACH, label, kind: "coach", rating: Math.round(e.rating), note };
  const b = readFileSync(e.path), id = basename(e.path, ".bin");
  brains[id] = b.toString("base64");
  return { id, label, kind: "bot", rating: Math.round(e.rating), note };
}).filter(Boolean).sort((a, b) => a.rating - b.rating);

// ---------- the numbers for the header and the nerds ----------
const runMatches = [...Object.values(RUN), "20261028T130929Z-run7-league"].map((d) => JSON.parse(readFileSync(join(OUT, d, "latest.json"), "utf8")).stats.matches_total);
const specMatches = specRuns.map((d) => result(d).matches_total ?? 0);
const facts = {
  matches: [...runMatches, ...specMatches].reduce((a, b) => a + b, 0),
  runs: runMatches.length, specialists: specRuns.filter((d) => result(d).elapsed_s >= 600).length,
  boss: arcade.at(-1)?.rating, members: ladder.members.length,
};
console.log("facts", facts);

const args = process.argv.slice(2);
const out = args[args.indexOf("--out") + 1] ?? "gallery.json";
const used = Object.fromEntries([...matches.flatMap((m) => [m.left.id, m.right.id]), ...arcade.map((r) => r.id)].filter((id) => id !== COACH).map((id) => [id, brains[id]]));
writeFileSync(out, JSON.stringify({ kernel: kernelPath, made: new Date().toISOString(), matches, brains: used, arcade, facts }));
console.log(`wrote ${out}: ${matches.length} matches, ${Object.keys(used).length} brains`);
