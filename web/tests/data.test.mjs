import { test } from "node:test";
import assert from "node:assert/strict";
import { parseSeed, parseHash, isLit, newestRun } from "../src/data.ts";

test("seeds parse as u64 from numbers and decimal strings", () => {
  assert.equal(parseSeed(123), 123n);
  assert.equal(parseSeed("11528747065380910690"), 11528747065380910690n);
  assert.equal(parseSeed("18446744073709551615"), 2n ** 64n - 1n);
  assert.throws(() => parseSeed("18446744073709551616"));
  assert.throws(() => parseSeed(-1));
  assert.throws(() => parseSeed(1.5));
});

test("hashes parse from 0x plus 16 hex digits", () => {
  assert.equal(parseHash("0xfaaaa7600154894c"), 0xfaaaa7600154894cn);
  assert.throws(() => parseHash("faaaa7600154894c"));
  assert.throws(() => parseHash("0x123"));
});

test("a cell is lit by its flag, or by fitness at the floor on older runs", () => {
  assert.equal(isLit({ fitness: -1, lit: true }), true);
  assert.equal(isLit({ fitness: 2, lit: false }), false);
  assert.equal(isLit({ fitness: 0 }), true);
  assert.equal(isLit({ fitness: -0.01 }), false);
});

test("the newest run wins", () => {
  assert.equal(newestRun([{ run_id: "20261027T055834Z-a" }, { run_id: "20261028T010000Z-b" }]), "20261028T010000Z-b");
  assert.throws(() => newestRun([]));
});

test("the main line follows the stronger parent back to generation zero", async () => {
  const { mainLine } = await import("../src/charts.ts");
  const tree = {
    nodes: [
      { id: "r1", parents: [], gen: 0, fitness: -1, champion: false, x: 0, y: 0 },
      { id: "r2", parents: [], gen: 0, fitness: -0.5, champion: false, x: 0, y: 1 },
      { id: "c", parents: ["r1", "r2"], gen: 1, fitness: 0.5, champion: false, x: 0.5, y: 0.5 },
      { id: "k", parents: ["c"], gen: 2, fitness: 1, champion: true, x: 1, y: 0.5 },
    ],
    edges: [], champion_lineage: ["r1", "r2", "c", "k"],
  };
  assert.deepEqual(mainLine(tree).map((n) => n.id), ["r2", "c", "k"]);
});

test("the coach line reads the coach's margin and stays quiet without one", async () => {
  const { coachLine } = await import("../src/describe.ts");
  const probes = ["scripted-v1", "a", "b", "c"];
  assert.equal(coachLine({ margins: [0.75, 1, 1, 1] }, probes, "scripted-v1"), "Against the coach: ahead, +0.75 points a match on the serves it was graded on.");
  assert.equal(coachLine({ margins: [-4.5, 1, 1, 1] }, probes, "scripted-v1"), "Against the coach: behind, -4.50 points a match on the serves it was graded on.");
  assert.equal(coachLine({}, probes, "scripted-v1"), null);
  assert.equal(coachLine({ margins: [1] }, undefined, "scripted-v1"), null);
});
