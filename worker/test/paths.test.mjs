import test from "node:test";
import assert from "node:assert/strict";
import {
  assetPath, cacheControl, contentType, dataKey, IMMUTABLE, normaliseBase, route, SHORT,
} from "../src/paths.ts";

const base = "/darwin-beach";

test("normaliseBase", () => {
  assert.equal(normaliseBase("/darwin-beach"), "/darwin-beach");
  assert.equal(normaliseBase("/darwin-beach/"), "/darwin-beach");
  assert.equal(normaliseBase("darwin-beach"), "/darwin-beach");
  assert.equal(normaliseBase(""), "");
  assert.equal(normaliseBase(undefined), "");
  assert.equal(normaliseBase("/"), "");
});

test("data paths map to R2 keys with or without the prefix", () => {
  const key = "20261027T051940Z-slime-v0/latest.json";
  assert.deepEqual(route(`/darwin-beach/data/${key}`, base), { kind: "data", key });
  assert.deepEqual(route(`/data/${key}`, base), { kind: "data", key }); // prefix already stripped
  assert.deepEqual(route("/data/index.json", base), { kind: "data", key: "index.json" });
  assert.deepEqual(route(`/data/${key}`, ""), { kind: "data", key }); // mounted at the root
});

test("a lookalike prefix is not stripped", () => {
  assert.deepEqual(route("/darwin-beachX/data/a.json", base), { kind: "asset", path: "/darwin-beachX/data/a.json" });
});

test("the bare base redirects so relative URLs resolve", () => {
  assert.deepEqual(route("/darwin-beach", base), { kind: "redirect", to: "/darwin-beach/" });
  assert.deepEqual(route("/darwin-beach/", base), { kind: "asset", path: "/" });
});

test("everything else is a static asset with the prefix removed", () => {
  assert.deepEqual(route("/darwin-beach/app.js", base), { kind: "asset", path: "/app.js" });
  assert.deepEqual(route("/app.js", base), { kind: "asset", path: "/app.js" });
  assert.equal(assetPath(""), "/");
});

test("no listing, no traversal, no odd keys", () => {
  for (const p of ["/data", "/data/", "/darwin-beach/data/", "/data/run/", "/data/run//x.json",
    "/data/../secret", "/data/a/../../b", "/data/%2e%2e/x", "/data/a%2F..%2Fb", "/data/a b.json",
    "/data/%E0%A4%A", "/data/a?b"]) {
    assert.deepEqual(route(p, base), { kind: "bad" }, p);
  }
  assert.equal(dataKey("run/genomes/ab12.bin"), "run/genomes/ab12.bin");
});

test("cache headers match the uploader", () => {
  assert.equal(cacheControl("run/genomes/ab12.bin"), IMMUTABLE);
  assert.equal(cacheControl("run/kernel/ab12.wasm"), IMMUTABLE);
  for (const k of ["run/latest.json", "run/curve.json", "run/tree.json", "run/tree-full.json", "index.json"]) {
    assert.equal(cacheControl(k), SHORT, k);
  }
  assert.equal(IMMUTABLE, "public, max-age=31536000, immutable");
  assert.equal(SHORT, "public, max-age=60");
});

test("content types", () => {
  assert.equal(contentType("a/latest.json"), "application/json");
  assert.equal(contentType("a/genomes/x.bin"), "application/octet-stream");
  assert.equal(contentType("a/kernel/x.wasm"), "application/wasm");
  assert.equal(contentType("noext"), "application/octet-stream");
});
