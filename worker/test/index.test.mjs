// The Worker end to end with stand-ins for R2 and the assets: no account needed.
import test from "node:test";
import assert from "node:assert/strict";
import worker from "../src/index.ts";

function env(objects = {}, base = "/darwin-beach") {
  const asked = [];
  return {
    asked,
    BASE_PATH: base,
    DATA: {
      async get(key) {
        const o = objects[key];
        return o ? { body: new Blob([o]).stream(), httpEtag: `"${key.length}"`, size: o.length } : null;
      },
      async head(key) {
        const o = objects[key];
        return o ? { httpEtag: `"${key.length}"`, size: o.length } : null;
      },
    },
    ASSETS: { async fetch(req) { asked.push(new URL(req.url).pathname); return new Response("<html>page</html>"); } },
  };
}
const call = (path, e, init) => worker.fetch(new Request(`https://play.example${path}`, init), e);

test("serves an index with a 60 second cache and an etag", async () => {
  const e = env({ "run/latest.json": '{"generation":1}' });
  const r = await call("/darwin-beach/data/run/latest.json", e);
  assert.equal(r.status, 200);
  assert.equal(r.headers.get("cache-control"), "public, max-age=60");
  assert.equal(r.headers.get("content-type"), "application/json");
  assert.equal(await r.text(), '{"generation":1}');
  assert.deepEqual(e.asked, []);
});

test("genomes are immutable", async () => {
  const r = await call("/data/run/genomes/ab.bin", env({ "run/genomes/ab.bin": "xxxx" }));
  assert.equal(r.headers.get("cache-control"), "public, max-age=31536000, immutable");
  assert.equal(r.headers.get("content-length"), "4");
});

test("HEAD has the headers and no body", async () => {
  const r = await call("/data/index.json", env({ "index.json": "[]" }), { method: "HEAD" });
  assert.equal(r.status, 200);
  assert.equal(r.headers.get("content-length"), "2");
  assert.equal(await r.text(), "");
});

test("a matching etag gets a 304", async () => {
  const e = env({ "index.json": "[]" });
  const first = await call("/data/index.json", e);
  const r = await call("/data/index.json", e, { headers: { "if-none-match": first.headers.get("etag") } });
  assert.equal(r.status, 304);
});

test("missing, listing and traversal are plain-text 404s", async () => {
  const e = env({ "index.json": "[]" });
  for (const p of ["/data/nope.json", "/data/", "/darwin-beach/data", "/data/a%2F..%2Fb", "/data/a//b"]) {
    const r = await call(p, e);
    assert.equal(r.status, 404, p);
    assert.match(r.headers.get("content-type"), /^text\/plain/);
    assert.equal(await r.text(), "Not found");
  }
});

test("only GET and HEAD on data", async () => {
  const r = await call("/data/index.json", env({ "index.json": "[]" }), { method: "POST", body: "x" });
  assert.equal(r.status, 405);
  assert.equal(r.headers.get("allow"), "GET, HEAD");
});

test("the page comes from the assets with the prefix removed", async () => {
  const e = env();
  assert.equal(await (await call("/darwin-beach/", e)).text(), "<html>page</html>");
  await call("/darwin-beach/app.js", e);
  await call("/app.js", e);
  assert.deepEqual(e.asked, ["/", "/app.js", "/app.js"]);
});

test("the bare base redirects to the slash", async () => {
  const r = await call("/darwin-beach?x=1", env());
  assert.equal(r.status, 301);
  assert.equal(new URL(r.headers.get("location")).pathname, "/darwin-beach/");
  assert.equal(new URL(r.headers.get("location")).search, "?x=1");
});

test("an empty BASE_PATH serves from the root", async () => {
  const e = env({ "index.json": "[]" }, "");
  assert.equal((await call("/data/index.json", e)).status, 200);
  await call("/", e);
  assert.deepEqual(e.asked, ["/"]);
});
