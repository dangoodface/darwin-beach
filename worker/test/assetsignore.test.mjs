// The deploy uploads web/dist as the site. The dev server links dist/data to the local runs,
// so dist/.assetsignore must keep data and the source maps out of the upload.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";

const web = new URL("../../web/", import.meta.url);
const WANTED = ["data", "data/*", "*.map"];

const lines = (text) => text.split("\n").map((l) => l.trim()).filter(Boolean);

test("the build script still writes dist/.assetsignore with the data rules", () => {
  const src = readFileSync(new URL("tools/copy-static.mjs", web), "utf8");
  assert.match(src, /\.assetsignore/, "web/tools/copy-static.mjs no longer writes dist/.assetsignore");
  for (const rule of WANTED) assert.ok(src.includes(rule), `the build script does not mention ${rule}`);
});

test("a built web/dist has .assetsignore with the data rules", (t) => {
  if (!existsSync(new URL("dist/", web))) {
    t.skip("web/dist is not built here; run: npm --prefix web ci && npm --prefix web run build");
    return;
  }
  const file = new URL("dist/.assetsignore", web);
  assert.ok(existsSync(file), "web/dist exists but .assetsignore is missing: a deploy would upload dist/data");
  const got = lines(readFileSync(file, "utf8"));
  for (const rule of WANTED) assert.ok(got.includes(rule), `.assetsignore lacks ${rule}`);
});
