// Packs the gallery into one HTML file: the page, the bundled script, the kernel and the brains.
// Usage: node tools/gallery/build.mjs --data gallery.json --out replays.html
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { homedir } from "node:os";
import { buildSync } from "esbuild";

const HERE = dirname(new URL(import.meta.url).pathname);
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const data = JSON.parse(readFileSync(args.data ?? "gallery.json", "utf8"));
const OUT = join(process.env.ARENA_DATA ?? join(homedir(), ".local/share/arena"), "out");
const run = data.matches.length && "20261027T110547Z-night-r4-cma";
const wasm = readFileSync(join(OUT, args.run ?? run, "kernel", `${data.kernel}.wasm`));

const app = buildSync({ entryPoints: [join(HERE, "../../src/gallery.ts")], bundle: true, format: "iife", target: "es2022", minify: true, write: false }).outputFiles[0].text;
// Nothing inside a script element may close it early.
const safe = (s) => s.replaceAll("</", "<\\/");
const html = readFileSync(join(HERE, "template.html"), "utf8")
  .replace("/*GALLERY*/", () => safe(JSON.stringify(data)))
  .replace("/*KERNEL*/", () => wasm.toString("base64"))
  .replace("/*APP*/", () => safe(app));
writeFileSync(args.out ?? "replays.html", html);
console.log(`wrote ${args.out ?? "replays.html"}: ${(html.length / 1024).toFixed(0)} KB, ${data.matches.length} matches`);
