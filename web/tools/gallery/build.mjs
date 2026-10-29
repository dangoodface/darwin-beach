// Packs the gallery into one HTML file: the page, the bundled script, the kernel and the brains.
// Usage: node tools/gallery/build.mjs --data gallery.json --out replays.html [--standalone https://site]
// Without --standalone the file is an Artifact body (the host adds the document around it);
// with it, a whole document with the share-card tags, for hosting at that address.
import { readFileSync, writeFileSync, copyFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { buildSync } from "esbuild";

const HERE = dirname(new URL(import.meta.url).pathname);
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const data = JSON.parse(readFileSync(args.data ?? "gallery.json", "utf8"));
// gallery.json names the kernel file it was curated on; the page must run those exact bytes.
const wasm = readFileSync(data.kernel);

const app = buildSync({ entryPoints: [join(HERE, "../../src/gallery.ts")], bundle: true, format: "iife", target: "es2022", minify: true, write: false }).outputFiles[0].text;
// Nothing inside a script element may close it early.
const safe = (s) => s.replaceAll("</", "<\\/");
const page = readFileSync(join(HERE, "template.html"), "utf8")
  .replace("/*GALLERY*/", () => safe(JSON.stringify(data)))
  .replace("/*KERNEL*/", () => wasm.toString("base64"))
  .replace("/*APP*/", () => safe(app));
const [head, body] = page.split("<!--BODY-->\n");
const site = args.standalone;
const card = site && [
  `<meta property="og:type" content="website">`,
  `<meta property="og:url" content="${site}/">`,
  `<meta property="og:title" content="Darwin Beach: can you beat a bot that taught itself volleyball?">`,
  `<meta property="og:description" content="Slimes evolved over 800 million matches. Pick one, play it on your phone, get your Elo.">`,
  `<meta property="og:image" content="${site}/og.png">`,
  `<meta property="og:image:width" content="1200">`,
  `<meta property="og:image:height" content="630">`,
  `<meta name="twitter:card" content="summary_large_image">`,
  `<meta name="theme-color" content="#0b1026">`,
  `<link rel="icon" href="data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" fill="#0b1026"/><path d="M2 13a6 6 0 0 1 12 0z" fill="#39ff9f"/><circle cx="10" cy="9" r="1.6" fill="#fff"/><circle cx="12" cy="4" r="1.6" fill="#ffc93c"/></svg>')}">`,
].join("\n");
const html = site
  ? `<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n${head.replace(/<meta name="claude:[^>]*>\n/, "")}${card}\n</head>\n<body>\n${body}</body>\n</html>\n`
  : head + body;
writeFileSync(args.out ?? "replays.html", html);
// The hall of fame's server replays wins: it needs each rival's brain and rating, and the same kernel bytes.
if (site) {
  const rivals = Object.fromEntries(data.arcade.map((r) => [r.id, { label: r.label, rating: r.rating, kind: r.kind, ...(r.kind === "coach" ? {} : { b64: data.brains[r.id] }) }]));
  writeFileSync(join(dirname(args.out), "rivals.json"), JSON.stringify(rivals));
}
if (args["kernel-out"]) copyFileSync(data.kernel, args["kernel-out"]);
console.log(`wrote ${args.out ?? "replays.html"}: ${(html.length / 1024).toFixed(0)} KB, ${data.matches.length} matches`);
