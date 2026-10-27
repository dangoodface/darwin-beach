// A tiny static server for working on the page: serves web/ (index.html,
// dist/, and data/, a symlink to $ARENA_DATA/out) at http://localhost:8787/.
// Rebuild with `npm run build` or keep `npm run watch` running alongside.
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";

const ROOT = resolve(new URL("..", import.meta.url).pathname);
const PORT = Number(process.env.PORT ?? 8787);
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".map": "application/json",
  ".json": "application/json", ".wasm": "application/wasm", ".bin": "application/octet-stream", ".css": "text/css",
};

createServer((req, res) => {
  const path = decodeURIComponent(req.url.split("?")[0]).replace(/\/{2,}/g, "/");
  const file = normalize(join(ROOT, path.endsWith("/") ? `${path}index.html` : path));
  if (!file.startsWith(ROOT)) return res.writeHead(403).end();
  try {
    if (!statSync(file).isFile()) throw new Error();
  } catch {
    return res.writeHead(404).end("not found");
  }
  res.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream", "cache-control": "no-cache" });
  createReadStream(file).pipe(res);
}).listen(PORT, () => console.log(`Darwin Beach page on http://localhost:${PORT}/`));
