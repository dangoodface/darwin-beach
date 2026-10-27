// A tiny static server for working on the page: serves web/dist/, the same
// folder the Worker serves in production, at http://localhost:8787/. It links
// dist/data to $ARENA_DATA/out so the page's data/ fetches work as they do
// behind the Worker. Run `npm run build` first, or keep `npm run watch` going.
import { createServer } from "node:http";
import { createReadStream, statSync, existsSync, symlinkSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { extname, join, normalize, resolve } from "node:path";

const ROOT = resolve(new URL("../dist", import.meta.url).pathname);
const DATA = process.env.ARENA_DATA ? join(process.env.ARENA_DATA, "out") : join(homedir(), ".local/share/arena/out");
mkdirSync(ROOT, { recursive: true });
if (!existsSync(join(ROOT, "data"))) symlinkSync(DATA, join(ROOT, "data"));
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
