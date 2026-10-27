// Copies the static files next to the bundle, so dist/ is the whole site: the
// Worker serves it as is in production, and the dev server serves it locally.
import { copyFileSync, mkdirSync, writeFileSync } from "node:fs";

const root = new URL("..", import.meta.url).pathname;
mkdirSync(`${root}dist`, { recursive: true });
copyFileSync(`${root}index.html`, `${root}dist/index.html`);
// The dev server links dist/data to the local runs; in production data/ comes from R2
// through the Worker, so a deploy from this machine must never upload that link.
writeFileSync(`${root}dist/.assetsignore`, "data\ndata/*\n*.map\n");
