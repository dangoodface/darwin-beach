// Pure functions: no Cloudflare types, so node --test runs them directly.

export const IMMUTABLE = "public, max-age=31536000, immutable";
export const SHORT = "public, max-age=60";

const TYPES: Record<string, string> = {
  ".json": "application/json",
  ".bin": "application/octet-stream",
  ".wasm": "application/wasm",
};

/** Normalise BASE_PATH to "/name" (no trailing slash), or "" for the root. */
export function normaliseBase(base: string | undefined): string {
  const b = (base ?? "").trim().replace(/\/+$/, "");
  return b === "" || b.startsWith("/") ? b : `/${b}`;
}

export type Route =
  | { kind: "redirect"; to: string } // /darwin-beach -> /darwin-beach/ so relative URLs resolve
  | { kind: "data"; key: string } // an R2 object key
  | { kind: "bad" } // a data path we refuse: traversal, listing, odd characters
  | { kind: "asset"; path: string }; // a path for the static assets, prefix removed

/**
 * Map a request path to what should answer it.
 * The prefix is optional: the play Worker may or may not have stripped it.
 */
export function route(pathname: string, base: string): Route {
  let p = pathname;
  if (base !== "") {
    if (p === base) return { kind: "redirect", to: `${base}/` };
    if (p.startsWith(`${base}/`)) p = p.slice(base.length);
  }
  if (p === "/data" || p === "/data/") return { kind: "bad" }; // no listing
  if (p.startsWith("/data/")) {
    const key = dataKey(p.slice("/data/".length));
    return key === null ? { kind: "bad" } : { kind: "data", key };
  }
  return { kind: "asset", path: p };
}

/** The R2 key for a data path, or null if it must be refused. */
export function dataKey(rest: string): string | null {
  let decoded: string;
  try {
    decoded = decodeURIComponent(rest);
  } catch {
    return null;
  }
  if (decoded === "" || decoded.endsWith("/")) return null; // a folder: no listing
  const parts = decoded.split("/");
  for (const part of parts) {
    if (part === "" || part === "." || part === "..") return null;
    if (!/^[A-Za-z0-9._-]+$/.test(part)) return null; // our keys are run ids, hex ids and plain names
  }
  return decoded;
}

/**
 * Same headers the uploader sets (arena/upload.py): genomes and kernels are named by
 * their hash, so they never change; the indexes change and may be 60 seconds old.
 */
export function cacheControl(key: string): string {
  return /(^|\/)(genomes|kernel)\//.test(key) ? IMMUTABLE : SHORT;
}

export function contentType(key: string): string {
  const dot = key.lastIndexOf(".");
  return (dot >= 0 && TYPES[key.slice(dot)]) || "application/octet-stream";
}

/** The path to ask the static assets for: "/" means the page. */
export function assetPath(path: string): string {
  return path === "" ? "/" : path;
}
