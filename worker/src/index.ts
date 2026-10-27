import { assetPath, cacheControl, contentType, normaliseBase, route } from "./paths.ts";

// Just what we use of the Workers runtime, so the only dev dependency is wrangler.
interface R2Object {
  body: ReadableStream;
  httpEtag: string;
  size: number;
}
interface R2Head {
  httpEtag: string;
  size: number;
}
interface R2Bucket {
  get(key: string): Promise<R2Object | null>;
  head(key: string): Promise<R2Head | null>;
}
interface Env {
  DATA: R2Bucket;
  ASSETS: { fetch(request: Request): Promise<Response> };
  BASE_PATH?: string;
}

const text404 = () =>
  new Response("Not found", { status: 404, headers: { "content-type": "text/plain; charset=utf-8" } });

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const r = route(url.pathname, normaliseBase(env.BASE_PATH));

    if (r.kind === "redirect") return Response.redirect(new URL(r.to + url.search, url).toString(), 301);
    if (r.kind === "bad") return text404();

    if (r.kind === "data") {
      if (request.method !== "GET" && request.method !== "HEAD") {
        return new Response("Method not allowed", { status: 405, headers: { allow: "GET, HEAD" } });
      }
      const head = request.method === "HEAD";
      const obj = head ? await env.DATA.head(r.key) : await env.DATA.get(r.key);
      if (obj === null) return text404();
      const headers = new Headers({
        "content-type": contentType(r.key),
        "cache-control": cacheControl(r.key),
        etag: obj.httpEtag,
        "content-length": String(obj.size),
      });
      if (request.headers.get("if-none-match") === obj.httpEtag) {
        return new Response(null, { status: 304, headers });
      }
      return new Response(head ? null : (obj as R2Object).body, { headers });
    }

    // Everything else is the page. The assets sit at the root of web/dist, so drop the prefix.
    const assetUrl = new URL(request.url);
    assetUrl.pathname = assetPath(r.path);
    return env.ASSETS.fetch(new Request(assetUrl, request));
  },
};
