# The Darwin Beach Worker

A small Cloudflare Worker with one job: answer every request under
`play.thegoodface.ai/darwin-beach/`. It is a scaffold. Nothing here has been
deployed, and deploying is Daniel's call.

## What it does

- `data/...` is read from the R2 bucket `arena-data`: the files the trainer
  writes (the map, the curve, the family tree, the bots, the game file). The
  Worker sends the same cache rules the uploader uses: files named by their
  content never change and are cached for a year; the three indexes may be
  60 seconds old.
- It never lists the bucket. A folder, a missing file, a path with `..` or an
  odd character gets a plain-text "Not found". Only GET and HEAD are allowed.
- Everything else is the page (`web/dist`), served as static assets.
- The path prefix is one setting, `BASE_PATH` in `wrangler.jsonc`
  (`/darwin-beach`). If a request arrives with the prefix it is removed; if the
  play Worker already removed it, nothing changes. Set it to an empty string to
  serve from the root of its own address instead. `/darwin-beach` without the
  slash redirects to `/darwin-beach/`, so the page's relative links work.

The page must fetch with relative addresses (`data/index.json`, never
`/data/index.json`), because it does not live at the root.

## Tests

No account needed:

```
make worker-test        # from the repo root
```

They cover the path rules, the cache headers and the whole handler with stand-ins
for R2 and the static files.

## Deploying later (Daniel's gate)

1. Create the bucket once: `npx wrangler r2 bucket create arena-data`
   (from `worker/`, after `npm install`). It stays private; only this Worker
   reads it. The uploader writes to it with `tools/upload.py --r2` (see
   `systemd/README.md` for the credentials).
2. Build the page so `web/dist` exists.
3. From `worker/`: `npx wrangler deploy`. The Worker has no public address of
   its own (`workers_dev` is off); it is reached only through the play Worker.

## The one-time wiring on the play side

The play Worker forwards the path to this one through a service binding:

1. In the play Worker's config, add a service binding, for example
   `{ "binding": "DARWIN_BEACH", "service": "darwin-beach" }`.
2. In its code, before its own routes: if the path is `/darwin-beach` or starts
   with `/darwin-beach/`, `return env.DARWIN_BEACH.fetch(request)`.

Pass the request through as it came. This Worker copes with the prefix being
there or not, so nothing else needs agreeing.

## What is in the folder

| File | What |
|---|---|
| `wrangler.jsonc` | name `darwin-beach`; static files from `../web/dist`; one R2 binding, `DATA` |
| `src/paths.ts` | the pure functions: path to bucket key, cache header, content type |
| `src/index.ts` | the handler |
| `test/` | `node --test` |

`wrangler` is the only dependency, pinned in `package.json` and `package-lock.json`.
The static-files setting also creates an `ASSETS` handle for the Worker; that is
part of serving the page, not another data store.
