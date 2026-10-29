# Deploying the page

The public page lives at https://play.thegoodface.ai/beach/, as static files inside the
`play-thegoodface` Worker (in the separate `websites` workspace, `play-thegoodface/public/beach/`).
It is the replay gallery built as a whole document (`build.mjs --standalone`); the Artifact build
and this one come from the same template and script.

From `web/`:

    node tools/gallery/curate.mjs --out /tmp/gallery.json      # plays every match on the kernel
    B=../../websites/play-thegoodface/public/beach             # relative to this worktree; adjust
    mkdir -p $B && cp deploy/og.png $B/
    node tools/gallery/build.mjs --data /tmp/gallery.json --out $B/index.html --standalone https://play.thegoodface.ai/beach
    # then deploy play-thegoodface the way its own notes say (wrangler deploy from that folder)

`curate.mjs` reads the run data in `~/.local/share/arena`, so it runs on the training machine.
`og.png` is the 1200 x 630 share card, a screenshot of the page with the boss previewing.
