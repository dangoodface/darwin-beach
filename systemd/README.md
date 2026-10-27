# Systemd units (user level)

Written, not installed. The lead decides when. They run the checkout's `.venv/bin/python` directly (uv lives under mise, whose path changes with its version).

| Unit | What |
|---|---|
| `arena-train.service` | the loop (`python -m arena.run`) under `MemoryMax=4G`, `CPUQuota=1400%` |
| `arena-upload.service` + `.timer` | `tools/upload.py` every ten minutes; local dry-run mirror until R2 exists |

Install (as Daniel, no sudo). Set `ARENA_REPO` once, to the checkout that should run
(the worktree now, `~/projects/arena` after the merge and kernel swap):

```
ARENA_REPO=$HOME/projects/arena-trainer        # the one variable to change
mkdir -p ~/.config/systemd/user
for u in arena-train.service arena-upload.service arena-upload.timer; do
  sed "s#%h/projects/arena-trainer#${ARENA_REPO}#g" systemd/$u > ~/.config/systemd/user/$u
done
systemctl --user daemon-reload
systemctl --user start arena-train.service
systemctl --user enable --now arena-upload.timer
```

The sed rewrites the repo path in `WorkingDirectory` and `ExecStart`, so the units in git
never need editing. The checkout needs its own `.venv` (`uv sync --dev` there).

Watch: `journalctl --user -u arena-train -f`. Stop: `systemctl --user stop arena-train`.

## R2 credentials (after Daniel creates the bucket)

The uploader reads four environment variables: `ARENA_R2_BUCKET`, `ARENA_R2_ENDPOINT`,
`ARENA_R2_KEY_ID`, `ARENA_R2_SECRET`. Put them in `~/.config/arena/r2.env` (mode 600, outside
the repo, never committed), written from `pass` by hand, for example
`pass show arena/r2-env > ~/.config/arena/r2.env`. Then change the upload service's `ExecStart` to
`... tools/upload.py --r2` and install the extra with `uv sync --extra r2`.
