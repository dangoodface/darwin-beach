# Systemd units (user level)

Written, not installed. The lead decides when. They run the worktree's `.venv/bin/python` directly (uv lives under mise, whose path changes with its version); after the branch merges, point `WorkingDirectory` and `ExecStart` at the final checkout.

| Unit | What |
|---|---|
| `arena-train.service` | the loop (`python -m arena.run`) under `MemoryMax=4G`, `CPUQuota=1400%` |
| `arena-upload.service` + `.timer` | `tools/upload.py` every ten minutes; local dry-run mirror until R2 exists |

Install (as Daniel, no sudo):

```
mkdir -p ~/.config/systemd/user
cp systemd/arena-* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start arena-train.service
systemctl --user enable --now arena-upload.timer
```

Watch: `journalctl --user -u arena-train -f`. Stop: `systemctl --user stop arena-train`.

## R2 credentials (after Daniel creates the bucket)

The uploader reads four environment variables: `ARENA_R2_BUCKET`, `ARENA_R2_ENDPOINT`,
`ARENA_R2_KEY_ID`, `ARENA_R2_SECRET`. Put them in `~/.config/arena/r2.env` (mode 600, outside
the repo, never committed), written from `pass` by hand, for example
`pass show arena/r2-env > ~/.config/arena/r2.env`. Then change the upload service's `ExecStart` to
`... tools/upload.py --r2` and install the extra with `uv sync --extra r2`.
