# Conventions for everyone building the arena

- One repo, `~/projects/arena`, local git for now; GitHub only when Daniel
  says so. Each track works on its own branch in its own worktree
  (`~/projects/arena-kernel`, `-trainer`, `-web`); the lead merges into `main`.
- Absolute paths in every command; `git -C <worktree>`; never `cd` in calls
  that may run in parallel. Never `sudo`. Credentials come from `pass` and
  are never written into a file or a commit.
- Data, venvs, build output and scratch live on the SSD: `.venv` in the
  worktree, `target/` in `kernel/`, `$ARENA_DATA` under `~/.local/share`.
  `/tmp` is RAM; nothing goes there.
- Python 3.13 through `uv`; one `pyproject.toml` at the root; add a
  dependency only by editing it and running `uv lock`, and say so in the
  commit. Rust through the pinned toolchain in `rust-toolchain.toml`. Node
  24 for the parity harness and the page's tests.
- Heavy runs: `systemd-run --user --scope -p MemoryMax=4G -p CPUQuota=1400%
  <command>`.
- Tests: `cargo test` in `kernel/`, `uv run pytest -q` at the root,
  `node --test web/tests` for the page. A change ships with its test.
- Prose: Daniel's voice. No em dashes, British spelling, dates as
  "26 October 2026". Plain names, short functions, comments that explain the
  game's quirks rather than the code.
- This is a standalone project: nothing is imported or copied from any other repository.
- Commits are small and say what and why; no pushes anywhere until the repo
  has a remote Daniel approved.
