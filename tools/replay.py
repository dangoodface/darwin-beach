"""Thin entry point: `uv run python tools/replay.py RUN_ID --all`. Logic is in arena.replay."""
import sys

from arena.replay import main

if __name__ == "__main__":
    sys.exit(main())
