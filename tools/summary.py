"""Thin entry point: `uv run python tools/summary.py RUN_ID [--json]`. Logic is in arena.summary."""
import sys

from arena.summary import main

if __name__ == "__main__":
    sys.exit(main())
