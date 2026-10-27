"""Thin entry point: `uv run python tools/upload.py --to FOLDER` or `--r2`. Logic is in arena.upload."""
from arena.upload import main

if __name__ == "__main__":
    main()
