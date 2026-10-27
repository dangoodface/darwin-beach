# Five commands for the arena. `make help` lists them.
.PHONY: help test train upload dev parity bench

help:
	@echo "make test      Python tests"
	@echo "make train     practice run on the pretend game (200 generations)"
	@echo "make upload    dry-run mirror of out/ into ~/.local/share/arena/mirror"
	@echo "make dev       the page (placeholder until web/ exists)"
	@echo "make parity    twin test, Python against Node (TODO: Track K's command)"
	@echo "make bench     speed test on 15 workers (TODO: Track K's command)"

test:
	uv run pytest -q

train:
	systemd-run --user --scope -q -p MemoryMax=4G -p CPUQuota=1400% \
	  uv run python -m arena.run --config configs/slime-v0.toml --generations 200

upload:
	uv run python tools/upload.py --to $${ARENA_DATA:-$$HOME/.local/share/arena}/mirror

dev:
	@echo "TODO: the page track adds 'npm run dev' in web/ (serves web/ plus a data/ link to out/)"

# TODO: fill these in from Track K's README section once the kernel lands.
parity:
	@echo "TODO: parity (see kernel README section)"; exit 1

bench:
	@echo "TODO: bench (see kernel README section)"; exit 1
