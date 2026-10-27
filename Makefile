# Five commands for the arena. `make help` lists them.
.PHONY: help test train practice upload dev verify kernel parity bench

WASM := kernel/target/wasm32-unknown-unknown/release/slime_kernel.wasm

help:
	@echo "make test      Python tests"
	@echo "make train     the real run on the kernel, one hour, under the training limits (SECONDS=... to change)"
	@echo "make practice  practice run on the pretend game (200 generations)"
	@echo "make upload    dry-run mirror of out/ into ~/.local/share/arena/mirror"
	@echo "make verify    re-run every live cell's replay and compare hashes: make verify RUN=<run_id>"
	@echo "make worker-test    the Worker's pure functions and handler (node --test, no account)"
	@echo "make summary    measurements of a run in plain words: make summary RUN=<run_id>"
	@echo "make dev       the page (placeholder until web/ exists)"
	@echo "make kernel    build the game file and say whether it matches one in artifacts/kernel/"
	@echo "make parity    twin test: 1,000 matches in wasmtime and in Node, tick by tick"
	@echo "make bench     speed test on 1, 8 and 15 workers, under the training limits"

test:
	uv run pytest -q

SECONDS ?= 3600
train:
	systemd-run --user --scope -q -p MemoryMax=4G -p CPUQuota=1400% \
	  uv run python -m arena.run --config configs/slime-v0.toml --max-seconds $(SECONDS)

practice:
	systemd-run --user --scope -q -p MemoryMax=4G -p CPUQuota=1400% \
	  uv run python -m arena.run --config configs/practice-fake.toml --generations 200

upload:
	uv run python tools/upload.py --to $${ARENA_DATA:-$$HOME/.local/share/arena}/mirror

verify:
	@test -n "$(RUN)" || { echo "usage: make verify RUN=<run_id>"; exit 2; }
	uv run python tools/replay.py $(RUN) --all

summary:
	@test -n "$(RUN)" || { echo "usage: make summary RUN=<run_id>"; exit 2; }
	uv run python tools/summary.py $(RUN)

dev:
	@echo "TODO: the page track adds 'npm run dev' in web/ (serves web/ plus a data/ link to out/)"

# +1.95.0 on purpose: RUSTUP_TOOLCHAIN in the shell would override
# kernel/rust-toolchain.toml, and another compiler makes another file.
kernel:
	cargo +1.95.0 build --release --target wasm32-unknown-unknown --manifest-path kernel/Cargo.toml
	@sha=$$(sha256sum $(WASM) | cut -d' ' -f1); \
	if [ -f artifacts/kernel/$$sha.wasm ]; then echo "kernel $$sha: the same bytes as artifacts/kernel/$$sha.wasm"; \
	else echo "kernel $$sha is new, not in artifacts/kernel/: a changed kernel needs a version bump (docs/ABI.md)"; fi

parity: kernel
	cargo +1.95.0 test --quiet --manifest-path kernel/Cargo.toml
	uv run pytest -q tests/test_kernel.py tests/test_parity.py

bench: kernel
	systemd-run --user --scope -q -p MemoryMax=4G -p CPUQuota=1400% \
	  uv run python tools/bench.py

worker-test:
	node --test worker/test/*.test.mjs
