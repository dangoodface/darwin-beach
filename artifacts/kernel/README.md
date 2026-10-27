# Kernels

Every kernel the arena has used, named by the sha256 of its bytes. They are
never deleted: genomes, snapshots and replays record the sha they were made
with, so an old replay can always find the kernel that plays it.

| Version | sha256 | Built from | Notes |
|---|---|---|---|
| 1 | `5487b2a61f66880a4bd468957b6f662cb0f5afd05ab8ae7ac6736a81b025674a` | `kernel/` at K4, 26 October 2026 | first release; scripted-v1 frozen |
| 2 | `93c29e323d3d1d7874286754ae395de50a8e3932b20cd75aa77787ce4387ebfb` | `kernel/` on branch kernel-juggle, 27 October 2026 | adds juggled_points and max_streak; plays every version 1 match to the same hash |

## Rebuilding a kernel to check it

```sh
cargo +1.95.0 build --release --target wasm32-unknown-unknown --manifest-path kernel/Cargo.toml
sha256sum kernel/target/wasm32-unknown-unknown/release/slime_kernel.wasm
```

The `+1.95.0` matters: a different compiler makes different bytes, and this
machine's shell sets `RUSTUP_TOOLCHAIN=stable`, which would override
`kernel/rust-toolchain.toml`. Version 1 was built with rustc 1.95.0
(59807616e 2026-04-14, LLVM 22.1.2); clean builds from any path give the
same sha.
