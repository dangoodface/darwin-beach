# Kernel ABI, version 1

One WebAssembly module, `slime_kernel.wasm`, built from `kernel/` (Rust,
target `wasm32-unknown-unknown`, plain `extern "C"` exports, no wasm-bindgen).
It exports functions and one linear memory. Python (wasmtime) and the browser
call the same exports on the same bytes, which is where parity comes from.

Physics, constants and update order are ported from `slimevolley.py` in
hardmaru's slimevolleygym (David Ha, 2020, Apache-2.0); the kernel's header
says so. Any deviation from that file is a comment in `kernel/src/physics.rs`
with the reason.

## Rules inside the kernel

- Everything is f64. Genome parameters are f32 in memory and upcast exactly.
- This kernel uses only `+ - * /`, `sqrt`, comparisons and integer ops. No
  libm calls. `softsign(x) = x / (1 + |x|)` is the activation. A later kernel
  may use `sin` and `cos` from Rust's libm, which is compiled into the module
  and therefore deterministic across engines, but a native build of such a
  kernel would then need the same libm; say so in that kernel's ABI note.
- NaN payloads are the one thing the wasm spec leaves to the engine. The
  kernel checks every state value for NaN or infinity before hashing a tick;
  on the first one it sets result flag bit 0, stops the match, and the hash
  covers only the ticks before it. Nothing is canonicalised silently.
- Integer tick counter. No accumulated `t += dt`.
- The only randomness is the kernel's PCG32, seeded from the u64 the caller
  passes. Nothing random is generated outside.
- Fixed update order (below). Ties on equality follow the Python source.

## Sides, actions, observations

- Side A is the left slime (`dir = -1`), side B the right slime (`dir = +1`).
  `play_match` plays the genome in buffer A on the left and buffer B on the
  right. Callers who want side balance play both orders with the same seed.
- Action bits: bit 0 forward (towards the net), bit 1 backward, bit 2 jump.
  Forward and backward together mean no horizontal move. Values 0 to 7.
- Observation, 12 f64 per side, mirrored exactly as slimevolleygym's
  `RelativeState` so one genome plays either side: `x*dir, y, vx*dir, vy,
  bx*dir, by, bvx*dir, bvy, ox*(-dir), oy, ovx*(-dir), ovy`, each divided
  by 10.

## Policy

MLP 12 to 32 to 3, softsign on the hidden layer, linear output; action bit i
is set when output i is greater than 0. Parameter layout, f32, little-endian,
515 values: `W1[32][12]` row-major, `b1[32]`, `W2[3][32]` row-major, `b2[3]`.
Each unit accumulates sequentially: `acc = b; for j: acc = acc + w[j] * x[j]`.
No fused multiply-add, no pairwise sums.

## Exports

Constants
- `kernel_version() -> u32`: 1.
- `genome_len() -> u32`: 515.
- `max_ticks_cap() -> u32`: 3000. `max_ticks` arguments above it are an error.
- `table_capacity() -> u32`: 1024 genomes. `batch_capacity() -> u32`: 256 jobs.

Memory (byte offsets into the exported `memory`; all static, no allocator)
- `genome_table_ptr() -> u32`: `f32[table_capacity][genome_len]`. Slot i is
  genome i. The caller fills slots with one bulk copy each.
- `jobs_ptr() -> u32`: `batch_capacity` jobs of 16 bytes: `seed u64, a u32,
  b u32` (slot indexes into the genome table).
- `results_ptr() -> u32`: `batch_capacity` result blocks of 152 bytes (layout
  below). `play_match` writes block 0; `play_batch` writes block i for job i.
- `tick_hash_ptr() -> u32`: `u64[max_ticks_cap]`, filled by `play_match`
  when `record_hashes` is nonzero (per-tick running hash, for parity tests).
- `state_ptr() -> u32`: the f64 state block of the interactive game (below).

Matches
- `play_match(seed: u64, a: u32, b: u32, max_ticks: u32, record_hashes: u32)
  -> u32`: resets with `seed`, plays table slot `a` on the left and `b` on
  the right until a side has no lives or `max_ticks` ticks; returns ticks
  played; fills result block 0.
- `play_batch(n: u32, max_ticks: u32) -> u32`: runs the first `n` jobs from
  the jobs buffer in order, each as `play_match` without hashes per tick;
  returns the number completed (less than `n` only on a bad job).

Interactive (for the browser, and for tests)
- `reset(seed: u64)`, `set_actions(a: u32, b: u32)`, `step() -> i32`:
  returns +1 when A wins the point, -1 when B wins it, 0 otherwise.
- `is_done(max_ticks: u32) -> u32`: 1 when a side has no lives or the tick
  count reached `max_ticks`.
- `policy_action(side: u32, slot: u32) -> u32`: the action genome `slot`
  would take now for `side` (0 = A, 1 = B), from the current state.
- `scripted_action(side: u32) -> u32`: the anchor bot: move towards the
  ball's x, jump when the ball is close and above; deterministic, no params.
  It is a ladder anchor and a probe opponent, so it never changes once
  released (a change is a kernel version bump).
- `state_hash() -> u64`: the running hash of the interactive game.

## Result block (152 bytes)

| offset | type | field |
|---|---|---|
| 0 | u64 | final_hash |
| 8 | u32 | ticks |
| 12 | u32 | points_a |
| 16 | u32 | points_b |
| 20 | u32 | flags: bit 0 NaN or infinity seen, bit 1 ended on max_ticks, bit 2 bad job |
| 24 | f64[8] | descriptors A |
| 88 | f64[8] | descriptors B |

## Descriptors, per side, over the whole match

| index | name | definition |
|---|---|---|
| 0 | court_position | mean over ticks of `(22.5 - x*dir) / 20.5`; 0 at the back wall, 1 at the net |
| 1 | airtime | fraction of ticks with `y > 1.5` |
| 2 | hits | count of ball bounces off this slime |
| 3 | mean_hit_height | mean ball y at those bounces, 0 if none |
| 4 | mean_speed_after_hit | mean ball speed right after those bounces, 0 if none |
| 5 | jumps | ground to air transitions |
| 6 | distance | sum of `|dx|` per tick |
| 7 | points | points won |

The x range comes from the slime's own-half clamp: `x*dir` runs from
`REF_WALL_WIDTH/2 + r = 2.0` at the net to `REF_W/2 - r = 22.5` at the wall.

## State block (f64 from `state_ptr`)

0 to 3 ball `x, y, vx, vy`; 4 to 7 A `x, y, vx, vy`; 8 to 11 B `x, y, vx,
vy`; 12 A lives; 13 B lives; 14 tick; 15 delay frames left; 16 last point
(+1, -1 or 0).

## Hash

FNV-1a 64 (offset `0xcbf29ce484222325`, prime `0x100000001b3`), chained
across ticks. After each `step`, in this order: `tick` as u32 little-endian,
the twelve f64 of ball, A and B as in the state block (little-endian bit
patterns), then A lives and B lives as u32. `final_hash` is the value at the
end; `tick_hash[tick]` is the running value after that tick when recording.

## RNG

PCG32, the reference `pcg32_random_r`: `state = state * 6364136223846793005
+ inc`; output from the old state: `xorshifted = (((old >> 18) ^ old) >> 27)
as u32`, `rot = (old >> 59) as u32`, result `(xorshifted >> rot) |
(xorshifted << ((32 - rot) & 31))`. Seeding, as the reference `pcg32_srandom_r`
with `initseq = 0xda3e39cb94b95bdb`: `state = 0; inc = (initseq << 1) | 1;
next(); state += seed; next()`. `u01() = (next_u32() >> 8) as f64 *
(1.0 / 16777216.0)`. The only use: the serve, `ball vx = -20 + 40 * u01()`,
once at `reset` and once after every point, in that order.

## Physics, in the order of one tick

Constants from slimevolleygym: `REF_W = 48`, `REF_H = 48`, `REF_U = 1.5`,
`REF_WALL_WIDTH = 1.0`, `REF_WALL_HEIGHT = 3.5`, `PLAYER_SPEED_X = 17.5`,
`PLAYER_SPEED_Y = 13.5`, `MAX_BALL_SPEED = 22.5`, `TIMESTEP = 1/30`,
`NUDGE = 0.1`, `FRICTION = 1.0`, `INIT_DELAY_FRAMES = 30`, `GRAVITY = -29.4`,
`MAXLIVES = 5`, ball radius 0.5, slime radius 1.5, fence stub a particle of
radius 0.5 at (0, 3.5). Write every constant as the same expression the
Python file uses (`-9.8*2*1.5`, `10*1.75`) so the double is the same.

1. A.update, then B.update (gravity, jump only from the ground, horizontal
   speed from the action, move, ground clamp, own-half clamp), exactly as
   `Agent.update`.
2. If the serve delay is over: ball gravity, `limitSpeed(0, MAX_BALL_SPEED)`,
   move. Otherwise the delay counter decrements, as `DelayScreen.status`.
3. Bounce the ball off A if colliding, then off B, then off the fence stub,
   each as `Particle.bounce` including its nudge loop. Guard: a zero-length
   normal uses (0, 1).
4. `checkEdges` on the ball: walls, ceiling, fence sides, floor. The floor
   decides the point: ball `x <= 0` is a point for B, otherwise for A.
5. On a point: new serve (RNG), lives, delay reset; `step` returns the sign.
6. `updateState` for A, then B (the observations for the next tick).
7. Descriptor accumulators, NaN check, hash.

Start: ball at (0, 12) with the random vx, A at (-12, 1.5), B at (12, 1.5),
delay 30 frames, 5 lives each.

## Versioning

Any change to physics, policy, descriptors, hash, RNG or the scripted bot
bumps `kernel_version`. A built module is stored as
`artifacts/kernel/<sha256>.wasm` and never deleted; every genome, snapshot
and replay records the kernel sha it was made with.
