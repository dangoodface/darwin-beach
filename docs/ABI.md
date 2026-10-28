# Kernel ABI, version 3

Rulings of 26 October 2026 (Opus's eight gaps and Sonnet's probe question)
are folded in below; the scripted bot is a reserved slot.

Version 2 (27 October 2026) adds two descriptors per side, juggled_points and
max_streak, for JOGA BONITO, and the `desc_count()` export; result blocks grow
from 152 to 184 bytes. The game does not move by a bit: physics, policy,
hash, RNG and the scripted bot are version 1's, so every version 1 match,
replay and stored hash plays out identically under version 2, with the same
first eight descriptors.

Version 3 (28 October 2026) widens the brain to 12-64-3 (1,027 parameters),
adds a per-side action hold (to `play_match`, the jobs and the interactive
game), raises `max_ticks_cap` to 12,000 and adds an eleventh descriptor,
spiked_points. Jobs grow to 24 bytes and result blocks to 200. The game is
unchanged: at hold 1, a version 2 genome padded with zeros (`pad_params`,
below) plays the bit-identical match, which night run 6's genomes confirm
1,000 matches out of 1,000. The hidden layer runs two units per wasm SIMD
register (simd128: standard f64x2 add, mul, div and abs, no relaxed SIMD),
each lane doing exactly one unit's multiplies and adds in the usual order.

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
- Lives saturate at zero, so stepping a finished interactive game cannot
  underflow. The interactive game is its own instance: `play_match` never
  disturbs a replay in progress. The state block is a copy refreshed after
  `reset` and `step`; writing to it changes nothing.

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

MLP 12 to 64 to 3 (version 3; 12 to 32 to 3 before), softsign on the hidden
layer, linear output; action bit i is set when output i is greater than 0.
Parameter layout, f32, little-endian, 1,027 values: `W1[64][12]` row-major
(offset 0), `b1[64]` (768), `W2[3][64]` row-major (832), `b2[3]` (1024).
Version 1 and 2 genomes have 515 values: `W1[32][12]`, `b1[32]`, `W2[3][32]`,
`b2[3]`. Each unit accumulates sequentially: `acc = b; for j: acc = acc +
w[j] * x[j]`. No fused multiply-add, no pairwise sums.

Padding a version 2 genome (`arena.genome.pad_params`): every block keeps its
old values first and gains zeros for hidden units 32 to 63, that is W1 rows
32 to 63, `b1[32..64]`, and the last 32 columns of each W2 row; `b2` stays
last. A zero unit outputs softsign(0) = 0 and adds 0 * 0 to each output, which
never changes which side of 0 an output is on, so the padded genome takes the
same actions and plays the same match, tick hash for tick hash.

## Action holds (version 3)

A side with hold `h` lets its brain choose an action on ticks where `tick % h
== 0`, from tick 0, and repeats that action on the ticks in between; `h = 1`
acts every tick, as before. `tick` is the number of ticks already played (the
state block's `tick` before the step). Holds are per side and per match, the
scripted bot included; a hold of 0 is a bad job. "Smooth Operator" plays at
hold 9 (0.3 s, human reflexes).

## Exports

Constants
- `kernel_version() -> u32`: 3 (2 and 1 for the earlier modules).
- `desc_count() -> u32`: 11, the descriptors per side in a result block (10 in
  version 2). A version 1 module has no such export and always 8.
- `job_bytes() -> u32`: 24, the size of one job (version 3). Older modules
  have no such export and 16-byte jobs.
- `genome_len() -> u32`: 1027 (515 before version 3).
- `max_ticks_cap() -> u32`: 12,000 (3,000 before version 3). `max_ticks`
  arguments above it are an error. Training plays 3,000; evaluations may
  play longer.
- `table_capacity() -> u32`: 1024 genomes. `batch_capacity() -> u32`: 256 jobs.
- `scripted_slot() -> u32`: `0xFFFFFFFF`. Wherever a slot is expected
  (`play_match`, a job, `policy_action`) this value means the scripted bot.

Memory (byte offsets into the exported `memory`; all static, no allocator)
- `genome_table_ptr() -> u32`: `f32[table_capacity][genome_len]`. Slot i is
  genome i. The caller fills slots with one bulk copy each.
- `jobs_ptr() -> u32`: `batch_capacity` jobs of 24 bytes: `seed u64, a u32,
  b u32, hold_a u32, hold_b u32` (slot indexes into the genome table, then
  the holds; 16 bytes without holds before version 3).
- `results_ptr() -> u32`: `batch_capacity` result blocks of 200 bytes (184 in
  version 2, 152 in version 1; layout below). `play_match` writes block 0; `play_batch` writes block i for job i.
- `tick_hash_ptr() -> u32`: `u64[max_ticks_cap]`, filled by `play_match`
  when `record_hashes` is nonzero. `tick_hash[i]` is the running hash after
  tick `i + 1` (the state block's `tick` is the 1-based count after a step).
- `state_ptr() -> u32`: the f64 state block of the interactive game (below).

Matches
- `play_match(seed: u64, a: u32, b: u32, max_ticks: u32, record_hashes: u32,
  hold_a: u32, hold_b: u32) -> u32`: resets with `seed`, plays table slot
  `a` on the left and `b` on the right, each holding its action for its hold
  (version 3; the two hold arguments are new), until a side has no lives or
  `max_ticks` ticks; returns ticks played; fills result block 0. A bad job
  (`max_ticks` of 0 or above the cap, a hold of 0, or a slot that is neither
  in the table nor `scripted_slot()`) writes block 0 with flag bit 2 set and
  returns 0.
- `play_batch(n: u32, max_ticks: u32) -> u32`: runs the first `n` jobs from
  the jobs buffer in order, each as `play_match` without hashes per tick;
  stops at the first bad job, flags that job's block, and returns how many
  completed before it. `n` above `batch_capacity` runs nothing and returns 0.

Interactive (for the browser, and for tests)
- `reset(seed: u64)`, `set_actions(a: u32, b: u32)`, `step() -> i32`:
  returns +1 when A wins the point, -1 when B wins it, 0 otherwise. `reset`
  also sets both holds to 1; `set_actions` remembers the actions it applies.
- `set_holds(hold_a: u32, hold_b: u32) -> u32` (version 3): the interactive
  game's holds, called after `reset`. Returns 1, or 0 and changes nothing
  when a hold is 0.
- `held_action(side: u32, slot: u32) -> u32` (version 3): `policy_action`
  under the side's hold: the brain's action on ticks divisible by the hold,
  otherwise the action the side last applied through `set_actions`. The
  page's replay loop is `reset(seed); set_holds(h_a, h_b);` then each tick
  `a = held_action(0, slot_a); b = held_action(1, slot_b); set_actions(a, b);
  step()`, which plays exactly what `play_match` played with those holds. A
  human side skips `held_action` and passes its keys to `set_actions`.
- `is_done(max_ticks: u32) -> u32`: 1 when a side has no lives or the tick
  count reached `max_ticks`.
- `policy_action(side: u32, slot: u32) -> u32`: the action genome `slot`
  would take now for `side` (0 = A, 1 = B), from the current state;
  `scripted_slot()` is accepted. A bad side or slot returns `0xFFFFFFFF`
  (which arrives as -1 in JS).
- `scripted_action(side: u32) -> u32`: the anchor bot, deterministic, no
  parameters: it tracks the ball's x when the ball is on its side and waits
  at a home x when the ball is on the far side, and jumps when the ball is
  close and above. The exact definition lives in `kernel/src/scripted.rs`
  and is frozen at kernel version 1: it is a ladder anchor and a probe
  opponent, so any change is a kernel version bump.
- `state_hash() -> u64`: the running hash of the interactive game.

## Result block (200 bytes; 184 in version 2, 152 in version 1)

| offset | type | field |
|---|---|---|
| 0 | u64 | final_hash |
| 8 | u32 | ticks |
| 12 | u32 | points_a |
| 16 | u32 | points_b |
| 20 | u32 | flags: bit 0 NaN or infinity seen, bit 1 ended on `max_ticks` with both sides still holding lives, bit 2 bad job |
| 24 | f64[11] | descriptors A (f64[10] in version 2, f64[8] in version 1) |
| 112 | f64[11] | descriptors B (at 104 in version 2, at 88 in version 1) |

## Descriptors, per side, over the whole match

| index | name | definition |
|---|---|---|
| 0 | court_position | mean over ticks of `(22.5 - x*dir) / 20.5`; 0 at the back wall, 1 at the net |
| 1 | airtime | fraction of ticks with `y > 1.5` |
| 2 | hits | count of ball bounces off this slime |
| 3 | mean_hit_height | mean ball y right after those bounces (after the nudge loop), 0 if none |
| 4 | mean_speed_after_hit | mean `sqrt(vx*vx + vy*vy)` of the ball after the velocity update of those bounces, 0 if none |
| 5 | jumps | count of ticks where the slime's y went from `<= 1.5` to `> 1.5` |
| 6 | distance | sum over ticks of `|x after update - x before|` |
| 7 | points | points won |
| 8 | juggled_points | points won with a juggle (version 2; below) |
| 9 | max_streak | the longest streak of own touches in the match, sent or not (version 2) |
| 10 | spiked_points | points won with a spike (version 3; below) |

The x range comes from the slime's own-half clamp: `x*dir` runs from
`REF_WALL_WIDTH/2 + r = 2.0` at the net to `REF_W/2 - r = 22.5` at the wall.

### Juggling (version 2, Daniel's definition)

Per side, per point. A streak is consecutive own touches (the bounces the
hits descriptor counts) with no net crossing in between; while a streak is
open the slime's lowest and highest x are tracked over every hashed tick from
its first touch on. The ball crosses the net out of a side's half when it
ends a tick on the other side of `x = 0` from where it ended the last one (A's
half is `x <= 0`, the same line that scores a landing for B; the serve starts
on it). That crossing sends the side's streak: its touches and its span
(highest minus lowest x) are kept, and a new streak can start. When the point
ends, the winner's most recently sent streak decides: the point is juggled
when it had at least `JUGGLE_MIN_TOUCHES = 4` touches and a span of at least
`JUGGLE_MIN_SPAN = 20.5 / 2 = 10.25`, half the slime's own court. Streaks,
spans and sent streaks reset with every point; juggled_points and max_streak
add up over the match. In a tick, touches count before the crossing, and the
crossing before the point. The tracker reads the state after the NaN check
and never writes the physics or the hash. Constants in `kernel/src/juggle.rs`.

### Spikes (version 3, Daniel's definition)

On the same per-side tracker: a point is spiked for the side that won it when
that side's most recently sent streak had at least `SPIKE_MIN_TOUCHES = 2`
touches, the spike included (one carry, then the spike), and its last touch
had the slime's centre within `SPIKE_MAX_FROM_NET = 4.0` of the net (`x*dir
<= 4.0`; the net is at 0 and the back wall at 24) and sent the ball off at
`SPIKE_MIN_SPEED = 35.0` or more (its speed right after the bounce, as
mean_speed_after_hit measures it). Like juggled_points, a match total;
everything else resets with each point.

## State block (f64 from `state_ptr`)

0 to 3 ball `x, y, vx, vy`; 4 to 7 A `x, y, vx, vy`; 8 to 11 B `x, y, vx,
vy`; 12 A lives; 13 B lives; 14 tick; 15 delay frames left; 16 last point
(+1, -1 or 0).

## Hash

FNV-1a 64 (offset `0xcbf29ce484222325`, prime `0x100000001b3`), chained
across ticks. After each `step`, in this order: `tick` as u32 little-endian,
the twelve f64 of ball, A and B as in the state block (little-endian bit
patterns), then A lives and B lives as u32. `final_hash` is the value at the
end; `tick_hash[i]` is the running value after tick `i + 1` when recording.

The NaN check runs before the descriptor accumulators and before the hash.
A tick that produced a NaN or an infinity is neither hashed nor counted in
`ticks`, its hits are discarded, flag bit 0 is set and the match stops.

## 64-bit values at the boundary

Hashes and seeds are u64 inside the kernel, but both wasmtime-py and JS
present i64 results as signed values. The Python `Kernel` masks results
with `& (2**64 - 1)` and passes a seed at or above 2^63 as its signed
equivalent; the page wraps results with `BigInt.asUintN(64, x)` and passes
seeds as BigInt, which wasm wraps by itself.

## RNG

PCG32, the reference `pcg32_random_r`: `state = state * 6364136223846793005
+ inc`; output from the old state: `xorshifted = (((old >> 18) ^ old) >> 27)
as u32`, `rot = (old >> 59) as u32`, result `(xorshifted >> rot) |
(xorshifted << ((32 - rot) & 31))`. Seeding, as the reference `pcg32_srandom_r`
with `initseq = 0xda3e39cb94b95bdb`: `state = 0; inc = (initseq << 1) | 1;
next(); state += seed; next()`. `u01() = (next_u32() >> 8) as f64 *
(1.0 / 16777216.0)`. The only use is the serve, two draws in this order:
`ball vx = -20 + 40 * u01()`, then `ball vy = 10 + 15 * u01()` (the source
draws `uniform(-20, 20)` then `uniform(10, 25)`), once at `reset` and once
after every point.

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
6. Only when no point was scored: `updateState` for A, then B (the source
   returns early on a point, so both observations stay one tick stale after
   every point; the kernel follows it).
7. Always: NaN check, then descriptor accumulators, then hash.

Start: ball at (0, 12) with the random vx and vy, A at (-12, 1.5), B at
(12, 1.5), delay 30 frames, 5 lives each.

## Versioning

Any change to physics, policy, descriptors, hash, RNG or the scripted bot
bumps `kernel_version`. Version 1 is `5487b2a6…`; version 2, which adds the
juggle descriptors and nothing else, is `93c29e32…`; version 3, the wider
brain, holds, the 12,000-tick cap and spiked_points, is `0f673ff4…` (full
shas in `artifacts/kernel/README.md`). A built module is stored as
`artifacts/kernel/<sha256>.wasm` and never deleted; every genome, snapshot
and replay records the kernel sha it was made with.
