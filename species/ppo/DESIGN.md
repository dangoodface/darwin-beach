# Species 2: a policy trained by reinforcement learning

Written 27 October 2026. The second species for Darwin Beach: same game, same brain, a different way of
getting there. Evolution mutates 515 numbers and keeps winners; here the same 515 numbers are trained by
PPO, a gradient method that learns from the points scored.

- **The brain is the kernel's own network** (docs/ABI.md): 12 inputs, 32 hidden units with softsign, 3
  outputs, 515 f32 parameters laid out W1, b1, W2, b2. Nothing is added, so the trained weights are a genome.
- **Training treats it as a stochastic policy**: each of the three action bits (forward, backward, jump) is an
  independent Bernoulli press with probability sigmoid(output). The value function is a separate small
  network used only in training and never exported.
- **The exported genome plays the mode**: press when the output is above zero, which is exactly the kernel's
  rule. So the kernel replays it tick for tick, and the map, the replay checker, play mode and the ladder
  accept it as they accept any genome. `export.py` proves it on 1,000 states of random play.
- **Reward is the kernel's point score**: +1 when the learner wins a point, -1 when it loses one, read from
  `step()`. An episode is a match: five lives each, or 3,000 ticks. The learner plays either side (mirrored
  observations), chosen at random per episode.
- **Observations are the kernel's** from the state block: mirrored, divided by 10, and stale for one tick
  after a point, because the kernel only refreshes them when no point was decided (`ObsFeed` copies that).
- **Opponents**: the coach (`scripted_action`) first, then elites from the arena runs (genomes in table slots,
  `policy_action` for the other side), later the learner's own past selves.
- **Rollouts on the CPU, updates on the GPU**: a few processes each own a kernel and play one match at a time
  through the interactive exports, with a numpy copy of the policy; the PPO update (clipped objective, GAE,
  entropy bonus) runs in torch on the RTX 5060 in `~/.cache/arena-preflight`, which gets only wasmtime and numpy.
- **Cost of a tick**: each wasmtime call from Python costs about 8 microseconds, and a tick needs three
  (opponent action, set_actions, step), so one process plays 12,000 to 13,500 ticks a second. Under a CPU
  quota of 200%, two processes are the right number (about 25,000 ticks a second together); more processes
  go slower. A kernel export that does the opponent action, the actions and the step in one call would
  roughly double that.
- **Checks**: a greedy, exact-arithmetic rollout reproduces `Kernel.play` (hash, ticks, points) on either side
  against an elite and against the coach; the policy's mode equals `policy_action` on mirrored observations;
  the exported file round-trips and is named by its genome id.
- **What it is not**: it does not replace evolution on the map. It is a second species to compare: does a
  gradient method find the reach-the-ball behaviour faster than selection does, and which styles does it miss?

### Addendum, 27 October 2026: after ppo-1, what to try next

**What ppo-1 showed.** 13,969 updates, 164.5M ticks in 2.2 hours, and the deterministic policy still lost 0 to 5
to the coach (best evaluation points for 0.75 of 5, last 0.625; entropy 2.06 of 2.08; KL 0.0002). The unshaped
reward almost never fires: a random policy scores about once in ten matches, so there is no gradient to follow.

**Why not the elite curriculum.** The archive elites are not an easier opponent. Their fitness is measured against
the scripted bot and the weakest of the 400 filled cells is -0.8 points a match, the lower quartile is +0.9. Even
the worst elite plays about as well as the coach, so the learner would meet the same wall under another name.

**Pick: reward shaping with an annealed schedule, on a softened coach.** Both parts live in `rollout.py` and
`train.py`, with no kernel change.
1. *Shaping*, potential-based so the optimum is unchanged: `r += gamma*phi(s') - phi(s)`, where phi rewards being
   under the ball while it is on our side and then the ball being on the opponent's side (all readable from the 12
   observations). Add a small contact bonus when the ball's vertical velocity flips next to our slime. The weight
   `w` starts at 1 (a contact is worth about a tenth of a point) and falls linearly to 0 over the first 60% of the
   run; the last 40% trains on the kernel's +1/-1 only, so what is exported was tuned on the real objective.
2. *Softened coach*: with probability `p` each tick the opponent's action is replaced by "no press". `p` starts at
   0.5 and anneals to 0 alongside `w`. It gives the learner points to win in the first minutes, which is the thing
   it has lacked. Evaluation always uses the unsoftened coach and is the only number that counts.

**First sign of life, within 30 minutes (about 45M ticks):** train points for above 2 of 5 against the softened
coach, entropy below 1.8, and evaluation points for above 1.5 against the real coach.

**To earn a place on the page, one 2-hour run must show:** evaluation win rate above 0 against the unsoftened coach
over 64 matches, both sides (0 to 5 does not qualify), at `w = 0` and `p = 0` for at least the last 30 minutes, and
the exported genome passing `export.py`'s 1,000-state check. If it only wins while shaping is on, it stays off the
page. If it beats the coach, it plays the archive elites once for the comparison with selection.

### Result, 27 October 2026, 05:20: ppo-2 did not earn a place

ppo-2 ran from 03:04 to 05:04 (two threads, 11,389 updates, 137 million ticks) with `w = 1` and `p = 0.5` fading
to zero at 72 minutes. Against the softened coach with hints it took 1.9 points a match at ten minutes; as the hints
and the softening faded its training score fell to 0.06 a match and did not recover. Against the real coach it took
0.5 points a match at every evaluation and 0.3 of 5 over the final 64 matches, win rate zero. Entropy sat at 2.04
for the whole run, so the policy never sharpened, and none of the 30-minute signs was met. Verdict: no place on the
page. The species is parked and the code is kept as a recorded dead end: the trainer, the shaping and the export
check all work, and the next attempt should change the objective rather than the hints (a reward that does not fade,
or self-play against generation-zero genomes rather than coach-level elites). Log: `$ARENA_DATA/species/ppo-2/`.
