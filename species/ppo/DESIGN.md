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
