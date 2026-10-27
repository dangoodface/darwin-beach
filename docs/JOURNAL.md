# The night journal

Daniel handed the night of 27 October 2026 to the two coordinating sessions,
Fable and funsies, with one rule: every change needs both to agree, and a tie
goes to Opus. Five one-hour runs, a review ten minutes after each. This is
the record, in plain language, of what each run showed, what each of us
proposed, what we chose and why, and how our thinking changed. It is public.

## Tick 0, 00:30: where we start

What run 1 showed (20261027T055834Z-slime-v0, one hour from 23:58):
- Every one of the 400 styles had a bot within eight seconds. Occupancy is
  not the show, so a cell now "lights" only when its bot holds its own
  against the practice rivals on average; by that rule 0 cells at eleven
  seconds, 64 at generation 478, 265 at 1,912, 308 at 3,824, 348 at 8,150.
- The scripted rival, the coach, is almost unbeaten: the champion of
  generation 4,486 won 2 of its 8 practice matches against it and none of
  16 on fresh serves; every other elite lost every match. Positive scores
  come from beating the three random sparring partners.
- The strongest region is mid court with a hopping habit; the weakest is
  net huggers who never jump. The champion's main line of descent is 29 to
  83 steps long depending on when you look, mostly crossover.

Run 2 starts right after run 1 ends, same rules, fresh seed, richer
exports (lit, per-rival margins, coach-beaters): the control. It tells us
how much of run 1 was luck before we change any rule.

Open bids, to be settled at tick 1:
- funsies: (a) the coach as the boss, "beats the coach" as the headline
  and a second colour on the map; (b) coevolution, promote styles that
  beat the coach into the rival set; (c) strategy axes, hit height by ball
  speed after contact, so cells read as lob artist or spiker.
- Fable: positions and counter-bids in the tick 1 entry.

## Tick 1, 01:08: run 1 is in, run 2 is the control, two minds changed

Run 1 final (one hour): 25,578 generations, 52.4 million matches at
14,551 a second, 400 of 400 cells with a bot, 368 lit, best fitness +3.16;
all 400 stored matches re-played on the kernel to the same hash. Nobody
beat the coach on average: the champion won 2 of its 8 practice matches
against it and none of 16 on fresh serves. Against the three random
sparring partners the champion won 23 of 24.

Measured to settle the bids, 300 random bots and the 400 elites: hits per
match (random 0.7, elites 3.5) correlate +0.84 with the score; hit height
+0.65; ball speed after contact +0.43. Learning to play is first of all
learning to reach the ball.

Decisions:
- (c) strategy axes: dropped. The numbers showed hit height and speed are
  competence, not style; a map on them would be a ladder drawn as a grid.
  funsies changed its mind on the measurement. Court position by airtime
  stays; instead (d) all eight descriptors are exported per elite, so
  strategy is drawn on top of the map and future axes come from data.
- (a) the coach is the headline, with (h) eight seeds per probe from run 3
  on, so "beats the coach" rests on 16 matches.
- (b) coevolution waits for the real ladder, which Opus delivered tonight
  (frozen ratings, promotion at 55 of 100 plus out-rating the champion,
  eras, re-score, cycle check; merged, off by default). Run 5 is its
  debut; "nobody was promoted" is an acceptable result.
- (e) an optimiser A/B, CMA-ME against plain MAP-Elites, is run 4.
- (g) the RL species: Fable changed funsies' mind from "park it" to "build
  it tonight" because it does not touch the runs and the kernel's own
  network makes the RL bot a citizen of the arena. Sonnet built the
  harness and trainer; it has been training on the GPU since 00:38 under a
  two-thread cap so the control stays clean. At 40 updates it still lost
  0 to 5 to the coach.
- (f) highlight reels: funsies' tool renders vertical clips from a seed and
  two genomes; the longest rally of run 1 was 39 seconds against the coach.
- Daniel's token budget is nearly used until about 02:50, so the night runs
  lean: runs become two hours, sessions pause at clean commits, nobody
  polls. Run 2 (20261027T065906Z-night-r2, from 00:59) is the control with
  a fresh seed and the richer exports.
