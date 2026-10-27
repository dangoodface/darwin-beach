# The night journal

Daniel handed the night of 27 October 2026 to the two coordinating sessions,
Fable and funsies, with one rule: every change needs both to agree, and a tie
goes to Opus. Five one-hour runs, a review ten minutes after each. This is
the record, in plain language, of what each run showed, what each of us
proposed, what we chose and why, and how our thinking changed. It is public.

## Morning summary, 27 Oct, 09:20 (Fable and funsies)

**The coach is beaten, twice.** Five runs overnight (run 1 plus four two-hour
runs), about 470 million matches, every stored match re-played to its hash.
- Plain breeding (runs 1 to 3) never beat the coach. Run 2's 21 "beaters" won
  at most half of 16 fresh matches: they had learned the test, so only fresh
  serves count now.
- Smarter breeding (CMA-ME, run 4, run 3's seed and rules) led on every figure
  at every equal generation count: 398 of 400 lit against 366, best +4.27
  against +2.50, 136 styles beat the coach on fresh serves, the best 15 of 16.
- Run 5 (CMA-ME plus the ladder, a second seed): ten promotions; the champion
  beats the coach 96 of 100 and 13 of 16 fresh. Only 4 styles beat it on fresh
  serves, because the ladder grades against champions, not the coach; run 5's
  lit and QD are not comparable with runs 1 to 4.
- The headline lesson: champions 6 to 9 each beat the one before and lost 100
  of 100 to the coach while their rating rose. The who-beats-whom table caught
  it; the rating did not.
- Reinforcement learning (ppo-1, ppo-2) never won a match against the coach;
  parked with the result written down.

**Built:** CMA-ME and the fresh-serve check in the exporter; the ladder's debut;
style cards showing the graded coach margin and the fresh-serve verdict (the
only line allowed to say "beats the coach"); highlight reels for every run in
~/.local/share/arena/reels/; the campaign page and the pitch page updated.

**Waiting for Daniel:** the promotion rule should also require holding the
coach (a two-hour run 6 to test it); keep the probe margin in era-0 fitness;
launching play.thegoodface.ai/darwin-beach/; an arena-scoped R2 token; the
pitch page wording (shared with his friends). Known bug for Sonnet: the
summary's lit timeline counts across eras (493 of 400).

**Budget:** sessions sat idle while compute ran; about three short messages
per tick; ticks 3 to 5 ran early on Fable's word instead of waiting for crons.

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

## Tick 2, 03:05: run 2 is in, the coach is even, run 3 is live

Run 2 final (20261027T065906Z-night-r2, two hours from 00:59): 48,848
generations, 100.0 million matches at 13,894 a second, 400 of 400 cells with
a bot (all by generation 88, 13 seconds), 321 lit, QD-score 2,260, best
fitness +2.78 at cell [14, 16], born generation 40,389; all 400 stored
matches re-played to the same hash. At the one-hour mark, where run 1
ended: 278 lit and 13 elites beating the coach on their evaluation matches,
against run 1's 368 lit and none. Lit and QD-score do not compare across
runs, because each run draws its own three random sparring partners; the
coach is the one yardstick every run shares.

The control's lesson is about the coach. Run 2 ended with 21 elites that
beat the coach on average over their 8 evaluation matches, the champion 8
of 8. Played on 16 fresh serves each (8 new seeds, both sides), none of the
21 wins more than 8: three draw 8 to 8 with a small positive margin and the
champion wins 7. So "beats the coach: yes" in the summary is selection on
the evaluation seeds, not a coach-beater. What is real: the best of run 2
play the coach even, where run 1's champion lost all 16 fresh matches. Run 3
scores every candidate on 16 coach matches instead of 8, which shrinks the
bias but cannot remove it, so a fresh-seed check is now part of every tick,
and the page should say "beats the coach" only about fresh serves (a bid for
tick 3: show both figures). The same caution applies to the ladder's
promotion seeds in run 5.

Run 3 (20261027T085957Z-night-r3) started at 02:59:57: two hours, seed
20261029, eight seeds per probe, all eight descriptors exported.

The RL species. ppo-1 ended at 02:50 after 13,969 updates and 164.5 million
ticks, still losing 0 to 5 (half a point a match). Sonnet's diagnosis: even
the weakest archive elite is far above a random policy, so a curriculum of
elites would not help; the wall is the sparse reward. The fix is committed
on species-ppo (8034e53): potential-based shaping (get under the ball, then
get it over the net, a small bonus for contact) and a coach that drops half
its actions, both fading to zero over the first 60 per cent of the run (72
minutes), then the plain win-or-lose game. funsies and Fable agreed to one
more two-hour run: ppo-2 started at 03:04:37. It earns a place on the page
only with a win rate above zero against the real coach over 64 matches,
with shaping off for the last 30 minutes. At 25 updates it already takes
2.5 points a match from the softened coach, which says nothing yet about
the real one.

Run 4. Opus's CMA-ME is merged (PR #2): four CMA-ES emitters through pyribs,
16 proposals each a generation, on the same archive, told what each
proposal did to it (new cell, improvement, rejection). Plain MAP-Elites runs
are bit-identical to before. On the pretend game over 60 generations
MAP-Elites filled 215 cells to CMA-ME's 206, so no prior either way. Run 4
shares run 3's seed, so rivals, evaluation seeds and the first random
generations match and only the optimiser differs; compare at equal
generation counts, since CMA-ME adds about 75 ms of Python a generation.

## Tick 3, 05:15: run 3 is in, the coach is unbeaten, the RL try failed, run 4 is live

Run 3 final (20261027T085957Z-night-r3, two hours from 02:59:57, 16 coach
matches per candidate): 24,873 generations, 101.9 million matches at 14,149
a second, 400 of 400 cells by generation 63 (15 seconds), 366 lit, QD-score
2,470, best fitness +2.64 at cell [14, 6], born generation 24,789; all 400
stored matches re-played to the same hash. Half the generations of run 2,
because every candidate plays 64 matches instead of 32.

On 16 graded matches nobody beats the coach: the best elite won 7 of 16
(margin -0.44), and on 16 fresh serves 6 of 16 (-0.44). Run 2's 21
"coach-beaters" were the noise of 8 matches; doubling the sample removed
them, as the tick 2 check predicted. The strongest region is mid court
with hops (+1.84 mean); the weakest is back court kangaroos (+0.45). Lit
and QD-score do not compare with run 2 (different rivals); run 4 shares run
3's seed, so from here the comparison is exact.

The RL species, second try (ppo-2, 03:04 to 05:04, 11,389 updates, 137
million ticks). Against the softened coach with hints it took 1.9 points a
match at ten minutes; as the hints and the softening faded (both zero at 72
minutes) its training score fell to 0.06 and never came back. Against the
real coach it took 0.5 points a match at every evaluation and 0.3 of 5 on
the final 64 matches, win rate zero. Policy entropy sat at 2.04 for two
hours, so the network never committed to a behaviour; Sonnet's 30-minute
signs (training points above 2, entropy below 1.8, evaluation above 1.5)
were all missed. Verdict: the gate is not met, the species is parked for
the night and there is no third trainer. What it taught stays in
species/ppo/DESIGN.md on its branch. Bid for tick 4: merge the branch with
the result written in, since it is tested and touches nothing else, or
leave that to Daniel in the morning.

Run 4 (20261027T110547Z-night-r4-cma) started at 05:05:46: CMA-ME on run
3's seed and rules, two hours. Main now carries Sonnet's fresh-serve check
(PR #3): every export plays the best elite and every graded coach-beater on
8 serves of their own, never used for selection, and writes the result
beside the graded one; run 4 is the first run to carry it. funsies' page
cards say "on the serves it was graded on" (merged here), and run 2's reels
are rendered: then-and-now, a 52-second rally at cell [14, 10] against the
coach, and the champion's 5 to 1 on a graded serve, captioned as such.

Run 5, the ladder's debut, is configured (configs/night/r5-ladder.toml):
run 3's rules on seed 20261031 with the ladder on, a challenge every 10
generations over 100 matches, promotion at 55 per cent plus out-rating the
champion. It starts when run 4 ends, about 07:06.

## Tick 4, 07:15: the coach is beaten, on one seed, by smarter breeding

Run 4 final (20261027T110547Z-night-r4-cma, 05:05:46 to 07:05:46, CMA-ME on
run 3's seed and rules): 26,588 generations, 108.9 million matches at 15,126
a second, 398 of 400 lit, QD-score 3,031, best fitness +4.27 at cell
[12, 10], born generation 23,951; all 400 stored matches re-played to the
same hash. 201 cells beat the coach on their graded serves and 136 of them
on fresh serves; the best elite won 15 of 16 fresh matches with a margin of
+3.75, and 16 of 16 graded. The first graded coach-beaters appeared before
generation 2,000 (about nine minutes); by 4,000 there were 47.

Side by side with run 3 at equal generation counts (same seed, so the same
rivals, evaluation seeds and first random generations; only the optimiser
differs), plain MAP-Elites first, CMA-ME second:

| generation | lit | QD-score | best fitness | graded coach-beaters |
|---|---|---|---|---|
| 1,000 | 248 / 316 | 2,043 / 2,198 | +1.22 / +1.94 | 0 / 0 |
| 4,000 | 319 / 372 | 2,237 / 2,576 | +1.81 / +3.36 | 0 / 47 |
| 12,000 | 347 / 393 | 2,371 / 2,865 | +2.20 / +3.89 | 0 / 135 |
| 24,000 | 366 / 398 | 2,462 / 3,010 | +2.50 / +4.27 | 0 / 195 |

CMA-ME led on every figure at every count, and ran faster per generation
(271 ms against 289) despite its 75 ms of Python, because stronger bots end
points sooner. The claim, agreed with funsies: a beat on fresh serves on one
seed; run 5 is the second seed. Why it works, in one sentence: plain
MAP-Elites mutates and crosses elites at a fixed step, while each CMA-ME
emitter adapts its step and direction to what improved the archive and
restarts from a fresh elite when it stalls; the champion's line of descent
is 77 genomes long, every step a CMA-ME proposal.

Decisions:
- Run 5 restarted. It began at 07:06:53 with plain MAP-Elites and the
  ladder on, as planned at tick 1; at generation 256 nothing was lit (with
  the ladder on, fitness is the margin against the ladder's members, which
  at era 0 is the coach alone). Three runs of plain MAP-Elites never beat
  the coach, so the ladder would most likely have shown no promotion in two
  hours. funsies and Fable agreed to stop it (20261027T130654Z-night-r5-ladder,
  abandoned) and restart with CMA-ME plus the ladder on the same seed
  20261031: 20261027T130856Z-night-r5-ladder, 07:08:55 to 09:08:55. It is
  the second CMA-ME seed and the first time promotions, eras, re-scoring and
  the cycle check run for real.
- Merged into main: the RL species branch with its negative result written
  into species/ppo/DESIGN.md (funsies: a dead end that is written down is
  worth keeping), and funsies' cards (331e825): each style shows its
  fresh-serve verdict, the only line allowed to say "beats the coach". Run
  3's reels are rendering.
- The "Smarter Breeding" level is cleared and the first point goes on the
  smart-breeding skill. Tick 5, the morning summary, comes ten minutes
  after run 5 ends.

## Tick 5, 09:15: the beat repeats on a second seed, and the ladder tells on itself

Run 5 is in: 20261027T130856Z-night-r5-ladder, 07:08:55 to 09:08:55, seed
20261031, CMA-ME with the ladder on. 16,505 generations, 105.8 million
matches at 14,695 a second; all 400 cells filled by generation 56; all 400
stored matches re-played to the same hash. Fewer generations than run 4
because every candidate also plays each hall-of-fame member: 64 matches a
candidate at era 0, 224 by era 10.

The coach is beaten again. The champion, cell [16, 10], born generation
14,889, won 96 of 100 against the coach on the ladder's 50 seeds played on
both sides, 16 of 16 on its graded serves and 13 of 16 on fresh serves
(margin +3.00). Two CMA-ME seeds, two coach-beaters; three plain MAP-Elites
runs, none. The breadth is not run 4's: 8 cells beat the coach on their
graded serves and 4 of them on fresh serves, against 201 and 136 in run 4.
The reason is the fitness definition, not the seed: with the ladder on,
fitness is the margin against the ladder's members, so since era 1 the
archive has been selected for beating the champions, and the coach is one
member of eleven. Run 5's lit count (293), QD-score (2,163) and best fitness
are margins against that ladder and do not compare with runs 1 to 4, where
fitness was the margin against the probe set.

Two lessons from the ladder's first outing.

First, the coach alone is a cliff. At era 0 the ladder is the coach, so
every candidate's fitness was its margin against the coach: negative for
everyone, nothing lit for 8,949 generations (best -2.38 at generation 1,000,
-1.38 at 4,000, -0.12 at 8,000). In run 4 fitness was the margin against the
probe set, three frozen random rivals plus the coach, which gives a gradient
from the first generation. Once the ladder had a champion, fitness had a
gradient again and nine more promotions followed within 40 minutes.

Second, the rating climbed while the champions got worse. Ten promotions,
each the fittest elite winning at least 55 per cent of 100 matches against
the champion and out-rating it; the share against the coach is from the
same ladder seeds:

| champion (era) | promoted at generation | share against the previous champion | share against the coach | rating |
|---|---|---|---|---|
| 1 | 8,949 | 0.56 (the coach was the champion) | 0.56 | 1,041 |
| 2 | 11,529 | 0.89 | 0.38 | 1,116 |
| 3 | 11,999 | 0.55 | 0.41 | 1,135 |
| 4 | 12,409 | 0.66 | 0.33 | 1,204 |
| 5 | 12,939 | 0.78 | 0.51 | 1,319 |
| 6 | 13,099 | 0.97 | 0.00 | 1,325 |
| 7 | 13,629 | 0.695 | 0.00 | 1,405 |
| 8 | 14,449 | 0.89 | 0.00 | 1,453 |
| 9 | 14,459 | 0.665 | 0.00 | 1,489 |
| 10 | 14,889 | 0.74 | 0.96 | 1,645 |

Champions 6 to 9 beat their predecessors by 69 to 97 per cent and lost 100
of 100 to the coach. The rule promotes a challenger that wins the
head-to-head and whose Elo rating, fitted against every member's frozen
rating, exceeds the champion's; beating five champions rated 1,041 to 1,319
outweighs losing every match to the 1,000-rated coach, and Elo assumes
transitivity, so the fit hides the cycle. The rating is the line the page
would have plotted; the who-beats-whom matrix is what caught it, which is
what it was built for. Champion 10 then beat every member (the coach 96 per
cent, the past champions 59 to 99) and held through the last 1,616
generations of checks.

The night in one paragraph. Six starts, five runs finished: run 1 (one hour,
plain MAP-Elites), run 2 (the two-hour control), run 3 (twice the coach
matches a candidate), run 4 (CMA-ME on run 3's seed), run 5 (CMA-ME with
the ladder, a fresh seed); run 5's first start, plain MAP-Elites with the
ladder, was stopped at generation 256; two reinforcement-learning trainers
ran and the species is parked with its result written in. Plain MAP-Elites
never beat the coach in three runs; CMA-ME did on both seeds it ran. The
fresh-serve check became the only line allowed to say "beats the coach".
The ladder's first outing found the rating's blind spot before anyone
plotted it.

Decisions, and what waits for the morning:
- The promotion rule (Daniel with Opus): a challenger must also hold the
  anchor, winning at least the threshold against the coach as well as the
  head-to-head, or beat every member. funsies and Fable recommend holding
  the anchor and a two-hour run 6 to test it; nothing changed tonight.
- Fitness at era 0: keep the probe margin in the fitness, or seed the ladder
  with the three frozen probes, so the map has a gradient from generation 0
  and lit counts compare across runs. Opus's call on the mechanism.
- The pitch page (funsies): two seeds, two coach-beaters, by different
  routes. Run 4's 136 styles and 15 of 16 stay run 4's figures; run 5's are
  the champion's 96 of 100 and 13 of 16, and four styles on fresh serves.
- The summary tool was written before eras existed: its lit timeline counts
  across re-scores (493 of 400 cells at the end) and its regions give 297
  lit against the export's 293. For Sonnet in the morning.
- The "Promotions" level is cleared and the honest-scores skill goes up one.
  "The Hall of Fame" stays in practice: the scores went up when the bots got
  better only against each other.
- Daniel's gates, unchanged: the public address
  (play.thegoodface.ai/darwin-beach/ as its own Worker, not deployed), a
  storage token scoped to the arena bucket (the stored keys are refused for
  it), and the pitch page's wording. Done without a gate and reversible:
  the RL species merged with its negative result, run 5's first start
  abandoned.
