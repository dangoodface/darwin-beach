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

## Afternoon, 27 Oct, 15:30: seven specialists, by request, and nobody loses

Daniel woke, read the night, and asked for specialists: bots trained hard for
one style each, with a rule that makes the style, and a twin trained the same
way without the rule so the rule's cost shows. He picked six of twenty
proposals and added a seventh of his own.

| style | the rule | twin |
|---|---|---|
| flat earther | never jumps: the jump output is wired off | yes |
| kangaroo | always jumps: the jump output is wired on | yes |
| long ball | hits hard and low: ball speed after a hit at least 34.8, hit height at most 3.83 | yes |
| lob artist | hits soft and high: speed at most 25.8, hit height at least 5.47 | yes |
| sprinter | ends points fast: a penalty for every tick a point lasts | yes |
| all-rounder | fitness is the smallest of its three margins, so it cannot specialise against one rival | no, by design |
| joga bonito (Daniel's) | at least a third of its points won after four or more consecutive own touches spanning half its court | yes |

Built for it before lunch and merged on main at ceeb456: kernel version 2
(Opus, PR #4), which counts juggled points and touch streaks as two new
descriptors, physics and hashes unchanged, so run 5's replays still verify;
and the specialist trainer (Sonnet, PR #5), an evolution strategy rather
than a map: 64 nudged copies of one starting bot in antithetic pairs, Adam
on the estimated gradient, 8 training serves resampled every generation, 16
graded serves every fifth generation, and 8 held-out fresh serves for the
verdict. The rivals are run 5's champion, run 4's champion and the coach, on
both sides. Window styles pay a penalty outside their window, pinned styles
have the output wired, the sprinter and joga bonito carry their rule in the
fitness. Each specialist starts from the best lit cell of run 5 (or run 4)
that fits its window, or from run 5's champion when no window applies.

The queue ran from 12:07 to 15:27 on 15 workers, twenty minutes per
specialist and ten per twin: 24,404 generations and 75.4 million matches,
6,285 a second (a generation is 64 candidates by 8 serves by 3 rivals by 2
sides, plus grading). Fresh serves below are 16 per rival, never seen in
training; the hall-of-fame column is 8 new seeds on both sides against each
of run 5's eleven members, era 0 the coach, era 10 the champion.

| specialist | run | fresh margin vs run 5 champion / run 4 champion / coach | the rule, as played on fresh serves | hall of fame: members beaten, weakest score |
|---|---|---|---|---|
| flat earther | 180711Z | +4.56 / +4.81 / +4.31 | 0 jumps in every match | 11 of 11, 16 of 16 against all |
| its twin | 182712Z | +4.81 / +4.69 / +4.88 | 20 jumps a match | 11 of 11, 15 of 16 (era 7) |
| all-rounder | 183713Z | +4.88 / +4.88 / +4.81 | smallest margin +4.81 | 11 of 11, 14 of 16 (era 8) |
| kangaroo | 185713Z | +3.56 / +4.88 / +4.50 | 81 jumps a match, airborne 97 per cent of ticks | 11 of 11, 14 of 16 (eras 8, 9) |
| its twin | 191714Z | +3.75 / +4.44 / +4.44 | 82 jumps a match | 11 of 11, 14 of 16 (era 0) |
| sprinter | 192716Z | +5.00 / +4.94 / +4.94 | 329 ticks a point, 15 hits a match, ball speed 34 after a hit | 11 of 11, 16 of 16 against all |
| its twin | 194716Z | +4.81 / +4.88 / +4.69 | 388 ticks a point, 22 hits, speed 27 | 11 of 11, 15 of 16 (eras 6, 7, 9) |
| long ball | 195717Z | +4.94 / +4.88 / +4.69 | speed 37.1, hit height 3.81, airborne 22 per cent | 11 of 11, 15 of 16 (era 7) |
| its twin | 201718Z | +4.69 / +4.75 / +4.69 | speed 33.5, hit height 3.97, airborne 36 per cent | 11 of 11, 12 of 16 (era 9) |
| lob artist | 202719Z | +4.88 / +5.00 / +4.69 | speed 23.2, hit height 6.05 | 11 of 11, 15 of 16 (eras 7, 8) |
| its twin | 204719Z | +4.81 / +5.00 / +4.81 | speed 32.4, hit height 5.94 | 11 of 11, 14 of 16 (era 9) |
| joga bonito | 205720Z | +4.75 / +4.94 / +4.25 | 2.25 of 4.73 points a match juggled (48 per cent), longest run 6.9 touches | 11 of 11, 14 of 16 (era 9) |
| its twin | 211721Z | +4.81 / +4.88 / +4.69 | 0.25 points juggled (5 per cent) | 11 of 11, 15 of 16 (eras 6, 7, 9) |

Run ids are `20261027T<time>-spec-<style>[-twin]` under
`$ARENA_DATA/specialists/`, each with `result.json` (the contract is in
docs/DATA.md), the winning genome, the kernel it ran on and `hof.json` from
the generality check.

Three things the afternoon says.

First, nobody loses, so strength no longer tells the styles apart. The
margin is out of 5, five points a match; a bot nudged for twenty minutes from
the champion against three fixed rivals reaches +4.8 or better against each,
and the never-jumper, the worst placed by its rule, reaches +4.3 against the
coach. The same bots beat all eleven hall-of-fame members on serves nobody
trained on; the weakest score in the whole table is the long ball twin's 12
of 16 against the era 9 champion. The members that take a match or two are
champions 7 to 9, the ones the coach beat 100 of 100: the night's cycle is
still there. The rivals chosen at noon were the strongest bots in the house;
by 15:30 every specialist is stronger. The next ladder needs them in it.

Second, the rules bite where they should, and the twins show what they cost.
The flat earther wins without a single jump while its twin learns to jump 20
times a match; the sprinter's points are 15 per cent shorter than its twin's,
with fewer touches and a harder ball; the long ball hits at 37 against its
twin's 33, the lob artist at 23 against 32; the juggler wins half its points
after a four-touch run across its half, its twin one in twenty. Measured
against the unconstrained nudge from the champion (+4.81 against the run 5
champion), the never-jumper gives up 0.25 of margin and the kangaroo 1.25,
the largest cost in the table; its points last twice as long, 639 ticks. The
kangaroo's twin is not a contrast: both started from the same high-airtime
elite at cell [14, 19] and the twin kept jumping, so that pair shows the
start cell, not the rule.

Third, the same file gives the same bot. The sprinter's twin and joga
bonito's twin are the same experiment, an unconstrained nudge from run 5's
champion under the same seed, and they produced the same genome byte for
byte (9e1c72c0, 2,060 bytes); one ran two generations longer because its ten
minutes went a little faster, and the best came before either cut-off. Ten
minutes of compute were spent twice; the page should show that bot once.

For the trainer, for Sonnet and Opus:
- Twins of champion-started styles are one run: cache by start, rule and
  seed, or skip the duplicate.
- The verdict rivals are too weak to rank specialists now. Rank them against
  each other, a thirteen-bot round robin on fresh serves, and put the winners
  in the ladder for run 6.
- The fresh-serve margin is nearly saturated at +5; a harder verdict is wins
  against the specialists, or longer matches.

The campaign page: "The Specialist" level cleared (its test was a specialist
beating its neighbours; these beat every rival and every champion), and the
"Learning by nudges" skill goes up one.

## Evening, 27 Oct, 22:30: run 6, the whole pool as the ladder, and no definitive player

Daniel, through funsies: "New round of training, ask fable to improve the
greater cohorts we now have the champs and the coach and several specialists
to beat. With elo system create the volleyball definitive player. Like the
same millions and millions of matches engine of evolution. But better."

**The design (commit 0bf1322).** Run 5's engine with three changes. The
ladder starts as a pool of every bot we have, 23 of them (run 5's ten
champions, run 4's, the twelve specialists and twins) plus the coach, rated
once by a round robin on the 50 ladder seeds with the coach fixed at 1000 and
the rest fitted jointly (`joint_elo`). Fitness weights each member's margin
by its rating, `10 ** ((rating - top) / 800)`, so the specialists are what
counts. A promotion must beat the top-rated member head to head, out-rate it
against the whole pool, and hold 55 per cent against the coach
(`hold_anchor`): the night's lesson, written into the rule. The map starts
full of run 4's and run 5's elites instead of random brains (362 of 400 cells
at generation 0). A verdict tool, `python -m arena.definitive`, plays the
field again on serves the run never drew. Six new tests, 130 passing.

**The pool, rated on the run's own seeds.** Flat earther twin 1876, lob
artist 1815, all-rounder and sprinter 1762, long ball 1760, sprinter twin
1740, flat earther 1720, joga bonito 1711, kangaroo twin 1708, lob artist
twin 1704, kangaroo 1691, long ball twin 1639; then a gap: run 5 champion #10
at 1213, #9 1140, #8 1074, #7 1005, the coach 1000, and six older champions
plus run 4's below the coach, down to 516.

**The run (`20261028T001309Z-run6-definitive`).** Four hours, 8,034
rounds, 111.3 million matches at 7,727 a second, twelve workers. The map was
full by round 8,000 (400 of 400). Cells that beat the pool on weighted
average went 0 at round 1,000, 6 at 2,000, 29 at 4,000, 77 at the end;
QD-score 580 to 1,626. Fifty-three cells beat the coach on their grading
matches, 34 of the 54 checked on fresh serves. Twenty-six map-bests
challenged the twin (100 seeded matches each, a tie counts half). Three won
the head to head, the best 60 per cent at round 1,559 (39 wins, 42 ties, 19
losses); none out-rated the twin against all 24 members, so no promotion and
no new era. Archive ratings of the best cells: 1,590 to 1,708, above every
run 5 champion by 400 points or more.

**The exploit.** The best-scoring cell of the whole run (round 7,809,
fitness 0.68) loses every fresh-serve match to the coach, 0 of 16, and 0 of
64 in the verdict, and 3 of 64 to the twin; it beats the specialists 55 per
cent on average. The weights did it: the coach at 1000 weighs 0.08 against
the twin's 1.0, so the fitness could ignore it. The gate held (that bot never
passed the head to head), but the search spent its last thousand rounds on
it. A rating-weighted fitness needs a floor under the anchor.

**The verdict** (32 held-out seeds, 64 matches a pair, 30 entrants: the 24
members plus the run's top six cells by archive rating, 435 pairs). Flat
earther twin 1566, lob artist 1545, all-rounder 1497, sprinter 1495, sprinter
twin 1484, long ball 1478, lob artist twin 1462, joga bonito 1453, run 6
elite #2 (the exploit) 1449, kangaroo 1448, flat earther 1438, kangaroo twin
1437, four more run 6 elites 1416 to 1407, elite #6 1370, long ball twin
1344, run 5 champion #10 1075, #9 1003, the coach 1000, the rest of run 5 and
run 4's champion below the coach, run 5 champion #1 last at 482. The scale
is compressed (1876 became 1566: other serves, a bigger field), the order
holds.

**Cycles, again.** The twin beats 26 of 29 (more wins than losses; a tie
counts half, the ladder's own definition). It loses to the all-rounder (2
of 64) and the lob artist (3 of 64), and run 6's elite #6, ranked 17th,
beats it 43 of 64. The lob artist beats 27 of 29, more than anyone, and has
the best worst matchup (49 per cent, against its own twin). Run 6's elite #1
(round 6,929) is the honest one: beats the coach 63 of 64, ranks 16th.

**So, the definitive player.** There is none in this field. By Elo it is
still the flat earther twin; by robustness, the lob artist; no bot beats
everyone, and the pool's own cycles mean "beats everyone" may not exist.
What run 6 produced is a map of 400 bots that all out-rate every past
champion and mostly beat the coach, and a rule that would not promote a
pretender.

Lessons for a run 7, if Daniel wants one: a floor on the coach's weight, or
the coach as a hard constraint on fitness; a promotion bar stated as
"out-rate the specialists as a group" is four hours of CMA-ME away from a
warm start, so give it a night; ties are a third of the twin's matches,
so longer matches or a tie-break would sharpen every verdict; and define
"definitive" before training it: best worst matchup is a target, Elo mean is
not.

Data: `out/20261028T001309Z-run6-definitive/` (latest.json, ladder.json,
curve.json, genomes) and `definitive.json` beside them (leaderboard, the
twin's record, the full matrix). The campaign page: "Honest scores" goes up
one, three log lines.

## Night, 28 Oct, 04:00: round 7, the go, and the oracle

Daniel, at 01:51, through funsies: "Okay set up crons with fable every 60 min
for 6 hours. Same protocol of yesterday you and fable coordinate if tied opus
does the break. Explore whatever you want, be creative, do not limit to
current scope. Be you :) good night." The plan he approved: track A, more
specialists (four assassins, four gap styles, One-Touch, Lazy, Smooth
Operator with a 9-tick hold, and his La Clavada), each with a twin where it
teaches something; track B, the league on a bigger brain, target 2500.
funsies syncs hourly from 02:53; Opus breaks ties.

**The twin patch test (02:03, 20 minutes).** funsies' idea: evolve from the
twin with the lob artist weighted three to one and see whether it keeps its
other wins. It did more than that. The result beats the lob artist 11 of 16
on fresh serves, the all-rounder 15 of 16, the sprinter 15 of 16, the coach
16 of 16, and on the official board it rates 2183, measurable, beating all
24 members. The twin's weakness was patchable in twenty minutes. It is the
board's first member above 2000.

**The official board** (`python -m arena.official`). A frozen ladder: the 23
members of run 6 at their ratings, then every promotion-grade bot at its
measured rating; 64 official serves no trainer ever draws, both sides, 6,000
ticks; a candidate's rating is the Elo that best explains its record against
the frozen members, the coach at 1000; it counts only while the candidate
scores under 90 per cent against at least three members, else it is "above
the ladder". The twin itself checks out at 1879 against the run 6 scale's
1876.

**The tick cap.** The first 6,000-tick rating came back as all ties: the
kernel refused anything over 3,000 ticks (`MAX_TICKS_CAP`). Worth knowing for
the record: at 3,000 ticks the twin against the lob artist ends 1-1 or 0-0
at the limit on most seeds. The top bots do not miss; a third of their
matches are stalemates. Kernel v3 raised the cap to 12,000.

**Kernel v3 (Opus, 02:30).** One version bump: a per-side action hold (the
brain decides every `hold` ticks and repeats in between; Smooth Operator's
constraint), a 64-wide hidden layer (515 to 1,027 parameters) with the old
units in the first 32 slots and the sums in the same order, so a run 6
genome padded with zeros plays a bit-identical match (1,000 of 1,000, every
descriptor), SIMD pairs of units at 1.11 times v2's cost on the training
mix (plain 64 cost 1.57, width 96 was 2.22 and is out), the cap, and an
eleventh descriptor, spiked_points, for La Clavada. Replay records carry
the holds; `tools/verify_specialist.py` replays a specialist's fresh record
from its own files.

**The assassins (02:22 to 03:42).** One bot each, trained twenty minutes
against a single target weighted three to one, with the coach as the gate.
Twin assassin: beats the twin 128 to 0 on the official serves and the coach
128 to 0, rates 1759 (1837 once the oracle rungs joined the board), and
loses to the all-rounder it started from, both sprinters, the lob artist
and joga bonito: narrow by design, a sparring partner. Lob artist assassin
1699. Sprinter assassin 1874. The all-rounder's assassin went 6 of 16
against its target: the all-rounder survived its killer, which says more
about the all-rounder than about the method.

**The recipe fix (02:00).** Rivals are the pool's top four plus the coach
as a hard gate (a candidate that loses to the coach scores that margin minus
ten, so it ranks below every candidate that holds), rivals and start genomes
can be files with weights, a win and stall verdict, and the styles spiker,
dinker (the two quadrants of hit height against speed that the long ball and
the lob artist did not cover; the driver and floater of the plan were those
two under other names), one_touch, lazy (a soft cap on travel per tick at
half the start's, the gate as the stall guard), assassin, clavada and
smooth_operator.

**The league code (02:30 to 03:30).** Fitness "softmin": a smooth minimum
of the mean margins against the eight highest-rated members, so the worst
matchup is what improves, with the coach gate; ladder matches at 6,000
ticks; newcomers from `runs/<id>/inbox`: genomes trained beside the run
join the ladder rated on the ladder seeds and the archive is re-scored
(exploiters, every 45 minutes, trained ten minutes against the current best
cell); padding of v2 genomes through the kernel's helper on load.

**The oracle (Opus, 03:10, plan B4).** Asked for an exploration, Opus built
a scripted bot that plans with the game's own arithmetic: for each incoming
ball it plays out about a thousand ways of meeting it, scores each return by
the opponent's chances to reach it, replans every 8 ticks, and waits where
the opponent's options hurt least. Handicap rungs add a perception delay or
aim noise. On the official board: rung 0 (no delay) 2220, beating all 25
members, the twin patched 127 to 1; delay 1, 2133; delay 2, 2021; delay 4,
1901; delay 8, 1747; delay 16, 1444; aim noise half a step at a 2-tick
delay, 1153; a full step, 868. Delay is a smooth handicap, about 100 Elo a
tick at first; noise is a cliff. Rung 0 beats everything head to head yet
rates only 37 above the twin patched, held down by the net blockers (lob
artist twin 0.63): the board's one number is non-transitive at the top. An
oracle side costs about thirty times a genome match and keeps state through
a match, so it cannot be replayed tick by tick. Decisions: rungs 0, 2 and 4
are board members at those ratings (27 members, fixed marks from 868 to
2220 for the first time); rungs 2 and 4 are promotion-only members of the
league's ladder (rated, faced in promotion checks, never played by the
candidates); rung 0 stays off the league, because a champion that must beat
the no-delay oracle head to head would never be promoted tonight. The next
oracle step, for another day: a second ply for the jumping blocker. Opus
stood down after that to spare the shared budget.

**The rehearsal (03:52).** The league config on the oracle build, 150
seconds on four loaded workers: 31 members rated in the opening round robin
(rungs 2 and 4 at 2052 and 1893, the twin patched on top at 2140), the
candidates playing 256 matches each without the rungs, no genome file
written for a rung, the store and the export carrying both kinds. The
league starts when the two specialist queues end, about 08:10, twelve
hours, from the main checkout.

Timeline: queue A (assassins, then spiker, dinker, One-Touch, Lazy with
twins) until about 06:45; queue 7b on v3 (Smooth Operator at hold 9, La
Clavada, with twins) until about 08:10; the league after. The all-rounder
and the lob artist still beat the twin; the twin patched does not lose to
anyone on the board but rung 0.

## Evening, 28 Oct, 19:30: the league, the crash, and what the night was worth

**The league** (`20261028T130929Z-run7-league`, then `20261028T161600Z-run7-league`).
Started 07:09 on the oracle build with 34 members: run 6's pool, the twin
patched, the one-touch and lazy twins, Lazy, four assassins, and the 2-tick
and 4-tick oracle rungs as promotion-only members. Fitness the soft minimum
over the eight highest-rated members with the coach gate; ladder matches at
6,000 ticks; an exploiter trained every 45 minutes against the best cell.

At 10:06 it died: the first exploiter to beat its target joined the ladder,
the store wrote its member row but not its genome, and the exporter crashed
on the missing parameters at round 3,329. The inbox test had passed because
its newcomer was a genome the shared test store already held. Fixed in
063746d (the store keeps a newcomer's genome, the exporter survives a
missing one and says so, the test drops a genome no store has seen) and
relaunched at 10:16, warm from the dead run's last export, with that
exploiter in the pool. Three exploiters joined the relaunched run (1788,
1793, 1899 on the run's scale); era 3.

Ended 19:16: 8,024 rounds, 157 million matches at 4,854 a second (6,000-tick
ladder matches and the exploiters' CPU), 400 of 400 cells, 73 lit (beating
the worst of their eight hardest matchups on average, up from 1 and still
rising), 252 cells beating the coach. Eighteen promotion checks, all lost,
the best 49 per cent against the one-touch twin. No promotion.

**The verdicts.** Official board: the league's best cells 1819, 1777, 1755,
beating 16 to 18 of 31; the twenty-minute recipe's bots 2183 (twin
patched), 2128 (one-touch twin), 2091 (lazy twin), 2054 (La Clavada), 2020
(Lazy). Held-out round robin of the run, 44 entrants: one-touch twin 1885
first, beating 41 of 43; lazy twin, twin patched, the 2-tick oracle, Lazy,
the 4-tick oracle; the league's elites 13th and 14th.

**What the night was worth.** Not 2500, and not a league champion. The
honest list: an official board with fixed marks from 868 to 2220 (the
oracle rungs) and 30 members; kernel v3 with holds and a brain twice as
wide that plays old genomes bit for bit; the oracle itself, a 2220 nobody
trained; four bots above 2000 from twenty minutes of nudges each, the twin
patched first among them; the knowledge that the map engine fills a map of
400 strong bots and does not catch the top of a ladder in a day, that the
twin recipe does in twenty minutes, that dinking loses and a 9-tick hold is
a crushing handicap, that the all-rounder survives its own assassin, and
that twins of styles that pick the same start are one bot.

**Next, Daniel's call.** The chain (`run7/chain.sh`): twenty minutes from
the board's best against the board's top eight, rate on the official
serves, add if measurable and higher, repeat. Smooth Operator at a 3- or
5-tick hold. Opus's second ply for the oracle's jumping-blocker weakness,
as a new build with its own rungs. A league that trains the exploiters
against the ladder's top rather than the map's best cell.
