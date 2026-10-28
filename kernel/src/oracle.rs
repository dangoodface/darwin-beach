//! The oracle (exploration, plan B4): a scripted bot that plans with the game's own
//! physics. When the ball is coming to its side it tries every way of meeting it from a
//! small family (walk to a spot, wait, then stand, run or jump into it), plays each one
//! through with the same arithmetic as the game, and judges the return by the opponent's
//! chances to touch it: a reach envelope (running from our touch, jumping, from where the
//! opponent is now) is checked against every tick of the return's flight on the other
//! side, and each chance is weighed by a danger map, worked out once from the physics:
//! how fast an answer from that spot could land where we could not reach. A return out of
//! reach beats any other. It plans again every few ticks while the ball is on its way, and
//! while the ball is the other side's business it waits where the opponent's chances on
//! the ball's path hurt least. Handicap rungs make a ladder of known strengths:
//!
//! - a perception delay: it sees the ball and the opponent as they were `delay` ticks
//!   ago and brings the ball up to now with its own (known) movements, so it is late
//!   only for the unexpected: the opponent's touches;
//! - aim noise: the spot it walks to is off by up to `noise`, drawn once per incoming
//!   ball from its own generator (seeded from the match seed and side; the serve's
//!   PCG32 is never touched).
//!
//! It reads only its side's mirrored observation (own half x > 0, net at 0), where the
//! court is symmetric, so one oracle plays either side.

use crate::physics::{
    BACKWARD, BALL_RADIUS, Body, FORWARD, GRAVITY, INIT_DELAY_FRAMES, JUMP, MAX_BALL_SPEED, NUDGE,
    PLAYER_SPEED_X, PLAYER_SPEED_Y, Particle, REF_U, REF_W, REF_WALL_HEIGHT, REF_WALL_WIDTH,
    SLIME_RADIUS, TIMESTEP, sqrt,
};
use crate::rng::Pcg32;

#[derive(Clone, Copy)]
pub struct Rung {
    pub delay: u32,
    pub noise: f64,
}

const fn rung(delay: u32, noise: f64) -> Rung {
    Rung { delay, noise }
}

/// The rungs, by slot: `oracle_slot_base() + i` plays rung i. Delays of 0 to 16 ticks,
/// then aim noise (a spot can only shift by whole steps, so noise under half a step,
/// about 0.29, changes nothing).
pub const RUNGS: [Rung; 11] = [
    rung(0, 0.0),
    rung(1, 0.0),
    rung(2, 0.0),
    rung(3, 0.0),
    rung(4, 0.0),
    rung(6, 0.0),
    rung(8, 0.0),
    rung(12, 0.0),
    rung(16, 0.0),
    rung(2, 0.5),
    rung(2, 1.0),
];

const HISTORY: usize = 17; // delays up to 16
const PLAN_LEN: usize = 160; // ticks it plays ahead (5.3 s)
/// Where it may wait while the ball is the other side's business, and where it heads
/// after a hit: home.
const WAIT_AT: [f64; 11] = [3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0];
const HOME: usize = 6;
const HOME_X: f64 = WAIT_AT[HOME];
const SERVE_Y: f64 = REF_W / 4.0; // where a serve hangs before it drops
const NEAR_NET: f64 = REF_WALL_WIDTH / 2.0 + SLIME_RADIUS; // 2.0, the own-half clamp
const BACK_WALL: f64 = REF_W / 2.0 - SLIME_RADIUS; // 22.5
const STEP: f64 = PLAYER_SPEED_X * TIMESTEP; // how far a slime moves in a tick
const TOUCH: f64 = SLIME_RADIUS + BALL_RADIUS; // centre distance at a touch
const JUMP_TICKS: usize = 28; // ticks a jump spends in the air
/// No slime, however it jumps, touches a ball whose centre is higher than this.
const REACH_Y: f64 = TOP[JUMP_TICKS] + TOUCH;
const RUN_LEAD: usize = 12; // the longest run-up into the ball it tries
const BRANCHES: [u32; 5] = [FORWARD, BACKWARD, JUMP, JUMP | FORWARD, JUMP | BACKWARD];
const LOST: f64 = -1000.0; // the ball comes down on its own side
const SHORT: f64 = -500.0; // the ball is still on its side when the plan runs out
const SAFE: f64 = 50.0; // a return the opponent cannot reach scores this
const SPARE: f64 = 0.3; // what each tick the opponent has to spare adds to the danger
/// While the ball is on its way it plans again this often, so the opponent it plays
/// against is where the opponent is now.
const REPLAN_EVERY: usize = 8;
/// The danger map's grid over where the opponent might touch our return (our frame):
/// x from -22.5 to 0 and height from 2 to 7.5, in half units.
const DANGER_X: usize = 46;
const DANGER_Y: usize = 12;
/// Slopes of the answers it fears, from steeply down to a high lob (the kernel keeps to
/// + - * / and sqrt, so no angles).
const SLOPES: [f64; 11] = [-1.5, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.5, 2.2, 3.5];
const SEEN: usize = 4096; // contacts remembered per plan, so each is played out once

/// A slime's centre height through a jump from the ground (the jump held on the first
/// tick only): `Agent::update`'s arithmetic, so it matches the game.
const JUMP_HEIGHT: [f64; JUMP_TICKS + 1] = {
    let mut h = [REF_U; JUMP_TICKS + 1];
    let (mut y, mut vy) = (REF_U, 0.0);
    let mut i = 1;
    while i <= JUMP_TICKS {
        vy += GRAVITY * TIMESTEP;
        if y <= REF_U + NUDGE * TIMESTEP {
            vy = if i == 1 { PLAYER_SPEED_Y } else { 0.0 };
        }
        y += vy * TIMESTEP;
        if y <= REF_U {
            y = REF_U;
            vy = 0.0;
        }
        h[i] = y;
        i += 1;
    }
    h
};

/// The highest a slime's centre can be `tau` ticks from now, from the ground.
const TOP: [f64; JUMP_TICKS + 1] = {
    let mut t = JUMP_HEIGHT;
    let mut i = 1;
    while i <= JUMP_TICKS {
        if t[i - 1] > t[i] {
            t[i] = t[i - 1];
        }
        i += 1;
    }
    t
};

/// A slime in the mirrored frame (dir +1: own half x > 0, forward is -x).
#[derive(Clone, Copy)]
struct Slime {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

impl Slime {
    fn from_obs(o: &[f64; 12]) -> Slime {
        Slime {
            x: o[0],
            y: o[1],
            vx: o[2],
            vy: o[3],
        }
    }

    /// `Agent::set_action` then `Agent::update` with dir +1: the same operations, so it
    /// ends exactly where the game puts a slime (mirroring is exact in IEEE doubles).
    fn step(&mut self, action: u32) {
        let (forward, backward) = (action & FORWARD != 0, action & BACKWARD != 0);
        let desired_vx = if forward && !backward {
            -PLAYER_SPEED_X
        } else if backward && !forward {
            PLAYER_SPEED_X
        } else {
            0.0
        };
        let desired_vy = if action & JUMP != 0 {
            PLAYER_SPEED_Y
        } else {
            0.0
        };
        self.vy += GRAVITY * TIMESTEP;
        if self.y <= REF_U + NUDGE * TIMESTEP {
            self.vy = desired_vy;
        }
        self.vx = desired_vx;
        self.x += self.vx * TIMESTEP;
        self.y += self.vy * TIMESTEP;
        if self.y <= REF_U {
            self.y = REF_U;
            self.vy = 0.0;
        }
        if self.x <= NEAR_NET {
            self.vx = 0.0;
            self.x = NEAR_NET;
        }
        if self.x >= BACK_WALL {
            self.vx = 0.0;
            self.x = BACK_WALL;
        }
    }

    const fn body(&self) -> Body {
        Body {
            x: self.x,
            y: self.y,
            vx: self.vx,
            vy: self.vy,
            r: SLIME_RADIUS,
        }
    }
}

/// Towards `x`, stopping within half a step: how it waits and how it leaves after a hit.
fn toward(me: &Slime, x: f64) -> u32 {
    if me.x > x + STEP / 2.0 {
        FORWARD
    } else if me.x < x - STEP / 2.0 {
        BACKWARD
    } else {
        0
    }
}

fn stub() -> Body {
    Particle::new(0.0, REF_WALL_HEIGHT, 0.0, 0.0, REF_WALL_WIDTH / 2.0).body()
}

fn state(b: &Particle) -> [f64; 4] {
    [b.x, b.y, b.vx, b.vy]
}

/// Ticks a slime at `from` (our frame), whose centre can be anywhere from `low` to `high`
/// and which is confined to [`left`, `right`], still lacks to touch the ball at `b` with
/// `tau` ticks to run: negative when it gets there in time, infinite when it could not
/// touch the ball there at all.
fn reach_lack(
    b: &Particle,
    from: f64,
    (low, high): (f64, f64),
    (left, right): (f64, f64),
    tau: usize,
) -> f64 {
    let dy = if b.y > high {
        b.y - high
    } else if b.y < low {
        low - b.y
    } else {
        0.0
    };
    if dy >= TOUCH {
        return f64::INFINITY;
    }
    let w = sqrt(TOUCH * TOUCH - dy * dy);
    let lo = (b.x - w).max(left);
    let hi = (b.x + w).min(right);
    if lo > hi {
        return f64::INFINITY;
    }
    let travel = if from < lo {
        lo - from
    } else if from > hi {
        from - hi
    } else {
        0.0
    };
    travel / STEP - tau as f64
}

type DangerMap = [[f64; DANGER_Y]; DANGER_X];

/// How badly the opponent can hurt us by touching the ball at each spot of the grid:
/// over answers at full speed along `SLOPES`, the most ticks we would lack, waiting at
/// `wait`, to touch the answer before it lands (above 0 it wins the point outright).
fn danger_map(wait: f64) -> DangerMap {
    let stub = stub();
    let mut map = [[f64::NEG_INFINITY; DANGER_Y]; DANGER_X];
    for (ix, column) in map.iter_mut().enumerate() {
        for (iy, cell) in column.iter_mut().enumerate() {
            for s in SLOPES {
                let speed = MAX_BALL_SPEED / sqrt(1.0 + s * s);
                let (x, y) = (-22.5 + 0.5 * ix as f64, 2.0 + 0.5 * iy as f64);
                let mut b = Particle::new(x, y, speed, s * speed, BALL_RADIUS);
                let mut best = SAFE;
                for tau in 1..=PLAN_LEN {
                    b.fly();
                    let landed = b.touch(None, &stub).0;
                    if b.x > 0.0 {
                        let heights = (REF_U, TOP[tau.min(JUMP_TICKS)]);
                        best = best.min(reach_lack(&b, wait, heights, (NEAR_NET, BACK_WALL), tau));
                    }
                    if landed != 0 {
                        if landed > 0 && best > *cell {
                            *cell = best;
                        }
                        break;
                    }
                }
            }
        }
    }
    map
}

/// The danger maps for every waiting spot, worked out once: they depend on nothing but
/// the game, so every oracle shares them.
fn danger_maps() -> &'static [DangerMap; WAIT_AT.len()] {
    static MAPS: std::sync::OnceLock<[DangerMap; WAIT_AT.len()]> = std::sync::OnceLock::new();
    MAPS.get_or_init(|| core::array::from_fn(|w| danger_map(WAIT_AT[w])))
}

fn danger(map: &DangerMap, b: &Particle) -> f64 {
    let ix = ((b.x + 22.5) * 2.0 + 0.5).clamp(0.0, (DANGER_X - 1) as f64) as usize;
    let iy = ((b.y - 2.0) * 2.0 + 0.5).clamp(0.0, (DANGER_Y - 1) as f64) as usize;
    map[ix][iy]
}

/// The opponent as the reach envelope sees it: where it stands (our frame) and, tick by
/// tick from now, the lowest and highest its centre can be. In the air it follows its
/// flight; on the ground it can be anywhere up to the top of a jump started then.
struct Opponent {
    x: f64,
    heights: [(f64, f64); PLAN_LEN + 1],
}

impl Opponent {
    fn new(view: &[f64; 12]) -> Opponent {
        let mut me = Slime {
            x: -view[8],
            y: view[9],
            vx: 0.0,
            vy: view[11],
        };
        let mut heights = [(REF_U, REF_U); PLAN_LEN + 1];
        let mut grounded = None;
        for (i, h) in heights.iter_mut().enumerate() {
            if i > 0 && grounded.is_none() {
                me.step(0);
            }
            if grounded.is_none() && me.y <= REF_U + NUDGE * TIMESTEP {
                grounded = Some(i);
            }
            *h = match grounded {
                Some(g) => (REF_U, TOP[(i - g).min(JUMP_TICKS)]),
                None => (me.y, me.y),
            };
        }
        Opponent {
            x: -view[8],
            heights,
        }
    }

    /// Ticks it still lacks to touch the ball at `b` at plan tick `i`, when it starts
    /// running at our touch, `tau` ticks before: negative when it can get there in
    /// time, infinite when it could not touch the ball there at all.
    fn lack(&self, b: &Particle, i: usize, tau: usize) -> f64 {
        reach_lack(
            b,
            self.x,
            self.heights[i.min(PLAN_LEN)],
            (-BACK_WALL, -NEAR_NET),
            tau,
        )
    }
}

/// One way of meeting the ball: walk `k` steps (forward when negative), wait until
/// `s` ticks have passed, then play `branch` (its jump on the first tick only) until
/// the ball is touched. A branch of 0 just stands.
#[derive(Clone, Copy)]
struct Cand {
    k: i32,
    s: usize,
    branch: u32,
}

impl Cand {
    /// The action for plan tick `i` (1-based) before the touch.
    fn action(&self, i: usize) -> u32 {
        let walk = self.k.unsigned_abs() as usize;
        if i <= walk {
            if self.k < 0 { FORWARD } else { BACKWARD }
        } else if self.branch == 0 || i <= self.s {
            0
        } else if i == self.s + 1 {
            self.branch
        } else {
            self.branch & !JUMP
        }
    }
}

/// What a plan starts from: the ball as believed now and how many more ticks it hangs
/// (a serve), our slime, and the opponent.
struct Start {
    ball: Particle,
    frozen: usize,
    me: Slime,
    opp: Opponent,
}

pub struct Oracle {
    rung: Rung,
    rng: Pcg32,
    /// The last observations, and for each how many observations in a row (ending with
    /// it) showed the serve hanging.
    history: [[f64; 12]; HISTORY],
    serve_run: [u32; HISTORY],
    seen: usize,
    noise: f64,
    incoming: bool,
    /// The plan: an action per tick and the ball it expects to see after each. While
    /// the ball it believes in matches, nothing unexpected happened and the plan stands.
    actions: [u32; PLAN_LEN],
    balls: [[f64; 4]; PLAN_LEN],
    len: usize,
    pos: usize,
    /// The plan tick of its own first touch, 0 when the plan has none.
    touch_at: usize,
    /// Where the plan waits once it has nothing else to do.
    home: f64,
    replans: u32,
    surprises: u32,
    /// The last choice: its score, walk, start of the branch and the branch (traces).
    choice: (f64, i32, usize, u32),
}

impl Oracle {
    pub fn new(rung: usize, seed: u64, side: u32) -> Oracle {
        Oracle {
            rung: RUNGS[rung],
            rng: Pcg32::with_stream(seed, 0x0dac_1e00 + side as u64),
            history: [[0.0; 12]; HISTORY],
            serve_run: [0; HISTORY],
            seen: 0,
            noise: 0.0,
            incoming: false,
            actions: [0; PLAN_LEN],
            balls: [[0.0; 4]; PLAN_LEN],
            len: 0,
            pos: 0,
            touch_at: 0,
            home: HOME_X,
            replans: 0,
            surprises: 0,
            choice: (0.0, 0, 0, 0),
        }
    }

    /// How many times it has planned (for tests and traces).
    pub fn replans(&self) -> u32 {
        self.replans
    }

    /// How many of those plans were forced by a ball it did not expect (for tests).
    pub fn surprises(&self) -> u32 {
        self.surprises
    }

    /// The last way of meeting the ball it chose: its score, the steps it walks, the
    /// tick its branch starts and the branch (for traces).
    pub fn choice(&self) -> (f64, i32, usize, u32) {
        self.choice
    }

    pub fn action(&mut self, obs: &[f64; 12]) -> u32 {
        let n = self.seen;
        self.seen += 1;
        let before = if n == 0 {
            0
        } else {
            self.serve_run[(n - 1) % HISTORY]
        };
        self.history[n % HISTORY] = *obs;
        self.serve_run[n % HISTORY] = if obs[4] == 0.0 && obs[5] == SERVE_Y {
            before + 1
        } else {
            0
        };

        // What it sees is `delay` ticks old; it brings the ball up to now with its own
        // movements. A serve hangs 30 ticks after a point; the first one 31, as the
        // first observation already shows it.
        let v = n - (self.rung.delay as usize).min(n);
        let view = self.history[v % HISTORY];
        let run = self.serve_run[v % HISTORY] as usize;
        let hang = if run == v + 1 {
            INIT_DELAY_FRAMES as usize + 1
        } else {
            INIT_DELAY_FRAMES as usize
        };
        let mut frozen = if run == 0 {
            0
        } else {
            hang.saturating_sub(run)
        };
        let stub = stub();
        let mut ball = Particle::new(view[4], view[5], view[6], view[7], BALL_RADIUS);
        for j in v + 1..=n {
            if frozen > 0 {
                frozen -= 1;
            } else {
                ball.fly();
            }
            ball.touch(
                Some(&Slime::from_obs(&self.history[j % HISTORY]).body()),
                &stub,
            );
        }

        let on_plan =
            self.pos >= 1 && self.pos <= self.len && self.balls[self.pos - 1] == state(&ball);
        // It also plans again every few ticks while the ball is on its way, and once its
        // return has crossed the net, to choose where to wait.
        let stale = self.pos >= REPLAN_EVERY && self.pos < self.touch_at;
        let crossed = self.touch_at > 0 && self.pos > self.touch_at && ball.x < 0.0;
        if !on_plan || stale || crossed {
            self.surprises += !on_plan as u32;
            let start = Start {
                ball,
                frozen,
                me: Slime::from_obs(obs),
                opp: Opponent::new(&view),
            };
            self.replan(&start);
            self.pos = 0;
        }
        let action = if self.pos < self.len {
            self.actions[self.pos]
        } else {
            toward(&Slime::from_obs(obs), self.home)
        };
        self.pos += 1;
        action
    }

    /// Chooses how to meet the ball (or where to wait) and writes the plan.
    fn replan(&mut self, start: &Start) {
        self.replans += 1;
        let stub = stub();
        // The ball's flight with nobody on the court, after each tick's move (where a
        // touch is checked) up to the tick it lands.
        let mut moved = [start.ball; PLAN_LEN + 1];
        let mut ball = start.ball;
        let mut end = PLAN_LEN;
        for (i, m) in moved.iter_mut().enumerate().skip(1) {
            if i > start.frozen {
                ball.fly();
            }
            *m = ball;
            if ball.touch(None, &stub).0 != 0 {
                end = i;
                break;
            }
        }
        // When and where it is within reach on our side.
        let mut window: Option<(usize, usize, f64, f64)> = None;
        for (i, b) in moved.iter().enumerate().take(end + 1).skip(1) {
            if b.x > 0.0 && b.y < REACH_Y {
                let (t_in, _, lo, hi) = window.unwrap_or((i, i, b.x, b.x));
                window = Some((t_in, i, lo.min(b.x), hi.max(b.x)));
            }
        }
        let Some((t_in, t_out, x_lo, x_hi)) = window else {
            // Not ours: wait where the opponent's most dangerous chance to touch the ball
            // as it flies now hurts least (it may start running now), near home on a tie.
            self.incoming = false;
            let mut best = (f64::INFINITY, HOME_X);
            for (map, &wait) in danger_maps().iter().zip(WAIT_AT.iter()) {
                let mut worst = f64::NEG_INFINITY;
                for (i, b) in moved.iter().enumerate().take(end + 1).skip(1) {
                    if b.x < 0.0 && start.opp.lack(b, i, i) <= 0.0 {
                        worst = worst.max(danger(map, b));
                    }
                }
                let key = worst.max(-SAFE) + 0.01 * (wait - HOME_X).abs();
                if key < best.0 {
                    best = (key, wait);
                }
            }
            self.record(start, None, best.1);
            return;
        };
        if !self.incoming {
            self.incoming = true;
            self.noise = if self.rung.noise > 0.0 {
                (2.0 * self.rng.u01() - 1.0) * self.rung.noise
            } else {
                0.0
            };
        }

        // Every spot on the grid of places it can step to that is near the ball while
        // it is within reach, and reachable before it leaves.
        let x0 = start.me.x;
        let reach = t_out as i32;
        let (mut k_lo, mut k_hi) = (0, 0);
        while k_lo > -reach
            && walk(x0, k_lo) > x_lo - TOUCH - STEP
            && walk(x0, k_lo - 1) < walk(x0, k_lo)
        {
            k_lo -= 1;
        }
        while k_hi < reach
            && walk(x0, k_hi) < x_hi + TOUCH + STEP
            && walk(x0, k_hi + 1) > walk(x0, k_hi)
        {
            k_hi += 1;
        }

        let mut seen = [0u64; SEEN];
        let mut best: Option<(f64, Cand)> = None;
        for k in k_lo..=k_hi {
            let walk_ticks = k.unsigned_abs() as usize;
            let still = Cand { k, s: 0, branch: 0 };
            let mut me = start.me;
            let mut score = LOST;
            for i in 0..end {
                // Branches that start after tick i.
                if i >= walk_ticks {
                    for branch in BRANCHES {
                        let lead = if branch & JUMP != 0 {
                            JUMP_TICKS
                        } else {
                            RUN_LEAD
                        };
                        if i + lead < t_in {
                            continue;
                        }
                        let c = Cand { k, s: i, branch };
                        if let Some(score) = self.branch_out(start, &moved, end, me, c, &mut seen)
                            && best.is_none_or(|(b, _)| score > b)
                        {
                            best = Some((score, c));
                        }
                    }
                }
                me.step(still.action(i + 1));
                if moved[i + 1].is_colliding(&me.body()) {
                    score = self
                        .after_touch(start, &moved, i + 1, me, &mut seen)
                        .unwrap_or(LOST);
                    break;
                }
            }
            if best.is_none_or(|(b, _)| score > b) {
                best = Some((score, still));
            }
        }

        let (score, mut c) = best.expect("at least one spot");
        self.choice = (score, c.k, c.s, c.branch);
        if self.noise != 0.0 {
            c.k += (self.noise / STEP).round() as i32;
            c.s = c.s.max(c.k.unsigned_abs() as usize);
        }
        self.record(start, Some(c), HOME_X);
    }

    /// Plays a branch from tick `c.s` (where `me` stands) to its touch, then the return.
    /// None when it never touches the ball or the touch was already played out.
    fn branch_out(
        &self,
        start: &Start,
        moved: &[Particle; PLAN_LEN + 1],
        end: usize,
        mut me: Slime,
        c: Cand,
        seen: &mut [u64; SEEN],
    ) -> Option<f64> {
        let jumping = c.branch & JUMP != 0;
        for j in c.s + 1..=end {
            me.step(c.action(j));
            if moved[j].is_colliding(&me.body()) {
                return self.after_touch(start, moved, j, me, seen);
            }
            // A run stopped by a wall, or a jump back on the ground, now just stands:
            // the plain stances cover that.
            if (!jumping && me.vx == 0.0) || (jumping && j > c.s + 1 && me.y == REF_U) {
                return None;
            }
        }
        None
    }

    /// Scores the return from a touch at tick `t` with our slime at `me`. None when the
    /// same touch (tick and slime, bit for bit) was already played out this plan.
    fn after_touch(
        &self,
        start: &Start,
        moved: &[Particle; PLAN_LEN + 1],
        t: usize,
        mut me: Slime,
        seen: &mut [u64; SEEN],
    ) -> Option<f64> {
        let key = [
            t as u64,
            me.x.to_bits(),
            me.y.to_bits(),
            me.vx.to_bits(),
            me.vy.to_bits(),
        ]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, &v| {
            (h ^ v).wrapping_mul(0x0100_0000_01b3)
        }) | 1;
        let mut slot = (key as usize) % SEEN;
        for _ in 0..SEEN {
            if seen[slot] == key {
                return None;
            }
            if seen[slot] == 0 {
                seen[slot] = key;
                break;
            }
            slot = (slot + 1) % SEEN;
        }
        let stub = stub();
        let mut ball = moved[t];
        let (mut landed, _) = ball.touch(Some(&me.body()), &stub);
        // The return is worth the worst of the opponent's chances to touch it, by the
        // danger map (a little worse the more time it has to spare); SAFE if it has none.
        let mut score = SAFE;
        let mut i = t;
        loop {
            if ball.x < 0.0 {
                let lack = start.opp.lack(&ball, i, i - t);
                if lack <= 0.0 {
                    score = score.min(SPARE * lack - danger(&danger_maps()[HOME], &ball));
                }
            }
            if landed != 0 {
                return Some(if landed > 0 {
                    LOST
                } else {
                    score - 0.001 * i as f64
                });
            }
            if i == PLAN_LEN {
                return Some(if ball.x > 0.0 {
                    SHORT
                } else {
                    score - 0.001 * i as f64
                });
            }
            i += 1;
            me.step(toward(&me, HOME_X));
            if i > start.frozen {
                ball.fly();
            }
            landed = ball.touch(Some(&me.body()), &stub).0;
        }
    }

    /// Plays the plan through once more, writing its actions and the ball it expects
    /// after each tick: `c` then off to `home`, or waiting at `home` when the ball is
    /// not coming.
    fn record(&mut self, start: &Start, c: Option<Cand>, home: f64) {
        let stub = stub();
        let (mut ball, mut me) = (start.ball, start.me);
        let mut touched = c.is_none();
        self.len = PLAN_LEN;
        self.touch_at = 0;
        self.home = home;
        for i in 1..=PLAN_LEN {
            let action = match c {
                Some(c) if !touched => c.action(i),
                _ => toward(&me, home),
            };
            me.step(action);
            if i > start.frozen {
                ball.fly();
            }
            let (landed, hit) = ball.touch(Some(&me.body()), &stub);
            if hit && !touched {
                touched = true;
                self.touch_at = i;
            }
            self.actions[i - 1] = action;
            self.balls[i - 1] = state(&ball);
            if landed != 0 {
                self.len = i;
                break;
            }
        }
    }
}

/// Where `k` steps from `x` lead, forward (towards the net) for negative `k`, with the
/// same arithmetic and clamps as `Agent::update`, so the slime stops exactly there.
fn walk(x: f64, k: i32) -> f64 {
    let mut me = Slime {
        x,
        y: REF_U,
        vx: 0.0,
        vy: 0.0,
    };
    for _ in 0..k.unsigned_abs() {
        me.step(if k < 0 { FORWARD } else { BACKWARD });
    }
    me.x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::Game;
    use crate::scripted::scripted_action;

    fn play(seed: u64, rung: usize, against: impl Fn(&[f64; 12]) -> u32) -> (u32, u32, u32) {
        let mut game = Game::new(seed);
        let mut oracle = Oracle::new(rung, seed, 0);
        while !game.is_done(3000) {
            let (a, b) = (oracle.action(&game.a.obs), against(&game.b.obs));
            game.set_actions(a, b);
            game.step();
        }
        (game.desc[0].points, game.desc[1].points, game.tick)
    }

    #[test]
    fn its_slime_moves_exactly_like_the_game_on_either_side() {
        let mut rng = Pcg32::new(11);
        let mut game = Game::new(4);
        let (mut a, mut b) = (Slime::from_obs(&game.a.obs), Slime::from_obs(&game.b.obs));
        for _ in 0..2000 {
            let (x, y) = ((rng.u01() * 8.0) as u32, (rng.u01() * 8.0) as u32);
            game.set_actions(x, y);
            game.step();
            a.step(x);
            b.step(y);
            for (s, agent) in [(a, &game.a), (b, &game.b)] {
                let mirrored = [agent.x * agent.dir, agent.y, agent.vx * agent.dir, agent.vy];
                assert_eq!([s.x, s.y, s.vx, s.vy], mirrored);
            }
        }
    }

    #[test]
    fn against_an_idle_slime_it_never_has_to_replan_mid_rally() {
        // Nothing unexpected can happen: only a new serve makes it plan again.
        let mut game = Game::new(9);
        let mut oracle = Oracle::new(0, 9, 1);
        let mut serves = 1;
        while !game.is_done(3000) {
            let b = oracle.action(&game.b.obs);
            game.set_actions(0, b);
            serves += (game.step() != 0) as u32;
        }
        assert!(game.desc[1].points == 5 && game.desc[0].points == 0);
        // A point leaves the observation stale for a tick: up to three surprises a serve.
        let surprises = oracle.surprises();
        assert!(
            surprises <= 3 * serves,
            "{surprises} surprises for {serves} serves"
        );
    }

    #[test]
    fn the_oracle_beats_the_scripted_bot_and_an_idle_slime() {
        let (mut won, mut lost) = (0, 0);
        for seed in 0..8 {
            let (a, b, _) = play(seed, 0, scripted_action);
            won += a;
            lost += b;
            let (a, b, _) = play(seed, 0, |_| 0);
            assert!(a > b, "seed {seed}: oracle {a}, idle {b}");
        }
        assert!(won > 4 * lost, "against scripted-v1: {won} to {lost}");
    }

    #[test]
    fn the_oracle_is_deterministic_and_rungs_differ() {
        assert_eq!(play(3, 1, scripted_action), play(3, 1, scripted_action));
        let a = play(3, 0, scripted_action);
        let b = play(3, 3, scripted_action);
        let c = play(3, 5, scripted_action);
        assert!(a != b || a != c);
    }
}
