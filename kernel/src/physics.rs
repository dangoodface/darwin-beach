//! The game, ported from `slimevolley.py` in hardmaru's slimevolleygym
//! (David Ha, 2020, Apache-2.0, https://github.com/hardmaru/slimevolleygym,
//! read at commit 8ac2243). Constants are written as the same expressions the
//! Python file uses, and every expression keeps Python's operation order, so
//! each double comes out the same.
//!
//! Deviations from the Python file, each needed for a deterministic kernel:
//! - `bounce` with coincident centres: Python raises ZeroDivisionError; here
//!   the normal is (0, 1), as docs/ABI.md says.
//! - The serve draws vx then vy from the kernel's PCG32 instead of numpy.
//! - Lives stop at 0 instead of going negative if a finished game keeps being
//!   stepped (only the interactive game can do that).
//! - Rendering, emotions and `betweenGameControl` are left out: they never
//!   touch the physics.

use crate::descriptors::Descriptors;
use crate::hash::{FNV_OFFSET, Fnv1a64};
use crate::rng::Pcg32;

pub const REF_W: f64 = 24.0 * 2.0;
pub const REF_H: f64 = REF_W;
pub const REF_U: f64 = 1.5; // ground height
pub const REF_WALL_WIDTH: f64 = 1.0;
pub const REF_WALL_HEIGHT: f64 = 3.5;
pub const PLAYER_SPEED_X: f64 = 10.0 * 1.75;
pub const PLAYER_SPEED_Y: f64 = 10.0 * 1.35;
pub const MAX_BALL_SPEED: f64 = 15.0 * 1.5;
pub const TIMESTEP: f64 = 1.0 / 30.0;
pub const NUDGE: f64 = 0.1;
pub const FRICTION: f64 = 1.0; // 1 means no friction
pub const INIT_DELAY_FRAMES: u32 = 30;
pub const GRAVITY: f64 = -9.8 * 2.0 * 1.5;
pub const MAXLIVES: u32 = 5;

pub const BALL_RADIUS: f64 = 0.5;
pub const SLIME_RADIUS: f64 = 1.5;

/// Action bits (docs/ABI.md): forward is towards the net.
pub const FORWARD: u32 = 1;
pub const BACKWARD: u32 = 2;
pub const JUMP: u32 = 4;

/// Result flag bit 0: a NaN or infinity reached the state.
pub const FLAG_NAN: u32 = 1;

/// IEEE square root, correctly rounded: wasm's `f64.sqrt`, x86's `sqrtsd`.
#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[inline(always)]
pub fn abs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

/// What the ball bounces off: a slime or the round stub on top of the fence.
#[derive(Clone, Copy)]
pub struct Body {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub r: f64,
}

/// The ball, and also the stub above the fence (`Particle` in Python).
#[derive(Clone, Copy)]
pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub prev_x: f64,
    pub prev_y: f64,
    pub vx: f64,
    pub vy: f64,
    pub r: f64,
}

impl Particle {
    pub const fn new(x: f64, y: f64, vx: f64, vy: f64, r: f64) -> Particle {
        Particle {
            x,
            y,
            prev_x: x,
            prev_y: y,
            vx,
            vy,
            r,
        }
    }

    pub const fn body(&self) -> Body {
        Body {
            x: self.x,
            y: self.y,
            vx: self.vx,
            vy: self.vy,
            r: self.r,
        }
    }

    fn move_(&mut self) {
        self.prev_x = self.x;
        self.prev_y = self.y;
        self.x += self.vx * TIMESTEP;
        self.y += self.vy * TIMESTEP;
    }

    fn apply_acceleration(&mut self, ax: f64, ay: f64) {
        self.vx += ax * TIMESTEP;
        self.vy += ay * TIMESTEP;
    }

    /// Walls, ceiling, floor and the two faces of the fence. Hitting the
    /// floor ends the point: -1 when the ball lands on the left (x <= 0),
    /// +1 on the right.
    fn check_edges(&mut self) -> i32 {
        if self.x <= (self.r - REF_W / 2.0) {
            self.vx *= -FRICTION;
            self.x = self.r - REF_W / 2.0 + NUDGE * TIMESTEP;
        }

        if self.x >= (REF_W / 2.0 - self.r) {
            self.vx *= -FRICTION;
            self.x = REF_W / 2.0 - self.r - NUDGE * TIMESTEP;
        }

        if self.y <= (self.r + REF_U) {
            self.vy *= -FRICTION;
            self.y = self.r + REF_U + NUDGE * TIMESTEP;
            return if self.x <= 0.0 { -1 } else { 1 };
        }
        if self.y >= (REF_H - self.r) {
            self.vy *= -FRICTION;
            self.y = REF_H - self.r - NUDGE * TIMESTEP;
        }
        // The fence only stops a ball that crossed its face during this tick.
        if (self.x <= (REF_WALL_WIDTH / 2.0 + self.r))
            && (self.prev_x > (REF_WALL_WIDTH / 2.0 + self.r))
            && (self.y <= REF_WALL_HEIGHT)
        {
            self.vx *= -FRICTION;
            self.x = REF_WALL_WIDTH / 2.0 + self.r + NUDGE * TIMESTEP;
        }

        if (self.x >= (-REF_WALL_WIDTH / 2.0 - self.r))
            && (self.prev_x < (-REF_WALL_WIDTH / 2.0 - self.r))
            && (self.y <= REF_WALL_HEIGHT)
        {
            self.vx *= -FRICTION;
            self.x = -REF_WALL_WIDTH / 2.0 - self.r - NUDGE * TIMESTEP;
        }
        0
    }

    fn get_dist2(&self, p: &Body) -> f64 {
        let dy = p.y - self.y;
        let dx = p.x - self.x;
        dx * dx + dy * dy
    }

    pub fn is_colliding(&self, p: &Body) -> bool {
        let r = self.r + p.r;
        r * r > self.get_dist2(p)
    }

    /// Push the ball out along the normal in NUDGE steps, then reflect its
    /// velocity relative to the other body. Python doubles the normal
    /// component ("added factor of 2"), which is a mirror reflection.
    pub fn bounce(&mut self, p: &Body) {
        let mut abx = self.x - p.x;
        let mut aby = self.y - p.y;
        let abd = sqrt(abx * abx + aby * aby);
        if abd == 0.0 {
            // Coincident centres: Python would raise ZeroDivisionError.
            abx = 0.0;
            aby = 1.0;
        } else {
            abx /= abd; // normalize
            aby /= abd;
        }
        let nx = abx;
        let ny = aby;
        abx *= NUDGE;
        aby *= NUDGE;
        while self.is_colliding(p) {
            self.x += abx;
            self.y += aby;
        }
        let mut ux = self.vx - p.vx;
        let mut uy = self.vy - p.vy;
        let un = ux * nx + uy * ny;
        let unx = nx * (un * 2.0);
        let uny = ny * (un * 2.0);
        ux -= unx;
        uy -= uny;
        self.vx = ux + p.vx;
        self.vy = uy + p.vy;
    }

    fn limit_speed(&mut self, min_speed: f64, max_speed: f64) {
        let mag2 = self.vx * self.vx + self.vy * self.vy;
        if mag2 > (max_speed * max_speed) {
            let mag = sqrt(mag2);
            self.vx /= mag;
            self.vy /= mag;
            self.vx *= max_speed;
            self.vy *= max_speed;
        }

        if mag2 < (min_speed * min_speed) {
            let mag = sqrt(mag2);
            self.vx /= mag;
            self.vy /= mag;
            self.vx *= min_speed;
            self.vy *= min_speed;
        }
    }
}

/// A slime. `dir` is -1 on the left and +1 on the right, so the same code
/// and the same genome serve both sides.
#[derive(Clone, Copy)]
pub struct Agent {
    pub dir: f64,
    pub x: f64,
    pub y: f64,
    pub r: f64,
    pub vx: f64,
    pub vy: f64,
    pub desired_vx: f64,
    pub desired_vy: f64,
    pub life: u32,
    /// `RelativeState`: the last observation, unscaled. Python refreshes it
    /// only at the end of a tick without a point, so after a point it is one
    /// tick stale; the kernel keeps that.
    pub obs: [f64; 12],
}

impl Agent {
    pub const fn new(dir: f64, x: f64, y: f64) -> Agent {
        Agent {
            dir,
            x,
            y,
            r: SLIME_RADIUS,
            vx: 0.0,
            vy: 0.0,
            desired_vx: 0.0,
            desired_vy: 0.0,
            life: MAXLIVES,
            obs: [0.0; 12],
        }
    }

    pub const fn body(&self) -> Body {
        Body {
            x: self.x,
            y: self.y,
            vx: self.vx,
            vy: self.vy,
            r: self.r,
        }
    }

    /// Forward and backward together cancel out, as in `Agent.setAction`.
    pub fn set_action(&mut self, action: u32) {
        let forward = action & FORWARD != 0;
        let backward = action & BACKWARD != 0;
        let jump = action & JUMP != 0;
        self.desired_vx = 0.0;
        self.desired_vy = 0.0;
        if forward && !backward {
            self.desired_vx = -PLAYER_SPEED_X;
        }
        if backward && !forward {
            self.desired_vx = PLAYER_SPEED_X;
        }
        if jump {
            self.desired_vy = PLAYER_SPEED_Y;
        }
    }

    fn move_(&mut self) {
        self.x += self.vx * TIMESTEP;
        self.y += self.vy * TIMESTEP;
    }

    /// Gravity, jump only from the ground, horizontal speed straight from the
    /// action (no inertia), then the ground and own-half clamps.
    fn update(&mut self) {
        self.vy += GRAVITY * TIMESTEP;

        if self.y <= REF_U + NUDGE * TIMESTEP {
            self.vy = self.desired_vy;
        }

        self.vx = self.desired_vx * self.dir;

        self.move_();

        if self.y <= REF_U {
            self.y = REF_U;
            self.vy = 0.0;
        }

        // stay in their own half:
        if self.x * self.dir <= (REF_WALL_WIDTH / 2.0 + self.r) {
            self.vx = 0.0;
            self.x = self.dir * (REF_WALL_WIDTH / 2.0 + self.r);
        }

        if self.x * self.dir >= (REF_W / 2.0 - self.r) {
            self.vx = 0.0;
            self.x = self.dir * (REF_W / 2.0 - self.r);
        }
    }

    /// The world seen from this side: x flipped so the net is at 0 and the
    /// own half is positive; the opponent is flipped into its own frame.
    pub const fn update_state(&mut self, ball: &Particle, opponent: &Body) {
        self.obs = [
            self.x * self.dir,
            self.y,
            self.vx * self.dir,
            self.vy,
            ball.x * self.dir,
            ball.y,
            ball.vx * self.dir,
            ball.vy,
            opponent.x * (-self.dir),
            opponent.y,
            opponent.vx * (-self.dir),
            opponent.vy,
        ];
    }
}

/// One rally's serve: `Game.newMatch` draws vx then vy, uniform as numpy's
/// `uniform(low, high) = low + (high - low) * u`.
const fn serve(rng: &mut Pcg32) -> Particle {
    let ball_vx = -20.0 + 40.0 * rng.u01();
    let ball_vy = 10.0 + 15.0 * rng.u01();
    Particle::new(0.0, REF_W / 4.0, ball_vx, ball_vy, BALL_RADIUS)
}

/// A whole game: physics, serve RNG, tick counter, running hash and the
/// descriptor accumulators.
#[derive(Clone, Copy)]
pub struct Game {
    pub ball: Particle,
    pub fence_stub: Particle,
    /// Side A, the left slime.
    pub a: Agent,
    /// Side B, the right slime.
    pub b: Agent,
    /// `DelayScreen.life`: ticks the ball stays frozen before a serve.
    pub delay: u32,
    pub rng: Pcg32,
    pub tick: u32,
    pub hash: u64,
    /// Sign of the most recent point (+1 A, -1 B), 0 before the first.
    pub last_point: i32,
    /// Whether the ball ended the last hashed tick in A's half (x <= 0, the same
    /// line that scores a landing for B). Juggling only, version 2.
    pub ball_on_a: bool,
    pub flags: u32,
    pub desc: [Descriptors; 2],
}

impl Game {
    pub const fn new(seed: u64) -> Game {
        let mut rng = Pcg32::new(seed);
        let fence_stub = Particle::new(0.0, REF_WALL_HEIGHT, 0.0, 0.0, REF_WALL_WIDTH / 2.0);
        let ball = serve(&mut rng);
        let mut a = Agent::new(-1.0, -REF_W / 4.0, 1.5);
        let mut b = Agent::new(1.0, REF_W / 4.0, 1.5);
        a.update_state(&ball, &b.body());
        b.update_state(&ball, &a.body());
        Game {
            ball,
            fence_stub,
            a,
            b,
            delay: INIT_DELAY_FRAMES,
            rng,
            tick: 0,
            hash: FNV_OFFSET,
            last_point: 0,
            ball_on_a: true,
            flags: 0,
            desc: [Descriptors::ZERO; 2],
        }
    }

    pub fn set_actions(&mut self, a: u32, b: u32) {
        self.a.set_action(a);
        self.b.set_action(b);
    }

    pub fn is_done(&self, max_ticks: u32) -> bool {
        self.a.life == 0 || self.b.life == 0 || self.tick >= max_ticks || self.flags & FLAG_NAN != 0
    }

    /// `DelayScreen.status`: true once the serve delay has run out.
    fn delay_over(&mut self) -> bool {
        if self.delay == 0 {
            return true;
        }
        self.delay -= 1;
        false
    }

    /// One tick of `Game.step`, then the NaN check, the descriptors and the
    /// hash. Returns +1 when A wins the point, -1 when B wins it, 0 otherwise.
    pub fn step(&mut self) -> i32 {
        if self.flags & FLAG_NAN != 0 {
            return 0;
        }
        let a_before = (self.a.x, self.a.y);
        let b_before = (self.b.x, self.b.y);

        self.a.update();
        self.b.update();

        if self.delay_over() {
            self.ball.apply_acceleration(0.0, GRAVITY);
            self.ball.limit_speed(0.0, MAX_BALL_SPEED);
            self.ball.move_();
        }

        // Hits are (ball y, ball speed) right after the bounce.
        let mut hits: [Option<(f64, f64)>; 2] = [None, None];
        let body_a = self.a.body();
        if self.ball.is_colliding(&body_a) {
            self.ball.bounce(&body_a);
            hits[0] = Some((self.ball.y, self.ball_speed()));
        }
        let body_b = self.b.body();
        if self.ball.is_colliding(&body_b) {
            self.ball.bounce(&body_b);
            hits[1] = Some((self.ball.y, self.ball_speed()));
        }
        let stub = self.fence_stub.body();
        if self.ball.is_colliding(&stub) {
            self.ball.bounce(&stub);
        }

        // The ball landing on the left (x <= 0) is a point for B, and Python
        // returns before `updateState`, so observations go stale for a tick.
        let point = self.ball.check_edges();
        let ball_x = self.ball.x; // where this tick left the ball, before any new serve
        if point != 0 {
            self.ball = serve(&mut self.rng);
            self.delay = INIT_DELAY_FRAMES;
            if point > 0 {
                self.b.life = self.b.life.saturating_sub(1);
            } else {
                self.a.life = self.a.life.saturating_sub(1);
            }
            self.last_point = point;
        } else {
            self.a.update_state(&self.ball, &self.b.body());
            self.b.update_state(&self.ball, &self.a.body());
        }

        if !self.state_is_finite() {
            // The hash covers only the ticks before this one.
            self.flags |= FLAG_NAN;
            return point;
        }
        self.tick += 1;
        self.desc[0].add_tick(&self.a, a_before, hits[0], point > 0);
        self.desc[1].add_tick(&self.b, b_before, hits[1], point < 0);
        self.juggle_tick(ball_x, point);
        self.hash_tick();
        point
    }

    /// A net crossing sends the streak of the side the ball left; a point lets the
    /// winner's last sent streak count. Reads the state, never writes the physics.
    fn juggle_tick(&mut self, ball_x: f64, point: i32) {
        let on_a = ball_x <= 0.0;
        if self.ball_on_a && !on_a {
            self.desc[0].juggle.sent();
        } else if !self.ball_on_a && on_a {
            self.desc[1].juggle.sent();
        }
        if point != 0 {
            self.desc[0].juggle.point_over(point > 0);
            self.desc[1].juggle.point_over(point < 0);
        }
        self.ball_on_a = self.ball.x <= 0.0; // after a point, the new serve at x = 0
    }

    fn ball_speed(&self) -> f64 {
        sqrt(self.ball.vx * self.ball.vx + self.ball.vy * self.ball.vy)
    }

    /// The twelve hashed values: ball, A, B, each x, y, vx, vy.
    pub const fn state_values(&self) -> [f64; 12] {
        [
            self.ball.x,
            self.ball.y,
            self.ball.vx,
            self.ball.vy,
            self.a.x,
            self.a.y,
            self.a.vx,
            self.a.vy,
            self.b.x,
            self.b.y,
            self.b.vx,
            self.b.vy,
        ]
    }

    fn state_is_finite(&self) -> bool {
        self.state_values().iter().all(|v| v.is_finite())
    }

    fn hash_tick(&mut self) {
        let mut h = Fnv1a64(self.hash);
        h.write_u32(self.tick);
        for v in self.state_values() {
            h.write_f64(v);
        }
        h.write_u32(self.a.life);
        h.write_u32(self.b.life);
        self.hash = h.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_python_doubles() {
        // Printed by CPython 3.13 from the expressions in slimevolley.py.
        assert_eq!(GRAVITY.to_bits(), (-29.400000000000002f64).to_bits());
        assert_eq!(PLAYER_SPEED_Y, 13.5);
        assert_eq!(TIMESTEP.to_bits(), 0.03333333333333333f64.to_bits());
        assert_eq!(REF_W / 4.0, 12.0);
    }

    #[test]
    fn serve_seed_gives_documented_velocity() {
        // Seed 1 is the one written down for this test: vx then vy.
        let game = Game::new(1);
        let mut rng = Pcg32::new(1);
        let vx = -20.0 + 40.0 * rng.u01();
        let vy = 10.0 + 15.0 * rng.u01();
        assert_eq!(game.ball.vx.to_bits(), vx.to_bits());
        assert_eq!(game.ball.vy.to_bits(), vy.to_bits());
        assert_eq!(game.ball.vx.to_bits(), SEED_1_VX.to_bits());
        assert_eq!(game.ball.vy.to_bits(), SEED_1_VY.to_bits());
        assert_eq!((game.ball.x, game.ball.y), (0.0, 12.0));
        assert!((-20.0..20.0).contains(&game.ball.vx) && (10.0..25.0).contains(&game.ball.vy));
    }

    // Seed 1 serves vx = 17.560949325561523, vy = 20.37936121225357, as an
    // independent Python PCG32 computes them (it also reproduces pcg32-demo).
    const SEED_1_VX: f64 = f64::from_bits(0x40318f9a60000000);
    const SEED_1_VY: f64 = f64::from_bits(0x4034611dd1000000);

    #[test]
    fn ball_dropped_on_still_slime_bounces() {
        let mut game = Game::new(3);
        game.delay = 0;
        // Drop the ball straight onto A's head.
        game.ball = Particle::new(game.a.x, 6.0, 0.0, 0.0, BALL_RADIUS);
        let mut bounced = false;
        for _ in 0..60 {
            let before = game.ball.vy;
            game.set_actions(0, 0);
            game.step();
            if before < 0.0 && game.ball.vy > 0.0 {
                bounced = true;
                assert!(game.ball.y > game.a.y + SLIME_RADIUS);
                break;
            }
        }
        assert!(bounced, "the ball never bounced off the slime");
        assert_eq!(game.desc[0].hits, 1);
        assert_eq!(game.a.life, MAXLIVES);
    }

    #[test]
    fn a_ball_bouncing_on_a_still_head_builds_a_streak_that_never_counts() {
        let mut game = Game::new(3);
        game.delay = 0;
        game.ball = Particle::new(game.a.x, 6.0, 0.0, 0.0, BALL_RADIUS);
        for _ in 0..600 {
            game.set_actions(0, 0);
            game.step();
        }
        let j = game.desc[0].juggle;
        assert!(
            j.max_streak >= 4 && j.max_streak == game.desc[0].hits,
            "{j:?}"
        );
        assert_eq!(j.juggled_points, 0); // never sent, never a point
        assert!(game.ball_on_a);
    }

    #[test]
    fn the_ball_changes_half_when_it_crosses_the_net() {
        let mut game = Game::new(4);
        game.delay = 0;
        game.ball = Particle::new(-1.0, 9.0, 12.0, 0.0, BALL_RADIUS);
        assert!(game.ball_on_a);
        for _ in 0..10 {
            game.step();
        }
        assert!(game.ball.x > 0.0 && !game.ball_on_a);
    }

    #[test]
    fn nan_injection_sets_flag_and_freezes_hash() {
        let mut game = Game::new(5);
        for _ in 0..10 {
            game.step();
        }
        let (tick, hash) = (game.tick, game.hash);
        game.ball.vx = f64::NAN;
        game.step();
        assert_eq!(game.flags & FLAG_NAN, FLAG_NAN);
        assert_eq!((game.tick, game.hash), (tick, hash));
        assert!(game.is_done(3000));
        assert_eq!(game.step(), 0);
        assert_eq!(game.tick, tick);
    }

    #[test]
    fn infinity_also_sets_the_flag() {
        // A slime's velocity is rebuilt from the action every tick and its x is
        // clamped, so inject into the ball, which the serve delay leaves alone.
        let mut game = Game::new(5);
        game.ball.vy = f64::INFINITY;
        game.step();
        assert_eq!(game.flags & FLAG_NAN, FLAG_NAN);
        assert_eq!(game.tick, 0);
    }

    #[test]
    fn idle_slimes_lose_points_and_lives_end_the_game() {
        let mut game = Game::new(11);
        while !game.is_done(3000) {
            game.set_actions(0, 0);
            game.step();
        }
        assert!(game.a.life == 0 || game.b.life == 0);
        assert!(game.tick < 3000);
        let points = game.desc[0].points + game.desc[1].points;
        assert_eq!(points, (MAXLIVES - game.a.life) + (MAXLIVES - game.b.life));
    }

    #[test]
    fn slimes_stay_in_their_own_half() {
        let mut game = Game::new(2);
        for t in 0..400 {
            // A runs at the net, B runs at its back wall, both jumping.
            game.set_actions(
                FORWARD | JUMP,
                if t % 2 == 0 {
                    BACKWARD
                } else {
                    BACKWARD | JUMP
                },
            );
            game.step();
            assert!(game.a.x * game.a.dir >= REF_WALL_WIDTH / 2.0 + SLIME_RADIUS);
            assert!(game.b.x * game.b.dir <= REF_W / 2.0 - SLIME_RADIUS);
            assert!(game.a.y >= REF_U && game.b.y >= REF_U);
        }
    }
}
