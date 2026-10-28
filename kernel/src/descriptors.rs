//! The behaviour descriptors per side (docs/ABI.md), accumulated over every
//! hashed tick of a match. They place a genome on the MAP-Elites grid. The first
//! eight are kernel version 1's; version 2 adds two from the juggle tracker and
//! version 3 one more, spiked_points.

use crate::juggle::Juggle;
use crate::physics::{Agent, REF_U, REF_W, REF_WALL_WIDTH, SLIME_RADIUS, abs};

pub const N_DESC: usize = 11;

/// `x*dir` at the back wall (22.5) and the depth of the own half (20.5): the
/// own-half clamp runs from REF_WALL_WIDTH/2 + r = 2.0 at the net to
/// REF_W/2 - r = 22.5 at the wall.
const COURT_BACK: f64 = REF_W / 2.0 - SLIME_RADIUS;
pub const COURT_DEPTH: f64 = COURT_BACK - (REF_WALL_WIDTH / 2.0 + SLIME_RADIUS);

#[derive(Clone, Copy)]
pub struct Descriptors {
    pub court_sum: f64,
    pub air_ticks: u32,
    pub hits: u32,
    pub hit_height_sum: f64,
    pub hit_speed_sum: f64,
    pub jumps: u32,
    pub distance: f64,
    pub points: u32,
    pub juggle: Juggle,
}

impl Descriptors {
    pub const ZERO: Descriptors = Descriptors {
        court_sum: 0.0,
        air_ticks: 0,
        hits: 0,
        hit_height_sum: 0.0,
        hit_speed_sum: 0.0,
        jumps: 0,
        distance: 0.0,
        points: 0,
        juggle: Juggle::ZERO,
    };

    /// `before` is the slime's (x, y) at the start of the tick; `hit` is the
    /// ball's (y, speed) right after bouncing off this slime this tick.
    pub fn add_tick(
        &mut self,
        agent: &Agent,
        before: (f64, f64),
        hit: Option<(f64, f64)>,
        won_point: bool,
    ) {
        self.court_sum += (COURT_BACK - agent.x * agent.dir) / COURT_DEPTH;
        if agent.y > REF_U {
            self.air_ticks += 1;
            if before.1 <= REF_U {
                self.jumps += 1;
            }
        }
        self.distance += abs(agent.x - before.0);
        if let Some((height, speed)) = hit {
            self.hits += 1;
            self.hit_height_sum += height;
            self.hit_speed_sum += speed;
        }
        if won_point {
            self.points += 1;
        }
        self.juggle
            .tick(agent.x, agent.dir, hit.map(|(_, speed)| speed));
    }

    /// The ABI order: court_position, airtime, hits, mean_hit_height,
    /// mean_speed_after_hit, jumps, distance, points, juggled_points, max_streak,
    /// spiked_points.
    pub fn finish(&self, ticks: u32) -> [f64; N_DESC] {
        let per_tick = |sum: f64| if ticks == 0 { 0.0 } else { sum / ticks as f64 };
        let per_hit = |sum: f64| {
            if self.hits == 0 {
                0.0
            } else {
                sum / self.hits as f64
            }
        };
        [
            per_tick(self.court_sum),
            per_tick(self.air_ticks as f64),
            self.hits as f64,
            per_hit(self.hit_height_sum),
            per_hit(self.hit_speed_sum),
            self.jumps as f64,
            self.distance,
            self.points as f64,
            self.juggle.juggled_points as f64,
            self.juggle.max_streak as f64,
            self.juggle.spiked_points as f64,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn court_position_runs_from_wall_to_net() {
        assert_eq!(COURT_BACK, 22.5);
        assert_eq!(COURT_DEPTH, 20.5);
        let mut d = Descriptors::ZERO;
        let mut wall = Agent::new(1.0, 22.5, REF_U);
        d.add_tick(&wall, (22.5, REF_U), None, false);
        assert_eq!(d.finish(1)[0], 0.0);
        let mut d = Descriptors::ZERO;
        wall.x = -2.0;
        wall.dir = -1.0;
        d.add_tick(&wall, (-2.0, REF_U), None, false);
        assert_eq!(d.finish(1)[0], 1.0);
    }

    #[test]
    fn empty_match_gives_zeros() {
        assert_eq!(Descriptors::ZERO.finish(0), [0.0; N_DESC]);
    }
}
