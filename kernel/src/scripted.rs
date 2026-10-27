//! The scripted anchor bot, scripted-v1, frozen at kernel version 1. It is
//! the ladder's fixed point and the first probe every genome is measured
//! against, and the page replays against it by default, so any change here is
//! a kernel version bump (the golden hashes in lib.rs will say so).
//!
//! It reads the same mirrored, unscaled observation a genome sees: its own x
//! runs from 2.0 at the net to 22.5 at its back wall, and the ball is over its
//! half when the ball's x is positive. The whole rule:
//!
//! - target x is the ball's x + 0.5 while the ball's x > 0, else 12.0 (home,
//!   where the slimes start);
//! - forward when x > target + 0.2, backward when x < target - 0.2;
//! - jump when the ball is above it by less than 3.5 and within 1.5 in x.
//!
//! Standing half a unit behind the ball means its head meets the ball on the
//! side away from the net, so the bounce sends the ball back over.

use crate::physics::{BACKWARD, FORWARD, JUMP, abs};

/// Where it waits while the ball is over the far half: its starting x.
pub const HOME_X: f64 = 12.0;
const BEHIND: f64 = 0.5;
const DEADBAND: f64 = 0.2;
const JUMP_ABOVE: f64 = 3.5;
const JUMP_WITHIN: f64 = 1.5;

pub fn scripted_action(obs: &[f64; 12]) -> u32 {
    scripted_with_home(obs, HOME_X)
}

/// The same rule with another home. Only tests use other homes, as variants
/// that must land in different columns of the map; the exports use `HOME_X`.
pub fn scripted_with_home(obs: &[f64; 12], home: f64) -> u32 {
    let (x, y, bx, by) = (obs[0], obs[1], obs[4], obs[5]);
    let target = if bx > 0.0 { bx + BEHIND } else { home };
    let mut action = 0;
    if x > target + DEADBAND {
        action |= FORWARD;
    } else if x < target - DEADBAND {
        action |= BACKWARD;
    }
    let dy = by - y;
    if dy > 0.0 && dy < JUMP_ABOVE && abs(bx - x) < JUMP_WITHIN {
        action |= JUMP;
    }
    action
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::Game;

    /// Plays side A with `left` and side B with `right` from `seed`; returns
    /// A's descriptors and both scores.
    fn play(
        seed: u64,
        left: impl Fn(&[f64; 12]) -> u32,
        right: impl Fn(&[f64; 12]) -> u32,
    ) -> ([f64; crate::descriptors::N_DESC], u32, u32) {
        let mut game = Game::new(seed);
        while !game.is_done(3000) {
            let (a, b) = (left(&game.a.obs), right(&game.b.obs));
            game.set_actions(a, b);
            game.step();
        }
        (
            game.desc[0].finish(game.tick),
            game.desc[0].points,
            game.desc[1].points,
        )
    }

    /// The map's x axis: court_position over [0, 1] in 20 columns, as
    /// configs/slime-v0.toml sets the grid.
    fn column(court_position: f64) -> usize {
        ((court_position * 20.0) as usize).min(19)
    }

    #[test]
    fn net_hugger_and_back_wall_variants_land_in_different_columns() {
        let mut net_mean = 0.0;
        let mut wall_mean = 0.0;
        for seed in 0..8 {
            let (net, ..) = play(seed, |o| scripted_with_home(o, 2.0), scripted_action);
            let (wall, ..) = play(seed, |o| scripted_with_home(o, 22.5), scripted_action);
            assert!(
                net[0] > wall[0],
                "seed {seed}: net {} wall {}",
                net[0],
                wall[0]
            );
            assert_ne!(column(net[0]), column(wall[0]), "seed {seed}");
            net_mean += net[0] / 8.0;
            wall_mean += wall[0] / 8.0;
        }
        assert!(
            column(net_mean) >= column(wall_mean) + 5,
            "net {net_mean} wall {wall_mean}"
        );
    }

    #[test]
    fn anchor_beats_an_idle_slime_from_either_side() {
        for seed in 0..10 {
            let (_, anchor, idle) = play(seed, scripted_action, |_| 0);
            assert!(anchor == 5 && idle < 5, "seed {seed}: {anchor}-{idle}");
            let (_, idle, anchor) = play(seed, |_| 0, scripted_action);
            assert!(anchor == 5 && idle < 5, "seed {seed}: {idle}-{anchor}");
        }
    }

    #[test]
    fn home_is_the_starting_x() {
        let game = Game::new(0);
        assert_eq!(game.a.obs[0], HOME_X);
        assert_eq!(game.b.obs[0], HOME_X);
    }
}
