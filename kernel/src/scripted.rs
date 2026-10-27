//! The scripted anchor bot, frozen at kernel version 1. It is the ladder's
//! fixed point and the probe opponent every genome is measured against, so
//! any change here is a kernel version bump.
//!
//! It reads the same mirrored observation a genome sees (unscaled): its own
//! x runs from 2.0 at the net to 22.5 at its back wall, and the ball is over
//! its half when the ball's x is positive.

use crate::physics::{BACKWARD, FORWARD, JUMP, abs};

/// Where it waits while the ball is over the far half.
pub const HOME_X: f64 = 12.0;
/// It stands this far behind the ball, so its head sends the ball back over.
const BEHIND: f64 = 0.5;
/// No move when within this of the target, so it does not dither.
const DEADBAND: f64 = 0.2;

/// Track the ball's x while the ball is over its own half, wait at home
/// otherwise, jump when the ball is close and above.
pub fn scripted_action(obs: &[f64; 12]) -> u32 {
    scripted_with_home(obs, HOME_X)
}

/// The same rule with another home: the test variants that check the
/// descriptors use it; the kernel's exports only ever use `HOME_X`.
pub fn scripted_with_home(obs: &[f64; 12], home: f64) -> u32 {
    let (x, y, bx, by) = (obs[0], obs[1], obs[4], obs[5]);
    let target = if bx > 0.0 { bx + BEHIND } else { home };
    let mut action = 0;
    if x > target + DEADBAND {
        action |= FORWARD;
    } else if x < target - DEADBAND {
        action |= BACKWARD;
    }
    let dx = bx - x;
    let dy = by - y;
    if dy > 0.0 && dy < 3.5 && abs(dx) < 1.5 {
        action |= JUMP;
    }
    action
}
