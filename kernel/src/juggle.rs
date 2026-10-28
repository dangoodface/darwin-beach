//! Juggling, counted per point (kernel version 2): JOGA BONITO, Daniel's specialist,
//! juggles the ball on its head four or more times while moving across its own court,
//! then kills the point. The eight aggregate descriptors cannot see that, so each side
//! keeps this tracker, fed once per hashed tick.
//!
//! The definition, fixed by Daniel: a streak is consecutive own touches (the bounces
//! the hits descriptor counts) with no net crossing in between; while a streak is open
//! the slime's lowest and highest x are tracked over every tick from its first touch
//! on. When the ball crosses the net out of this side's half, the streak is sent: its
//! touches and its span (highest minus lowest x) are kept, and a new streak can start.
//! When a point ends, the winner's most recently sent streak decides: the point is
//! juggled when it had at least JUGGLE_MIN_TOUCHES touches and a span of at least
//! JUGGLE_MIN_SPAN. Everything but the match totals resets with every point.
//!
//! Spikes (version 3, Daniel's, on the same machinery): the winner's most recently
//! sent streak had at least SPIKE_MIN_TOUCHES touches, the spike included (so one
//! carry, then the spike), and that last touch had the slime's centre within
//! SPIKE_MAX_FROM_NET of the net and sent the ball off at SPIKE_MIN_SPEED or more
//! (its speed right after the bounce).

use crate::descriptors::COURT_DEPTH;

pub const JUGGLE_MIN_TOUCHES: u32 = 4;
/// Half of the slime's own court: x*dir runs over COURT_DEPTH (2.0 to 22.5).
pub const JUGGLE_MIN_SPAN: f64 = COURT_DEPTH / 2.0;
/// Touches in the sent streak, the spike included: at least one carry before it.
pub const SPIKE_MIN_TOUCHES: u32 = 2;
/// The slime's `x * dir` at the spiking touch: the net is at 0, the back wall at 24.
pub const SPIKE_MAX_FROM_NET: f64 = 4.0;
/// The ball's speed right after the spiking bounce.
pub const SPIKE_MIN_SPEED: f64 = 35.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Juggle {
    streak: u32,
    lo: f64,
    hi: f64,
    sent_streak: u32,
    sent_span: f64,
    last_from_net: f64,
    last_speed: f64,
    sent_from_net: f64,
    sent_speed: f64,
    /// Points this side won with a juggle, over the match.
    pub juggled_points: u32,
    /// The longest streak of the match, sent or not.
    pub max_streak: u32,
    /// Points this side won with a spike, over the match (version 3).
    pub spiked_points: u32,
}

impl Juggle {
    pub const ZERO: Juggle = Juggle {
        streak: 0,
        lo: 0.0,
        hi: 0.0,
        sent_streak: 0,
        sent_span: 0.0,
        last_from_net: 0.0,
        last_speed: 0.0,
        sent_from_net: 0.0,
        sent_speed: 0.0,
        juggled_points: 0,
        max_streak: 0,
        spiked_points: 0,
    };

    /// One hashed tick: this slime's x after its update and its `dir`, and the ball's
    /// speed right after bouncing off it, if it did this tick.
    pub fn tick(&mut self, x: f64, dir: f64, touch_speed: Option<f64>) {
        if let Some(speed) = touch_speed {
            self.last_from_net = x * dir;
            self.last_speed = speed;
            if self.streak == 0 {
                self.lo = x;
                self.hi = x;
            }
            self.streak += 1;
            if self.streak > self.max_streak {
                self.max_streak = self.streak;
            }
        }
        if self.streak > 0 {
            if x < self.lo {
                self.lo = x;
            }
            if x > self.hi {
                self.hi = x;
            }
        }
    }

    /// The ball crossed the net out of this side's half.
    pub fn sent(&mut self) {
        self.sent_streak = self.streak;
        self.sent_span = if self.streak > 0 {
            self.hi - self.lo
        } else {
            0.0
        };
        self.sent_from_net = self.last_from_net;
        self.sent_speed = self.last_speed;
        self.streak = 0;
    }

    /// The point ended; `won` says whether this side won it.
    pub fn point_over(&mut self, won: bool) {
        if won && self.sent_streak >= JUGGLE_MIN_TOUCHES && self.sent_span >= JUGGLE_MIN_SPAN {
            self.juggled_points += 1;
        }
        if won
            && self.sent_streak >= SPIKE_MIN_TOUCHES
            && self.sent_from_net <= SPIKE_MAX_FROM_NET
            && self.sent_speed >= SPIKE_MIN_SPEED
        {
            self.spiked_points += 1;
        }
        self.streak = 0;
        self.sent_streak = 0;
        self.sent_span = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Side A's touches at the given xs on consecutive ticks, the ball leaving gently.
    fn touches(j: &mut Juggle, xs: &[f64]) {
        for &x in xs {
            j.tick(x, -1.0, Some(12.0));
        }
    }

    const WIDE: [f64; 4] = [-20.0, -16.0, -12.5, -9.0]; // span 11.0, above 10.25

    #[test]
    fn thresholds() {
        assert_eq!(JUGGLE_MIN_TOUCHES, 4);
        assert_eq!(JUGGLE_MIN_SPAN, 10.25);
    }

    #[test]
    fn four_touches_with_a_wide_span_on_a_won_point_count() {
        let mut j = Juggle::ZERO;
        touches(&mut j, &WIDE);
        j.sent();
        j.point_over(true);
        assert_eq!((j.juggled_points, j.max_streak), (1, 4));
    }

    #[test]
    fn three_touches_do_not() {
        let mut j = Juggle::ZERO;
        touches(&mut j, &[-22.0, -16.0, -9.0]);
        j.sent();
        j.point_over(true);
        assert_eq!((j.juggled_points, j.max_streak), (0, 3));
    }

    #[test]
    fn four_touches_without_the_span_do_not() {
        let mut j = Juggle::ZERO;
        touches(&mut j, &[-12.0, -13.0, -11.0, -12.5]);
        j.tick(-20.0, -1.0, None); // ticks between touches count for the span too, but stay short here
        j.tick(-10.0, -1.0, None);
        j.sent();
        j.point_over(true);
        assert_eq!(j.juggled_points, 0);
        let mut j = Juggle::ZERO; // the same moves after the first touch do reach the span
        touches(&mut j, &[-12.0, -13.0, -11.0, -12.5]);
        j.tick(-22.0, -1.0, None);
        j.tick(-10.0, -1.0, None);
        j.sent();
        j.point_over(true);
        assert_eq!(j.juggled_points, 1);
    }

    #[test]
    fn the_span_starts_at_the_first_touch() {
        let mut j = Juggle::ZERO;
        j.tick(-22.0, -1.0, None); // before the streak opens: not part of the span
        touches(&mut j, &[-12.0, -12.0, -12.0, -12.0]);
        j.sent();
        j.point_over(true);
        assert_eq!(j.juggled_points, 0);
    }

    #[test]
    fn a_streak_broken_by_a_crossing_does_not_carry_over() {
        let mut j = Juggle::ZERO;
        touches(&mut j, &WIDE[..2]);
        j.sent(); // over the net and back
        touches(&mut j, &WIDE[2..]);
        j.sent();
        j.point_over(true);
        assert_eq!((j.juggled_points, j.max_streak), (0, 2));
    }

    #[test]
    fn the_most_recent_sent_streak_decides() {
        let mut j = Juggle::ZERO;
        touches(&mut j, &WIDE); // a juggle sent over...
        j.sent();
        j.tick(-12.0, -1.0, Some(12.0)); // ...comes back, one touch sends it again
        j.sent();
        j.point_over(true);
        assert_eq!(j.juggled_points, 0);
    }

    #[test]
    fn the_losers_streak_never_counts() {
        let mut j = Juggle::ZERO;
        touches(&mut j, &[-22.0, -18.0, -14.0, -10.0, -6.0]);
        j.sent();
        j.point_over(false);
        assert_eq!(j.juggled_points, 0);
    }

    #[test]
    fn max_streak_does_not_depend_on_winning_or_sending() {
        let mut j = Juggle::ZERO;
        touches(&mut j, &[-12.0; 6]); // dropped on its own side: never sent, point lost
        j.point_over(false);
        assert_eq!((j.juggled_points, j.max_streak), (0, 6));
        touches(&mut j, &[-12.0; 2]);
        j.point_over(true);
        assert_eq!(j.max_streak, 6);
    }

    #[test]
    fn everything_but_the_totals_resets_with_each_point() {
        let mut j = Juggle::ZERO;
        touches(&mut j, &WIDE);
        j.sent();
        j.point_over(false); // lost this one
        j.point_over(true); // the next point is won without touching the ball
        assert_eq!((j.juggled_points, j.max_streak), (0, 4));
    }

    /// Side A carries the ball once, then spikes from `from_net` at `speed`, and it goes over.
    fn carry_and_spike(j: &mut Juggle, from_net: f64, speed: f64) {
        touches(j, &[-6.0]);
        j.tick(-from_net, -1.0, Some(speed));
        j.sent();
    }

    #[test]
    fn spike_thresholds() {
        assert_eq!(
            (SPIKE_MIN_TOUCHES, SPIKE_MAX_FROM_NET, SPIKE_MIN_SPEED),
            (2, 4.0, 35.0)
        );
    }

    #[test]
    fn a_carry_and_a_fast_spike_at_the_net_win_a_spiked_point() {
        let mut j = Juggle::ZERO;
        carry_and_spike(&mut j, 3.5, 36.0);
        j.point_over(true);
        assert_eq!((j.spiked_points, j.juggled_points), (1, 0));
        let mut j = Juggle::ZERO; // more carries are fine
        touches(&mut j, &[-12.0, -9.0, -6.0]);
        j.tick(-3.0, -1.0, Some(40.0));
        j.sent();
        j.point_over(true);
        assert_eq!(j.spiked_points, 1);
    }

    #[test]
    fn a_spike_needs_a_carry_the_net_and_the_speed() {
        for (carries, from_net, speed) in [(0, 3.5, 36.0), (1, 4.5, 36.0), (1, 3.5, 34.9)] {
            let mut j = Juggle::ZERO;
            touches(&mut j, &[-6.0][..carries]);
            j.tick(-from_net, -1.0, Some(speed));
            j.sent();
            j.point_over(true);
            assert_eq!(
                j.spiked_points, 0,
                "{carries} carries, {from_net} from the net, speed {speed}"
            );
        }
        let mut j = Juggle::ZERO; // right at the limits counts
        carry_and_spike(&mut j, 4.0, 35.0);
        j.point_over(true);
        assert_eq!(j.spiked_points, 1);
    }

    #[test]
    fn the_losers_spike_and_an_old_spike_do_not_count() {
        let mut j = Juggle::ZERO;
        carry_and_spike(&mut j, 3.0, 40.0);
        j.point_over(false);
        assert_eq!(j.spiked_points, 0);
        carry_and_spike(&mut j, 3.0, 40.0); // a spike that comes back...
        j.tick(-12.0, -1.0, Some(12.0)); // ...and goes over again with one plain touch
        j.sent();
        j.point_over(true);
        assert_eq!(j.spiked_points, 0);
    }

    #[test]
    fn side_b_measures_from_its_own_side_of_the_net() {
        let mut j = Juggle::ZERO;
        j.tick(6.0, 1.0, Some(12.0));
        j.tick(3.0, 1.0, Some(38.0));
        j.sent();
        j.point_over(true);
        assert_eq!(j.spiked_points, 1);
    }
}
