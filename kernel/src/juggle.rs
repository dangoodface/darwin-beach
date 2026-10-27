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
//! JUGGLE_MIN_SPAN. Everything but the two match totals resets with every point.

use crate::descriptors::COURT_DEPTH;

pub const JUGGLE_MIN_TOUCHES: u32 = 4;
/// Half of the slime's own court: x*dir runs over COURT_DEPTH (2.0 to 22.5).
pub const JUGGLE_MIN_SPAN: f64 = COURT_DEPTH / 2.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Juggle {
    streak: u32,
    lo: f64,
    hi: f64,
    sent_streak: u32,
    sent_span: f64,
    /// Points this side won with a juggle, over the match.
    pub juggled_points: u32,
    /// The longest streak of the match, sent or not.
    pub max_streak: u32,
}

impl Juggle {
    pub const ZERO: Juggle = Juggle {
        streak: 0,
        lo: 0.0,
        hi: 0.0,
        sent_streak: 0,
        sent_span: 0.0,
        juggled_points: 0,
        max_streak: 0,
    };

    /// One hashed tick: this slime's x after its update, and whether the ball bounced
    /// off it this tick.
    pub fn tick(&mut self, x: f64, touched: bool) {
        if touched {
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
        self.streak = 0;
    }

    /// The point ended; `won` says whether this side won it.
    pub fn point_over(&mut self, won: bool) {
        if won && self.sent_streak >= JUGGLE_MIN_TOUCHES && self.sent_span >= JUGGLE_MIN_SPAN {
            self.juggled_points += 1;
        }
        self.streak = 0;
        self.sent_streak = 0;
        self.sent_span = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Touches at the given xs on consecutive ticks.
    fn touches(j: &mut Juggle, xs: &[f64]) {
        for &x in xs {
            j.tick(x, true);
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
        j.tick(-20.0, false); // ticks between touches count for the span too, but stay short here
        j.tick(-10.0, false);
        j.sent();
        j.point_over(true);
        assert_eq!(j.juggled_points, 0);
        let mut j = Juggle::ZERO; // the same moves after the first touch do reach the span
        touches(&mut j, &[-12.0, -13.0, -11.0, -12.5]);
        j.tick(-22.0, false);
        j.tick(-10.0, false);
        j.sent();
        j.point_over(true);
        assert_eq!(j.juggled_points, 1);
    }

    #[test]
    fn the_span_starts_at_the_first_touch() {
        let mut j = Juggle::ZERO;
        j.tick(-22.0, false); // before the streak opens: not part of the span
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
        j.tick(-12.0, true); // ...comes back, one touch sends it again
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
}
