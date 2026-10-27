//! PCG32, the reference `pcg32_random_r` and `pcg32_srandom_r` from pcg-random.org.
//! The serve is the only thing in the game that is random.

const MULTIPLIER: u64 = 6364136223846793005;

/// The stream every kernel game uses (docs/ABI.md, RNG).
pub const INITSEQ: u64 = 0xda3e39cb94b95bdb;

#[derive(Clone, Copy)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub const fn new(seed: u64) -> Pcg32 {
        Pcg32::with_stream(seed, INITSEQ)
    }

    /// `pcg32_srandom_r(rng, initstate, initseq)`.
    pub const fn with_stream(initstate: u64, initseq: u64) -> Pcg32 {
        let mut rng = Pcg32 {
            state: 0,
            inc: (initseq << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(initstate);
        rng.next_u32();
        rng
    }

    pub const fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << (32u32.wrapping_sub(rot) & 31))
    }

    /// 24 random bits scaled into [0, 1).
    pub const fn u01(&mut self) -> f64 {
        (self.next_u32() >> 8) as f64 * (1.0 / 16777216.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_demo() {
        // pcg32-demo from pcg-c-basic: seed 42, sequence 54.
        let mut rng = Pcg32::with_stream(42, 54);
        let got: [u32; 6] = core::array::from_fn(|_| rng.next_u32());
        assert_eq!(
            got,
            [
                0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e
            ]
        );
    }

    #[test]
    fn u01_stays_in_range() {
        let mut rng = Pcg32::new(7);
        for _ in 0..100_000 {
            let u = rng.u01();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
