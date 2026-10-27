//! FNV-1a 64, chained across ticks: the fingerprint that proves two engines
//! played the same match.

pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Copy)]
pub struct Fnv1a64(pub u64);

impl Fnv1a64 {
    #[inline(always)]
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    #[inline(always)]
    pub fn write_u32(&mut self, value: u32) {
        self.write_bytes(&value.to_le_bytes());
    }

    #[inline(always)]
    pub fn write_f64(&mut self, value: f64) {
        self.write_bytes(&value.to_bits().to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv(bytes: &[u8]) -> u64 {
        let mut h = Fnv1a64(FNV_OFFSET);
        h.write_bytes(bytes);
        h.0
    }

    #[test]
    fn known_vectors() {
        assert_eq!(fnv(b""), 0xcbf29ce484222325);
        assert_eq!(fnv(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn little_endian_words() {
        let mut h = Fnv1a64(FNV_OFFSET);
        h.write_u32(0x0403_0201);
        assert_eq!(h.0, fnv(&[1, 2, 3, 4]));
    }
}
