//! A genome's policy: the MLP that turns a side's mirrored observation into
//! button presses. The same genome plays either side.

use crate::physics::abs;

/// MLP 12 to 32 to 3: `W1[32][12]`, `b1[32]`, `W2[3][32]`, `b2[3]`.
pub const N_IN: usize = 12;
pub const N_HIDDEN: usize = 32;
pub const N_OUT: usize = 3;
pub const GENOME_LEN: usize = N_HIDDEN * N_IN + N_HIDDEN + N_OUT * N_HIDDEN + N_OUT;

const W1: usize = 0;
const B1: usize = W1 + N_HIDDEN * N_IN;
const W2: usize = B1 + N_HIDDEN;
const B2: usize = W2 + N_OUT * N_HIDDEN;

/// RelativeState.getObservation divides by 10 so the inputs are near 1.
const SCALE: f64 = 10.0;

/// A genome's parameters upcast once per match; f32 to f64 is exact.
pub fn upcast(params: &[f32]) -> [f64; GENOME_LEN] {
    let mut w = [0.0; GENOME_LEN];
    for (out, p) in w.iter_mut().zip(params) {
        *out = *p as f64;
    }
    w
}

/// Softsign hidden layer, linear output, button i pressed when output i > 0.
/// Each unit sums sequentially from its bias: no fused multiply-add, no
/// pairwise sums, so every engine adds in the same order. Written as
/// `acc = acc + w * x` to mirror docs/ABI.md.
#[allow(clippy::assign_op_pattern)]
pub fn net_action(w: &[f64; GENOME_LEN], obs: &[f64; N_IN]) -> u32 {
    let mut x = [0.0; N_IN];
    for j in 0..N_IN {
        x[j] = obs[j] / SCALE;
    }
    let mut h = [0.0; N_HIDDEN];
    for i in 0..N_HIDDEN {
        let row = &w[W1 + i * N_IN..W1 + (i + 1) * N_IN];
        let mut acc = w[B1 + i];
        for j in 0..N_IN {
            acc = acc + row[j] * x[j];
        }
        h[i] = acc / (1.0 + abs(acc));
    }
    let mut action = 0;
    for i in 0..N_OUT {
        let row = &w[W2 + i * N_HIDDEN..W2 + (i + 1) * N_HIDDEN];
        let mut acc = w[B2 + i];
        for j in 0..N_HIDDEN {
            acc = acc + row[j] * h[j];
        }
        if acc > 0.0 {
            action |= 1 << i;
        }
    }
    action
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{BACKWARD, FORWARD, JUMP};

    #[test]
    fn layout_is_515() {
        assert_eq!(GENOME_LEN, 515);
        assert_eq!((B1, W2, B2), (384, 416, 512));
    }

    #[test]
    fn zero_genome_presses_nothing() {
        let w = [0.0; GENOME_LEN];
        assert_eq!(net_action(&w, &[1.0; N_IN]), 0);
    }

    #[test]
    fn output_biases_press_buttons() {
        let mut w = [0.0; GENOME_LEN];
        w[B2] = 1.0; // forward
        w[B2 + 2] = 0.5; // jump
        assert_eq!(net_action(&w, &[0.0; N_IN]), FORWARD | JUMP);
    }

    #[test]
    fn hidden_layer_is_softsign() {
        // One hidden unit reads input 0 with weight 10: obs 10 -> x = 1 -> acc 10 -> 10/11.
        // Output 1 reads it with weight 1 and bias -10/11 + tiny: pressed only above 10/11.
        let mut w = [0.0; GENOME_LEN];
        w[W1] = 10.0;
        w[W2 + N_HIDDEN] = 1.0;
        w[B2 + 1] = -(10.0 / 11.0) + 1e-12;
        let mut obs = [0.0; N_IN];
        obs[0] = 10.0;
        assert_eq!(net_action(&w, &obs), BACKWARD);
        w[B2 + 1] = -(10.0 / 11.0) - 1e-12;
        assert_eq!(net_action(&w, &obs), 0);
    }

    #[test]
    fn nan_parameters_press_nothing() {
        let w = [f64::NAN; GENOME_LEN];
        assert_eq!(net_action(&w, &[1.0; N_IN]), 0);
    }
}
