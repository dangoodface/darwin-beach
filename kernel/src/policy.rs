//! A genome's policy: the MLP that turns a side's mirrored observation into
//! button presses. The same genome plays either side. The loops index the
//! weights the way docs/ABI.md writes them (`W1[i][j]`, `x[j]`).
#![allow(clippy::needless_range_loop)]

use crate::physics::abs;

/// MLP 12 to 64 to 3 since kernel version 3 (12 to 32 to 3 before): `W1[64][12]`,
/// `b1[64]`, `W2[3][64]`, `b2[3]`, 1,027 parameters. A version 2 genome padded with
/// zeros (`pad_v2`) keeps its 32 units in slots 0 to 31 and plays the same match: a
/// zero unit outputs softsign(0) = 0, and adding 0 * 0 to an output never changes
/// which side of 0 it is on.
pub const N_IN: usize = 12;
pub const N_HIDDEN: usize = 64;
pub const N_OUT: usize = 3;
pub const GENOME_LEN: usize = N_HIDDEN * N_IN + N_HIDDEN + N_OUT * N_HIDDEN + N_OUT;

/// The version 1 and 2 brain: 32 hidden units, 515 parameters.
pub const N_HIDDEN_V2: usize = 32;
pub const GENOME_LEN_V2: usize = N_HIDDEN_V2 * N_IN + N_HIDDEN_V2 + N_OUT * N_HIDDEN_V2 + N_OUT;

const W1: usize = 0;
const B1: usize = W1 + N_HIDDEN * N_IN;
const W2: usize = B1 + N_HIDDEN;
const B2: usize = W2 + N_OUT * N_HIDDEN;

/// RelativeState.getObservation divides by 10 so the inputs are near 1.
const SCALE: f64 = 10.0;

/// A version 2 genome in the version 3 layout: each block keeps its old values
/// first and gains zeros for hidden units 32 to 63 (W1 rows, b1 entries, and the
/// last 32 columns of each W2 row).
pub fn pad_v2(v2: &[f32]) -> [f32; GENOME_LEN] {
    assert_eq!(v2.len(), GENOME_LEN_V2);
    let (h2, n2) = (N_HIDDEN_V2, N_HIDDEN_V2 * N_IN);
    let mut out = [0.0f32; GENOME_LEN];
    out[W1..W1 + n2].copy_from_slice(&v2[..n2]);
    out[B1..B1 + h2].copy_from_slice(&v2[n2..n2 + h2]);
    for i in 0..N_OUT {
        let from = n2 + h2 + i * h2;
        out[W2 + i * N_HIDDEN..W2 + i * N_HIDDEN + h2].copy_from_slice(&v2[from..from + h2]);
    }
    out[B2..B2 + N_OUT].copy_from_slice(&v2[n2 + h2 + N_OUT * h2..]);
    out
}

/// A genome's net, upcast once per match (f32 to f64 is exact). The hidden
/// weights are stored a pair of units at a time, `w1[p][j] = [W1[2p][j],
/// W1[2p+1][j]]`, so the wasm build can run two units in one SIMD register.
#[derive(Clone, Copy)]
pub struct Net {
    w1: [[[f64; 2]; N_IN]; N_HIDDEN / 2],
    b1: [f64; N_HIDDEN],
    w2: [[f64; N_HIDDEN]; N_OUT],
    b2: [f64; N_OUT],
}

pub fn upcast(params: &[f32]) -> Net {
    let mut wide = [0.0; GENOME_LEN];
    for (out, p) in wide.iter_mut().zip(params) {
        *out = *p as f64;
    }
    from_f64(&wide)
}

/// The net of an already upcast parameter vector, in the genome layout.
pub fn from_f64(params: &[f64; GENOME_LEN]) -> Net {
    let mut net = Net {
        w1: [[[0.0; 2]; N_IN]; N_HIDDEN / 2],
        b1: [0.0; N_HIDDEN],
        w2: [[0.0; N_HIDDEN]; N_OUT],
        b2: [0.0; N_OUT],
    };
    for i in 0..N_HIDDEN {
        for j in 0..N_IN {
            net.w1[i / 2][j][i % 2] = params[W1 + i * N_IN + j];
        }
        net.b1[i] = params[B1 + i];
    }
    for i in 0..N_OUT {
        for j in 0..N_HIDDEN {
            net.w2[i][j] = params[W2 + i * N_HIDDEN + j];
        }
        net.b2[i] = params[B2 + i];
    }
    net
}

/// Softsign hidden layer, linear output, button i pressed when output i > 0.
/// Each unit sums sequentially from its bias, `acc = acc + w * x` over its
/// inputs in order: no fused multiply-add, no pairwise sums, so every engine
/// adds in the same order. The wasm build runs two units per f64x2 register,
/// each lane doing exactly the scalar unit's multiplies and adds.
pub fn net_action(net: &Net, obs: &[f64; N_IN]) -> u32 {
    let mut x = [0.0; N_IN];
    for j in 0..N_IN {
        x[j] = obs[j] / SCALE;
    }
    let mut h = [0.0; N_HIDDEN];
    hidden_layer(net, &x, &mut h);
    output_layer(net, &h)
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::assign_op_pattern)]
fn hidden_layer(net: &Net, x: &[f64; N_IN], h: &mut [f64; N_HIDDEN]) {
    for i in 0..N_HIDDEN {
        let mut acc = net.b1[i];
        for j in 0..N_IN {
            acc = acc + net.w1[i / 2][j][i % 2] * x[j];
        }
        h[i] = acc / (1.0 + abs(acc));
    }
}

#[cfg(target_arch = "wasm32")]
fn hidden_layer(net: &Net, x: &[f64; N_IN], h: &mut [f64; N_HIDDEN]) {
    // SAFETY: simd128 is part of every engine this module runs in (wasmtime,
    // Node, current browsers); an engine without it refuses the module at load.
    unsafe { hidden_layer_simd(net, x, h) }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
fn hidden_layer_simd(net: &Net, x: &[f64; N_IN], h: &mut [f64; N_HIDDEN]) {
    use core::arch::wasm32::*;
    let one = f64x2_splat(1.0);
    for p in 0..N_HIDDEN / 2 {
        let mut acc = f64x2(net.b1[2 * p], net.b1[2 * p + 1]);
        for j in 0..N_IN {
            let [w0, w1] = net.w1[p][j];
            acc = f64x2_add(acc, f64x2_mul(f64x2(w0, w1), f64x2_splat(x[j])));
        }
        let s = f64x2_div(acc, f64x2_add(one, f64x2_abs(acc)));
        h[2 * p] = f64x2_extract_lane::<0>(s);
        h[2 * p + 1] = f64x2_extract_lane::<1>(s);
    }
}

#[allow(clippy::assign_op_pattern)]
fn output_layer(net: &Net, h: &[f64; N_HIDDEN]) -> u32 {
    let mut action = 0;
    for i in 0..N_OUT {
        let mut acc = net.b2[i];
        for j in 0..N_HIDDEN {
            acc = acc + net.w2[i][j] * h[j];
        }
        if acc > 0.0 {
            action |= 1 << i;
        }
    }
    action
}

const _: () = assert!(N_HIDDEN.is_multiple_of(2));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{BACKWARD, FORWARD, JUMP};

    #[test]
    fn layout_is_1027() {
        assert_eq!((GENOME_LEN, GENOME_LEN_V2), (1027, 515));
        assert_eq!((B1, W2, B2), (768, 832, 1024));
    }

    #[test]
    #[allow(clippy::assign_op_pattern)]
    fn a_padded_v2_genome_acts_like_the_v2_net() {
        // The v2 net computed by hand on the v2 layout, against the padded v3 net.
        let v2: Vec<f32> = (0..GENOME_LEN_V2)
            .map(|i| ((i * 53 % 97) as f32) / 40.0 - 1.2)
            .collect();
        let w = upcast(&pad_v2(&v2));
        for k in 0..200 {
            let obs: [f64; N_IN] = core::array::from_fn(|j| ((k * 7 + j * 13) % 41) as f64 - 20.0);
            let x: [f64; N_IN] = core::array::from_fn(|j| obs[j] / SCALE);
            let mut h = [0.0; N_HIDDEN_V2];
            for i in 0..N_HIDDEN_V2 {
                let mut acc = v2[384 + i] as f64;
                for j in 0..N_IN {
                    acc = acc + v2[i * N_IN + j] as f64 * x[j];
                }
                h[i] = acc / (1.0 + abs(acc));
            }
            let mut want = 0;
            for i in 0..N_OUT {
                let mut acc = v2[512 + i] as f64;
                for j in 0..N_HIDDEN_V2 {
                    acc = acc + v2[416 + i * N_HIDDEN_V2 + j] as f64 * h[j];
                }
                if acc > 0.0 {
                    want |= 1 << i;
                }
            }
            assert_eq!(net_action(&w, &obs), want, "observation {k}");
        }
    }

    #[test]
    fn zero_genome_presses_nothing() {
        let w = upcast(&[0.0; GENOME_LEN]);
        assert_eq!(net_action(&w, &[1.0; N_IN]), 0);
    }

    #[test]
    fn output_biases_press_buttons() {
        let mut w = [0.0; GENOME_LEN];
        w[B2] = 1.0; // forward
        w[B2 + 2] = 0.5; // jump
        assert_eq!(net_action(&from_f64(&w), &[0.0; N_IN]), FORWARD | JUMP);
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
        assert_eq!(net_action(&from_f64(&w), &obs), BACKWARD);
        w[B2 + 1] = -(10.0 / 11.0) - 1e-12;
        assert_eq!(net_action(&from_f64(&w), &obs), 0);
    }

    #[test]
    fn nan_parameters_press_nothing() {
        let w = [f64::NAN; GENOME_LEN];
        assert_eq!(net_action(&from_f64(&w), &[1.0; N_IN]), 0);
    }
}
