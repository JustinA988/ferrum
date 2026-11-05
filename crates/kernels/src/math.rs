//! Formulas shared by the scalar reference and the AVX2 backend.
//!
//! Sharing them keeps angles, the SiLU definition, and softmax's maximum on
//! one code path so both backends agree where the contract demands identical
//! results.

/// `x * sigmoid(x)` with `sigmoid(x) = 1 / (1 + exp(-x))`.
#[inline]
pub(crate) fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Maximum used by stable softmax.
///
/// NaNs do not compare greater than anything, so they are skipped. `+inf`
/// wins. If no value compares greater than `-inf` (all NaN, all `-inf`, or
/// empty), the result stays non-finite and softmax writes zeros.
pub(crate) fn softmax_max(x: &[f32]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &v in x {
        if v > max {
            max = v;
        }
    }
    max
}

/// Llama half-split angle at pair index `index` (`0 .. rope_dim/2`).
///
/// Returns `(sin, cos)` of `pos / theta ^ (2 * index / rope_dim)`.
pub(crate) fn rope_sin_cos(pos: u32, theta: f32, index: usize, rope_dim: usize) -> (f32, f32) {
    let freq = pos as f32 / theta.powf((2 * index) as f32 / rope_dim as f32);
    freq.sin_cos()
}
