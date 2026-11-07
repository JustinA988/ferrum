//! Formulas both backends call.
//!
//! SiLU, the RoPE angle, and softmax's max live here so the scalar and AVX2
//! paths share one definition where the contract requires the same result.

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

/// RoPE angle at pair index `index` (`0 .. rope_dim/2`).
///
/// Returns `(sin, cos)` of `pos / theta ^ (2 * index / rope_dim)`.
/// Pairing is consecutive `(2i, 2i+1)` (llama.cpp `GGML_ROPE_TYPE_NORMAL`).
pub(crate) fn rope_sin_cos(pos: u32, theta: f32, index: usize, rope_dim: usize) -> (f32, f32) {
    let freq = pos as f32 / theta.powf((2 * index) as f32 / rope_dim as f32);
    freq.sin_cos()
}
