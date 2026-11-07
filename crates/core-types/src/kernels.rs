//! Kernel backend choice and the numeric operations a backend must provide.

use crate::dtype::DType;
use crate::error::FerrumError;

/// Runtime kernel choice. Revision 1 never selects [`Backend::Avx512`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Scalar,
    Avx2Fma,
    Avx512,
    Neon,
}

/// One-token Llama RoPE. `q` and `k` are head-major and mutated in place.
#[derive(Debug)]
pub struct RopeArgs<'a> {
    pub q: &'a mut [f32],
    pub k: &'a mut [f32],
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub pos: u32,
    pub theta: f32,
}

/// `y[token * rows + m] = dot(W[m], x[token * cols + k])`.
/// `n_tokens == 0` writes nothing and returns `Ok`.
#[derive(Debug)]
pub struct GemmArgs<'a> {
    pub dtype: DType,
    pub w: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub x: &'a [f32],
    pub n_tokens: usize,
    pub y: &'a mut [f32],
}

/// Stateless numeric kernels. Implementors use unaligned SIMD loads.
///
/// `f32` slices are the activation boundary. Quantized weights stay packed bytes.
/// Length mismatches and unsupported dtypes return [`FerrumError::Kernel`]. They do not panic.
///
/// GEMV / GEMM layout: `cols` (K) is contiguous in the GGUF weight. For each row `m` in
/// `0..rows`, `y[m] = dot(W[m, 0..cols], x[0..cols])`. Q8_0 blocks run along K.
/// GEMM batch layout is row-major tokens: `x[token * cols + k]`, `y[token * rows + m]`.
pub trait Kernels: Send + Sync {
    fn backend(&self) -> Backend;

    fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), FerrumError>;

    fn silu(&self, x: &[f32], out: &mut [f32]) -> Result<(), FerrumError>;

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), FerrumError>;

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), FerrumError>;

    fn scale(&self, x: &mut [f32], factor: f32) -> Result<(), FerrumError>;

    /// Stable softmax. A non-finite maximum writes zeros.
    fn softmax_inplace(&self, x: &mut [f32]) -> Result<(), FerrumError>;

    /// Dot product. Empty inputs are an error.
    fn dot(&self, a: &[f32], b: &[f32]) -> Result<f32, FerrumError>;

    /// Llama `GGML_ROPE_TYPE_NORMAL` (adjacent-pair) RoPE on one token. See `docs/ARCHITECTURE.md`.
    fn rope(&self, args: RopeArgs<'_>) -> Result<(), FerrumError>;

    fn gemv(
        &self,
        dtype: DType,
        w: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), FerrumError>;

    fn gemm(&self, args: GemmArgs<'_>) -> Result<(), FerrumError>;

    fn dequant_f32(&self, dtype: DType, w: &[u8], out: &mut [f32]) -> Result<(), FerrumError>;
}
