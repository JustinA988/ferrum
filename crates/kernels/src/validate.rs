//! Shape checks run before any arithmetic.
//!
//! Each backend calls these first, so a bad length, a zero dimension, or an
//! unsupported dtype becomes the same `FerrumError::Kernel` on scalar and AVX2.

use ferrum_core_types::{DType, FerrumError, GemmArgs, RopeArgs, Q8_0_BLOCK, Q8_0_BLOCK_BYTES};

fn kernel(message: &str) -> FerrumError {
    FerrumError::kernel(message)
}

pub(crate) fn length_mismatch() -> FerrumError {
    kernel("length mismatch")
}

fn shape_overflow() -> FerrumError {
    kernel("shape overflow")
}

fn unsupported_dtype() -> FerrumError {
    kernel("unsupported dtype")
}

fn empty_dot() -> FerrumError {
    kernel("empty dot")
}

fn q8_k() -> FerrumError {
    kernel("q8_0 k is not a multiple of 32")
}

pub(crate) fn same_len(first: usize, rest: &[usize]) -> Result<(), FerrumError> {
    if rest.iter().all(|len| *len == first) {
        Ok(())
    } else {
        Err(length_mismatch())
    }
}

pub(crate) fn check_rms(x: usize, weight: usize, out: usize) -> Result<(), FerrumError> {
    same_len(x, &[weight, out])?;
    if x == 0 {
        return Err(kernel("empty rms_norm"));
    }
    Ok(())
}

pub(crate) fn check_dot(a: usize, b: usize) -> Result<(), FerrumError> {
    same_len(a, &[b])?;
    if a == 0 {
        return Err(empty_dot());
    }
    Ok(())
}

pub(crate) fn check_softmax(n: usize) -> Result<(), FerrumError> {
    if n == 0 {
        Err(kernel("empty softmax"))
    } else {
        Ok(())
    }
}

/// Theta, then parity, then shape. Both backends call this before touching `q` or `k`.
pub(crate) fn check_rope(args: &RopeArgs<'_>) -> Result<(), FerrumError> {
    // `<=` misses NaN, and NaN is not a positive base.
    if args.theta <= 0.0 || args.theta.is_nan() {
        return Err(kernel("theta must be positive"));
    }
    if !args.rope_dim.is_multiple_of(2) {
        return Err(kernel("rope_dim must be even"));
    }
    if args.rope_dim > args.head_dim {
        return Err(kernel("rope_dim exceeds head_dim"));
    }
    let q_len = args
        .n_heads
        .checked_mul(args.head_dim)
        .ok_or_else(shape_overflow)?;
    let k_len = args
        .n_kv_heads
        .checked_mul(args.head_dim)
        .ok_or_else(shape_overflow)?;
    if args.q.len() != q_len || args.k.len() != k_len {
        return Err(length_mismatch());
    }
    Ok(())
}

fn weight_nbytes(dtype: DType, rows: usize, cols: usize) -> Result<usize, FerrumError> {
    match dtype {
        DType::F32 => rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(shape_overflow),
        DType::F16 => rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(shape_overflow),
        DType::Q8_0 => {
            let row_bytes = (cols / Q8_0_BLOCK)
                .checked_mul(Q8_0_BLOCK_BYTES)
                .ok_or_else(shape_overflow)?;
            rows.checked_mul(row_bytes).ok_or_else(shape_overflow)
        }
        DType::Other(_) => Err(unsupported_dtype()),
    }
}

/// GEMV checks. `cols == 0` is an empty dot. Q8_0 then requires `cols % 32 == 0`.
pub(crate) fn check_gemv(
    dtype: DType,
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    y: &[f32],
) -> Result<(), FerrumError> {
    check_matrix(dtype, w, rows, cols)?;
    if x.len() != cols || y.len() != rows {
        return Err(length_mismatch());
    }
    Ok(())
}

/// GEMM checks. The caller already returned `Ok` for `n_tokens == 0`.
pub(crate) fn check_gemm(args: &GemmArgs<'_>) -> Result<(), FerrumError> {
    check_matrix(args.dtype, args.w, args.rows, args.cols)?;
    let x_len = args
        .n_tokens
        .checked_mul(args.cols)
        .ok_or_else(shape_overflow)?;
    let y_len = args
        .n_tokens
        .checked_mul(args.rows)
        .ok_or_else(shape_overflow)?;
    if args.x.len() != x_len || args.y.len() != y_len {
        return Err(length_mismatch());
    }
    Ok(())
}

fn check_matrix(dtype: DType, w: &[u8], rows: usize, cols: usize) -> Result<(), FerrumError> {
    if !dtype.is_compute_supported() {
        return Err(unsupported_dtype());
    }
    if cols == 0 {
        return Err(empty_dot());
    }
    if dtype == DType::Q8_0 && !cols.is_multiple_of(Q8_0_BLOCK) {
        return Err(q8_k());
    }
    let nbytes = weight_nbytes(dtype, rows, cols)?;
    if w.len() != nbytes {
        return Err(length_mismatch());
    }
    Ok(())
}

pub(crate) fn check_dequant(dtype: DType, w: &[u8], out: &[f32]) -> Result<(), FerrumError> {
    if !dtype.is_compute_supported() {
        return Err(unsupported_dtype());
    }
    let nbytes = match dtype {
        DType::F32 => out.len().checked_mul(4).ok_or_else(shape_overflow)?,
        DType::F16 => out.len().checked_mul(2).ok_or_else(shape_overflow)?,
        DType::Q8_0 => {
            if !out.len().is_multiple_of(Q8_0_BLOCK) {
                return Err(q8_k());
            }
            (out.len() / Q8_0_BLOCK)
                .checked_mul(Q8_0_BLOCK_BYTES)
                .ok_or_else(shape_overflow)?
        }
        DType::Other(_) => return Err(unsupported_dtype()),
    };
    if w.len() != nbytes {
        return Err(length_mismatch());
    }
    Ok(())
}
