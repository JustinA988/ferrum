//! Scalar reference. Reductions accumulate left to right in `f32`.

use ferrum_core_types::{
    Backend, DType, FerrumError, GemmArgs, Kernels, RopeArgs, Q8_0_BLOCK, Q8_0_BLOCK_BYTES,
};

use crate::math;
use crate::pack;
use crate::par;
use crate::validate;

pub(crate) struct Scalar;

impl Kernels for Scalar {
    fn backend(&self) -> Backend {
        Backend::Scalar
    }

    fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), FerrumError> {
        validate::check_rms(x.len(), weight.len(), out.len())?;
        rms(x, weight, eps, out);
        Ok(())
    }

    fn silu(&self, x: &[f32], out: &mut [f32]) -> Result<(), FerrumError> {
        validate::same_len(x.len(), &[out.len()])?;
        for (slot, &v) in out.iter_mut().zip(x.iter()) {
            *slot = math::silu(v);
        }
        Ok(())
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), FerrumError> {
        validate::same_len(a.len(), &[b.len(), out.len()])?;
        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
        Ok(())
    }

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), FerrumError> {
        validate::same_len(a.len(), &[b.len(), out.len()])?;
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
        Ok(())
    }

    fn scale(&self, x: &mut [f32], factor: f32) -> Result<(), FerrumError> {
        for v in x.iter_mut() {
            *v *= factor;
        }
        Ok(())
    }

    fn softmax_inplace(&self, x: &mut [f32]) -> Result<(), FerrumError> {
        validate::check_softmax(x.len())?;
        softmax(x);
        Ok(())
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> Result<f32, FerrumError> {
        validate::check_dot(a.len(), b.len())?;
        Ok(dot(a, b))
    }

    fn rope(&self, args: RopeArgs<'_>) -> Result<(), FerrumError> {
        validate::check_rope(&args)?;
        rope_tensor(args.q, args.head_dim, args.rope_dim, args.pos, args.theta);
        rope_tensor(args.k, args.head_dim, args.rope_dim, args.pos, args.theta);
        Ok(())
    }

    fn gemv(
        &self,
        dtype: DType,
        w: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), FerrumError> {
        validate::check_gemv(dtype, w, rows, cols, x, y)?;
        gemv_into(dtype, w, cols, x, y);
        Ok(())
    }

    fn gemm(&self, args: GemmArgs<'_>) -> Result<(), FerrumError> {
        if args.n_tokens == 0 {
            return Ok(());
        }
        validate::check_gemm(&args)?;
        let GemmArgs {
            dtype,
            w,
            rows,
            cols,
            x,
            y,
            n_tokens: _,
        } = args;
        par::for_each_token(rows, cols, x, y, |x_tok, y_tok| {
            gemv_into(dtype, w, cols, x_tok, y_tok);
        });
        Ok(())
    }

    fn dequant_f32(&self, dtype: DType, w: &[u8], out: &mut [f32]) -> Result<(), FerrumError> {
        validate::check_dequant(dtype, w, out)?;
        dequant(dtype, w, out);
        Ok(())
    }
}

pub(crate) fn softmax(x: &mut [f32]) {
    let max = math::softmax_max(x);
    if !max.is_finite() {
        x.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn rms(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len().min(weight.len()).min(out.len());
    if n == 0 {
        return;
    }
    let mut sumsq = 0.0f32;
    for &v in x.iter().take(n) {
        sumsq += v * v;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv * weight[i];
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += a[i] * b[i];
    }
    acc
}

pub(crate) fn rope_tensor(
    data: &mut [f32],
    head_dim: usize,
    rope_dim: usize,
    pos: u32,
    theta: f32,
) {
    if head_dim == 0 || rope_dim == 0 || rope_dim > head_dim {
        return;
    }
    let half = rope_dim / 2;
    for head in data.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let (sin, cos) = math::rope_sin_cos(pos, theta, i, rope_dim);
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = b * cos + a * sin;
        }
    }
}

pub(crate) fn gemv_into(dtype: DType, w: &[u8], cols: usize, x: &[f32], y: &mut [f32]) {
    let Some(stride) = pack::row_stride(dtype, cols) else {
        return;
    };
    if stride == 0 {
        return;
    }
    for (row, slot) in w.chunks_exact(stride).zip(y.iter_mut()) {
        *slot = dot_row(dtype, row, x);
    }
}

fn dot_row(dtype: DType, row: &[u8], x: &[f32]) -> f32 {
    match dtype {
        DType::F32 => dot_f32_row(row, x),
        DType::F16 => dot_f16_row(row, x),
        DType::Q8_0 => dot_q8_row(row, x),
        DType::Other(_) => 0.0,
    }
}

fn dot_f32_row(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (k, &xv) in x.iter().enumerate() {
        let Some(w) = pack::f32_at(row, k) else {
            break;
        };
        acc += w * xv;
    }
    acc
}

fn dot_f16_row(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (k, &xv) in x.iter().enumerate() {
        let Some(w) = pack::f16_at(row, k) else {
            break;
        };
        acc += w * xv;
    }
    acc
}

fn dot_q8_row(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    let (blocks, _) = row.as_chunks::<Q8_0_BLOCK_BYTES>();
    for block in blocks {
        if x_off >= x.len() {
            break;
        }
        let Some(scale) = pack::q8_scale(block) else {
            break;
        };
        for j in 0..Q8_0_BLOCK {
            let Some(&xv) = x.get(x_off + j) else {
                return acc;
            };
            let Some(&byte) = block.get(2 + j) else {
                return acc;
            };
            acc += (scale * pack::q8_quant(byte)) * xv;
        }
        x_off += Q8_0_BLOCK;
    }
    acc
}

pub(crate) fn dequant(dtype: DType, w: &[u8], out: &mut [f32]) {
    match dtype {
        DType::F32 => {
            for (i, slot) in out.iter_mut().enumerate() {
                let Some(v) = pack::f32_at(w, i) else {
                    break;
                };
                *slot = v;
            }
        }
        DType::F16 => {
            for (i, slot) in out.iter_mut().enumerate() {
                let Some(v) = pack::f16_at(w, i) else {
                    break;
                };
                *slot = v;
            }
        }
        DType::Q8_0 => {
            let mut i = 0usize;
            let (blocks, _) = w.as_chunks::<Q8_0_BLOCK_BYTES>();
            for block in blocks {
                if i + Q8_0_BLOCK > out.len() {
                    break;
                }
                let Some(scale) = pack::q8_scale(block) else {
                    break;
                };
                for j in 0..Q8_0_BLOCK {
                    let Some(&byte) = block.get(2 + j) else {
                        return;
                    };
                    out[i + j] = scale * pack::q8_quant(byte);
                }
                i += Q8_0_BLOCK;
            }
        }
        DType::Other(_) => {}
    }
}
