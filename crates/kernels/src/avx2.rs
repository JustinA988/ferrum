//! AVX2+FMA backend. Loads and stores are unaligned.
//!
//! `add` and `mul` use `addps` / `mulps` (not FMA) so finite lanes match scalar
//! IEEE results. FMA is used in the exponential for SiLU and softmax.
//! MXCSR is left alone: flush-to-zero would break bitwise `add` and `mul`.
//!
//! # Safety
//!
//! Every `target_feature` function in this module requires AVX2 and FMA on the
//! host CPU. [`crate::select`] is the only constructor, and it checks both
//! features. Detection is per CPU, so rayon workers may call these functions.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use ferrum_core_types::{
    Backend, DType, FerrumError, GemmArgs, Kernels, RopeArgs, Q8_0_BLOCK, Q8_0_BLOCK_BYTES,
};

use crate::math;
use crate::pack;
use crate::par;
use crate::scalar;
use crate::validate;

const LANES: usize = 8;

pub(crate) struct Avx2Fma;

impl Kernels for Avx2Fma {
    fn backend(&self) -> Backend {
        Backend::Avx2Fma
    }

    fn rms_norm(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), FerrumError> {
        validate::check_rms(x.len(), weight.len(), out.len())?;
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        unsafe { rms_avx2(x, weight, eps, out) }
        Ok(())
    }

    fn silu(&self, x: &[f32], out: &mut [f32]) -> Result<(), FerrumError> {
        validate::same_len(x.len(), &[out.len()])?;
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        unsafe { silu_avx2(x, out) }
        Ok(())
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), FerrumError> {
        validate::same_len(a.len(), &[b.len(), out.len()])?;
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        unsafe { zip_avx2(a, b, out, ZipOp::Mul) }
        Ok(())
    }

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), FerrumError> {
        validate::same_len(a.len(), &[b.len(), out.len()])?;
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        unsafe { zip_avx2(a, b, out, ZipOp::Add) }
        Ok(())
    }

    fn scale(&self, x: &mut [f32], factor: f32) -> Result<(), FerrumError> {
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        unsafe { scale_avx2(x, factor) }
        Ok(())
    }

    fn softmax_inplace(&self, x: &mut [f32]) -> Result<(), FerrumError> {
        validate::check_softmax(x.len())?;
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        unsafe { softmax_avx2(x) }
        Ok(())
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> Result<f32, FerrumError> {
        validate::check_dot(a.len(), b.len())?;
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        Ok(unsafe { sum_products(a, b) })
    }

    fn rope(&self, args: RopeArgs<'_>) -> Result<(), FerrumError> {
        validate::check_rope(&args)?;
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        unsafe {
            rope_avx2(args.q, args.head_dim, args.rope_dim, args.pos, args.theta);
            rope_avx2(args.k, args.head_dim, args.rope_dim, args.pos, args.theta);
        }
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
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        unsafe { dequant_avx2(dtype, w, out) }
        Ok(())
    }
}

fn gemv_into(dtype: DType, w: &[u8], cols: usize, x: &[f32], y: &mut [f32]) {
    let Some(stride) = pack::row_stride(dtype, cols) else {
        return;
    };
    if stride == 0 {
        return;
    }
    for (row, slot) in w.chunks_exact(stride).zip(y.iter_mut()) {
        // SAFETY: `select` built this backend only after AVX2 and FMA were detected.
        // `dot_row_avx2` issues only in-bounds unaligned loads for this row and `x`.
        *slot = unsafe { dot_row_avx2(dtype, row, x) };
    }
}

#[derive(Clone, Copy)]
enum ZipOp {
    Add,
    Mul,
}

/// Finite `mulps` / `addps` match scalar `*` / `+` under the default MXCSR.
/// Non-finite inputs take the scalar op so NaN payloads stay identical.
fn pairwise_term(op: ZipOp, simd: f32, a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() {
        let exact = match op {
            ZipOp::Add => a + b,
            ZipOp::Mul => a * b,
        };
        debug_assert_eq!(
            simd.to_bits(),
            exact.to_bits(),
            "avx2 arithmetic diverged from scalar"
        );
        simd
    } else {
        match op {
            ZipOp::Add => a + b,
            ZipOp::Mul => a * b,
        }
    }
}

fn product_term(simd: f32, a: f32, b: f32) -> f32 {
    pairwise_term(ZipOp::Mul, simd, a, b)
}

/// # Safety
///
/// AVX2 and FMA are enabled. `tmp` has room for 8 `f32`s.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn lanes(v: __m256) -> [f32; LANES] {
    let mut tmp = [0.0f32; LANES];
    // SAFETY: AVX2 was detected before this backend is used. `tmp` is in-bounds
    // for 8 f32s (32 bytes). The store is unaligned.
    unsafe {
        _mm256_storeu_ps(tmp.as_mut_ptr(), v);
    }
    tmp
}

/// Add `a[i] * b[i]` left to right, the same order as the scalar dot.
///
/// # Safety
///
/// AVX2 and FMA are enabled. `a` and `b` may have any length; the loop stops
/// at the shorter one and never loads past it.
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_products(acc: &mut f32, a: &[f32], b: &[f32]) {
    let n = a.len().min(b.len());
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: AVX2 and FMA were detected before this backend is used.
        // `a` and `b` are in-bounds for 8 f32s (32 bytes) at `i`. Both loads
        // are unaligned. `lanes` stores each vector into an 8-wide stack slot.
        let (pa, pb, prod) = unsafe {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vp = _mm256_mul_ps(va, vb);
            (lanes(va), lanes(vb), lanes(vp))
        };
        for k in 0..LANES {
            *acc += product_term(prod[k], pa[k], pb[k]);
        }
        i += LANES;
    }
    while i < n {
        *acc += a[i] * b[i];
        i += 1;
    }
}

/// # Safety
///
/// AVX2 and FMA are enabled.
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_products(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    // SAFETY: same feature and slice preconditions as `accumulate_products`.
    unsafe { accumulate_products(&mut acc, a, b) };
    acc
}

/// # Safety
///
/// AVX2 and FMA are enabled.
#[target_feature(enable = "avx2,fma")]
unsafe fn zip_avx2(a: &[f32], b: &[f32], out: &mut [f32], op: ZipOp) {
    let n = a.len().min(b.len()).min(out.len());
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: AVX2 and FMA were detected before this backend is used.
        // `a`, `b`, and `out` are in-bounds for 8 f32s (32 bytes) at `i`.
        // Loads and the lane store are unaligned.
        let (pa, pb, pr) = unsafe {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vr = match op {
                ZipOp::Add => _mm256_add_ps(va, vb),
                ZipOp::Mul => _mm256_mul_ps(va, vb),
            };
            (lanes(va), lanes(vb), lanes(vr))
        };
        for k in 0..LANES {
            out[i + k] = pairwise_term(op, pr[k], pa[k], pb[k]);
        }
        i += LANES;
    }
    while i < n {
        out[i] = match op {
            ZipOp::Add => a[i] + b[i],
            ZipOp::Mul => a[i] * b[i],
        };
        i += 1;
    }
}

/// # Safety
///
/// AVX2 and FMA are enabled.
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_avx2(x: &mut [f32], factor: f32) {
    if !factor.is_finite() {
        for v in x.iter_mut() {
            *v *= factor;
        }
        return;
    }
    let n = x.len();
    let mut i = 0usize;
    let factor_v = _mm256_set1_ps(factor);
    while i + LANES <= n {
        // SAFETY: AVX2 and FMA were detected before this backend is used.
        // `x` is in-bounds for 8 f32s (32 bytes) at `i`. The load and the
        // lane store are unaligned.
        let (px, pr) = unsafe {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let vr = _mm256_mul_ps(vx, factor_v);
            (lanes(vx), lanes(vr))
        };
        for k in 0..LANES {
            x[i + k] = product_term(pr[k], px[k], factor);
        }
        i += LANES;
    }
    while i < n {
        x[i] *= factor;
        i += 1;
    }
}

/// # Safety
///
/// AVX2 and FMA are enabled.
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_avx2(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len().min(weight.len()).min(out.len());
    if n == 0 {
        return;
    }
    // SAFETY: AVX2 and FMA were detected before this backend is used.
    // `sum_products` only reads in-bounds unaligned groups of 8 f32s.
    let sumsq = unsafe { sum_products(x, x) };
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    if !inv.is_finite() {
        for i in 0..n {
            out[i] = x[i] * inv * weight[i];
        }
        return;
    }
    let inv_v = _mm256_set1_ps(inv);
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: AVX2 and FMA were detected before this backend is used.
        // `x`, `weight`, and `out` are in-bounds for 8 f32s (32 bytes) at `i`.
        // Loads and the lane store are unaligned.
        let (px, pw, pr) = unsafe {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let vw = _mm256_loadu_ps(weight.as_ptr().add(i));
            let scaled = _mm256_mul_ps(vx, inv_v);
            let vr = _mm256_mul_ps(scaled, vw);
            (lanes(vx), lanes(vw), lanes(vr))
        };
        for k in 0..LANES {
            let exact = (px[k] * inv) * pw[k];
            if px[k].is_finite() && pw[k].is_finite() {
                debug_assert_eq!(pr[k].to_bits(), exact.to_bits());
                out[i + k] = pr[k];
            } else {
                out[i + k] = exact;
            }
        }
        i += LANES;
    }
    while i < n {
        out[i] = x[i] * inv * weight[i];
        i += 1;
    }
}

/// Cephes `exp` for 8 lanes. NaN inputs are returned unchanged.
///
/// # Safety
///
/// AVX2 and FMA are enabled.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp256(x: __m256) -> __m256 {
    // Decimal values are the sse_mathfun / Cephes f32 constants.
    // Shortest f32 spellings of the Cephes / sse_mathfun coefficients.
    const EXP_HI: f32 = 88.376_26;
    const EXP_LO: f32 = -88.376_26;
    const LOG2EF: f32 = std::f32::consts::LOG2_E;
    const C1: f32 = 0.693_359_4;
    const C2: f32 = -2.121_944_4e-4;
    const P0: f32 = 1.987_569_1e-4;
    const P1: f32 = 1.398_199_9e-3;
    const P2: f32 = 8.333_452e-3;
    const P3: f32 = 4.166_579_6e-2;
    const P4: f32 = 1.666_666_6e-1;
    const P5: f32 = 0.5;

    let original = x;
    let mut x = _mm256_min_ps(x, _mm256_set1_ps(EXP_HI));
    x = _mm256_max_ps(x, _mm256_set1_ps(EXP_LO));

    let mut n = _mm256_mul_ps(x, _mm256_set1_ps(LOG2EF));
    n = _mm256_add_ps(n, _mm256_set1_ps(0.5));
    let n_floor = _mm256_floor_ps(n);
    let one = _mm256_set1_ps(1.0);
    let adjust = _mm256_and_ps(_mm256_cmp_ps(n_floor, n, _CMP_GT_OQ), one);
    n = _mm256_sub_ps(n_floor, adjust);

    x = _mm256_fnmadd_ps(n, _mm256_set1_ps(C1), x);
    x = _mm256_fnmadd_ps(n, _mm256_set1_ps(C2), x);

    let z = _mm256_mul_ps(x, x);
    let mut y = _mm256_fmadd_ps(x, _mm256_set1_ps(P0), _mm256_set1_ps(P1));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P2));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P3));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P4));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(P5));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    let mut imm = _mm256_cvttps_epi32(n);
    imm = _mm256_add_epi32(imm, _mm256_set1_epi32(0x7f));
    imm = _mm256_slli_epi32(imm, 23);
    y = _mm256_mul_ps(y, _mm256_castsi256_ps(imm));

    let nan = _mm256_cmp_ps(original, original, _CMP_UNORD_Q);
    _mm256_blendv_ps(y, original, nan)
}

/// # Safety
///
/// AVX2 and FMA are enabled.
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_avx2(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: AVX2 and FMA were detected before this backend is used.
        // `x` is in-bounds for 8 f32s (32 bytes) at `i`. The load is unaligned.
        let approx = unsafe {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let neg = _mm256_sub_ps(_mm256_setzero_ps(), vx);
            let denom = _mm256_add_ps(_mm256_set1_ps(1.0), exp256(neg));
            lanes(_mm256_div_ps(vx, denom))
        };
        for k in 0..LANES {
            let v = x[i + k];
            if v.is_finite() && (-20.0..20.0).contains(&v) {
                out[i + k] = approx[k];
            } else {
                out[i + k] = math::silu(v);
            }
        }
        i += LANES;
    }
    while i < n {
        out[i] = math::silu(x[i]);
        i += 1;
    }
}

/// # Safety
///
/// AVX2 and FMA are enabled. `x` is non-empty; the caller checked.
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_avx2(x: &mut [f32]) {
    let max = math::softmax_max(x);
    if !max.is_finite() || x.iter().any(|v| v.is_nan()) {
        scalar::softmax(x);
        return;
    }
    let n = x.len();
    let max_v = _mm256_set1_ps(max);
    let mut sum = 0.0f32;
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: AVX2 and FMA were detected before this backend is used.
        // `x` is in-bounds for 8 f32s (32 bytes) at `i`. The load and the
        // store of 8 f32s are unaligned.
        let exps = unsafe {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            let e = exp256(_mm256_sub_ps(v, max_v));
            _mm256_storeu_ps(x.as_mut_ptr().add(i), e);
            lanes(e)
        };
        for value in exps {
            sum += value;
        }
        i += LANES;
    }
    while i < n {
        let e = (x[i] - max).exp();
        x[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    unsafe { scale_avx2(x, inv) }
}

/// # Safety
///
/// AVX2 and FMA are enabled. `rope_dim` is even and `<= head_dim`.
#[target_feature(enable = "avx2,fma")]
unsafe fn rope_avx2(data: &mut [f32], head_dim: usize, rope_dim: usize, pos: u32, theta: f32) {
    if head_dim == 0 || rope_dim == 0 || rope_dim > head_dim {
        return;
    }
    let half = rope_dim / 2;
    for head in data.chunks_exact_mut(head_dim) {
        let mut i = 0usize;
        while i + LANES <= half {
            let mut sin_a = [0.0f32; LANES];
            let mut cos_a = [0.0f32; LANES];
            for k in 0..LANES {
                let (sin, cos) = math::rope_sin_cos(pos, theta, i + k, rope_dim);
                sin_a[k] = sin;
                cos_a[k] = cos;
            }
            // SAFETY: AVX2 and FMA were detected before this backend is used.
            // Pair `i` and pair `i + half` are inside this head for 8 f32s
            // (32 bytes): `i + 8 <= half` and `half + i + 8 <= rope_dim <= head_dim`.
            // Loads and stores are unaligned. The two halves do not overlap.
            unsafe {
                let base = head.as_mut_ptr();
                let a = _mm256_loadu_ps(base.add(i));
                let b = _mm256_loadu_ps(base.add(i + half));
                let sin_v = _mm256_loadu_ps(sin_a.as_ptr());
                let cos_v = _mm256_loadu_ps(cos_a.as_ptr());
                let new_a = _mm256_sub_ps(_mm256_mul_ps(a, cos_v), _mm256_mul_ps(b, sin_v));
                let new_b = _mm256_add_ps(_mm256_mul_ps(b, cos_v), _mm256_mul_ps(a, sin_v));
                let pa = lanes(a);
                let pb = lanes(b);
                let ps = lanes(sin_v);
                let pc = lanes(cos_v);
                let na = lanes(new_a);
                let nb = lanes(new_b);
                for k in 0..LANES {
                    let scalar_a = pa[k] * pc[k] - pb[k] * ps[k];
                    let scalar_b = pb[k] * pc[k] + pa[k] * ps[k];
                    let finite = pa[k].is_finite()
                        && pb[k].is_finite()
                        && ps[k].is_finite()
                        && pc[k].is_finite();
                    if finite {
                        debug_assert_eq!(na[k].to_bits(), scalar_a.to_bits());
                        debug_assert_eq!(nb[k].to_bits(), scalar_b.to_bits());
                        head[i + k] = na[k];
                        head[i + half + k] = nb[k];
                    } else {
                        head[i + k] = scalar_a;
                        head[i + half + k] = scalar_b;
                    }
                }
            }
            i += LANES;
        }
        while i < half {
            let (sin, cos) = math::rope_sin_cos(pos, theta, i, rope_dim);
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = b * cos + a * sin;
            i += 1;
        }
    }
}

/// # Safety
///
/// AVX2 and FMA are enabled. `row` and `x` come from a validated GEMV, but the
/// loops still stop before an out-of-bounds load.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_avx2(dtype: DType, row: &[u8], x: &[f32]) -> f32 {
    match dtype {
        DType::F32 => unsafe { dot_f32_row(row, x) },
        DType::F16 => unsafe { dot_f16_row(row, x) },
        DType::Q8_0 => unsafe { dot_q8_row(row, x) },
        DType::Other(_) => 0.0,
    }
}

/// # Safety
///
/// AVX2 and FMA are enabled.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_row(row: &[u8], x: &[f32]) -> f32 {
    let n = x.len().min(row.len() / 4);
    let mut acc = 0.0f32;
    let mut k = 0usize;
    while k + LANES <= n {
        // SAFETY: AVX2 and FMA were detected before this backend is used.
        // The weight pointer is in-bounds for 8 f32s (32 bytes) at element `k`,
        // and `x` is in-bounds for the same width. Both loads are unaligned.
        let (pw, px, prod) = unsafe {
            let wp = row.as_ptr().add(k * 4).cast::<f32>();
            let vw = _mm256_loadu_ps(wp);
            let vx = _mm256_loadu_ps(x.as_ptr().add(k));
            let vp = _mm256_mul_ps(vw, vx);
            (lanes(vw), lanes(vx), lanes(vp))
        };
        for lane in 0..LANES {
            acc += product_term(prod[lane], pw[lane], px[lane]);
        }
        k += LANES;
    }
    while k < n {
        if let Some(w) = pack::f32_at(row, k) {
            acc += w * x[k];
        } else {
            break;
        }
        k += 1;
    }
    acc
}

/// # Safety
///
/// AVX2 and FMA are enabled. Every half is converted with `f16_to_f32`.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f16_row(row: &[u8], x: &[f32]) -> f32 {
    let n = x.len().min(row.len() / 2);
    let mut acc = 0.0f32;
    let mut k = 0usize;
    while k + LANES <= n {
        let mut weights = [0.0f32; LANES];
        let mut complete = true;
        for (lane, slot) in weights.iter_mut().enumerate() {
            match pack::f16_at(row, k + lane) {
                Some(w) => *slot = w,
                None => complete = false,
            }
        }
        if !complete {
            break;
        }
        // SAFETY: AVX2 and FMA were detected. Both slices are 8 f32s (32 bytes).
        // `accumulate_products` loads them unaligned.
        unsafe { accumulate_products(&mut acc, &weights, &x[k..k + LANES]) };
        k += LANES;
    }
    while k < n {
        let Some(w) = pack::f16_at(row, k) else {
            break;
        };
        acc += w * x[k];
        k += 1;
    }
    acc
}

/// # Safety
///
/// AVX2 and FMA are enabled.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_row(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut x_off = 0usize;
    let (blocks, _) = row.as_chunks::<Q8_0_BLOCK_BYTES>();
    for block in blocks {
        if x_off + Q8_0_BLOCK > x.len() {
            break;
        }
        let Some(scale) = pack::q8_scale(block) else {
            break;
        };
        let qs = &block[2..2 + Q8_0_BLOCK];
        let x_block = &x[x_off..x_off + Q8_0_BLOCK];
        if !scale.is_finite() {
            for j in 0..Q8_0_BLOCK {
                acc += (scale * pack::q8_quant(qs[j])) * x_block[j];
            }
        } else {
            let mut j = 0usize;
            while j + LANES <= Q8_0_BLOCK {
                // SAFETY: AVX2 and FMA were detected before this backend is used.
                // `qs` has 32 bytes; the 8-byte load at `j` (0, 8, 16, 24) is
                // in-bounds and unaligned. `x_block` is in-bounds for 8 f32s
                // (32 bytes) at `j`. The lane store is unaligned.
                let (pq, px, prod) = unsafe {
                    let qv = _mm_loadl_epi64(qs.as_ptr().add(j).cast::<__m128i>());
                    let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qv));
                    let vx = _mm256_loadu_ps(x_block.as_ptr().add(j));
                    let scaled = _mm256_mul_ps(qf, _mm256_set1_ps(scale));
                    let vp = _mm256_mul_ps(scaled, vx);
                    (lanes(qf), lanes(vx), lanes(vp))
                };
                for lane in 0..LANES {
                    let q = pack::q8_quant(qs[j + lane]);
                    debug_assert_eq!(pq[lane].to_bits(), q.to_bits());
                    acc += product_term(prod[lane], scale * q, px[lane]);
                }
                j += LANES;
            }
        }
        x_off += Q8_0_BLOCK;
    }
    acc
}

/// # Safety
///
/// AVX2 and FMA are enabled. Lengths were validated, and each load is still
/// bounded by the slice length.
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_avx2(dtype: DType, w: &[u8], out: &mut [f32]) {
    match dtype {
        DType::F32 => unsafe { dequant_f32(w, out) },
        DType::F16 => {
            for (i, slot) in out.iter_mut().enumerate() {
                let Some(v) = pack::f16_at(w, i) else {
                    break;
                };
                *slot = v;
            }
        }
        DType::Q8_0 => unsafe { dequant_q8(w, out) },
        DType::Other(_) => {}
    }
}

/// # Safety
///
/// AVX2 and FMA are enabled.
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_f32(w: &[u8], out: &mut [f32]) {
    let n = out.len().min(w.len() / 4);
    let mut i = 0usize;
    while i + LANES <= n {
        // SAFETY: AVX2 and FMA were detected before this backend is used.
        // Element `i` is in-bounds for 8 f32s (32 bytes) in both `w` and `out`.
        // The load and the store are unaligned.
        unsafe {
            let p = w.as_ptr().add(i * 4).cast::<f32>();
            let v = _mm256_loadu_ps(p);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), v);
        }
        i += LANES;
    }
    while i < n {
        if let Some(v) = pack::f32_at(w, i) {
            out[i] = v;
        } else {
            break;
        }
        i += 1;
    }
}

/// # Safety
///
/// AVX2 and FMA are enabled.
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_q8(w: &[u8], out: &mut [f32]) {
    let mut i = 0usize;
    let (blocks, _) = w.as_chunks::<Q8_0_BLOCK_BYTES>();
    for block in blocks {
        if i + Q8_0_BLOCK > out.len() {
            break;
        }
        let Some(scale) = pack::q8_scale(block) else {
            break;
        };
        let qs = &block[2..2 + Q8_0_BLOCK];
        if !scale.is_finite() {
            for j in 0..Q8_0_BLOCK {
                out[i + j] = scale * pack::q8_quant(qs[j]);
            }
        } else {
            let mut j = 0usize;
            while j + LANES <= Q8_0_BLOCK {
                // SAFETY: AVX2 and FMA were detected before this backend is used.
                // `qs` has 32 bytes; the 8-byte load at `j` is in-bounds and
                // unaligned. `out` is in-bounds for 8 f32s (32 bytes) at `i + j`.
                // The store is unaligned.
                unsafe {
                    let qv = _mm_loadl_epi64(qs.as_ptr().add(j).cast::<__m128i>());
                    let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qv));
                    let scaled = _mm256_mul_ps(qf, _mm256_set1_ps(scale));
                    let pq = lanes(qf);
                    let ps = lanes(scaled);
                    for lane in 0..LANES {
                        let q = pack::q8_quant(qs[j + lane]);
                        debug_assert_eq!(pq[lane].to_bits(), q.to_bits());
                        out[i + j + lane] = product_term(ps[lane], scale, q);
                    }
                }
                j += LANES;
            }
        }
        i += Q8_0_BLOCK;
    }
}
