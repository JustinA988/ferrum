//! Direct checks of the scalar reference and the AVX2 backend.
//!
//! These call the kernel functions with fixed inputs. The property tests that
//! compare both backends live in `tests/accept`.

use ferrum_core_types::{f16_to_f32, Backend, DType, FerrumError, GemmArgs, Kernels, RopeArgs};

use super::{scalar, select};

fn avx2() -> Option<Box<dyn Kernels>> {
    let kernel = select();
    if kernel.backend() == Backend::Avx2Fma {
        Some(kernel)
    } else {
        None
    }
}

fn assert_send_sync<T: Send + Sync>() {}

struct Lcg(u32);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (self.0 >> 8) as f32 / 16_777_216.0;
        unit * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

fn bits_eq(got: &[f32], exp: &[f32]) {
    assert_eq!(got.len(), exp.len());
    for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "index {i}: got {g} ({:#010x}) exp {e} ({:#010x})",
            g.to_bits(),
            e.to_bits()
        );
    }
}

fn elem_tol(reference: f32) -> f32 {
    1e-6_f32.max(1e-5 * 1.0_f32.max(reference.abs()))
}

fn reduced_tol(reference: f32, n: usize) -> f32 {
    let rel = if n <= 4096 { 1e-4 } else { 1e-3 };
    1e-6_f32.max(rel * 1.0_f32.max(reference.abs()))
}

fn assert_elem_close(got: &[f32], exp: &[f32]) {
    assert_eq!(got.len(), exp.len());
    for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
        assert!(
            elem_close(*g, *e),
            "index {i}: got {g} exp {e} tol {}",
            elem_tol(*e)
        );
    }
}

fn elem_close(got: f32, exp: f32) -> bool {
    if got.to_bits() == exp.to_bits() || (got.is_nan() && exp.is_nan()) {
        return true;
    }
    (got - exp).abs() <= elem_tol(exp)
}

fn reduced_close(got: f32, exp: f32, n: usize) -> bool {
    if got.to_bits() == exp.to_bits() || (got.is_nan() && exp.is_nan()) {
        return true;
    }
    (got - exp).abs() <= reduced_tol(exp, n)
}

fn assert_reduced_close(got: &[f32], exp: &[f32], n: usize) {
    assert_eq!(got.len(), exp.len());
    for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
        assert!(
            reduced_close(*g, *e, n),
            "index {i}: got {g} exp {e} tol {} n {n}",
            reduced_tol(*e, n)
        );
    }
}

type KernelOp = Box<dyn Fn(&dyn Kernels) -> Result<(), FerrumError>>;

fn err_string(result: Result<(), FerrumError>) -> String {
    result.expect_err("expected kernel error").to_string()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn push_f16(out: &mut Vec<u8>, bits: u16) {
    out.extend_from_slice(&bits.to_le_bytes());
}

fn push_q8(out: &mut Vec<u8>, scale_bits: u16, qs: &[i8]) {
    assert_eq!(qs.len() % 32, 0);
    for block in qs.chunks(32) {
        push_f16(out, scale_bits);
        for q in block {
            out.push(q.to_le_bytes()[0]);
        }
    }
}

#[test]
fn kernels_are_send_sync() {
    assert_send_sync::<Box<dyn Kernels>>();
}

#[test]
fn select_matches_features_and_never_avx512() {
    assert_eq!(scalar().backend(), Backend::Scalar);
    let selected = select();
    assert_ne!(selected.backend(), Backend::Avx512);
    let avx2_cpu = cfg!(any(target_arch = "x86", target_arch = "x86_64"))
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma");
    if avx2_cpu {
        assert_eq!(selected.backend(), Backend::Avx2Fma);
    } else {
        assert_eq!(selected.backend(), Backend::Scalar);
    }
}

#[test]
fn scalar_elementwise_matches_formulas() {
    let k = scalar();
    let a = [1.0, -2.0, 0.5, -0.0];
    let b = [3.0, 4.0, -1.0, 2.0];
    let mut out = [0.0; 4];
    k.add(&a, &b, &mut out).unwrap();
    assert_eq!(out, [4.0, 2.0, -0.5, 2.0]);
    k.mul(&a, &b, &mut out).unwrap();
    assert_eq!(out[0], 3.0);
    assert_eq!(out[1], -8.0);
    assert_eq!(out[2], -0.5);

    let mut x = [2.0, -4.0, 0.0];
    k.scale(&mut x, 0.5).unwrap();
    assert_eq!(x, [1.0, -2.0, 0.0]);

    let input = [0.0, 1.0, -1.0, 20.0, -20.0];
    let mut got = [0.0; 5];
    k.silu(&input, &mut got).unwrap();
    for (i, &v) in input.iter().enumerate() {
        let expect = v / (1.0 + (-v).exp());
        assert_eq!(got[i].to_bits(), expect.to_bits());
    }
}

#[test]
fn scalar_rms_softmax_dot_and_rope() {
    let k = scalar();
    let x = [3.0f32, 4.0];
    let weight = [2.0, 0.5];
    let mut out = [0.0; 2];
    k.rms_norm(&x, &weight, 0.0, &mut out).unwrap();
    let inv = 1.0 / ((9.0 + 16.0) / 2.0f32).sqrt();
    assert_eq!(out[0].to_bits(), (3.0 * inv * 2.0f32).to_bits());
    assert_eq!(out[1].to_bits(), (4.0 * inv * 0.5f32).to_bits());

    let mut one = [7.5f32];
    k.softmax_inplace(&mut one).unwrap();
    assert_eq!(one[0].to_bits(), 1.0f32.to_bits());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut v = [bad];
        k.softmax_inplace(&mut v).unwrap();
        assert_eq!(v[0].to_bits(), 0.0f32.to_bits(), "{bad}");
    }
    let mut logits = [1.0f32, 2.0, 3.0];
    k.softmax_inplace(&mut logits).unwrap();
    let max = 3.0f32;
    let exps = [(1.0 - max).exp(), (2.0 - max).exp(), (3.0 - max).exp()];
    let sum = exps[0] + exps[1] + exps[2];
    for (got, exp) in logits.iter().zip(exps) {
        assert_eq!(got.to_bits(), (exp / sum).to_bits());
    }

    assert_eq!(k.dot(&[1.0, 2.0, 3.0], &[4.0, -1.0, 0.5]).unwrap(), 3.5);

    let mut q = vec![1.0f32, 0.0, 0.0, 0.0];
    let mut key = q.clone();
    let original = q.clone();
    k.rope(RopeArgs {
        q: &mut q,
        k: &mut key,
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: 4,
        rope_dim: 4,
        pos: 0,
        theta: 10_000.0,
    })
    .unwrap();
    bits_eq(&q, &original);
    bits_eq(&key, &original);

    let mut q = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tail = q[4..].to_vec();
    k.rope(RopeArgs {
        q: &mut q,
        k: &mut [0.0, 0.0, 0.0, 0.0, 8.0, 9.0],
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: 6,
        rope_dim: 4,
        pos: 1,
        theta: 10_000.0,
    })
    .unwrap();
    bits_eq(&q[4..], &tail);
    let (sin0, cos0) = rope_ref(1, 10_000.0, 0, 4);
    let (sin1, cos1) = rope_ref(1, 10_000.0, 1, 4);
    // NORMAL / adjacent pairs: (0,1) then (2,3).
    let a0 = 1.0f32;
    let b0 = 2.0f32;
    let a1 = 3.0f32;
    let b1 = 4.0f32;
    assert_eq!(q[0].to_bits(), (a0 * cos0 - b0 * sin0).to_bits());
    assert_eq!(q[1].to_bits(), (a0 * sin0 + b0 * cos0).to_bits());
    assert_eq!(q[2].to_bits(), (a1 * cos1 - b1 * sin1).to_bits());
    assert_eq!(q[3].to_bits(), (a1 * sin1 + b1 * cos1).to_bits());
}

fn rope_ref(pos: u32, theta: f32, index: usize, rope_dim: usize) -> (f32, f32) {
    let freq = pos as f32 / theta.powf((2 * index) as f32 / rope_dim as f32);
    freq.sin_cos()
}

#[test]
fn scalar_gemv_gemm_and_dequant_layout() {
    let k = scalar();
    let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = f32_bytes(&weights);
    let x = [1.0f32, 1.0, 1.0];
    let mut y = [0.0; 2];
    k.gemv(DType::F32, &w, 2, 3, &x, &mut y).unwrap();
    assert_eq!(y, [6.0, 15.0]);

    let x2 = [1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0];
    let mut y2 = [0.0; 4];
    k.gemm(GemmArgs {
        dtype: DType::F32,
        w: &w,
        rows: 2,
        cols: 3,
        x: &x2,
        n_tokens: 2,
        y: &mut y2,
    })
    .unwrap();
    assert_eq!(&y2, &[6.0, 15.0, 1.0, 4.0]);

    let mut sentinel = [9.0, 8.0, 7.0];
    k.gemm(GemmArgs {
        dtype: DType::Other(99),
        w: &[],
        rows: 3,
        cols: 0,
        x: &[],
        n_tokens: 0,
        y: &mut sentinel,
    })
    .unwrap();
    assert_eq!(sentinel, [9.0, 8.0, 7.0]);

    let mut half = Vec::new();
    push_f16(&mut half, 0x3c00);
    push_f16(&mut half, 0xc000);
    let mut out = [0.0; 2];
    k.dequant_f32(DType::F16, &half, &mut out).unwrap();
    assert_eq!(out[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(out[1].to_bits(), (-2.0f32).to_bits());

    let qs: Vec<i8> = (0..32).map(|i| i as i8 - 8).collect();
    let mut packed = Vec::new();
    push_q8(&mut packed, 0x3c00, &qs);
    let mut deq = [0.0; 32];
    k.dequant_f32(DType::Q8_0, &packed, &mut deq).unwrap();
    for (i, q) in qs.iter().enumerate() {
        assert_eq!(deq[i].to_bits(), f32::from(*q).to_bits());
    }
    let x = vec![1.0; 32];
    let mut y = [0.0];
    k.gemv(DType::Q8_0, &packed, 1, 32, &x, &mut y).unwrap();
    // ggml path: quantize x to q8_0 then integer dot. For x==1, qs_x==127 and
    // scale_x == f16(1/127), so result is sum(qs)*127*scale_x (not bare sum(qs)).
    let amax = 1.0f32;
    let d = amax / 127.0;
    let scale_x = f16_to_f32(f32_to_f16_bits(d));
    let sumi: i32 = qs.iter().map(|q| i32::from(*q) * 127).sum();
    let expect = (sumi as f32) * (1.0 * scale_x);
    assert_eq!(y[0].to_bits(), expect.to_bits());
}

#[test]
fn both_backends_return_the_same_errors() {
    let cases: Vec<(&str, KernelOp)> = vec![
        (
            "add",
            Box::new(|k| {
                let mut out = [0.0];
                k.add(&[1.0, 2.0], &[3.0], &mut out)
            }),
        ),
        (
            "mul",
            Box::new(|k| {
                let mut out = [0.0, 0.0];
                k.mul(&[1.0], &[2.0, 3.0], &mut out)
            }),
        ),
        (
            "silu",
            Box::new(|k| {
                let mut out = [];
                k.silu(&[1.0], &mut out)
            }),
        ),
        (
            "rms empty",
            Box::new(|k| k.rms_norm(&[], &[], 1e-5, &mut [])),
        ),
        (
            "rms mismatch",
            Box::new(|k| k.rms_norm(&[1.0], &[1.0, 2.0], 1e-5, &mut [0.0])),
        ),
        ("dot empty", Box::new(|k| k.dot(&[], &[]).map(|_| ()))),
        (
            "dot mismatch",
            Box::new(|k| k.dot(&[1.0], &[1.0, 2.0]).map(|_| ())),
        ),
        ("softmax empty", Box::new(|k| k.softmax_inplace(&mut []))),
        (
            "theta",
            Box::new(|k| {
                k.rope(RopeArgs {
                    q: &mut [0.0, 0.0],
                    k: &mut [0.0, 0.0],
                    n_heads: 1,
                    n_kv_heads: 1,
                    head_dim: 2,
                    rope_dim: 2,
                    pos: 1,
                    theta: 0.0,
                })
            }),
        ),
        (
            "theta nan",
            Box::new(|k| {
                k.rope(RopeArgs {
                    q: &mut [0.0, 0.0],
                    k: &mut [0.0, 0.0],
                    n_heads: 1,
                    n_kv_heads: 1,
                    head_dim: 2,
                    rope_dim: 2,
                    pos: 1,
                    theta: f32::NAN,
                })
            }),
        ),
        (
            "odd rope",
            Box::new(|k| {
                k.rope(RopeArgs {
                    q: &mut [0.0, 0.0, 0.0, 0.0],
                    k: &mut [0.0, 0.0, 0.0, 0.0],
                    n_heads: 1,
                    n_kv_heads: 1,
                    head_dim: 4,
                    rope_dim: 3,
                    pos: 1,
                    theta: 10_000.0,
                })
            }),
        ),
        (
            "rope exceeds",
            Box::new(|k| {
                k.rope(RopeArgs {
                    q: &mut [0.0, 0.0],
                    k: &mut [0.0, 0.0],
                    n_heads: 1,
                    n_kv_heads: 1,
                    head_dim: 2,
                    rope_dim: 4,
                    pos: 1,
                    theta: 10_000.0,
                })
            }),
        ),
        (
            "rope len",
            Box::new(|k| {
                k.rope(RopeArgs {
                    q: &mut [0.0],
                    k: &mut [0.0, 0.0],
                    n_heads: 1,
                    n_kv_heads: 1,
                    head_dim: 2,
                    rope_dim: 2,
                    pos: 1,
                    theta: 10_000.0,
                })
            }),
        ),
        (
            "dtype",
            Box::new(|k| k.gemv(DType::Other(12), &[0, 1, 2, 3], 1, 1, &[1.0], &mut [0.0])),
        ),
        (
            "empty k",
            Box::new(|k| k.gemv(DType::F32, &[], 1, 0, &[], &mut [0.0])),
        ),
        (
            "q8 k",
            Box::new(|k| k.gemv(DType::Q8_0, &[0; 17], 1, 16, &[0.0; 16], &mut [0.0])),
        ),
        (
            "gemv len",
            Box::new(|k| k.gemv(DType::F32, &[0; 8], 2, 1, &[1.0], &mut [0.0])),
        ),
        (
            "overflow",
            Box::new(|k| k.gemv(DType::F32, &[], usize::MAX, usize::MAX, &[], &mut [])),
        ),
        (
            "dequant q8",
            Box::new(|k| k.dequant_f32(DType::Q8_0, &[], &mut [0.0; 31])),
        ),
        (
            "dequant len",
            Box::new(|k| k.dequant_f32(DType::F16, &[0, 0, 0], &mut [0.0])),
        ),
        (
            "gemm len",
            Box::new(|k| {
                k.gemm(GemmArgs {
                    dtype: DType::F32,
                    w: &f32_bytes(&[1.0, 0.0]),
                    rows: 1,
                    cols: 2,
                    x: &[1.0, 2.0, 3.0],
                    n_tokens: 1,
                    y: &mut [0.0],
                })
            }),
        ),
    ];

    let scalar_k = scalar();
    let fast = select();
    for (name, op) in &cases {
        let left = err_string(op(scalar_k.as_ref()));
        let right = err_string(op(fast.as_ref()));
        assert_eq!(left, right, "{name}");
        assert!(left.starts_with("kernel: "), "{name}: {left}");
    }

    let mut out = [4.0f32];
    assert!(scalar_k.add(&[1.0, 2.0], &[3.0], &mut out).is_err());
    assert_eq!(out[0].to_bits(), 4.0f32.to_bits());
}

#[test]
fn odd_f32_and_f16_gemv_do_not_panic() {
    let k = scalar();
    for cols in [1usize, 3, 7, 9, 15, 17] {
        let values: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.25 - 0.5).collect();
        let mut y = [0.0];
        k.gemv(DType::F32, &f32_bytes(&values), 1, cols, &values, &mut y)
            .unwrap();
        let expect: f32 = values.iter().map(|v| v * v).sum();
        assert_eq!(y[0].to_bits(), expect.to_bits(), "f32 cols {cols}");

        let mut packed = Vec::new();
        for value in &values {
            let bits = f32_to_f16_bits(*value);
            push_f16(&mut packed, bits);
        }
        let mut y = [0.0];
        k.gemv(DType::F16, &packed, 1, cols, &values, &mut y)
            .unwrap();
        let expect: f32 = values
            .iter()
            .map(|v| f16_to_f32(f32_to_f16_bits(*v)) * v)
            .sum();
        assert_eq!(y[0].to_bits(), expect.to_bits(), "f16 cols {cols}");
    }
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        let half_frac = (frac >> 13) as u16;
        return sign | 0x7c00 | half_frac | u16::from(half_frac == 0 && frac != 0);
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let frac = frac | 0x80_0000;
        let shift = (1 - half_exp) as u32;
        let half = (frac >> (shift + 13)) as u16;
        let round = (frac >> (shift + 12)) & 1;
        return sign | (half + round as u16);
    }
    let half = ((half_exp as u32) << 10) | (frac >> 13);
    let round = (frac >> 12) & 1;
    sign | (half as u16).wrapping_add(round as u16)
}

#[test]
fn every_f16_dequant_is_f16_to_f32() {
    let k = scalar();
    let mut packed = Vec::with_capacity(65536 * 2);
    for bits in 0u16..=u16::MAX {
        push_f16(&mut packed, bits);
    }
    let mut out = vec![0.0; 65536];
    k.dequant_f32(DType::F16, &packed, &mut out).unwrap();
    for bits in 0u16..=u16::MAX {
        assert_eq!(out[bits as usize].to_bits(), f16_to_f32(bits).to_bits());
    }
}

#[test]
fn q8_dequant_covers_every_code() {
    let k = scalar();
    let codes: Vec<i8> = (0..256).map(|i| i as i8).collect();
    for &scale_bits in &[0x3c00u16, 0x3800, 0xc000, 0x0001, 0x7c00, 0x7e00] {
        let mut packed = Vec::new();
        push_q8(&mut packed, scale_bits, &codes);
        let mut out = vec![0.0; codes.len()];
        k.dequant_f32(DType::Q8_0, &packed, &mut out).unwrap();
        let scale = f16_to_f32(scale_bits);
        for (i, q) in codes.iter().enumerate() {
            assert_eq!(out[i].to_bits(), (scale * f32::from(*q)).to_bits());
        }
    }
}

#[test]
fn avx2_matches_scalar_where_required() {
    let Some(fast) = avx2() else {
        return;
    };
    let slow = scalar();
    let mut rng = Lcg(0x51ed_5eed);

    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -2.0,
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        1e-40,
        -1e-40,
        1e20,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_1234),
    ];
    let mut a = specials.to_vec();
    a.extend(rng.vec(24));
    let mut b = specials.iter().rev().copied().collect::<Vec<_>>();
    b.extend(rng.vec(24));
    assert!(a.len() >= 16 && a.len() == b.len());

    let mut out_s = vec![0.0; a.len()];
    let mut out_f = vec![0.0; a.len()];
    slow.add(&a, &b, &mut out_s).unwrap();
    fast.add(&a, &b, &mut out_f).unwrap();
    bits_eq(&out_f, &out_s);
    slow.mul(&a, &b, &mut out_s).unwrap();
    fast.mul(&a, &b, &mut out_f).unwrap();
    bits_eq(&out_f, &out_s);

    let mut scale_s = a.clone();
    let mut scale_f = a.clone();
    slow.scale(&mut scale_s, -3.0).unwrap();
    fast.scale(&mut scale_f, -3.0).unwrap();
    bits_eq(&scale_f, &scale_s);

    for n in [1usize, 7, 8, 9, 31, 32, 33, 128, 4096, 4097, 5000] {
        let left = rng.vec(n);
        let right = rng.vec(n);
        let dot_s = slow.dot(&left, &right).unwrap();
        let dot_f = fast.dot(&left, &right).unwrap();
        assert_eq!(dot_f.to_bits(), dot_s.to_bits(), "dot {n}");

        let weight = rng.vec(n);
        let mut rms_s = vec![0.0; n];
        let mut rms_f = vec![0.0; n];
        slow.rms_norm(&left, &weight, 1e-5, &mut rms_s).unwrap();
        fast.rms_norm(&left, &weight, 1e-5, &mut rms_f).unwrap();
        bits_eq(&rms_f, &rms_s);

        let mut soft_s = left.clone();
        let mut soft_f = left.clone();
        slow.softmax_inplace(&mut soft_s).unwrap();
        fast.softmax_inplace(&mut soft_f).unwrap();
        assert_reduced_close(&soft_f, &soft_s, n);

        let mut silu_s = vec![0.0; n];
        let mut silu_f = vec![0.0; n];
        slow.silu(&left, &mut silu_s).unwrap();
        fast.silu(&left, &mut silu_f).unwrap();
        assert_elem_close(&silu_f, &silu_s);
    }

    let mut dense = Vec::new();
    for i in -800..=800 {
        dense.push(i as f32 * 0.05);
    }
    dense.extend([
        40.0,
        -40.0,
        80.0,
        -80.0,
        100.0,
        -100.0,
        f32::MAX,
        f32::MIN,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e-20,
        -1e-20,
    ]);
    let mut silu_s = vec![0.0; dense.len()];
    let mut silu_f = vec![0.0; dense.len()];
    slow.silu(&dense, &mut silu_s).unwrap();
    fast.silu(&dense, &mut silu_f).unwrap();
    assert_elem_close(&silu_f, &silu_s);

    for &(heads, kv, head_dim, rope_dim, pos) in &[
        (1, 1, 4, 4, 1),
        (2, 1, 8, 4, 3),
        (4, 2, 32, 16, 7),
        (1, 1, 64, 64, 19),
    ] {
        let mut q_s = rng.vec(heads * head_dim);
        let mut k_s = rng.vec(kv * head_dim);
        let mut q_f = q_s.clone();
        let mut k_f = k_s.clone();
        let theta = 10_000.0;
        slow.rope(RopeArgs {
            q: &mut q_s,
            k: &mut k_s,
            n_heads: heads,
            n_kv_heads: kv,
            head_dim,
            rope_dim,
            pos,
            theta,
        })
        .unwrap();
        fast.rope(RopeArgs {
            q: &mut q_f,
            k: &mut k_f,
            n_heads: heads,
            n_kv_heads: kv,
            head_dim,
            rope_dim,
            pos,
            theta,
        })
        .unwrap();
        bits_eq(&q_f, &q_s);
        bits_eq(&k_f, &k_s);
    }
}

#[test]
fn avx2_dequant_and_matmul_match_scalar_bits() {
    let Some(fast) = avx2() else {
        return;
    };
    let slow = scalar();
    let mut rng = Lcg(99);

    let n = 40;
    let mut raw = vec![0u8; n * 4 + 1];
    for (i, byte) in raw.iter_mut().enumerate() {
        *byte = (rng.next_f32().to_bits() >> (i % 24)) as u8;
    }
    let w = &raw[1..1 + n * 4];
    let mut out_s = vec![0.0; n];
    let mut out_f = vec![0.0; n];
    slow.dequant_f32(DType::F32, w, &mut out_s).unwrap();
    fast.dequant_f32(DType::F32, w, &mut out_f).unwrap();
    bits_eq(&out_f, &out_s);

    let mut packed = Vec::new();
    for bits in 0u16..=1024 {
        push_f16(&mut packed, bits);
        push_f16(&mut packed, bits ^ 0x8000);
    }
    let mut out_s = vec![0.0; packed.len() / 2];
    let mut out_f = out_s.clone();
    slow.dequant_f32(DType::F16, &packed, &mut out_s).unwrap();
    fast.dequant_f32(DType::F16, &packed, &mut out_f).unwrap();
    bits_eq(&out_f, &out_s);

    let codes: Vec<i8> = (0..256).map(|i| i as i8).collect();
    for &scale_bits in &[0x3c00u16, 0x3800, 0x0001, 0x7c00, 0xfbff, 0x0400] {
        let mut q8 = vec![0u8; 1];
        push_q8(&mut q8, scale_bits, &codes);
        let packed = &q8[1..];
        let mut out_s = vec![0.0; codes.len()];
        let mut out_f = vec![0.0; codes.len()];
        slow.dequant_f32(DType::Q8_0, packed, &mut out_s).unwrap();
        fast.dequant_f32(DType::Q8_0, packed, &mut out_f).unwrap();
        bits_eq(&out_f, &out_s);
    }

    for cols in [1usize, 7, 8, 9, 31, 33, 64, 1000] {
        let rows = 3;
        let matrix = rng.vec(rows * cols);
        let x = rng.vec(cols);
        let bytes = f32_bytes(&matrix);
        let mut y_s = vec![0.0; rows];
        let mut y_f = vec![0.0; rows];
        slow.gemv(DType::F32, &bytes, rows, cols, &x, &mut y_s)
            .unwrap();
        fast.gemv(DType::F32, &bytes, rows, cols, &x, &mut y_f)
            .unwrap();
        bits_eq(&y_f, &y_s);

        let mut half = Vec::new();
        for value in &matrix {
            push_f16(&mut half, f32_to_f16_bits(*value));
        }
        slow.gemv(DType::F16, &half, rows, cols, &x, &mut y_s)
            .unwrap();
        fast.gemv(DType::F16, &half, rows, cols, &x, &mut y_f)
            .unwrap();
        bits_eq(&y_f, &y_s);
    }

    for cols in [32usize, 64, 256, 4096] {
        let rows = 2;
        let mut packed = Vec::new();
        let mut qs = Vec::new();
        for _ in 0..(rows * cols) {
            qs.push((rng.next_f32() * 100.0) as i8);
        }
        for block in qs.chunks(32) {
            push_q8(&mut packed, 0x3c00, block);
        }
        let x = rng.vec(cols);
        let mut y_s = vec![0.0; rows];
        let mut y_f = vec![0.0; rows];
        slow.gemv(DType::Q8_0, &packed, rows, cols, &x, &mut y_s)
            .unwrap();
        fast.gemv(DType::Q8_0, &packed, rows, cols, &x, &mut y_f)
            .unwrap();
        bits_eq(&y_f, &y_s);

        let n_tokens = 4;
        let x = rng.vec(n_tokens * cols);
        let mut g_s = vec![0.0; n_tokens * rows];
        let mut g_f = vec![0.0; n_tokens * rows];
        slow.gemm(GemmArgs {
            dtype: DType::Q8_0,
            w: &packed,
            rows,
            cols,
            x: &x,
            n_tokens,
            y: &mut g_s,
        })
        .unwrap();
        fast.gemm(GemmArgs {
            dtype: DType::Q8_0,
            w: &packed,
            rows,
            cols,
            x: &x,
            n_tokens,
            y: &mut g_f,
        })
        .unwrap();
        bits_eq(&g_f, &g_s);
    }

    let rows = 5;
    let cols = 7;
    let n_tokens = 3;
    let matrix = rng.vec(rows * cols);
    let bytes = f32_bytes(&matrix);
    let x = rng.vec(n_tokens * cols);
    let mut y_s = vec![0.0; n_tokens * rows];
    let mut y_f = vec![0.0; n_tokens * rows];
    slow.gemm(GemmArgs {
        dtype: DType::F32,
        w: &bytes,
        rows,
        cols,
        x: &x,
        n_tokens,
        y: &mut y_s,
    })
    .unwrap();
    fast.gemm(GemmArgs {
        dtype: DType::F32,
        w: &bytes,
        rows,
        cols,
        x: &x,
        n_tokens,
        y: &mut y_f,
    })
    .unwrap();
    bits_eq(&y_f, &y_s);
    for token in 0..n_tokens {
        let mut one = [0.0; 5];
        slow.gemv(
            DType::F32,
            &bytes,
            rows,
            cols,
            &x[token * cols..(token + 1) * cols],
            &mut one,
        )
        .unwrap();
        bits_eq(&y_s[token * rows..(token + 1) * rows], &one);
    }
}
