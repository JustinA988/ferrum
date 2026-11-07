//! Checks scalar and AVX2 kernels against the tolerance table.
//!
//! The scalar backend is the reference. AVX2 must land inside those bounds.

#![forbid(unsafe_code)]

use ferrum_core_types::{f16_to_f32, Backend, DType, FerrumError, GemmArgs, Kernels, RopeArgs};
use proptest::prelude::*;

const SILU_SCALE_REL: f32 = 1.0e-5;
const REDUCTION_REL: f32 = 1.0e-4;

fn cases(n: u32) -> ProptestConfig {
    let mut config = ProptestConfig::with_cases(n);
    config.failure_persistence = None;
    config
}

fn limit(scalar: f32, rel: f32) -> f32 {
    // max(1e-6, rel * max(1, abs(scalar)))
    1.0e-6_f32.max(rel * 1.0_f32.max(scalar.abs()))
}

fn assert_close(got: f32, scalar: f32, rel: f32, ctx: &str) {
    let allowed = limit(scalar, rel);
    let diff = (got - scalar).abs();
    assert!(
        diff <= allowed,
        "{ctx}: select {got} scalar {scalar} abs diff {diff} allowed {allowed}"
    );
}

fn assert_same_bits(got: &[f32], scalar: &[f32], ctx: &str) {
    assert_eq!(got.len(), scalar.len(), "{ctx}");
    for (i, (got, scalar)) in got.iter().zip(scalar.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            scalar.to_bits(),
            "{ctx} [{i}]: select {got} scalar {scalar}"
        );
    }
}

fn assert_same_err<T: std::fmt::Debug>(
    scalar: Result<T, FerrumError>,
    select: Result<T, FerrumError>,
    ctx: &str,
) {
    match (scalar, select) {
        (Err(left), Err(right)) => {
            assert!(
                matches!(left, FerrumError::Kernel(_)),
                "{ctx}: scalar error {left}"
            );
            assert!(
                matches!(right, FerrumError::Kernel(_)),
                "{ctx}: select error {right}"
            );
            assert_eq!(left.to_string(), right.to_string(), "{ctx}");
        }
        (left, right) => panic!("{ctx}: expected both Err, scalar={left:?} select={right:?}"),
    }
}

fn both(check: impl Fn(&dyn Kernels)) {
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    check(scalar.as_ref());
    check(select.as_ref());
}

fn pack_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn pack_f16_bits(bits: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len() * 2);
    for value in bits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn pack_q8(scale_bits: u16, quants: &[i8]) -> Vec<u8> {
    assert_eq!(quants.len() % 32, 0, "q8 row must be whole blocks");
    let mut bytes = Vec::with_capacity((quants.len() / 32) * 34);
    for block in quants.chunks(32) {
        bytes.extend_from_slice(&scale_bits.to_le_bytes());
        for quant in block {
            bytes.push(*quant as u8);
        }
    }
    bytes
}

fn q8_reference(weights: &[u8]) -> Vec<f32> {
    let (blocks, rest) = weights.as_chunks::<34>();
    assert!(
        rest.is_empty(),
        "q8_0 weight length is not a multiple of 34"
    );
    let mut out = vec![0.0; blocks.len() * 32];
    for (block_index, block) in blocks.iter().enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for lane in 0..32 {
            let quant = block[2 + lane] as i8 as f32;
            out[block_index * 32 + lane] = scale * quant;
        }
    }
    out
}

fn f16_reference(weights: &[u8]) -> Vec<f32> {
    let (chunks, rest) = weights.as_chunks::<2>();
    assert!(rest.is_empty(), "f16 weight length is odd");
    chunks
        .iter()
        .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk)))
        .collect()
}

fn assert_dequant_bits(dtype: DType, weights: &[u8], expected: &[f32]) {
    let mut scalar_out = vec![0.0; expected.len()];
    let mut select_out = vec![0.0; expected.len()];
    ferrum_kernels::scalar()
        .dequant_f32(dtype, weights, &mut scalar_out)
        .unwrap_or_else(|err| panic!("scalar dequant {dtype:?}: {err}"));
    ferrum_kernels::select()
        .dequant_f32(dtype, weights, &mut select_out)
        .unwrap_or_else(|err| panic!("select dequant {dtype:?}: {err}"));
    assert_same_bits(&scalar_out, expected, &format!("scalar dequant {dtype:?}"));
    assert_same_bits(&select_out, expected, &format!("select dequant {dtype:?}"));
}

fn check_add_mul(a: &[f32], b: &[f32]) {
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    let mut scalar_out = vec![0.0; a.len()];
    let mut select_out = vec![0.0; a.len()];
    scalar.add(a, b, &mut scalar_out).expect("scalar add");
    select.add(a, b, &mut select_out).expect("select add");
    assert_same_bits(&select_out, &scalar_out, "add");
    scalar.mul(a, b, &mut scalar_out).expect("scalar mul");
    select.mul(a, b, &mut select_out).expect("select mul");
    assert_same_bits(&select_out, &scalar_out, "mul");
}

fn check_silu_scale(x: &[f32], factor: f32) {
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    let mut scalar_out = vec![0.0; x.len()];
    let mut select_out = vec![0.0; x.len()];
    scalar.silu(x, &mut scalar_out).expect("scalar silu");
    select.silu(x, &mut select_out).expect("select silu");
    for (i, (got, reference)) in select_out.iter().zip(scalar_out.iter()).enumerate() {
        assert_close(*got, *reference, SILU_SCALE_REL, &format!("silu [{i}]"));
    }
    let mut scalar_scaled = x.to_vec();
    let mut select_scaled = x.to_vec();
    scalar
        .scale(&mut scalar_scaled, factor)
        .expect("scalar scale");
    select
        .scale(&mut select_scaled, factor)
        .expect("select scale");
    for (i, (got, reference)) in select_scaled.iter().zip(scalar_scaled.iter()).enumerate() {
        assert_close(*got, *reference, SILU_SCALE_REL, &format!("scale [{i}]"));
    }
}

fn check_reductions(a: &[f32], weight: &[f32]) {
    assert!(a.len() <= 4096, "reduced axis");
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    let scalar_dot = scalar.dot(a, weight).expect("scalar dot");
    let select_dot = select.dot(a, weight).expect("select dot");
    assert_close(select_dot, scalar_dot, REDUCTION_REL, "dot");

    let mut scalar_out = vec![0.0; a.len()];
    let mut select_out = vec![0.0; a.len()];
    scalar
        .rms_norm(a, weight, 1.0e-5, &mut scalar_out)
        .expect("scalar rms_norm");
    select
        .rms_norm(a, weight, 1.0e-5, &mut select_out)
        .expect("select rms_norm");
    for (i, (got, reference)) in select_out.iter().zip(scalar_out.iter()).enumerate() {
        assert_close(*got, *reference, REDUCTION_REL, &format!("rms_norm [{i}]"));
    }

    let mut scalar_soft = a.to_vec();
    let mut select_soft = a.to_vec();
    scalar
        .softmax_inplace(&mut scalar_soft)
        .expect("scalar softmax");
    select
        .softmax_inplace(&mut select_soft)
        .expect("select softmax");
    for (i, (got, reference)) in select_soft.iter().zip(scalar_soft.iter()).enumerate() {
        assert_close(*got, *reference, REDUCTION_REL, &format!("softmax [{i}]"));
    }
}

fn check_gemv(dtype: DType, weights: &[u8], rows: usize, cols: usize, x: &[f32]) {
    assert!(cols <= 4096, "reduced axis");
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    let mut scalar_out = vec![0.0; rows];
    let mut select_out = vec![0.0; rows];
    scalar
        .gemv(dtype, weights, rows, cols, x, &mut scalar_out)
        .unwrap_or_else(|err| panic!("scalar gemv {dtype:?} {rows}x{cols}: {err}"));
    select
        .gemv(dtype, weights, rows, cols, x, &mut select_out)
        .unwrap_or_else(|err| panic!("select gemv {dtype:?} {rows}x{cols}: {err}"));
    for (i, (got, reference)) in select_out.iter().zip(scalar_out.iter()).enumerate() {
        assert_close(
            *got,
            *reference,
            REDUCTION_REL,
            &format!("gemv {dtype:?} [{i}] rows {rows} cols {cols}"),
        );
    }
}

fn check_gemm(dtype: DType, weights: &[u8], rows: usize, cols: usize, x: &[f32], n_tokens: usize) {
    assert!(cols <= 4096, "reduced axis");
    assert_eq!(x.len(), n_tokens * cols);
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    let mut scalar_out = vec![0.0; n_tokens * rows];
    let mut select_out = vec![0.0; n_tokens * rows];
    scalar
        .gemm(GemmArgs {
            dtype,
            w: weights,
            rows,
            cols,
            x,
            n_tokens,
            y: &mut scalar_out,
        })
        .unwrap_or_else(|err| panic!("scalar gemm {dtype:?}: {err}"));
    select
        .gemm(GemmArgs {
            dtype,
            w: weights,
            rows,
            cols,
            x,
            n_tokens,
            y: &mut select_out,
        })
        .unwrap_or_else(|err| panic!("select gemm {dtype:?}: {err}"));
    for (i, (got, reference)) in select_out.iter().zip(scalar_out.iter()).enumerate() {
        assert_close(
            *got,
            *reference,
            REDUCTION_REL,
            &format!("gemm {dtype:?} [{i}] {rows}x{cols} tokens {n_tokens}"),
        );
    }
}

fn rope_on<'a>(
    kernels: &'a dyn Kernels,
    q: &'a mut [f32],
    k: &'a mut [f32],
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
) -> Result<(), FerrumError> {
    kernels.rope(RopeArgs {
        q,
        k,
        n_heads: 1,
        n_kv_heads: 1,
        head_dim,
        rope_dim,
        pos: 1,
        theta,
    })
}

#[derive(Clone, Debug)]
struct RopeCase {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: u32,
    theta: f32,
    q: Vec<f32>,
    k: Vec<f32>,
}

fn odd_vec(lo: i32, hi: i32, denom: f32) -> impl Strategy<Value = Vec<f32>> {
    (0usize..16).prop_flat_map(move |pair| {
        let len = pair * 2 + 1;
        proptest::collection::vec((lo..hi).prop_map(move |value| value as f32 / denom), len)
    })
}

fn small_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec((-32..32).prop_map(|value| value as f32 / 8.0), len)
}

fn rope_strategy() -> impl Strategy<Value = RopeCase> {
    (1usize..=3, 1usize..=3, 1usize..=4, 0u32..8, 1i32..64).prop_flat_map(
        |(n_heads, n_kv_heads, pairs, pos, theta)| {
            let rope_dim = pairs * 2;
            let head_dim = rope_dim + (pairs % 2);
            (
                Just(n_heads),
                Just(n_kv_heads),
                Just(head_dim),
                Just(rope_dim),
                Just(pos),
                Just(theta as f32),
                small_vec(n_heads * head_dim),
                small_vec(n_kv_heads * head_dim),
            )
                .prop_map(
                    |(n_heads, n_kv_heads, head_dim, rope_dim, pos, theta, q, k)| RopeCase {
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        rope_dim,
                        pos,
                        theta,
                        q,
                        k,
                    },
                )
        },
    )
}

fn gemv_f32_strategy() -> impl Strategy<Value = (usize, usize, Vec<f32>, Vec<f32>)> {
    (0usize..3, 0usize..8).prop_flat_map(|(row_pair, col_pair)| {
        let rows = row_pair * 2 + 1;
        let cols = col_pair * 2 + 1;
        (
            Just(rows),
            Just(cols),
            small_vec(rows * cols),
            small_vec(cols),
        )
    })
}

fn gemm_f32_strategy() -> impl Strategy<Value = (usize, usize, usize, Vec<f32>, Vec<f32>)> {
    (0usize..3, 0usize..6, 1usize..=3).prop_flat_map(|(row_pair, col_pair, n_tokens)| {
        let rows = row_pair * 2 + 1;
        let cols = col_pair * 2 + 1;
        (
            Just(rows),
            Just(cols),
            Just(n_tokens),
            small_vec(rows * cols),
            small_vec(n_tokens * cols),
        )
    })
}

fn q8_weights(rows: usize, cols: usize) -> impl Strategy<Value = Vec<u8>> {
    let scales = [0x3c00u16, 0x3800, 0x4000, 0xbc00, 0x3e00];
    proptest::collection::vec(any::<u8>(), rows * cols).prop_map(move |quants| {
        let mut weights = Vec::new();
        for (row, chunk) in quants.chunks(cols).enumerate() {
            let scale = scales[row % scales.len()];
            for block in chunk.chunks(32) {
                weights.extend_from_slice(&scale.to_le_bytes());
                weights.extend_from_slice(block);
            }
        }
        weights
    })
}

fn q8_gemv_strategy() -> impl Strategy<Value = (usize, usize, Vec<u8>, Vec<f32>)> {
    (1usize..=3, 1usize..=2).prop_flat_map(|(rows, blocks)| {
        let cols = blocks * 32;
        (
            Just(rows),
            Just(cols),
            q8_weights(rows, cols),
            small_vec(cols),
        )
    })
}

#[test]
fn select_follows_revision_1_features() {
    assert_eq!(ferrum_kernels::scalar().backend(), Backend::Scalar);
    let backend = ferrum_kernels::select().backend();
    assert_ne!(backend, Backend::Avx512);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            assert_eq!(backend, Backend::Avx2Fma);
        } else {
            assert_eq!(backend, Backend::Scalar);
        }
    }
}

#[test]
fn f32_odd_lengths_match_scalar() {
    for len in [1usize, 3, 7, 9, 15, 17, 31, 33] {
        let left: Vec<f32> = (0..len).map(|i| (i as f32) * 0.05 - 0.2).collect();
        let right: Vec<f32> = (0..len).map(|i| (i as f32) * -0.03 + 0.4).collect();
        check_add_mul(&left, &right);
        check_silu_scale(&left, 1.5);
        check_silu_scale(&left, -0.25);
        check_reductions(&left, &right);
    }
    let mixed = vec![0.0, 1.0, -1.0, 4.0, 30.0, -30.0, 0.5, 8.0, -2.5];
    check_silu_scale(&mixed, 2.0);
    check_add_mul(&mixed, &mixed);
}

#[test]
fn empty_dot_is_err() {
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    assert_same_err(scalar.dot(&[], &[]), select.dot(&[], &[]), "empty dot");
}

#[test]
fn softmax_of_one_finite_element_is_one() {
    for value in [0.0f32, 1.0, -4.0, 12.5, 1.0e-20, -100.0, 50.0] {
        both(|kernels| {
            let mut x = [value];
            kernels.softmax_inplace(&mut x).expect("softmax");
            assert_eq!(x[0], 1.0, "softmax([{value}])");
        });
    }
}

#[test]
fn non_finite_softmax_max_writes_zeros() {
    let cases: &[&[f32]] = &[
        &[f32::NAN],
        &[f32::INFINITY],
        &[f32::NEG_INFINITY, f32::NEG_INFINITY],
        &[f32::NAN, f32::NAN],
        &[1.0, f32::INFINITY, -3.0],
        &[f32::NEG_INFINITY, f32::INFINITY],
    ];
    for input in cases {
        both(|kernels| {
            let mut x = input.to_vec();
            kernels.softmax_inplace(&mut x).expect("softmax");
            for value in &x {
                assert_eq!(*value, 0.0, "input {input:?}");
            }
        });
    }
}

#[test]
fn gemm_zero_tokens_is_ok_and_writes_nothing() {
    let rows = 4;
    let cols = 5;
    let weights = pack_f32(&vec![0.25; rows * cols]);
    both(|kernels| {
        let mut empty = Vec::new();
        kernels
            .gemm(GemmArgs {
                dtype: DType::F32,
                w: &weights,
                rows,
                cols,
                x: &[],
                n_tokens: 0,
                y: &mut empty,
            })
            .expect("n_tokens 0 with empty y");
        let mut y = vec![f32::from_bits(0x3f80_0001); rows];
        let before = y.clone();
        kernels
            .gemm(GemmArgs {
                dtype: DType::F32,
                w: &weights,
                rows,
                cols,
                x: &[],
                n_tokens: 0,
                y: &mut y,
            })
            .expect("n_tokens 0 writes nothing");
        assert_eq!(y, before);
    });
}

#[test]
fn length_mismatch_is_the_same_err() {
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    let left = [1.0f32];
    let right = [1.0f32, 2.0];
    let mut scalar_out = [0.0f32];
    let mut select_out = [0.0f32];
    assert_same_err(
        scalar.add(&left, &right, &mut scalar_out),
        select.add(&left, &right, &mut select_out),
        "add length",
    );
    assert_same_err(
        scalar.dot(&left, &right),
        select.dot(&left, &right),
        "dot length",
    );

    let mut scalar_q = [0.0f32; 3];
    let mut scalar_k = [0.0f32; 4];
    let mut select_q = [0.0f32; 3];
    let mut select_k = [0.0f32; 4];
    assert_same_err(
        rope_on(
            scalar.as_ref(),
            &mut scalar_q,
            &mut scalar_k,
            4,
            4,
            10_000.0,
        ),
        rope_on(
            select.as_ref(),
            &mut select_q,
            &mut select_k,
            4,
            4,
            10_000.0,
        ),
        "rope length",
    );
}

#[test]
fn rope_rejects_odd_dim_and_non_positive_theta() {
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    for rope_dim in [1usize, 3, 5] {
        let mut scalar_q = vec![0.1; 8];
        let mut scalar_k = vec![0.2; 8];
        let mut select_q = scalar_q.clone();
        let mut select_k = scalar_k.clone();
        assert_same_err(
            rope_on(
                scalar.as_ref(),
                &mut scalar_q,
                &mut scalar_k,
                8,
                rope_dim,
                10_000.0,
            ),
            rope_on(
                select.as_ref(),
                &mut select_q,
                &mut select_k,
                8,
                rope_dim,
                10_000.0,
            ),
            &format!("odd rope_dim {rope_dim}"),
        );
    }
    for theta in [0.0f32, -1.0, f32::NAN, f32::NEG_INFINITY] {
        let mut scalar_q = vec![0.1; 8];
        let mut scalar_k = vec![0.2; 8];
        let mut select_q = scalar_q.clone();
        let mut select_k = scalar_k.clone();
        assert_same_err(
            rope_on(scalar.as_ref(), &mut scalar_q, &mut scalar_k, 8, 4, theta),
            rope_on(select.as_ref(), &mut select_q, &mut select_k, 8, 4, theta),
            &format!("theta {theta}"),
        );
    }
}

#[test]
fn f32_and_f16_gemv_odd_lengths() {
    for rows in [1usize, 3, 5] {
        for cols in [1usize, 3, 7, 9, 15, 17, 31, 33] {
            let values: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.02 - 0.3).collect();
            let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.03 - 0.1).collect();
            check_gemv(DType::F32, &pack_f32(&values), rows, cols, &x);
        }
    }
    let pattern = [0x3c00u16, 0x4000, 0x3800, 0xbc00, 0x3e00, 0xc000];
    for rows in [1usize, 3] {
        for cols in [1usize, 7, 9, 17, 33] {
            let bits: Vec<u16> = (0..rows * cols)
                .map(|i| pattern[i % pattern.len()])
                .collect();
            let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.05 - 0.2).collect();
            check_gemv(DType::F16, &pack_f16_bits(&bits), rows, cols, &x);
            let tokens = 2;
            let x: Vec<f32> = (0..tokens * cols)
                .map(|i| (i as f32) * 0.04 - 0.15)
                .collect();
            check_gemm(DType::F16, &pack_f16_bits(&bits), rows, cols, &x, tokens);
        }
    }
}

#[test]
fn q8_gemv_and_gemm_match_within_tolerance() {
    for cols in [32usize, 64] {
        for rows in [1usize, 3] {
            let mut quants = Vec::with_capacity(rows * cols);
            for i in 0..rows * cols {
                quants.push(((i % 17) as i8) - 8);
            }
            let mut weights = Vec::new();
            for row in quants.chunks(cols) {
                weights.extend(pack_q8(0x3c00, row));
            }
            let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.2).collect();
            check_gemv(DType::Q8_0, &weights, rows, cols, &x);
            let n_tokens = 2;
            let x: Vec<f32> = (0..n_tokens * cols)
                .map(|i| (i as f32) * 0.01 - 0.2)
                .collect();
            check_gemm(DType::Q8_0, &weights, rows, cols, &x, n_tokens);
        }
    }
}

#[test]
fn q8_k_not_multiple_of_32_is_err_on_both() {
    let scalar = ferrum_kernels::scalar();
    let select = ferrum_kernels::select();
    for cols in [1usize, 17, 31, 33, 63] {
        let rows = 2usize;
        let x = vec![0.1f32; cols];
        let weights = vec![0u8; 16];
        let mut scalar_y = vec![0.0f32; rows];
        let mut select_y = vec![0.0f32; rows];
        assert_same_err(
            scalar.gemv(DType::Q8_0, &weights, rows, cols, &x, &mut scalar_y),
            select.gemv(DType::Q8_0, &weights, rows, cols, &x, &mut select_y),
            &format!("gemv q8_0 cols {cols}"),
        );
        let mut scalar_y = vec![0.0f32; rows];
        let mut select_y = vec![0.0f32; rows];
        assert_same_err(
            scalar.gemm(GemmArgs {
                dtype: DType::Q8_0,
                w: &weights,
                rows,
                cols,
                x: &x,
                n_tokens: 1,
                y: &mut scalar_y,
            }),
            select.gemm(GemmArgs {
                dtype: DType::Q8_0,
                w: &weights,
                rows,
                cols,
                x: &x,
                n_tokens: 1,
                y: &mut select_y,
            }),
            &format!("gemm q8_0 cols {cols}"),
        );
    }
    for len in [1usize, 31, 33] {
        let weights = vec![0u8; len];
        let mut scalar_out = vec![0.0f32; len];
        let mut select_out = vec![0.0f32; len];
        assert_same_err(
            scalar.dequant_f32(DType::Q8_0, &weights, &mut scalar_out),
            select.dequant_f32(DType::Q8_0, &weights, &mut select_out),
            &format!("dequant q8_0 len {len}"),
        );
    }
}

#[test]
fn f16_dequant_matches_f16_to_f32_bits() {
    let mut weights = Vec::with_capacity(65536 * 2);
    for bits in 0u16..=u16::MAX {
        weights.extend_from_slice(&bits.to_le_bytes());
    }
    assert_dequant_bits(DType::F16, &weights, &f16_reference(&weights));
}

#[test]
fn q8_dequant_matches_f16_to_f32_scale_times_i8() {
    let mut quants = vec![0i8; 32];
    quants[0] = 0;
    quants[1] = 1;
    quants[2] = -1;
    quants[3] = 127;
    quants[4] = -128;
    for (lane, quant) in quants.iter_mut().enumerate().skip(5) {
        *quant = (lane as i8).wrapping_mul(7).wrapping_sub(20);
    }
    let mut weights = Vec::with_capacity(65536 * 34);
    for scale in 0u16..=u16::MAX {
        weights.extend(pack_q8(scale, &quants));
    }
    let expected = q8_reference(&weights);
    let mut scalar_out = vec![0.0; expected.len()];
    let mut select_out = vec![0.0; expected.len()];
    ferrum_kernels::scalar()
        .dequant_f32(DType::Q8_0, &weights, &mut scalar_out)
        .expect("scalar q8 dequant");
    ferrum_kernels::select()
        .dequant_f32(DType::Q8_0, &weights, &mut select_out)
        .expect("select q8 dequant");
    for (index, ((scalar_value, select_value), expected_value)) in scalar_out
        .iter()
        .zip(select_out.iter())
        .zip(expected.iter())
        .enumerate()
    {
        let scale = index / 32;
        let lane = index % 32;
        assert_eq!(
            scalar_value.to_bits(),
            expected_value.to_bits(),
            "scalar q8 scale {scale:#06x} lane {lane} quant {}",
            quants[lane]
        );
        assert_eq!(
            select_value.to_bits(),
            expected_value.to_bits(),
            "select q8 scale {scale:#06x} lane {lane} quant {}",
            quants[lane]
        );
    }
}

#[test]
fn f32_dequant_odd_lengths_are_bitwise() {
    for len in [1usize, 3, 7, 9, 15, 17, 31, 33] {
        let values: Vec<f32> = (0..len).map(|i| (i as f32) * 0.125 - 1.0).collect();
        assert_dequant_bits(DType::F32, &pack_f32(&values), &values);
    }
}

proptest! {
    #![proptest_config(cases(24))]
    #[test]
    fn prop_add_mul_bitwise((left, right) in odd_pair_strategy()) {
        check_add_mul(&left, &right);
    }
}

proptest! {
    #![proptest_config(cases(24))]
    #[test]
    fn prop_silu_scale(x in odd_vec(-320, 320, 8.0), factor in -64i32..64) {
        check_silu_scale(&x, factor as f32 / 8.0);
    }
}

proptest! {
    #![proptest_config(cases(24))]
    #[test]
    fn prop_dot_rms_softmax((left, right) in odd_pair_range(-64, 64, 16.0)) {
        check_reductions(&left, &right);
    }
}

proptest! {
    #![proptest_config(cases(16))]
    #[test]
    fn prop_rope(case in rope_strategy()) {
        let scalar = ferrum_kernels::scalar();
        let select = ferrum_kernels::select();
        let mut scalar_q = case.q.clone();
        let mut scalar_k = case.k.clone();
        let mut select_q = case.q.clone();
        let mut select_k = case.k.clone();
        scalar
            .rope(RopeArgs {
                q: &mut scalar_q,
                k: &mut scalar_k,
                n_heads: case.n_heads,
                n_kv_heads: case.n_kv_heads,
                head_dim: case.head_dim,
                rope_dim: case.rope_dim,
                pos: case.pos,
                theta: case.theta,
            })
            .expect("scalar rope");
        select
            .rope(RopeArgs {
                q: &mut select_q,
                k: &mut select_k,
                n_heads: case.n_heads,
                n_kv_heads: case.n_kv_heads,
                head_dim: case.head_dim,
                rope_dim: case.rope_dim,
                pos: case.pos,
                theta: case.theta,
            })
            .expect("select rope");
        for (i, (got, reference)) in select_q.iter().zip(scalar_q.iter()).enumerate() {
            assert_close(*got, *reference, REDUCTION_REL, &format!("rope q [{i}]"));
        }
        for (i, (got, reference)) in select_k.iter().zip(scalar_k.iter()).enumerate() {
            assert_close(*got, *reference, REDUCTION_REL, &format!("rope k [{i}]"));
        }
    }
}

proptest! {
    #![proptest_config(cases(16))]
    #[test]
    fn prop_gemv_f32_odd((rows, cols, matrix, x) in gemv_f32_strategy()) {
        check_gemv(DType::F32, &pack_f32(&matrix), rows, cols, &x);
    }
}

proptest! {
    #![proptest_config(cases(12))]
    #[test]
    fn prop_gemm_f32_odd((rows, cols, n_tokens, matrix, x) in gemm_f32_strategy()) {
        check_gemm(DType::F32, &pack_f32(&matrix), rows, cols, &x, n_tokens);
    }
}

proptest! {
    #![proptest_config(cases(12))]
    #[test]
    fn prop_gemv_q8((rows, cols, weights, x) in q8_gemv_strategy()) {
        check_gemv(DType::Q8_0, &weights, rows, cols, &x);
        let x_batch = x.repeat(2);
        check_gemm(DType::Q8_0, &weights, rows, cols, &x_batch, 2);
    }
}

proptest! {
    #![proptest_config(cases(16))]
    #[test]
    fn prop_dequant_f32_odd(values in odd_vec(-128, 128, 8.0)) {
        assert_dequant_bits(DType::F32, &pack_f32(&values), &values);
    }
}

proptest! {
    #![proptest_config(cases(16))]
    #[test]
    fn prop_dequant_f16_odd(bits in odd_u16()) {
        let weights = pack_f16_bits(&bits);
        assert_dequant_bits(DType::F16, &weights, &f16_reference(&weights));
    }
}

proptest! {
    #![proptest_config(cases(12))]
    #[test]
    fn prop_dequant_q8(weights in (1usize..=3).prop_flat_map(|blocks| {
        proptest::collection::vec(any::<u8>(), blocks * 34)
    })) {
        assert_dequant_bits(DType::Q8_0, &weights, &q8_reference(&weights));
    }
}

fn odd_pair_strategy() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    odd_pair_range(-256, 256, 8.0)
}

fn odd_pair_range(lo: i32, hi: i32, denom: f32) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (0usize..16).prop_flat_map(move |pair| {
        let len = pair * 2 + 1;
        let values = move || {
            proptest::collection::vec((lo..hi).prop_map(move |value| value as f32 / denom), len)
        };
        (values(), values())
    })
}

fn odd_u16() -> impl Strategy<Value = Vec<u16>> {
    (0usize..12).prop_flat_map(|pair| proptest::collection::vec(any::<u16>(), pair * 2 + 1))
}
