//! GEMV bytes per second and GEMM FLOP/s for the backend `select` returns.
//!
//! Criterion times one shape of each. The five-trial median in `docs/STATUS.md`
//! is a separate release measurement. This file does not record that number.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use ferrum_core_types::{DType, GemmArgs};

const GEMV_ROWS: usize = 256;
const GEMV_COLS: usize = 256;
const GEMM_ROWS: usize = 128;
const GEMM_COLS: usize = 128;
const GEMM_TOKENS: usize = 16;

fn bench_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10)
}

fn gemv_gemm(c: &mut Criterion) {
    let kernels = ferrum_kernels::select();

    let weight_bytes = GEMV_ROWS * GEMV_COLS * 4;
    let mut w = vec![0u8; weight_bytes];
    for (i, byte) in w.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(3).wrapping_add(1);
    }
    let x: Vec<f32> = (0..GEMV_COLS).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let mut y = vec![0.0f32; GEMV_ROWS];
    kernels
        .gemv(DType::F32, &w, GEMV_ROWS, GEMV_COLS, &x, &mut y)
        .expect("gemv");
    let gemv_bytes = (w.len() + x.len() * std::mem::size_of::<f32>()) as u64;
    let mut gemv = c.benchmark_group("gemv");
    gemv.throughput(Throughput::Bytes(gemv_bytes));
    gemv.bench_function("f32", |b| {
        b.iter(|| {
            kernels
                .gemv(
                    DType::F32,
                    black_box(&w),
                    GEMV_ROWS,
                    GEMV_COLS,
                    black_box(&x),
                    &mut y,
                )
                .unwrap();
            black_box(&y);
        });
    });
    gemv.finish();

    let gemm_weight_bytes = GEMM_ROWS * GEMM_COLS * 4;
    let mut gw = vec![0u8; gemm_weight_bytes];
    for (i, byte) in gw.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(5).wrapping_add(2);
    }
    let gx: Vec<f32> = (0..GEMM_TOKENS * GEMM_COLS)
        .map(|i| (i as f32) * 0.002 - 0.25)
        .collect();
    let mut gy = vec![0.0f32; GEMM_TOKENS * GEMM_ROWS];
    kernels
        .gemm(GemmArgs {
            dtype: DType::F32,
            w: &gw,
            rows: GEMM_ROWS,
            cols: GEMM_COLS,
            x: &gx,
            n_tokens: GEMM_TOKENS,
            y: &mut gy,
        })
        .expect("gemm");
    let elements = 2u64 * GEMM_ROWS as u64 * GEMM_COLS as u64 * GEMM_TOKENS as u64;
    let mut gemm = c.benchmark_group("gemm");
    gemm.throughput(Throughput::Elements(elements));
    gemm.bench_function("f32", |b| {
        b.iter(|| {
            kernels
                .gemm(GemmArgs {
                    dtype: DType::F32,
                    w: black_box(&gw),
                    rows: GEMM_ROWS,
                    cols: GEMM_COLS,
                    x: black_box(&gx),
                    n_tokens: GEMM_TOKENS,
                    y: &mut gy,
                })
                .unwrap();
            black_box(&gy);
        });
    });
    gemm.finish();
}

criterion_group! {
    name = kernel_benches;
    config = bench_config();
    targets = gemv_gemm
}
criterion_main!(kernel_benches);
