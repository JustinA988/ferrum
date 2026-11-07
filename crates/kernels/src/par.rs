//! GEMM split across tokens.
//!
//! Rayon runs on the inference thread that called the kernel, not on an async
//! worker. Each token writes its own slice of `y`, so workers do not share an
//! output element.

use rayon::prelude::*;

pub(crate) fn for_each_token(
    rows: usize,
    cols: usize,
    x: &[f32],
    y: &mut [f32],
    f: impl Fn(&[f32], &mut [f32]) + Sync,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    y.par_chunks_mut(rows)
        .zip(x.par_chunks(cols))
        .for_each(|(y_tok, x_tok)| {
            if y_tok.len() == rows && x_tok.len() == cols {
                f(x_tok, y_tok);
            }
        });
}
