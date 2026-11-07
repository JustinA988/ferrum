//! Q8_0 matmul matching llama.cpp/ggml: quantize the f32 activation to Q8_0,
//! then `q8_0 · q8_0` with integer products and f16 scales.
//!
//! Weight-only dequant · f32 (the old path) disagrees with ggml on near-ties and
//! fails M4 top-1 on short prompts.

use std::cell::RefCell;

use ferrum_core_types::{f16_to_f32, Q8_0_BLOCK, Q8_0_BLOCK_BYTES};

use crate::pack;

thread_local! {
    /// Reused across GEMV/GEMM calls on this thread (rayon workers each own one).
    static Q8_ACT_SCRATCH: RefCell<Q8Activation> = RefCell::new(Q8Activation::empty());
}

/// Activation quantized like `quantize_row_q8_0_ref`.
pub(crate) struct Q8Activation {
    /// Per-block scales after f16 round-trip (ggml stores `d` as f16).
    scales: Vec<f32>,
    qs: Vec<i8>,
}

impl Q8Activation {
    fn empty() -> Self {
        Self {
            scales: Vec::new(),
            qs: Vec::new(),
        }
    }

    /// Fill `scales`/`qs` in place, growing capacity only when needed.
    pub(crate) fn quantize_into(&mut self, x: &[f32]) {
        debug_assert!(x.len().is_multiple_of(Q8_0_BLOCK));
        let n_blocks = x.len() / Q8_0_BLOCK;
        self.scales.clear();
        self.qs.clear();
        if self.scales.capacity() < n_blocks {
            self.scales.reserve(n_blocks - self.scales.capacity());
        }
        if self.qs.capacity() < x.len() {
            self.qs.reserve(x.len() - self.qs.capacity());
        }
        let (blocks, _) = x.as_chunks::<Q8_0_BLOCK>();
        for block in blocks {
            let mut amax = 0.0f32;
            for &v in block {
                amax = amax.max(v.abs());
            }
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            // Match ggml: quantize with `id` from the pre-f16 `d`, store f16(`d`).
            self.scales.push(f16_to_f32(f32_to_f16_bits(d)));
            for &v in block {
                let q = (v * id).round() as i32;
                self.qs.push(q.clamp(i8::MIN as i32, i8::MAX as i32) as i8);
            }
        }
    }
}

/// Quantize `x` into thread-local scratch and run `f` on the view.
pub(crate) fn with_quantized_activation<R>(x: &[f32], f: impl FnOnce(&Q8Activation) -> R) -> R {
    Q8_ACT_SCRATCH.with(|cell| {
        let mut act = cell.borrow_mut();
        act.quantize_into(x);
        f(&act)
    })
}

/// Dot one Q8_0 weight row against a pre-quantized activation.
pub(crate) fn dot_q8_row_q8_act(row: &[u8], x: &Q8Activation) -> f32 {
    debug_assert_eq!(
        row.len() / Q8_0_BLOCK_BYTES,
        x.scales.len(),
        "Q8 weight blocks must match activation blocks"
    );
    debug_assert_eq!(x.qs.len(), x.scales.len() * Q8_0_BLOCK);
    let mut acc = 0.0f32;
    let (blocks, _) = row.as_chunks::<Q8_0_BLOCK_BYTES>();
    for (bi, block) in blocks.iter().enumerate() {
        let Some(w_scale) = pack::q8_scale(block) else {
            debug_assert!(false, "truncated Q8 weight block");
            break;
        };
        let Some(&x_scale) = x.scales.get(bi) else {
            debug_assert!(false, "Q8 activation scale missing");
            break;
        };
        let qs_off = bi * Q8_0_BLOCK;
        let Some(x_qs) = x.qs.get(qs_off..qs_off + Q8_0_BLOCK) else {
            debug_assert!(false, "Q8 activation qs missing");
            break;
        };
        let mut sumi = 0i32;
        for (j, &xq) in x_qs.iter().enumerate().take(Q8_0_BLOCK) {
            let Some(&byte) = block.get(2 + j) else {
                debug_assert!(false, "truncated Q8 weight qs");
                return acc;
            };
            let wq = i8::from_ne_bytes([byte]) as i32;
            sumi += wq * (xq as i32);
        }
        acc += (sumi as f32) * (w_scale * x_scale);
    }
    acc
}

/// Round-to-nearest-even-ish half conversion matching common ggml soft-float.
/// Ties round away from zero on the LSB path used by the kernel tests' helper.
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
