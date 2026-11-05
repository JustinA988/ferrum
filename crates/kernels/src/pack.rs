//! Little-endian loads for packed weights. Short buffers return `None`.

use ferrum_core_types::{f16_to_f32, DType, Q8_0_BLOCK, Q8_0_BLOCK_BYTES};

#[inline]
pub(crate) fn f32_at(bytes: &[u8], index: usize) -> Option<f32> {
    let off = index.checked_mul(4)?;
    let chunk = bytes.get(off..off.checked_add(4)?)?;
    Some(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

#[inline]
pub(crate) fn f16_at(bytes: &[u8], index: usize) -> Option<f32> {
    let off = index.checked_mul(2)?;
    let chunk = bytes.get(off..off.checked_add(2)?)?;
    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
    Some(f16_to_f32(bits))
}

/// Q8_0 scale at the start of a 34-byte block. Callers pass a full block.
#[inline]
pub(crate) fn q8_scale(block: &[u8]) -> Option<f32> {
    let chunk = block.get(..2)?;
    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
    Some(f16_to_f32(bits))
}

#[inline]
pub(crate) fn q8_quant(byte: u8) -> f32 {
    f32::from(i8::from_ne_bytes([byte]))
}

/// Byte length of one weight row, or `None` if the multiply overflows.
/// A Q8_0 `cols` that is not a multiple of 32 yields `Some(0)` only when
/// `cols < 32`; validation rejects that before compute.
pub(crate) fn row_stride(dtype: DType, cols: usize) -> Option<usize> {
    match dtype {
        DType::F32 => cols.checked_mul(4),
        DType::F16 => cols.checked_mul(2),
        DType::Q8_0 => (cols / Q8_0_BLOCK).checked_mul(Q8_0_BLOCK_BYTES),
        DType::Other(_) => None,
    }
}
