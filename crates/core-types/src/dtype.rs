use crate::error::FerrumError;

/// Compute dtypes. `Other` is identified so a dump can name it, and is not computed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    Q8_0,
    Other(u32),
}

pub const Q8_0_BLOCK: usize = 32;
/// Little-endian IEEE binary16 scale, then 32 `i8` values. No padding.
pub const Q8_0_BLOCK_BYTES: usize = 34;

impl DType {
    pub fn from_ggml(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            other => Self::Other(other),
        }
    }

    pub fn ggml_type_id(self) -> Option<u32> {
        match self {
            Self::F32 => Some(0),
            Self::F16 => Some(1),
            Self::Q8_0 => Some(8),
            Self::Other(_) => None,
        }
    }

    pub fn is_compute_supported(self) -> bool {
        self.ggml_type_id().is_some()
    }
}

pub fn ggml_type_name(id: u32) -> Option<&'static str> {
    Some(match id {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        4 => "q4_2",
        5 => "q4_3",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_K",
        11 => "q3_K",
        12 => "q4_K",
        13 => "q5_K",
        14 => "q6_K",
        15 => "q8_K",
        16 => "iq2_xxs",
        17 => "iq2_xs",
        18 => "iq3_xxs",
        19 => "iq1_s",
        20 => "iq4_nl",
        21 => "iq3_s",
        22 => "iq2_s",
        23 => "iq4_xs",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        29 => "iq1_m",
        30 => "bf16",
        34 => "tq1_0",
        35 => "tq2_0",
        39 => "mxfp4",
        _ => return None,
    })
}

pub fn shape_elements(shape: &[u64]) -> Result<u64, FerrumError> {
    if shape.is_empty() || shape.len() > 4 {
        return Err(FerrumError::gguf(
            "tensor rank must be from 1 to 4 inclusive",
        ));
    }
    let mut n = 1u64;
    for dim in shape {
        if *dim == 0 {
            return Err(FerrumError::gguf("tensor dimension is zero"));
        }
        n = n
            .checked_mul(*dim)
            .ok_or_else(|| FerrumError::gguf("tensor element count overflow"))?;
    }
    Ok(n)
}

pub fn tensor_nbytes(dtype: DType, n_elements: u64) -> Result<u64, FerrumError> {
    match dtype {
        DType::F32 => n_elements
            .checked_mul(4)
            .ok_or_else(|| FerrumError::gguf("tensor size overflow")),
        DType::F16 => n_elements
            .checked_mul(2)
            .ok_or_else(|| FerrumError::gguf("tensor size overflow")),
        DType::Q8_0 => {
            let block = u64::try_from(Q8_0_BLOCK).expect("block fits");
            if !n_elements.is_multiple_of(block) {
                return Err(FerrumError::gguf(
                    "q8_0 element count is not a multiple of 32",
                ));
            }
            let block_bytes = u64::try_from(Q8_0_BLOCK_BYTES).expect("block bytes fit");
            (n_elements / block)
                .checked_mul(block_bytes)
                .ok_or_else(|| FerrumError::gguf("tensor size overflow"))
        }
        DType::Other(id) => Err(FerrumError::gguf(format!(
            "no compute size for ggml type {id}"
        ))),
    }
}

/// IEEE binary16 to binary32. Every binary16 value is exactly representable in binary32.
///
/// `f16` in std is still unstable on this toolchain, so the conversion lives here
/// and every backend must call it for Q8_0 scales.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) & 1;
    let exp = u32::from((bits >> 10) & 0x1f);
    let frac = u32::from(bits & 0x3ff);
    let out = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let scale = f32::from_bits((127 - 24) << 23);
            let frac16 = u16::try_from(frac).expect("binary16 fraction fits in u16");
            let mag = f32::from(frac16) * scale;
            mag.to_bits() | (sign << 31)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_f16_values() {
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0.0f32.to_bits());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
    }

    #[test]
    fn every_finite_f16_matches_scale_formula() {
        for bits in 0u16..=u16::MAX {
            let got = f16_to_f32(bits);
            let exp = (bits >> 10) & 0x1f;
            let frac = bits & 0x3ff;
            if exp == 31 {
                if frac == 0 {
                    assert!(got.is_infinite());
                } else {
                    assert!(got.is_nan());
                }
                continue;
            }
            let sign = if bits >> 15 == 1 { -1.0f64 } else { 1.0 };
            let mag = if exp == 0 {
                f64::from(frac) * 2f64.powi(-24)
            } else {
                let e = i32::from(exp) - 15;
                (1.0 + f64::from(frac) / 1024.0) * 2f64.powi(e)
            };
            let expected = (sign * mag) as f32;
            assert_eq!(got.to_bits(), expected.to_bits(), "bits {bits:#06x}");
        }
    }

    #[test]
    fn nbytes_rules() {
        assert_eq!(tensor_nbytes(DType::F32, 3).unwrap(), 12);
        assert_eq!(tensor_nbytes(DType::F16, 3).unwrap(), 6);
        assert_eq!(tensor_nbytes(DType::Q8_0, 32).unwrap(), 34);
        assert_eq!(tensor_nbytes(DType::Q8_0, 64).unwrap(), 68);
        assert!(tensor_nbytes(DType::Q8_0, 31).is_err());
        assert!(tensor_nbytes(DType::F32, u64::MAX).is_err());
        assert!(tensor_nbytes(DType::Other(12), 32).is_err());
        assert!(shape_elements(&[]).is_err());
        assert!(shape_elements(&[0, 1]).is_err());
        assert_eq!(shape_elements(&[2, 3, 4]).unwrap(), 24);
    }
}
