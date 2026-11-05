//! Stored byte sizes for ggml tensor types.
//!
//! Sizes match upstream ggml `static_assert`s with `QK_K` = 256. Deprecated
//! ids 4 and 5 (`q4_2`, `q4_3`) and the removed interleave ids are size 0 in
//! current ggml, so they are omitted: a reader cannot bounds-check a layout
//! it cannot size.

use ferrum_core_types::FerrumError;

/// `(ggml type id, elements per block, bytes per block)`.
pub(super) const LAYOUTS: &[(u32, u64, u64)] = &[
    (0, 1, 4),      // f32
    (1, 1, 2),      // f16
    (2, 32, 18),    // q4_0
    (3, 32, 20),    // q4_1
    (6, 32, 22),    // q5_0
    (7, 32, 24),    // q5_1
    (8, 32, 34),    // q8_0
    (9, 32, 36),    // q8_1
    (10, 256, 84),  // q2_K
    (11, 256, 110), // q3_K
    (12, 256, 144), // q4_K
    (13, 256, 176), // q5_K
    (14, 256, 210), // q6_K
    (15, 256, 292), // q8_K
    (16, 256, 66),  // iq2_xxs
    (17, 256, 74),  // iq2_xs
    (18, 256, 98),  // iq3_xxs
    (19, 256, 50),  // iq1_s
    (20, 32, 18),   // iq4_nl
    (21, 256, 110), // iq3_s
    (22, 256, 82),  // iq2_s
    (23, 256, 136), // iq4_xs
    (24, 1, 1),     // i8
    (25, 1, 2),     // i16
    (26, 1, 4),     // i32
    (27, 1, 8),     // i64
    (28, 1, 8),     // f64
    (29, 256, 56),  // iq1_m
    (30, 1, 2),     // bf16
    (34, 256, 54),  // tq1_0
    (35, 256, 66),  // tq2_0
    (39, 32, 17),   // mxfp4
];

pub(super) fn ggml_block(type_id: u32) -> Option<(u64, u64)> {
    LAYOUTS
        .iter()
        .find(|row| row.0 == type_id)
        .map(|row| (row.1, row.2))
}

/// Bytes occupied by `n_elements` values of `type_id`.
///
/// The product is checked. A count that is not a multiple of the block length
/// is an error, including for types this crate does not compute on.
pub(super) fn stored_nbytes(type_id: u32, n_elements: u64) -> Result<u64, FerrumError> {
    let Some((block, block_bytes)) = ggml_block(type_id) else {
        return Err(FerrumError::gguf(format!(
            "ggml type {type_id} has no stored size"
        )));
    };
    if block == 0 || !n_elements.is_multiple_of(block) {
        return Err(FerrumError::gguf(
            "tensor element count is not a multiple of the block size",
        ));
    }
    (n_elements / block)
        .checked_mul(block_bytes)
        .ok_or_else(|| FerrumError::gguf("tensor size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core_types::{ggml_type_name, tensor_nbytes, DType};

    #[test]
    fn published_block_sizes() {
        let expect = [
            (0, 1, 4),
            (2, 32, 18),
            (8, 32, 34),
            (12, 256, 144),
            (15, 256, 292),
            (21, 256, 110),
            (34, 256, 54),
            (35, 256, 66),
            (39, 32, 17),
        ];
        for row in expect {
            assert!(LAYOUTS.contains(&row), "{row:?}");
        }
    }

    #[test]
    fn one_and_two_blocks_and_a_short_block() {
        for &(id, block, bytes) in LAYOUTS {
            assert_eq!(stored_nbytes(id, block).unwrap(), bytes, "type {id}");
            assert_eq!(
                stored_nbytes(id, block * 2).unwrap(),
                bytes * 2,
                "type {id}"
            );
            if block > 1 {
                assert!(stored_nbytes(id, block - 1).is_err(), "type {id}");
            }
        }
    }

    #[test]
    fn compute_sizes_match_core_types() {
        let cases = [
            (0u32, DType::F32, 1u64),
            (0, DType::F32, 3),
            (0, DType::F32, 7),
            (1, DType::F16, 1),
            (1, DType::F16, 3),
            (8, DType::Q8_0, 32),
            (8, DType::Q8_0, 64),
            (8, DType::Q8_0, 256),
        ];
        for (id, dtype, n) in cases {
            assert_eq!(
                stored_nbytes(id, n).unwrap(),
                tensor_nbytes(dtype, n).unwrap(),
                "type {id} n {n}"
            );
        }
        assert!(stored_nbytes(8, 31).is_err());
        assert!(stored_nbytes(0, (u64::MAX / 2) + 1).is_err());
        assert!(stored_nbytes(4, 32).is_err());
        assert!(stored_nbytes(5, 32).is_err());
        assert!(stored_nbytes(100, 1).is_err());
    }

    #[test]
    fn named_types_are_sized_except_removed_q4() {
        for id in 0..64u32 {
            let named = ggml_type_name(id).is_some();
            let sized = ggml_block(id).is_some();
            if id == 4 || id == 5 {
                assert!(named);
                assert!(!sized);
            } else if named {
                assert!(sized, "ggml type {id} is named but has no size");
            }
        }
    }
}
