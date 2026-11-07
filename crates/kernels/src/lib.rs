//! Scalar reference and the AVX2+FMA backend.
//!
//! `scalar` implements `Kernels` in plain `f32` and is the reference the SIMD
//! path is checked against. `select` returns AVX2+FMA when the CPU has both
//! features, and scalar otherwise. Revision 1 never returns `Backend::Avx512`.
//! `aarch64` gets the scalar backend.
//!
//! `unsafe` stays in the AVX2 module. GEMM splits tokens with rayon on the
//! thread that called the kernel. It does not start work on an async runtime.

#![deny(unsafe_code)]

mod math;
mod pack;
mod par;
mod q8_dot;
mod scalar;
mod validate;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod avx2;

use ferrum_core_types::Kernels;

/// Reference backend. Every trait method is implemented in scalar `f32`.
pub fn scalar() -> Box<dyn Kernels> {
    Box::new(scalar::Scalar)
}

/// Fastest backend this revision can run on the host CPU.
///
/// AVX2+FMA is selected only when `avx2` and `fma` are both detected.
/// Any other host, including `aarch64` without a Neon kernel, gets [`scalar`].
pub fn select() -> Box<dyn Kernels> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Box::new(avx2::Avx2Fma);
        }
    }
    scalar()
}

#[cfg(test)]
mod tests;
