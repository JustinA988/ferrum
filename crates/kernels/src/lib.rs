//! Scalar and AVX2+FMA kernels.
//!
//! [`scalar`] is the reference implementation of [`ferrum_core_types::Kernels`].
//! [`select`] returns the AVX2+FMA backend only when both features are detected
//! at runtime. Revision 1 never returns [`ferrum_core_types::Backend::Avx512`].
//! Neon is not implemented, so `aarch64` gets the scalar backend.
//!
//! `unsafe` is confined to the AVX2 module. GEMM fans tokens out through rayon
//! on the calling thread; it does not run on an async runtime.

#![deny(unsafe_code)]

mod math;
mod pack;
mod par;
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
