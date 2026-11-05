//! Numeric kernels. Owned by A2.
//!
//! `unsafe` is allowed in SIMD implementations. Each block needs a `// SAFETY:`
//! comment. SIMD loads are unaligned. Required constructors, once implemented:
//! `scalar() -> Box<dyn Kernels>` and `select() -> Box<dyn Kernels>`.
//! `select` returns AVX2+FMA only when both features are detected, Neon on
//! aarch64 when that backend exists, and scalar otherwise. It does not return
//! [`ferrum_core_types::Backend::Avx512`] in interface revision 1.
