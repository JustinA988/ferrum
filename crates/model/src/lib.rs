//! Llama forward pass over GGUF weights.
//!
//! `Llama` owns the weight source and a pre-allocated KV cache, and it
//! implements `CausalModel`. Math comes from `ferrum-kernels`. Storage comes
//! from `ferrum-kvcache`. This crate does not open files or start threads.

#![forbid(unsafe_code)]

mod llama;
mod weights;

pub use llama::Llama;

#[cfg(test)]
mod tests;
