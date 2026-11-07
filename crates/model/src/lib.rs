//! Llama forward pass over GGUF weights.
//!
//! [`Llama`] owns the weights and a pre-allocated key/value cache. It
//! implements [`ferrum_core_types::CausalModel`].

#![forbid(unsafe_code)]

mod llama;
mod weights;

pub use llama::Llama;

#[cfg(test)]
mod tests;
