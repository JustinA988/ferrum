//! Llama forward pass. Owned by A4.
//!
//! Generic over the weight source and the kernel set:
//! [`Llama`]`<M: GgufModel>` owns its weights and a [`PreallocatedKvPool`].
//! It implements [`ferrum_core_types::CausalModel`].

#![forbid(unsafe_code)]

mod llama;
mod weights;

pub use llama::Llama;

#[cfg(test)]
mod tests;
