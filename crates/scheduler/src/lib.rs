//! Batches prefill and decode across active sequences.
//!
//! [`ContinuousBatcher`] runs every waiting decode before starting new
//! prefill, and stops the step when `max_batch_tokens` is spent.

#![forbid(unsafe_code)]

mod batcher;

pub use batcher::ContinuousBatcher;
