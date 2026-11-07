//! Continuous batching over a `CausalModel`.
//!
//! `ContinuousBatcher` keeps a queue and the sequences already in flight. Each
//! `step` finishes waiting decodes before it starts prefill, and stops when
//! `max_batch_tokens` is spent. A failure mid-step rolls back tokens that step
//! already sampled, so a retry does not emit them again.

#![forbid(unsafe_code)]

mod batcher;

pub use batcher::ContinuousBatcher;
