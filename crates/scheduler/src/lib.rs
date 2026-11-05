//! Continuous batching. Owned by A5.
//!
//! `ContinuousBatcher` implements [`ferrum_core_types::Scheduler`].
//! One step admits decode tokens before new prefill, inside `max_batch_tokens`.

#![forbid(unsafe_code)]

mod batcher;

pub use batcher::ContinuousBatcher;
