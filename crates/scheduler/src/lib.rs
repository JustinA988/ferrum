//! Continuous batching. Owned by A5.
//!
//! `ContinuousBatcher<M: CausalModel>` implements [`ferrum_core_types::Scheduler`].
//! One step admits decode tokens before new prefill, inside `max_batch_tokens`.

#![forbid(unsafe_code)]
