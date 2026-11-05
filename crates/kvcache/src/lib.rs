//! Pre-allocated KV cache. Owned by A5.
//!
//! The concrete pool implements [`ferrum_core_types::KvPool`].
//! Allocation happens in the constructor. `append_layer` and `commit_token` do not allocate.

#![forbid(unsafe_code)]
