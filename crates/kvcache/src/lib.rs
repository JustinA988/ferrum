//! Pre-allocated key/value cache for Llama layers.
//!
//! `PreallocatedKvPool` reserves every slot in its constructor. Appending a
//! layer or committing a token does not allocate. The model reaches this crate
//! through the `KvPool` trait in `ferrum-core-types`.

#![forbid(unsafe_code)]

mod pool;

pub use pool::PreallocatedKvPool;
