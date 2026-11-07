//! Fixed-size key/value cache for Llama layers.
//!
//! Storage is allocated in the constructor. Appending one layer or committing
//! a token does not allocate.

#![forbid(unsafe_code)]

mod pool;

pub use pool::PreallocatedKvPool;
