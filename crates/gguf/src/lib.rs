//! Memory-mapped GGUF loader.
//!
//! `unsafe` is used only to create the read-only map. [`load`] accepts
//! little-endian GGUF versions 2 and 3. Metadata is owned. Tensor payloads are
//! borrowed from the map.
//!
//! [`GgufFile`] implements [`ferrum_core_types::GgufModel`] and
//! [`ferrum_core_types::MetadataView`]. `dump` equals
//! [`ferrum_core_types::format_gguf_dump`] for the same version, alignment,
//! metadata length, and tensors.

mod layout;
mod parse;

pub use parse::{load, GgufFile};

#[cfg(test)]
mod tests;
