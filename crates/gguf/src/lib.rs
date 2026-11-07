//! Memory-mapped GGUF reader.
//!
//! `load` opens a little-endian GGUF v2 or v3 file and returns `GgufFile`.
//! Metadata is copied into owned values. Tensor bytes are borrowed from the
//! map, so the file must stay unchanged for the life of that map. `unsafe` is
//! used only to create the map.
//!
//! `GgufFile` implements `GgufModel` and `MetadataView`. Its `dump` text matches
//! `format_gguf_dump` for the same version, alignment, metadata, and tensors.

mod layout;
mod parse;

pub use parse::{load, GgufFile};

#[cfg(test)]
mod tests;
