//! GGUF loader. Owned by A1.
//!
//! `unsafe` is allowed only while creating the read-only mmap. Each block needs
//! a `// SAFETY:` comment. Public entry point, once implemented:
//! `load(path: &Path) -> Result<GgufFile, FerrumError>`.
//! `GgufFile` implements [`ferrum_core_types::GgufModel`] and
//! [`ferrum_core_types::MetadataView`]. `dump` must equal
//! [`ferrum_core_types::format_gguf_dump`].
