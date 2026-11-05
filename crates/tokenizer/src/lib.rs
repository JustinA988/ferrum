//! SentencePiece tokenizer. Owned by A3.
//!
//! Required constructor, once implemented:
//! `from_metadata(meta: &dyn MetadataView) -> Result<SentencePieceTokenizer, FerrumError>`.
//! The type implements [`ferrum_core_types::Tokenizer`]. Revision 1 accepts
//! `tokenizer.ggml.model == "llama"` and rejects every other tokenizer model.

#![forbid(unsafe_code)]
