//! Encode and decode contract, and the GGML token-type integers.
//!
//! `encode` turns text into ids. `decode` turns ids back into text.
//! `TOKEN_NORMAL` through `TOKEN_BYTE` are the `tokenizer.ggml.token_type`
//! values stored in GGUF. The SentencePiece implementation is `ferrum-tokenizer`.

use crate::error::FerrumError;

pub const TOKEN_NORMAL: i32 = 1;
pub const TOKEN_UNKNOWN: i32 = 2;
pub const TOKEN_CONTROL: i32 = 3;
pub const TOKEN_USER_DEFINED: i32 = 4;
pub const TOKEN_UNUSED: i32 = 5;
pub const TOKEN_BYTE: i32 = 6;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, FerrumError>;

    fn decode(&self, ids: &[u32]) -> Result<String, FerrumError>;

    fn decode_token(&self, id: u32) -> Result<String, FerrumError>;

    fn vocab_size(&self) -> u32;

    fn bos_id(&self) -> Option<u32>;

    fn eos_id(&self) -> Option<u32>;

    fn is_special(&self, id: u32) -> bool;
}
