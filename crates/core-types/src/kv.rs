//! Key/value cache trait. Append does not advance the sequence; commit does.

use crate::error::FerrumError;
use crate::ids::SlotId;

/// Pre-allocated KV pool. `append` and `commit_token` must not allocate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPoolConfig {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub max_slots: usize,
}

pub trait KvPool {
    fn config(&self) -> &KvPoolConfig;

    fn alloc(&mut self) -> Result<SlotId, FerrumError>;

    fn free(&mut self, slot: SlotId) -> Result<(), FerrumError>;

    fn seq_len(&self, slot: SlotId) -> Result<usize, FerrumError>;

    /// `k` and `v` are post-projection, head-major `[n_kv_heads, head_dim]`.
    /// `k` is post-RoPE. `v` is not rotated. Does not advance `seq_len`.
    fn append_layer(
        &mut self,
        slot: SlotId,
        layer: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), FerrumError>;

    /// Advances `seq_len` by one after every layer has appended this token.
    fn commit_token(&mut self, slot: SlotId) -> Result<(), FerrumError>;

    fn k(&self, slot: SlotId, layer: usize, pos: usize) -> Result<&[f32], FerrumError>;

    fn v(&self, slot: SlotId, layer: usize, pos: usize) -> Result<&[f32], FerrumError>;
}
