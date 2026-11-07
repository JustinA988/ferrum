//! Dense f32 key/value storage, one slot per sequence.

use ferrum_core_types::{FerrumError, KvPool, KvPoolConfig, SlotId};

/// Dense f32 KV store. Each slot holds `n_layers * max_seq_len` positions; each
/// position stores contiguous K then V for all KV heads.
pub struct PreallocatedKvPool {
    config: KvPoolConfig,
    free: Vec<bool>,
    seq_len: Vec<usize>,
    layers_appended: Vec<usize>,
    storage: Vec<f32>,
}

impl PreallocatedKvPool {
    pub fn new(config: KvPoolConfig) -> Result<Self, FerrumError> {
        if config.n_layers == 0
            || config.n_kv_heads == 0
            || config.head_dim == 0
            || config.max_seq_len == 0
            || config.max_slots == 0
        {
            return Err(FerrumError::cache("kv pool dimension is zero"));
        }
        let block = kv_block_len(&config);
        let elems_per_slot = config
            .n_layers
            .checked_mul(config.max_seq_len)
            .and_then(|n| n.checked_mul(block))
            .ok_or_else(|| FerrumError::cache("kv pool size overflow"))?;
        let total = elems_per_slot
            .checked_mul(config.max_slots)
            .ok_or_else(|| FerrumError::cache("kv pool size overflow"))?;
        Ok(Self {
            free: vec![true; config.max_slots],
            seq_len: vec![0; config.max_slots],
            layers_appended: vec![0; config.max_slots],
            storage: vec![0.0; total],
            config,
        })
    }

    fn slot_index(slot: SlotId) -> Result<usize, FerrumError> {
        usize::try_from(slot.0).map_err(|_| FerrumError::cache("slot id out of range"))
    }

    fn block_offset(&self, slot: usize, layer: usize, pos: usize) -> Result<usize, FerrumError> {
        let block = kv_block_len(&self.config);
        let per_slot = self
            .config
            .n_layers
            .checked_mul(self.config.max_seq_len)
            .and_then(|n| n.checked_mul(block))
            .ok_or_else(|| FerrumError::cache("kv offset overflow"))?;
        let layer_base = layer
            .checked_mul(self.config.max_seq_len)
            .and_then(|n| n.checked_mul(block))
            .ok_or_else(|| FerrumError::cache("kv offset overflow"))?;
        let pos_off = pos
            .checked_mul(block)
            .ok_or_else(|| FerrumError::cache("kv offset overflow"))?;
        slot.checked_mul(per_slot)
            .and_then(|base| base.checked_add(layer_base))
            .and_then(|base| base.checked_add(pos_off))
            .ok_or_else(|| FerrumError::cache("kv offset overflow"))
    }
}

fn kv_block_len(config: &KvPoolConfig) -> usize {
    config.n_kv_heads * config.head_dim * 2
}

impl KvPool for PreallocatedKvPool {
    fn config(&self) -> &KvPoolConfig {
        &self.config
    }

    fn alloc(&mut self) -> Result<SlotId, FerrumError> {
        let index = self
            .free
            .iter()
            .position(|free| *free)
            .ok_or_else(|| FerrumError::unavailable("no free kv slots"))?;
        self.free[index] = false;
        self.seq_len[index] = 0;
        self.layers_appended[index] = 0;
        let block = kv_block_len(&self.config);
        let per_slot = self.config.n_layers * self.config.max_seq_len * block;
        let base = index * per_slot;
        for v in &mut self.storage[base..base + per_slot] {
            *v = 0.0;
        }
        Ok(SlotId(
            u32::try_from(index).map_err(|_| FerrumError::cache("slot id overflow"))?,
        ))
    }

    fn free(&mut self, slot: SlotId) -> Result<(), FerrumError> {
        let index = Self::slot_index(slot)?;
        if index >= self.free.len() || self.free[index] {
            return Err(FerrumError::cache("free of unallocated slot"));
        }
        self.free[index] = true;
        self.seq_len[index] = 0;
        self.layers_appended[index] = 0;
        Ok(())
    }

    fn seq_len(&self, slot: SlotId) -> Result<usize, FerrumError> {
        let index = Self::slot_index(slot)?;
        self.seq_len
            .get(index)
            .copied()
            .ok_or_else(|| FerrumError::cache("unknown slot"))
    }

    fn append_layer(
        &mut self,
        slot: SlotId,
        layer: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), FerrumError> {
        let index = Self::slot_index(slot)?;
        if self.free[index] {
            return Err(FerrumError::cache("append on free slot"));
        }
        if layer >= self.config.n_layers {
            return Err(FerrumError::cache("layer out of range"));
        }
        let expected = self.config.n_kv_heads * self.config.head_dim;
        if k.len() != expected || v.len() != expected {
            return Err(FerrumError::cache("kv vector length mismatch"));
        }
        let pos = self.seq_len[index];
        if pos >= self.config.max_seq_len {
            return Err(FerrumError::cache("sequence length exceeds max"));
        }
        if self.layers_appended[index] != layer {
            return Err(FerrumError::cache("layers must append in order"));
        }
        let off = self.block_offset(index, layer, pos)?;
        let block = kv_block_len(&self.config);
        self.storage[off..off + expected].copy_from_slice(k);
        self.storage[off + expected..off + block].copy_from_slice(v);
        self.layers_appended[index] += 1;
        Ok(())
    }

    fn commit_token(&mut self, slot: SlotId) -> Result<(), FerrumError> {
        let index = Self::slot_index(slot)?;
        if self.free[index] {
            return Err(FerrumError::cache("commit on free slot"));
        }
        if self.layers_appended[index] != self.config.n_layers {
            return Err(FerrumError::cache("not every layer appended"));
        }
        let next = self.seq_len[index]
            .checked_add(1)
            .ok_or_else(|| FerrumError::cache("sequence length overflow"))?;
        if next > self.config.max_seq_len {
            return Err(FerrumError::cache("sequence length exceeds max"));
        }
        self.seq_len[index] = next;
        self.layers_appended[index] = 0;
        Ok(())
    }

    fn k(&self, slot: SlotId, layer: usize, pos: usize) -> Result<&[f32], FerrumError> {
        let index = Self::slot_index(slot)?;
        if self.free[index] {
            return Err(FerrumError::cache("read on free slot"));
        }
        if pos >= self.seq_len[index] {
            return Err(FerrumError::cache("position not yet committed"));
        }
        if layer >= self.config.n_layers {
            return Err(FerrumError::cache("layer out of range"));
        }
        let off = self.block_offset(index, layer, pos)?;
        let n = self.config.n_kv_heads * self.config.head_dim;
        Ok(&self.storage[off..off + n])
    }

    fn v(&self, slot: SlotId, layer: usize, pos: usize) -> Result<&[f32], FerrumError> {
        let index = Self::slot_index(slot)?;
        if self.free[index] {
            return Err(FerrumError::cache("read on free slot"));
        }
        if pos >= self.seq_len[index] {
            return Err(FerrumError::cache("position not yet committed"));
        }
        if layer >= self.config.n_layers {
            return Err(FerrumError::cache("layer out of range"));
        }
        let off = self.block_offset(index, layer, pos)?;
        let n = self.config.n_kv_heads * self.config.head_dim;
        Ok(&self.storage[off + n..off + n + n])
    }
}

impl PreallocatedKvPool {
    /// Drop a partial append so the next token can start at layer 0.
    pub fn abandon_partial(&mut self, slot: SlotId) -> Result<(), FerrumError> {
        let index = Self::slot_index(slot)?;
        if index >= self.free.len() || self.free[index] {
            return Err(FerrumError::cache("abandon on free slot"));
        }
        self.layers_appended[index] = 0;
        Ok(())
    }

    /// Undo one committed token. Stored K/V at that position is overwritten on the next append.
    pub fn uncommit_token(&mut self, slot: SlotId) -> Result<(), FerrumError> {
        let index = Self::slot_index(slot)?;
        if index >= self.free.len() || self.free[index] {
            return Err(FerrumError::cache("uncommit on free slot"));
        }
        if self.seq_len[index] == 0 {
            return Err(FerrumError::cache("nothing to uncommit"));
        }
        self.seq_len[index] -= 1;
        self.layers_appended[index] = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> KvPoolConfig {
        KvPoolConfig {
            n_layers: 2,
            n_kv_heads: 1,
            head_dim: 4,
            max_seq_len: 3,
            max_slots: 2,
        }
    }

    #[test]
    fn append_commit_round_trip() {
        let mut pool = PreallocatedKvPool::new(tiny_config()).unwrap();
        let slot = pool.alloc().unwrap();
        assert_eq!(pool.seq_len(slot).unwrap(), 0);
        let k = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0, 7.0, 8.0];
        pool.append_layer(slot, 0, &k, &v).unwrap();
        pool.append_layer(slot, 1, &k, &v).unwrap();
        pool.commit_token(slot).unwrap();
        assert_eq!(pool.seq_len(slot).unwrap(), 1);
        assert_eq!(pool.k(slot, 0, 0).unwrap(), k);
        assert_eq!(pool.v(slot, 1, 0).unwrap(), v);
        pool.free(slot).unwrap();
    }

    #[test]
    fn append_must_be_in_layer_order_and_commit_needs_all_layers() {
        let mut pool = PreallocatedKvPool::new(tiny_config()).unwrap();
        let slot = pool.alloc().unwrap();
        let k = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0, 7.0, 8.0];
        assert!(pool.append_layer(slot, 1, &k, &v).is_err());
        pool.append_layer(slot, 0, &k, &v).unwrap();
        assert!(pool.commit_token(slot).is_err());
        pool.append_layer(slot, 1, &k, &v).unwrap();
        pool.commit_token(slot).unwrap();
        assert!(pool.k(slot, 0, 1).is_err());
    }

    #[test]
    fn uncommit_and_abandon_let_the_next_append_start_over() {
        let mut pool = PreallocatedKvPool::new(tiny_config()).unwrap();
        let slot = pool.alloc().unwrap();
        let k = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0, 7.0, 8.0];
        pool.append_layer(slot, 0, &k, &v).unwrap();
        pool.abandon_partial(slot).unwrap();
        pool.append_layer(slot, 0, &k, &v).unwrap();
        pool.append_layer(slot, 1, &k, &v).unwrap();
        pool.commit_token(slot).unwrap();
        pool.uncommit_token(slot).unwrap();
        assert_eq!(pool.seq_len(slot).unwrap(), 0);
        pool.append_layer(slot, 0, &k, &v).unwrap();
    }

    #[test]
    fn constructor_allocates_dense_storage_without_later_growth() {
        let cfg = tiny_config();
        let expected =
            cfg.max_slots * cfg.n_layers * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * 2;
        let mut pool = PreallocatedKvPool::new(cfg).unwrap();
        assert_eq!(pool.storage.len(), expected);
        let slot = pool.alloc().unwrap();
        let k = [9.0, 8.0, 7.0, 6.0];
        let v = [1.0, 2.0, 3.0, 4.0];
        pool.append_layer(slot, 0, &k, &v).unwrap();
        pool.append_layer(slot, 1, &k, &v).unwrap();
        pool.commit_token(slot).unwrap();
        assert_eq!(pool.storage.len(), expected);
    }
}
