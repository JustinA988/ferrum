//! Llama hyperparameters and the causal-model trait a forward pass implements.

use crate::error::FerrumError;
use crate::ids::SlotId;
use crate::metadata::{
    meta_f32, meta_str, meta_u32, optional_bool, optional_str, optional_u32, required, MetadataView,
};

/// Llama hyperparameters read from GGUF metadata. Counts accept uint32 or uint64.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub name: String,
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_ff: u32,
    pub n_ctx_train: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub tokenizer_model: String,
}

pub fn llama_tensor(layer: u32, name: &str) -> String {
    format!("blk.{layer}.{name}.weight")
}

pub const TOKEN_EMBD: &str = "token_embd.weight";
pub const OUTPUT: &str = "output.weight";
pub const OUTPUT_NORM: &str = "output_norm.weight";

/// Lowest index wins ties. NaN entries are ignored.
pub fn argmax(logits: &[f32]) -> Result<usize, FerrumError> {
    let mut best_index = 0usize;
    let mut best = f32::NEG_INFINITY;
    let mut saw = false;
    for (index, value) in logits.iter().copied().enumerate() {
        if value.is_nan() {
            continue;
        }
        if !saw || value > best {
            best = value;
            best_index = index;
            saw = true;
        }
    }
    if !saw {
        return Err(FerrumError::model("logits are empty or all NaN"));
    }
    Ok(best_index)
}

pub fn llama_config(meta: &dyn MetadataView) -> Result<LlamaConfig, FerrumError> {
    let architecture = meta_str(meta, "general.architecture")?;
    if architecture != "llama" {
        return Err(FerrumError::gguf(format!(
            "unsupported architecture {architecture}"
        )));
    }
    let name = optional_str(meta, "general.name")?
        .unwrap_or("")
        .to_string();
    if optional_bool(meta, "llama.use_parallel_residual")? == Some(true) {
        return Err(FerrumError::gguf("parallel residual is not supported"));
    }
    if let Some(experts) = optional_u32(meta, "llama.expert_count")? {
        if experts != 0 {
            return Err(FerrumError::gguf("mixture-of-experts is not supported"));
        }
    }
    if let Some(scaling) = optional_str(meta, "llama.rope.scaling.type")? {
        if scaling != "none" {
            return Err(FerrumError::gguf(format!(
                "unsupported rope scaling {scaling}"
            )));
        }
    }

    let n_ctx_train = meta_u32(meta, "llama.context_length")?;
    let n_embd = meta_u32(meta, "llama.embedding_length")?;
    let n_layer = meta_u32(meta, "llama.block_count")?;
    let n_head = meta_u32(meta, "llama.attention.head_count")?;
    let n_head_kv = optional_u32(meta, "llama.attention.head_count_kv")?.unwrap_or(n_head);
    let n_ff = meta_u32(meta, "llama.feed_forward_length")?;
    let rms_norm_eps = meta_f32(meta, "llama.attention.layer_norm_rms_epsilon")?;
    let rope_freq_base = match meta.get("llama.rope.freq_base") {
        None => 10_000.0,
        Some(value) => value
            .as_f32()
            .map_err(|_| FerrumError::gguf("llama.rope.freq_base: expected finite f32"))?,
    };
    let head_dim = match optional_u32(meta, "llama.attention.key_length")? {
        Some(len) => len,
        None => {
            if n_head == 0 || !n_embd.is_multiple_of(n_head) {
                return Err(FerrumError::gguf(
                    "embedding length is not divisible by head count",
                ));
            }
            n_embd / n_head
        }
    };
    let rope_dim = optional_u32(meta, "llama.rope.dimension_count")?.unwrap_or(head_dim);
    if let Some(value_len) = optional_u32(meta, "llama.attention.value_length")? {
        if value_len != head_dim {
            return Err(FerrumError::gguf(
                "separate value head length is not supported",
            ));
        }
    }

    let tokens = required(meta, "tokenizer.ggml.tokens")?.as_str_vec()?;
    let n_vocab =
        u32::try_from(tokens.len()).map_err(|_| FerrumError::gguf("vocab does not fit u32"))?;
    let tokenizer_model = meta_str(meta, "tokenizer.ggml.model")?.to_string();
    let bos_token_id = meta_u32(meta, "tokenizer.ggml.bos_token_id")?;
    let eos_token_id = meta_u32(meta, "tokenizer.ggml.eos_token_id")?;

    if n_embd == 0
        || n_layer == 0
        || n_head == 0
        || n_head_kv == 0
        || n_ff == 0
        || n_ctx_train == 0
        || n_vocab == 0
        || head_dim == 0
        || rope_dim == 0
    {
        return Err(FerrumError::gguf("llama hyperparameter is zero"));
    }
    if n_embd != n_head * head_dim {
        return Err(FerrumError::gguf(
            "embedding length is not head_count * head_dim",
        ));
    }
    if !n_head.is_multiple_of(n_head_kv) {
        return Err(FerrumError::gguf(
            "head count is not divisible by kv head count",
        ));
    }
    if !head_dim.is_multiple_of(2) || !rope_dim.is_multiple_of(2) || rope_dim > head_dim {
        return Err(FerrumError::gguf("invalid rope dimension"));
    }
    if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 || rms_norm_eps >= 1.0 {
        return Err(FerrumError::gguf("invalid rms norm epsilon"));
    }
    if !rope_freq_base.is_finite() || rope_freq_base <= 0.0 {
        return Err(FerrumError::gguf("invalid rope frequency base"));
    }
    if tokenizer_model.is_empty() {
        return Err(FerrumError::gguf("tokenizer model is empty"));
    }
    if bos_token_id >= n_vocab || eos_token_id >= n_vocab {
        return Err(FerrumError::gguf("special token id is outside the vocab"));
    }

    Ok(LlamaConfig {
        name,
        n_vocab,
        n_embd,
        n_layer,
        n_head,
        n_head_kv,
        n_ff,
        n_ctx_train,
        head_dim,
        rope_dim,
        rms_norm_eps,
        rope_freq_base,
        bos_token_id,
        eos_token_id,
        tokenizer_model,
    })
}

/// Teacher-forced Llama forward. `logits_out` is row-major `[token][vocab]`.
///
/// `position0` must equal the slot's current sequence length. The call appends
/// one KV position per input token. Cached K is post-RoPE; cached V is not.
pub trait CausalModel {
    fn config(&self) -> &LlamaConfig;

    fn alloc_slot(&mut self) -> Result<SlotId, FerrumError>;

    fn free_slot(&mut self, slot: SlotId) -> Result<(), FerrumError>;

    fn seq_len(&self, slot: SlotId) -> Result<usize, FerrumError>;

    fn forward(
        &mut self,
        token_ids: &[u32],
        position0: u32,
        slot: SlotId,
        logits_out: &mut [f32],
    ) -> Result<(), FerrumError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{MetadataArray, MetadataType, MetadataValue};
    use std::collections::HashMap;

    struct MapMeta(HashMap<String, MetadataValue>);

    impl MetadataView for MapMeta {
        fn get(&self, key: &str) -> Option<&MetadataValue> {
            self.0.get(key)
        }
    }

    fn tiny() -> MapMeta {
        let mut map = HashMap::new();
        let put = |map: &mut HashMap<String, MetadataValue>, key: &str, value: MetadataValue| {
            map.insert(key.to_string(), value);
        };
        put(
            &mut map,
            "general.architecture",
            MetadataValue::String("llama".into()),
        );
        put(
            &mut map,
            "general.name",
            MetadataValue::String("tiny".into()),
        );
        put(&mut map, "llama.context_length", MetadataValue::Uint64(32));
        put(&mut map, "llama.embedding_length", MetadataValue::Uint32(8));
        put(&mut map, "llama.block_count", MetadataValue::Uint32(2));
        put(
            &mut map,
            "llama.feed_forward_length",
            MetadataValue::Uint32(16),
        );
        put(
            &mut map,
            "llama.attention.head_count",
            MetadataValue::Uint32(2),
        );
        put(
            &mut map,
            "llama.attention.head_count_kv",
            MetadataValue::Uint32(1),
        );
        put(
            &mut map,
            "llama.attention.layer_norm_rms_epsilon",
            MetadataValue::Float32(1e-5),
        );
        put(
            &mut map,
            "tokenizer.ggml.model",
            MetadataValue::String("llama".into()),
        );
        put(
            &mut map,
            "tokenizer.ggml.bos_token_id",
            MetadataValue::Uint32(1),
        );
        put(
            &mut map,
            "tokenizer.ggml.eos_token_id",
            MetadataValue::Uint32(2),
        );
        put(
            &mut map,
            "tokenizer.ggml.tokens",
            MetadataValue::Array(MetadataArray {
                elem_type: MetadataType::String,
                values: vec![
                    MetadataValue::String("<unk>".into()),
                    MetadataValue::String("<s>".into()),
                    MetadataValue::String("</s>".into()),
                    MetadataValue::String("a".into()),
                ],
            }),
        );
        MapMeta(map)
    }

    #[test]
    fn reads_llama_metadata() {
        let config = llama_config(&tiny()).unwrap();
        assert_eq!(config.n_vocab, 4);
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.n_head_kv, 1);
        assert_eq!(config.rope_dim, 4);
        assert_eq!(config.rope_freq_base, 10_000.0);
        assert_eq!(config.bos_token_id, 1);
        assert!(llama_config(&MapMeta(HashMap::new())).is_err());
    }

    #[test]
    fn argmax_breaks_ties_toward_zero() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, f32::NAN]).unwrap(), 1);
        assert!(argmax(&[]).is_err());
        assert!(argmax(&[f32::NAN]).is_err());
    }
}
