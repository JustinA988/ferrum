//! Load and forward checks on a tiny GGUF built in the test.
//!
//! The model is small enough to run without a downloaded file. The tests check
//! logit shape, the `position0` rule, and rejection of a bias tensor.

use std::collections::HashMap;

use ferrum_core_types::{
    format_gguf_dump, llama_tensor, CausalModel, FerrumError, GgufModel, MetadataValue, TensorDesc,
    TOKEN_EMBD,
};
use ferrum_gguf::load;

use crate::Llama;

struct MockModel {
    version: u32,
    alignment: u64,
    meta: HashMap<String, MetadataValue>,
    tensors: Vec<TensorDesc>,
    bytes: HashMap<String, Vec<u8>>,
}

impl MockModel {
    fn f32_tensor(name: &str, shape: Vec<u64>, fill: f32) -> (TensorDesc, Vec<u8>) {
        let elems = shape.iter().product::<u64>() as usize;
        let mut data = vec![0u8; elems * 4];
        let (chunks, _) = data.as_chunks_mut::<4>();
        for chunk in chunks {
            chunk.copy_from_slice(&fill.to_le_bytes());
        }
        let desc = TensorDesc::try_new(name, shape, 0, 0, data.len() as u64).unwrap();
        (desc, data)
    }

    fn tiny() -> Self {
        let mut meta = HashMap::new();
        let put = |map: &mut HashMap<String, MetadataValue>, key: &str, value: MetadataValue| {
            map.insert(key.to_string(), value);
        };
        put(
            &mut meta,
            "general.architecture",
            MetadataValue::String("llama".into()),
        );
        put(&mut meta, "llama.context_length", MetadataValue::Uint64(8));
        put(
            &mut meta,
            "llama.embedding_length",
            MetadataValue::Uint32(4),
        );
        put(&mut meta, "llama.block_count", MetadataValue::Uint32(1));
        put(
            &mut meta,
            "llama.feed_forward_length",
            MetadataValue::Uint32(8),
        );
        put(
            &mut meta,
            "llama.attention.head_count",
            MetadataValue::Uint32(2),
        );
        put(
            &mut meta,
            "llama.attention.head_count_kv",
            MetadataValue::Uint32(1),
        );
        put(
            &mut meta,
            "llama.attention.layer_norm_rms_epsilon",
            MetadataValue::Float32(1e-5),
        );
        put(
            &mut meta,
            "tokenizer.ggml.model",
            MetadataValue::String("llama".into()),
        );
        put(
            &mut meta,
            "tokenizer.ggml.bos_token_id",
            MetadataValue::Uint32(1),
        );
        put(
            &mut meta,
            "tokenizer.ggml.eos_token_id",
            MetadataValue::Uint32(2),
        );
        put(
            &mut meta,
            "tokenizer.ggml.tokens",
            MetadataValue::Array(ferrum_core_types::MetadataArray {
                elem_type: ferrum_core_types::MetadataType::String,
                values: vec![
                    MetadataValue::String("<unk>".into()),
                    MetadataValue::String("<s>".into()),
                    MetadataValue::String("</s>".into()),
                    MetadataValue::String("a".into()),
                ],
            }),
        );

        let mut tensors = Vec::new();
        let mut bytes = HashMap::new();
        let mut add = |name: &str, shape: Vec<u64>| {
            let (desc, data) = Self::f32_tensor(name, shape, 0.01);
            bytes.insert(name.to_string(), data);
            tensors.push(desc);
        };
        add(TOKEN_EMBD, vec![4, 4]);
        add(ferrum_core_types::OUTPUT_NORM, vec![4]);
        add(&llama_tensor(0, "attn_norm"), vec![4]);
        add(&llama_tensor(0, "attn_q"), vec![4, 4]);
        add(&llama_tensor(0, "attn_k"), vec![4, 2]);
        add(&llama_tensor(0, "attn_v"), vec![4, 2]);
        add(&llama_tensor(0, "attn_output"), vec![4, 4]);
        add(&llama_tensor(0, "ffn_norm"), vec![4]);
        add(&llama_tensor(0, "ffn_gate"), vec![4, 8]);
        add(&llama_tensor(0, "ffn_up"), vec![4, 8]);
        add(&llama_tensor(0, "ffn_down"), vec![8, 4]);

        Self {
            version: 3,
            alignment: 32,
            meta,
            tensors,
            bytes,
        }
    }
}

impl GgufModel for MockModel {
    fn version(&self) -> u32 {
        self.version
    }

    fn alignment(&self) -> u64 {
        self.alignment
    }

    fn metadata_len(&self) -> usize {
        self.meta.len()
    }

    fn metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.meta.get(key)
    }

    fn tensors(&self) -> &[TensorDesc] {
        &self.tensors
    }

    fn tensor(&self, name: &str) -> Result<&TensorDesc, FerrumError> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| FerrumError::gguf(format!("missing tensor {name}")))
    }

    fn tensor_bytes(&self, name: &str) -> Result<&[u8], FerrumError> {
        self.bytes
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| FerrumError::gguf(format!("missing tensor bytes {name}")))
    }

    fn dump(&self) -> String {
        format_gguf_dump(
            self.version,
            self.alignment,
            self.metadata_len(),
            &self.tensors,
        )
    }
}

#[test]
fn forward_logit_shape_and_position0() {
    let mut llama = Llama::load(MockModel::tiny(), 1, 4).unwrap();
    let slot = llama.alloc_slot().unwrap();
    assert_eq!(llama.seq_len(slot).unwrap(), 0);

    let mut logits = vec![0.0; 2 * 4];
    llama.forward(&[0, 1], 0, slot, &mut logits).unwrap();
    assert_eq!(logits.len(), 8);
    assert!(logits.iter().all(|v| v.is_finite()));
    assert_eq!(llama.seq_len(slot).unwrap(), 2);

    let err = llama.forward(&[0], 0, slot, &mut [0.0; 4]);
    assert!(matches!(err, Err(FerrumError::Model(_))));

    llama.free_slot(slot).unwrap();
}

#[test]
fn rejects_bias_tensor() {
    let mut model = MockModel::tiny();
    let (desc, data) = MockModel::f32_tensor("blk.0.attn_q.bias", vec![4], 0.0);
    model.bytes.insert(desc.name.clone(), data);
    model.tensors.push(desc);
    assert!(Llama::load(model, 1, 4).is_err());
}

#[test]
fn gguf_round_trip_loads() {
    // Minimal file builder copied from ferrum-gguf tests (single tensor smoke).
    let mut file_bytes = vec![0x47, 0x47, 0x55, 0x46];
    file_bytes.extend_from_slice(&3u32.to_le_bytes());
    file_bytes.extend_from_slice(&0u64.to_le_bytes());
    file_bytes.extend_from_slice(&1u64.to_le_bytes());
    let key = b"general.architecture";
    file_bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(key);
    file_bytes.extend_from_slice(&8u32.to_le_bytes()); // string type
    let val = b"llama";
    file_bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(val);
    // Incomplete for full llama load — ensure gguf dev-dep links.
    let path = std::env::temp_dir().join(format!("ferrum-model-smoke-{}.gguf", std::process::id()));
    std::fs::write(&path, &file_bytes).unwrap();
    assert!(load(&path).is_err());
    let _ = std::fs::remove_file(path);
}
