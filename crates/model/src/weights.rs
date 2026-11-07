//! Weight checks, embedding lookup, and the matrix-vector used by each layer.

use std::collections::HashSet;

use ferrum_core_types::{
    f16_to_f32, llama_tensor, DType, FerrumError, GgufModel, Kernels, LlamaConfig, TensorDesc,
    OUTPUT, OUTPUT_NORM, Q8_0_BLOCK, Q8_0_BLOCK_BYTES, TOKEN_EMBD,
};

pub fn linear<M: GgufModel>(
    model: &M,
    kernels: &dyn Kernels,
    name: &str,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), FerrumError> {
    let desc = model.tensor(name)?;
    let (rows, cols) = gemv_dims(&desc.shape)?;
    let w = model.tensor_bytes(name)?;
    kernels.gemv(desc.dtype, w, rows, cols, x, y)
}

/// Column gather for `token_embd.weight` stored as GGUF `[n_embd, n_vocab]` (ne0 contiguous).
pub fn embed_token<M: GgufModel>(
    model: &M,
    token_id: u32,
    config: &LlamaConfig,
    out: &mut [f32],
) -> Result<(), FerrumError> {
    let n_embd = config.n_embd as usize;
    let n_vocab = config.n_vocab as usize;
    if out.len() != n_embd {
        return Err(FerrumError::model("embed output length mismatch"));
    }
    if token_id as usize >= n_vocab {
        return Err(FerrumError::model("token id outside vocab"));
    }
    let desc = model.tensor(TOKEN_EMBD)?;
    if desc.shape.len() != 2
        || desc.shape[0] as usize != n_embd
        || desc.shape[1] as usize != n_vocab
    {
        return Err(FerrumError::model("token_embd shape mismatch"));
    }
    let w = model.tensor_bytes(TOKEN_EMBD)?;
    let t = token_id as usize;
    match desc.dtype {
        DType::F32 => {
            for (d, slot) in out.iter_mut().enumerate().take(n_embd) {
                let idx = d + t * n_embd;
                let off = idx
                    .checked_mul(4)
                    .ok_or_else(|| FerrumError::model("embed overflow"))?;
                let chunk = w
                    .get(off..off + 4)
                    .ok_or_else(|| FerrumError::model("embed weight out of range"))?;
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        DType::F16 => {
            for (d, slot) in out.iter_mut().enumerate().take(n_embd) {
                let idx = d + t * n_embd;
                let off = idx
                    .checked_mul(2)
                    .ok_or_else(|| FerrumError::model("embed overflow"))?;
                let chunk = w
                    .get(off..off + 2)
                    .ok_or_else(|| FerrumError::model("embed weight out of range"))?;
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                *slot = f16_to_f32(bits);
            }
        }
        DType::Q8_0 => {
            for (d, slot) in out.iter_mut().enumerate().take(n_embd) {
                let idx = d + t * n_embd;
                let block = idx / Q8_0_BLOCK;
                let lane = idx % Q8_0_BLOCK;
                let off = block
                    .checked_mul(Q8_0_BLOCK_BYTES)
                    .ok_or_else(|| FerrumError::model("embed overflow"))?;
                let chunk = w
                    .get(off..off + Q8_0_BLOCK_BYTES)
                    .ok_or_else(|| FerrumError::model("embed weight out of range"))?;
                let scale_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let scale = f16_to_f32(scale_bits);
                let q = i8::from_ne_bytes([chunk[2 + lane]]);
                *slot = scale * f32::from(q);
            }
        }
        DType::Other(_) => return Err(FerrumError::model("unsupported embed dtype")),
    }
    Ok(())
}

pub fn dequant_weight<M: GgufModel>(
    model: &M,
    kernels: &dyn Kernels,
    name: &str,
    out: &mut [f32],
) -> Result<(), FerrumError> {
    let desc = model.tensor(name)?;
    let w = model.tensor_bytes(name)?;
    kernels.dequant_f32(desc.dtype, w, out)
}

const ATTN_SUFFIXES: &[&str] = &[
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

/// GGUF `[ne0, ne1]` → GEMV `(rows, cols)` with `cols` contiguous in the weight.
pub fn gemv_dims(shape: &[u64]) -> Result<(usize, usize), FerrumError> {
    match shape.len() {
        1 => {
            let cols =
                usize::try_from(shape[0]).map_err(|_| FerrumError::model("shape overflow"))?;
            Ok((1, cols))
        }
        2 => {
            let cols =
                usize::try_from(shape[0]).map_err(|_| FerrumError::model("shape overflow"))?;
            let rows =
                usize::try_from(shape[1]).map_err(|_| FerrumError::model("shape overflow"))?;
            Ok((rows, cols))
        }
        _ => Err(FerrumError::model("tensor rank must be 1 or 2")),
    }
}

pub fn expected_tensor_names(config: &LlamaConfig, has_output: bool) -> HashSet<String> {
    let mut names = HashSet::new();
    names.insert(TOKEN_EMBD.to_string());
    names.insert(OUTPUT_NORM.to_string());
    if has_output {
        names.insert(OUTPUT.to_string());
    }
    for layer in 0..config.n_layer {
        for suffix in ATTN_SUFFIXES {
            names.insert(llama_tensor(layer, suffix));
        }
    }
    names
}

pub fn validate_weights<M: GgufModel>(
    model: &M,
    config: &LlamaConfig,
) -> Result<String, FerrumError> {
    let has_output = model.tensor(OUTPUT).is_ok();
    let expected = expected_tensor_names(config, has_output);
    for tensor in model.tensors() {
        let name = &tensor.name;
        if name.contains(".bias") {
            return Err(FerrumError::model(format!("unexpected bias tensor {name}")));
        }
        if !expected.contains(name) {
            return Err(FerrumError::model(format!("unexpected weight {name}")));
        }
    }
    for name in &expected {
        if name == OUTPUT && !has_output {
            continue;
        }
        let desc = model.tensor(name)?;
        check_tensor_shape(name, desc, config)?;
    }
    Ok(if has_output {
        OUTPUT.to_string()
    } else {
        TOKEN_EMBD.to_string()
    })
}

fn check_tensor_shape(
    name: &str,
    desc: &TensorDesc,
    config: &LlamaConfig,
) -> Result<(), FerrumError> {
    let n_embd = config.n_embd as u64;
    let n_vocab = config.n_vocab as u64;
    let n_head = config.n_head as u64;
    let n_kv = config.n_head_kv as u64;
    let head_dim = config.head_dim as u64;
    let n_ff = config.n_ff as u64;
    let q_out = n_head * head_dim;
    let kv_out = n_kv * head_dim;

    let ok = match name {
        TOKEN_EMBD | OUTPUT => desc.shape == [n_embd, n_vocab],
        OUTPUT_NORM => desc.shape == [n_embd],
        other if other.ends_with(".attn_norm.weight") || other.ends_with(".ffn_norm.weight") => {
            desc.shape == [n_embd]
        }
        other if other.ends_with(".attn_q.weight") => desc.shape == [n_embd, q_out],
        other if other.ends_with(".attn_k.weight") || other.ends_with(".attn_v.weight") => {
            desc.shape == [n_embd, kv_out]
        }
        other if other.ends_with(".attn_output.weight") => desc.shape == [q_out, n_embd],
        other if other.ends_with(".ffn_gate.weight") || other.ends_with(".ffn_up.weight") => {
            desc.shape == [n_embd, n_ff]
        }
        other if other.ends_with(".ffn_down.weight") => desc.shape == [n_ff, n_embd],
        _ => false,
    };
    if !ok {
        return Err(FerrumError::model(format!(
            "tensor {name} shape {:?} does not match config",
            desc.shape
        )));
    }
    if !desc.dtype.is_compute_supported() {
        return Err(FerrumError::model(format!(
            "tensor {name} dtype {:?} is not supported",
            desc.dtype
        )));
    }
    Ok(())
}
