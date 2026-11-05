use std::collections::HashSet;

use ferrum_core_types::{
    llama_tensor, FerrumError, GgufModel, Kernels, LlamaConfig, TensorDesc, OUTPUT, OUTPUT_NORM,
    TOKEN_EMBD,
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
