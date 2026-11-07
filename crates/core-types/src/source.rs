//! Read-only view of a GGUF file.
//!
//! `TensorDesc` is one tensor's name, shape, dtype, and offset. `GgufModel`
//! returns metadata and borrowed tensor bytes. The bytes stay in the mapped
//! file. Callers do not copy the payload. `format_gguf_dump` is the text a
//! loader's `dump` method must match.

use crate::dtype::{ggml_type_name, shape_elements, tensor_nbytes, DType};
use crate::error::FerrumError;
use crate::metadata::MetadataValue;

/// One tensor in GGUF dimension order. `shape[0]` is the contiguous dimension (`ne[0]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub ggml_type: u32,
    /// Byte offset from the start of the tensor data section. A multiple of the file alignment.
    pub data_offset: u64,
    pub nbytes: u64,
}

impl TensorDesc {
    pub fn try_new(
        name: impl Into<String>,
        shape: Vec<u64>,
        ggml_type: u32,
        data_offset: u64,
        nbytes: u64,
    ) -> Result<Self, FerrumError> {
        let name = name.into();
        let dtype = DType::from_ggml(ggml_type);
        let elements = shape_elements(&shape)?;
        if let Some(id) = dtype.ggml_type_id() {
            if id != ggml_type {
                return Err(FerrumError::gguf("dtype does not match ggml type"));
            }
            let expected = tensor_nbytes(dtype, elements)?;
            if expected != nbytes {
                return Err(FerrumError::gguf(format!(
                    "tensor {name} byte count {nbytes} does not match layout {expected}"
                )));
            }
        } else if nbytes == 0 {
            return Err(FerrumError::gguf("unsupported tensor has zero bytes"));
        }
        Ok(Self {
            name,
            shape,
            dtype,
            ggml_type,
            data_offset,
            nbytes,
        })
    }
}

pub trait GgufModel: Send + Sync {
    fn version(&self) -> u32;
    fn alignment(&self) -> u64;
    fn metadata_len(&self) -> usize;
    fn metadata(&self, key: &str) -> Option<&MetadataValue>;
    fn tensors(&self) -> &[TensorDesc];
    fn tensor(&self, name: &str) -> Result<&TensorDesc, FerrumError>;
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], FerrumError>;
    fn dump(&self) -> String;
}

/// Header plus one tensor per line, GGUF order, trailing newline.
pub fn format_gguf_dump(
    version: u32,
    alignment: u64,
    metadata_len: usize,
    tensors: &[TensorDesc],
) -> String {
    let mut out = format!(
        "ferrum-gguf version={version} alignment={alignment} metadata={metadata_len} tensors={n}\n",
        n = tensors.len()
    );
    for tensor in tensors {
        let dims = tensor
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let dtype = ggml_type_name(tensor.ggml_type).unwrap_or("unknown");
        out.push_str(&format!("{} [{}] {}\n", tensor.name, dims, dtype));
    }
    out
}
