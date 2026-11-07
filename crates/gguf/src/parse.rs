//! Little-endian GGUF v2 and v3 decoder.
//!
//! Reads the header, the metadata pairs, and the tensor directory from a byte
//! slice. A count becomes an allocation only after it fits in the bytes still
//! left. A short buffer, an overflow, bad UTF-8, or an illegal bool is
//! `FerrumError::Gguf`. Malformed input returns that error. It does not panic.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::path::Path;

use ferrum_core_types::{
    format_gguf_dump, shape_elements, FerrumError, GgufModel, MetadataArray, MetadataType,
    MetadataValue, MetadataView, TensorDesc,
};
use memmap2::Mmap;

use crate::layout::stored_nbytes;

const MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Key length (`u64`) plus value type (`u32`) plus the smallest value (`u8`).
const MIN_KV_BYTES: u64 = 13;

/// Empty name (`u64`), `n_dims` (`u32`), type (`u32`), and offset (`u64`).
/// A valid tensor is larger; this is the smallest record the directory can hold.
const MIN_TENSOR_INFO_BYTES: u64 = 24;

/// Real GGUF metadata is one or two levels deep. The cap keeps a corrupt
/// nested array from overflowing the stack.
const MAX_ARRAY_DEPTH: u32 = 64;

/// Memory-mapped GGUF file. Fields stay private. The value is `Send + Sync`.
pub struct GgufFile {
    mmap: Mmap,
    /// Opened read-only and retained for the life of `mmap`.
    ///
    /// Declared after `mmap` so the map is dropped first. On Windows the
    /// original handle must stay open for that whole life; memmap2 also
    /// duplicates it, and this field is the handle ferrum opened.
    #[allow(dead_code)]
    file: File,
    version: u32,
    alignment: u64,
    metadata: HashMap<String, MetadataValue>,
    tensors: Vec<TensorDesc>,
    data_start: u64,
}

/// Memory-map `path` and parse a little-endian GGUF v2 or v3 file.
///
/// Corrupt input returns [`FerrumError::Gguf`] and does not panic.
pub fn load(path: &Path) -> Result<GgufFile, FerrumError> {
    let file = OpenOptions::new()
        .read(true)
        .write(false)
        .open(path)
        .map_err(|err| FerrumError::gguf(format!("failed to open {}: {err}", path.display())))?;
    // SAFETY: the file was opened read-only, ferrum does not write it, and it is
    // not truncated or replaced for the life of the map. Dropping the map ends
    // that use. `file` is stored in `GgufFile` for that same lifetime.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|err| FerrumError::gguf(format!("failed to map {}: {err}", path.display())))?;
    parse(file, mmap)
}

fn parse(file: File, mmap: Mmap) -> Result<GgufFile, FerrumError> {
    let parsed = {
        let mut reader = Reader::new(&mmap);
        parse_view(&mut reader)?
    };
    Ok(GgufFile {
        mmap,
        file,
        version: parsed.version,
        alignment: parsed.alignment,
        metadata: parsed.metadata,
        tensors: parsed.tensors,
        data_start: parsed.data_start,
    })
}

struct Parsed {
    version: u32,
    alignment: u64,
    metadata: HashMap<String, MetadataValue>,
    tensors: Vec<TensorDesc>,
    data_start: u64,
}

struct RawTensor {
    name: String,
    shape: Vec<u64>,
    ggml_type: u32,
    data_offset: u64,
    nbytes: u64,
}

fn parse_view(reader: &mut Reader<'_>) -> Result<Parsed, FerrumError> {
    let magic = reader.read_exact(4)?;
    if magic != MAGIC.as_slice() {
        return Err(FerrumError::gguf("bad gguf magic"));
    }
    let version = reader.read_u32()?;
    if version != 2 && version != 3 {
        return Err(FerrumError::gguf(format!(
            "unsupported gguf version {version}"
        )));
    }

    let tensor_count = reader.read_u64()?;
    let kv_count = reader.read_u64()?;
    let metadata = read_metadata(reader, kv_count)?;
    let alignment = alignment_of(&metadata)?;
    let raw_tensors = read_tensor_directory(reader, tensor_count)?;

    let file_len = u64::try_from(reader.data.len())
        .map_err(|_| FerrumError::gguf("file length exceeds u64"))?;
    let info_end = u64::try_from(reader.pos)
        .map_err(|_| FerrumError::gguf("tensor directory offset exceeds u64"))?;
    let data_start = align_offset(info_end, alignment)?;
    if data_start > file_len {
        return Err(FerrumError::gguf("alignment padding exceeds file"));
    }
    let info_end_us = reader.pos;
    let data_start_us = usize::try_from(data_start)
        .map_err(|_| FerrumError::gguf("data section offset exceeds usize"))?;
    let padding = reader
        .data
        .get(info_end_us..data_start_us)
        .ok_or_else(|| FerrumError::gguf("alignment padding out of range"))?;
    if padding.iter().any(|byte| *byte != 0) {
        return Err(FerrumError::gguf("alignment padding is not zero"));
    }

    let data_len = file_len - data_start;
    let mut tensors = Vec::new();
    tensors
        .try_reserve(raw_tensors.len())
        .map_err(|_| FerrumError::gguf("tensor allocation failed"))?;
    for raw in raw_tensors {
        if !raw.data_offset.is_multiple_of(alignment) {
            return Err(FerrumError::gguf(format!(
                "tensor {} data_offset is not aligned",
                raw.name
            )));
        }
        let end = raw.data_offset.checked_add(raw.nbytes).ok_or_else(|| {
            FerrumError::gguf(format!("tensor {} data range overflows", raw.name))
        })?;
        if end > data_len {
            return Err(FerrumError::gguf(format!(
                "tensor {} extends past end of file",
                raw.name
            )));
        }
        tensors.push(TensorDesc::try_new(
            raw.name,
            raw.shape,
            raw.ggml_type,
            raw.data_offset,
            raw.nbytes,
        )?);
    }

    Ok(Parsed {
        version,
        alignment,
        metadata,
        tensors,
        data_start,
    })
}

fn read_metadata(
    reader: &mut Reader<'_>,
    kv_count: u64,
) -> Result<HashMap<String, MetadataValue>, FerrumError> {
    let n_kv = bound_count(kv_count, MIN_KV_BYTES, reader.remaining_u64()?)?;
    let mut metadata = HashMap::new();
    metadata
        .try_reserve(n_kv)
        .map_err(|_| FerrumError::gguf("metadata allocation failed"))?;
    for _ in 0..n_kv {
        let key = reader.read_string()?;
        let type_id = reader.read_u32()?;
        let value_type = MetadataType::from_u32(type_id)
            .ok_or_else(|| FerrumError::gguf(format!("unknown metadata type {type_id}")))?;
        let value = read_value(reader, value_type, 0)?;
        if metadata.contains_key(&key) {
            return Err(FerrumError::gguf(format!("duplicate metadata key {key}")));
        }
        metadata.insert(key, value);
    }
    Ok(metadata)
}

fn read_tensor_directory(
    reader: &mut Reader<'_>,
    tensor_count: u64,
) -> Result<Vec<RawTensor>, FerrumError> {
    let n_tensors = bound_count(tensor_count, MIN_TENSOR_INFO_BYTES, reader.remaining_u64()?)?;
    let mut seen = HashSet::new();
    seen.try_reserve(n_tensors)
        .map_err(|_| FerrumError::gguf("tensor allocation failed"))?;
    let mut raw = Vec::new();
    raw.try_reserve(n_tensors)
        .map_err(|_| FerrumError::gguf("tensor allocation failed"))?;
    for _ in 0..n_tensors {
        raw.push(read_tensor_info(reader, &mut seen)?);
    }
    Ok(raw)
}

fn read_tensor_info(
    reader: &mut Reader<'_>,
    seen: &mut HashSet<String>,
) -> Result<RawTensor, FerrumError> {
    let name = reader.read_string()?;
    if !seen.insert(name.clone()) {
        return Err(FerrumError::gguf(format!("duplicate tensor name {name}")));
    }
    let n_dims = reader.read_u32()?;
    if n_dims == 0 || n_dims > 4 {
        return Err(FerrumError::gguf(
            "tensor rank must be from 1 to 4 inclusive",
        ));
    }
    let n_dims_us = bound_count(u64::from(n_dims), 8, reader.remaining_u64()?)?;
    let mut shape = Vec::new();
    shape
        .try_reserve(n_dims_us)
        .map_err(|_| FerrumError::gguf("tensor allocation failed"))?;
    for _ in 0..n_dims_us {
        shape.push(reader.read_u64()?);
    }
    let ggml_type = reader.read_u32()?;
    let data_offset = reader.read_u64()?;
    let elements = shape_elements(&shape)?;
    let nbytes = stored_nbytes(ggml_type, elements)?;
    Ok(RawTensor {
        name,
        shape,
        ggml_type,
        data_offset,
        nbytes,
    })
}

fn read_value(
    reader: &mut Reader<'_>,
    value_type: MetadataType,
    depth: u32,
) -> Result<MetadataValue, FerrumError> {
    match value_type {
        MetadataType::Uint8 => Ok(MetadataValue::Uint8(reader.read_u8()?)),
        MetadataType::Int8 => Ok(MetadataValue::Int8(reader.read_i8()?)),
        MetadataType::Uint16 => Ok(MetadataValue::Uint16(reader.read_u16()?)),
        MetadataType::Int16 => Ok(MetadataValue::Int16(reader.read_i16()?)),
        MetadataType::Uint32 => Ok(MetadataValue::Uint32(reader.read_u32()?)),
        MetadataType::Int32 => Ok(MetadataValue::Int32(reader.read_i32()?)),
        MetadataType::Float32 => Ok(MetadataValue::Float32(reader.read_f32()?)),
        MetadataType::Bool => Ok(MetadataValue::Bool(reader.read_bool()?)),
        MetadataType::String => Ok(MetadataValue::String(reader.read_string()?)),
        MetadataType::Uint64 => Ok(MetadataValue::Uint64(reader.read_u64()?)),
        MetadataType::Int64 => Ok(MetadataValue::Int64(reader.read_i64()?)),
        MetadataType::Float64 => Ok(MetadataValue::Float64(reader.read_f64()?)),
        MetadataType::Array => read_array(reader, depth),
    }
}

fn read_array(reader: &mut Reader<'_>, depth: u32) -> Result<MetadataValue, FerrumError> {
    if depth >= MAX_ARRAY_DEPTH {
        return Err(FerrumError::gguf("metadata array nested too deeply"));
    }
    let elem_id = reader.read_u32()?;
    let elem_type = MetadataType::from_u32(elem_id)
        .ok_or_else(|| FerrumError::gguf(format!("unknown metadata type {elem_id}")))?;
    let count = reader.read_u64()?;
    let n = bound_count(count, min_value_bytes(elem_type), reader.remaining_u64()?)?;
    let mut values = Vec::new();
    values
        .try_reserve(n)
        .map_err(|_| FerrumError::gguf("metadata allocation failed"))?;
    for _ in 0..n {
        values.push(read_value(reader, elem_type, depth + 1)?);
    }
    Ok(MetadataValue::Array(MetadataArray { elem_type, values }))
}

fn min_value_bytes(value_type: MetadataType) -> u64 {
    match value_type {
        MetadataType::Uint8 | MetadataType::Int8 | MetadataType::Bool => 1,
        MetadataType::Uint16 | MetadataType::Int16 => 2,
        MetadataType::Uint32 | MetadataType::Int32 | MetadataType::Float32 => 4,
        MetadataType::Uint64
        | MetadataType::Int64
        | MetadataType::Float64
        | MetadataType::String => 8,
        MetadataType::Array => 12,
    }
}

fn alignment_of(metadata: &HashMap<String, MetadataValue>) -> Result<u64, FerrumError> {
    match metadata.get("general.alignment") {
        None => Ok(32),
        Some(MetadataValue::Uint32(value)) => validate_alignment(u64::from(*value)),
        Some(MetadataValue::Uint64(value)) => validate_alignment(*value),
        Some(_) => Err(FerrumError::gguf(
            "general.alignment must be uint32 or uint64",
        )),
    }
}

fn validate_alignment(alignment: u64) -> Result<u64, FerrumError> {
    if alignment == 0 || !alignment.is_multiple_of(8) {
        return Err(FerrumError::gguf(
            "alignment must be a non-zero multiple of 8",
        ));
    }
    Ok(alignment)
}

/// `offset + (alignment - (offset % alignment)) % alignment`.
fn align_offset(offset: u64, alignment: u64) -> Result<u64, FerrumError> {
    let alignment = validate_alignment(alignment)?;
    let rem = offset % alignment;
    let pad = (alignment - rem) % alignment;
    offset
        .checked_add(pad)
        .ok_or_else(|| FerrumError::gguf("aligned offset overflows"))
}

/// Reject `count` before any `Vec` or map reserves it.
///
/// `count * min_elem <= remaining` is required, with the product checked by
/// comparing against `remaining / min_elem` so the multiplication cannot wrap.
fn bound_count(count: u64, min_elem: u64, remaining: u64) -> Result<usize, FerrumError> {
    if min_elem == 0 {
        return Err(FerrumError::gguf("element size is zero"));
    }
    if count > remaining / min_elem {
        return Err(FerrumError::gguf("count exceeds remaining bytes"));
    }
    usize::try_from(count).map_err(|_| FerrumError::gguf("count does not fit usize"))
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn remaining_u64(&self) -> Result<u64, FerrumError> {
        u64::try_from(self.remaining())
            .map_err(|_| FerrumError::gguf("remaining length exceeds u64"))
    }

    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], FerrumError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| FerrumError::gguf("read offset overflows"))?;
        let data = self.data;
        if end > data.len() {
            return Err(FerrumError::gguf("unexpected end of gguf file"));
        }
        let start = self.pos;
        self.pos = end;
        data.get(start..end)
            .ok_or_else(|| FerrumError::gguf("unexpected end of gguf file"))
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], FerrumError> {
        let bytes = self.read_exact(N)?;
        if bytes.len() != N {
            return Err(FerrumError::gguf("unexpected end of gguf file"));
        }
        let mut buf = [0u8; N];
        buf.copy_from_slice(bytes);
        Ok(buf)
    }

    fn read_u8(&mut self) -> Result<u8, FerrumError> {
        self.read_exact(1)?
            .first()
            .copied()
            .ok_or_else(|| FerrumError::gguf("unexpected end of gguf file"))
    }

    fn read_u16(&mut self) -> Result<u16, FerrumError> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> Result<u32, FerrumError> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, FerrumError> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_i8(&mut self) -> Result<i8, FerrumError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_i16(&mut self) -> Result<i16, FerrumError> {
        Ok(i16::from_le_bytes(self.read_array()?))
    }

    fn read_i32(&mut self) -> Result<i32, FerrumError> {
        Ok(i32::from_le_bytes(self.read_array()?))
    }

    fn read_i64(&mut self) -> Result<i64, FerrumError> {
        Ok(i64::from_le_bytes(self.read_array()?))
    }

    fn read_f32(&mut self) -> Result<f32, FerrumError> {
        Ok(f32::from_le_bytes(self.read_array()?))
    }

    fn read_f64(&mut self) -> Result<f64, FerrumError> {
        Ok(f64::from_le_bytes(self.read_array()?))
    }

    fn read_bool(&mut self) -> Result<bool, FerrumError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(FerrumError::gguf("invalid bool byte")),
        }
    }

    fn read_string(&mut self) -> Result<String, FerrumError> {
        let len = self.read_u64()?;
        let n = bound_count(len, 1, self.remaining_u64()?)?;
        let bytes = self.read_exact(n)?;
        let text = std::str::from_utf8(bytes).map_err(|_| FerrumError::gguf("invalid utf-8"))?;
        Ok(text.to_owned())
    }
}

impl GgufModel for GgufFile {
    fn version(&self) -> u32 {
        self.version
    }

    fn alignment(&self) -> u64 {
        self.alignment
    }

    fn metadata_len(&self) -> usize {
        self.metadata.len()
    }

    fn metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    fn tensors(&self) -> &[TensorDesc] {
        &self.tensors
    }

    fn tensor(&self, name: &str) -> Result<&TensorDesc, FerrumError> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| FerrumError::gguf(format!("missing tensor {name}")))
    }

    fn tensor_bytes(&self, name: &str) -> Result<&[u8], FerrumError> {
        let tensor = self.tensor(name)?;
        let start = self
            .data_start
            .checked_add(tensor.data_offset)
            .ok_or_else(|| FerrumError::gguf("tensor offset overflow"))?;
        let end = start
            .checked_add(tensor.nbytes)
            .ok_or_else(|| FerrumError::gguf("tensor end overflow"))?;
        let start =
            usize::try_from(start).map_err(|_| FerrumError::gguf("tensor offset exceeds usize"))?;
        let end =
            usize::try_from(end).map_err(|_| FerrumError::gguf("tensor end exceeds usize"))?;
        self.mmap
            .get(start..end)
            .ok_or_else(|| FerrumError::gguf("tensor bytes out of range"))
    }

    fn dump(&self) -> String {
        format_gguf_dump(
            self.version,
            self.alignment,
            self.metadata.len(),
            &self.tensors,
        )
    }
}

impl MetadataView for GgufFile {
    fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_offset_matches_spec_and_rejects_overflow() {
        assert_eq!(align_offset(0, 32).unwrap(), 0);
        assert_eq!(align_offset(32, 32).unwrap(), 32);
        assert_eq!(align_offset(24, 32).unwrap(), 32);
        assert_eq!(align_offset(41, 8).unwrap(), 48);
        assert_eq!(align_offset(40, 8).unwrap(), 40);
        assert!(align_offset(1, 0).is_err());
        assert!(align_offset(1, 12).is_err());
        assert!(align_offset(u64::MAX - 3, 8).is_err());
    }

    #[test]
    fn gguf_file_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GgufFile>();
    }
}
