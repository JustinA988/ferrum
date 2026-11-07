//! GGUF metadata types and helpers for reading required fields.

use crate::error::FerrumError;

/// GGUF metadata value type tag. Numeric values match the GGUF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataType {
    pub fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetadataArray {
    pub elem_type: MetadataType,
    pub values: Vec<MetadataValue>,
}

/// Owned metadata. Tensor payloads stay borrowed from the mmap; metadata does not.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn as_u64(&self) -> Result<u64, FerrumError> {
        match self {
            Self::Uint8(v) => Ok(u64::from(*v)),
            Self::Uint16(v) => Ok(u64::from(*v)),
            Self::Uint32(v) => Ok(u64::from(*v)),
            Self::Uint64(v) => Ok(*v),
            Self::Int8(v) => u64::try_from(*v).map_err(|_| FerrumError::gguf("negative integer")),
            Self::Int16(v) => u64::try_from(*v).map_err(|_| FerrumError::gguf("negative integer")),
            Self::Int32(v) => u64::try_from(*v).map_err(|_| FerrumError::gguf("negative integer")),
            Self::Int64(v) => u64::try_from(*v).map_err(|_| FerrumError::gguf("negative integer")),
            _ => Err(FerrumError::gguf("expected integer")),
        }
    }

    pub fn as_u32(&self) -> Result<u32, FerrumError> {
        let value = self.as_u64()?;
        u32::try_from(value).map_err(|_| FerrumError::gguf("integer does not fit u32"))
    }

    pub fn as_i32(&self) -> Result<i32, FerrumError> {
        match self {
            Self::Int8(v) => Ok(i32::from(*v)),
            Self::Int16(v) => Ok(i32::from(*v)),
            Self::Int32(v) => Ok(*v),
            Self::Int64(v) => {
                i32::try_from(*v).map_err(|_| FerrumError::gguf("integer does not fit i32"))
            }
            Self::Uint8(v) => Ok(i32::from(*v)),
            Self::Uint16(v) => Ok(i32::from(*v)),
            Self::Uint32(v) => {
                i32::try_from(*v).map_err(|_| FerrumError::gguf("integer does not fit i32"))
            }
            Self::Uint64(v) => {
                i32::try_from(*v).map_err(|_| FerrumError::gguf("integer does not fit i32"))
            }
            _ => Err(FerrumError::gguf("expected integer")),
        }
    }

    pub fn as_f32(&self) -> Result<f32, FerrumError> {
        match self {
            Self::Float32(v) if v.is_finite() => Ok(*v),
            Self::Float64(v)
                if v.is_finite() && *v <= f64::from(f32::MAX) && *v >= f64::from(f32::MIN) =>
            {
                Ok(*v as f32)
            }
            _ => Err(FerrumError::gguf("expected finite float")),
        }
    }

    pub fn as_str(&self) -> Result<&str, FerrumError> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(FerrumError::gguf("expected string")),
        }
    }

    pub fn as_array(&self) -> Result<&[MetadataValue], FerrumError> {
        match self {
            Self::Array(array) => Ok(&array.values),
            _ => Err(FerrumError::gguf("expected array")),
        }
    }

    pub fn as_str_vec(&self) -> Result<Vec<&str>, FerrumError> {
        self.as_array()?
            .iter()
            .map(Self::as_str)
            .collect::<Result<Vec<_>, _>>()
    }

    pub fn as_f32_vec(&self) -> Result<Vec<f32>, FerrumError> {
        self.as_array()?
            .iter()
            .map(Self::as_f32)
            .collect::<Result<Vec<_>, _>>()
    }

    pub fn as_i32_vec(&self) -> Result<Vec<i32>, FerrumError> {
        self.as_array()?
            .iter()
            .map(Self::as_i32)
            .collect::<Result<Vec<_>, _>>()
    }
}

pub trait MetadataView {
    fn get(&self, key: &str) -> Option<&MetadataValue>;
}

pub fn required<'a>(
    meta: &'a dyn MetadataView,
    key: &str,
) -> Result<&'a MetadataValue, FerrumError> {
    meta.get(key)
        .ok_or_else(|| FerrumError::gguf(format!("missing metadata key {key}")))
}

pub fn meta_u32(meta: &dyn MetadataView, key: &str) -> Result<u32, FerrumError> {
    required(meta, key)?
        .as_u32()
        .map_err(|_| FerrumError::gguf(format!("{key}: expected u32")))
}

pub fn meta_f32(meta: &dyn MetadataView, key: &str) -> Result<f32, FerrumError> {
    required(meta, key)?
        .as_f32()
        .map_err(|_| FerrumError::gguf(format!("{key}: expected finite f32")))
}

pub fn meta_str<'a>(meta: &'a dyn MetadataView, key: &str) -> Result<&'a str, FerrumError> {
    required(meta, key)?
        .as_str()
        .map_err(|_| FerrumError::gguf(format!("{key}: expected string")))
}

pub fn optional_u32(meta: &dyn MetadataView, key: &str) -> Result<Option<u32>, FerrumError> {
    match meta.get(key) {
        None => Ok(None),
        Some(value) => value
            .as_u32()
            .map(Some)
            .map_err(|_| FerrumError::gguf(format!("{key}: expected u32"))),
    }
}

pub fn optional_str<'a>(
    meta: &'a dyn MetadataView,
    key: &str,
) -> Result<Option<&'a str>, FerrumError> {
    match meta.get(key) {
        None => Ok(None),
        Some(value) => value
            .as_str()
            .map(Some)
            .map_err(|_| FerrumError::gguf(format!("{key}: expected string"))),
    }
}

pub fn optional_bool(meta: &dyn MetadataView, key: &str) -> Result<Option<bool>, FerrumError> {
    match meta.get(key) {
        None => Ok(None),
        Some(MetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(FerrumError::gguf(format!("{key}: expected bool"))),
    }
}
