//! Parser tests for alignment, metadata, tensor layout, and corrupt input.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ferrum_core_types::{
    format_gguf_dump, ggml_type_name, llama_config, DType, FerrumError, GgufModel, MetadataValue,
    MetadataView,
};

use crate::layout::LAYOUTS;
use crate::load;
use crate::GgufFile;

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

struct TempModel {
    path: PathBuf,
}

impl TempModel {
    fn new(bytes: &[u8]) -> Self {
        let n = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ferrum-gguf-{}-{}-{nanos}.gguf",
            std::process::id(),
            n
        ));
        std::fs::write(&path, bytes).expect("write temp gguf");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// `file` is declared first so it drops before the temp path is removed.
struct Loaded {
    file: GgufFile,
    _tmp: TempModel,
}

#[track_caller]
fn load_ok(bytes: &[u8]) -> Loaded {
    let tmp = TempModel::new(bytes);
    let file = load(tmp.path()).expect("gguf load");
    Loaded { file, _tmp: tmp }
}

#[track_caller]
fn load_gguf_err(bytes: &[u8]) -> String {
    let tmp = TempModel::new(bytes);
    match load(tmp.path()) {
        Ok(_) => panic!("expected gguf error"),
        Err(err) => gguf_message(err),
    }
}

#[track_caller]
fn gguf_message(err: FerrumError) -> String {
    match err {
        FerrumError::Gguf(message) => message,
        other => panic!("expected FerrumError::Gguf, got {other}"),
    }
}

#[track_caller]
fn unwrap_gguf_err(result: Result<GgufFile, FerrumError>) -> FerrumError {
    match result {
        Ok(_) => panic!("expected gguf error"),
        Err(err) => err,
    }
}

fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_str(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn header(version: u32, tensors: u64, kvs: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0x47, 0x47, 0x55, 0x46]);
    push_u32(&mut out, version);
    push_u64(&mut out, tensors);
    push_u64(&mut out, kvs);
    out
}

enum Val {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(u8),
    Str(&'static str),
    Raw(&'static [u8]),
    U64(u64),
    I64(i64),
    F64(f64),
    Arr(u32, Vec<Val>),
}

impl Val {
    fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::Str(_) | Self::Raw(_) => 8,
            Self::Arr(_, _) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }

    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Self::U8(value) => push_u8(out, *value),
            Self::I8(value) => push_u8(out, *value as u8),
            Self::U16(value) => push_u16(out, *value),
            Self::I16(value) => out.extend_from_slice(&value.to_le_bytes()),
            Self::U32(value) => push_u32(out, *value),
            Self::I32(value) => out.extend_from_slice(&value.to_le_bytes()),
            Self::F32(value) => out.extend_from_slice(&value.to_le_bytes()),
            Self::Bool(value) => push_u8(out, *value),
            Self::Str(value) => push_str(out, value),
            Self::Raw(value) => {
                push_u64(out, value.len() as u64);
                out.extend_from_slice(value);
            }
            Self::U64(value) => push_u64(out, *value),
            Self::I64(value) => out.extend_from_slice(&value.to_le_bytes()),
            Self::F64(value) => out.extend_from_slice(&value.to_le_bytes()),
            Self::Arr(elem, values) => {
                push_u32(out, *elem);
                push_u64(out, values.len() as u64);
                for value in values {
                    value.write(out);
                }
            }
        }
    }
}

fn write_kv(out: &mut Vec<u8>, key: &str, value: &Val) {
    push_str(out, key);
    push_u32(out, value.type_id());
    value.write(out);
}

struct Ten {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    offset: u64,
    bytes: Vec<u8>,
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    let rem = offset % alignment;
    offset + (alignment - rem) % alignment
}

/// `pad_alignment` is the alignment the reader is expected to use.
/// `None` leaves the file unpadded for cases that must fail before the data section.
fn build(
    version: u32,
    kvs: &[(&str, Val)],
    tensors: &[Ten],
    pad_alignment: Option<u64>,
) -> Vec<u8> {
    let mut out = header(version, tensors.len() as u64, kvs.len() as u64);
    for (key, value) in kvs {
        write_kv(&mut out, key, value);
    }
    for tensor in tensors {
        push_str(&mut out, &tensor.name);
        push_u32(&mut out, u32::try_from(tensor.dims.len()).unwrap());
        for dim in &tensor.dims {
            push_u64(&mut out, *dim);
        }
        push_u32(&mut out, tensor.ggml_type);
        push_u64(&mut out, tensor.offset);
    }
    if let Some(alignment) = pad_alignment {
        let data_start = align_up(u64::try_from(out.len()).unwrap(), alignment);
        out.resize(usize::try_from(data_start).unwrap(), 0);
        for tensor in tensors {
            if tensor.bytes.is_empty() {
                continue;
            }
            let Some(rel_end) = tensor
                .offset
                .checked_add(u64::try_from(tensor.bytes.len()).unwrap())
            else {
                continue;
            };
            if rel_end > 8 * 1024 * 1024 {
                continue;
            }
            let Some(abs_end) = data_start.checked_add(rel_end) else {
                continue;
            };
            let abs_start = data_start + tensor.offset;
            let abs_start = usize::try_from(abs_start).unwrap();
            let abs_end = usize::try_from(abs_end).unwrap();
            if abs_end > out.len() {
                out.resize(abs_end, 0);
            }
            out[abs_start..abs_end].copy_from_slice(&tensor.bytes);
        }
    }
    out
}

fn f32_tensor(name: &str, shape: &[u64], offset: u64, bytes: &[u8]) -> Ten {
    Ten {
        name: name.to_string(),
        dims: shape.to_vec(),
        ggml_type: 0,
        offset,
        bytes: bytes.to_vec(),
    }
}

fn assert_dump(file: &GgufFile) {
    assert_eq!(
        file.dump(),
        format_gguf_dump(
            file.version(),
            file.alignment(),
            file.metadata_len(),
            file.tensors()
        )
    );
    assert!(file.dump().ends_with('\n'));
}

#[test]
fn loads_f32_tensor_relative_to_the_data_section() {
    let payload = [1u8, 2, 3, 4];
    let second = [5u8, 6, 7, 8];
    let bytes = build(
        3,
        &[("general.alignment", Val::U32(32)), ("name", Val::Str("x"))],
        &[
            f32_tensor("a", &[1], 0, &payload),
            f32_tensor("b", &[1], 32, &second),
        ],
        Some(32),
    );
    let loaded = load_ok(&bytes);
    let file = &loaded.file;
    assert_eq!(file.version(), 3);
    assert_eq!(file.alignment(), 32);
    assert_eq!(file.metadata_len(), 2);
    assert_eq!(
        MetadataView::get(file, "name"),
        Some(&MetadataValue::String("x".into()))
    );
    assert_eq!(file.metadata("name"), MetadataView::get(file, "name"));
    assert!(file.metadata("missing").is_none());
    assert_eq!(file.tensor_bytes("a").unwrap(), &payload);
    assert_eq!(file.tensor_bytes("b").unwrap(), &second);
    assert_eq!(file.tensor("a").unwrap().dtype, DType::F32);
    assert_eq!(file.tensor("a").unwrap().nbytes, 4);
    assert!(file.tensor("missing").is_err());
    assert_dump(file);
    assert_eq!(
        file.dump(),
        "ferrum-gguf version=3 alignment=32 metadata=2 tensors=2\na [1] f32\nb [1] f32\n"
    );

    let expected = payload.to_vec();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            assert_eq!(file.tensor_bytes("a").unwrap(), expected.as_slice());
        });
        scope.spawn(|| {
            assert!(file.dump().contains("alignment=32"));
        });
    });
}

#[test]
fn version_two_and_odd_f32_and_f16_lengths() {
    let bytes = build(
        2,
        &[],
        &[
            f32_tensor("f", &[3], 0, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            Ten {
                name: "h".into(),
                dims: vec![3],
                ggml_type: 1,
                offset: 32,
                bytes: vec![9, 8, 7, 6, 5, 4],
            },
        ],
        Some(32),
    );
    let loaded = load_ok(&bytes);
    assert_eq!(loaded.file.version(), 2);
    assert_eq!(loaded.file.alignment(), 32);
    assert_eq!(loaded.file.metadata_len(), 0);
    assert_eq!(loaded.file.tensor("f").unwrap().dtype, DType::F32);
    assert_eq!(loaded.file.tensor("f").unwrap().nbytes, 12);
    assert_eq!(loaded.file.tensor("h").unwrap().dtype, DType::F16);
    assert_eq!(loaded.file.tensor("h").unwrap().nbytes, 6);
    assert_eq!(loaded.file.tensor_bytes("h").unwrap(), &[9, 8, 7, 6, 5, 4]);
    assert_dump(&loaded.file);
}

#[test]
fn empty_directory_uses_default_alignment_and_zero_padding() {
    let bytes = header(3, 0, 0);
    assert_eq!(bytes.len(), 24);
    assert!(load_gguf_err(&bytes).contains("padding"));

    let mut padded = bytes.clone();
    padded.resize(32, 0);
    let loaded = load_ok(&padded);
    assert_eq!(loaded.file.version(), 3);
    assert_eq!(loaded.file.alignment(), 32);
    assert_eq!(loaded.file.metadata_len(), 0);
    assert!(loaded.file.tensors().is_empty());
    assert_eq!(
        loaded.file.dump(),
        "ferrum-gguf version=3 alignment=32 metadata=0 tensors=0\n"
    );

    let mut dirty = padded.clone();
    dirty[24] = 1;
    assert!(load_gguf_err(&dirty).contains("not zero"));

    let mut version_two = header(2, 0, 0);
    version_two.resize(32, 0);
    assert_eq!(load_ok(&version_two).file.version(), 2);
}

#[test]
fn alignment_key_accepts_u32_and_u64_multiples_of_eight() {
    let payload = [9u8, 8, 7, 6];
    let bytes = build(
        3,
        &[("general.alignment", Val::U64(64))],
        &[f32_tensor("w", &[1], 0, &payload)],
        Some(64),
    );
    let loaded = load_ok(&bytes);
    assert_eq!(loaded.file.alignment(), 64);
    assert_eq!(loaded.file.tensor_bytes("w").unwrap(), &payload);

    let narrow = build(
        3,
        &[("general.alignment", Val::U32(8))],
        &[f32_tensor("w", &[1], 0, &payload)],
        Some(8),
    );
    assert_eq!(load_ok(&narrow).file.alignment(), 8);

    for bad in [
        Val::U32(0),
        Val::U32(12),
        Val::U64(0),
        Val::U64(4),
        Val::U16(32),
    ] {
        let err = load_gguf_err(&build(3, &[("general.alignment", bad)], &[], None));
        assert!(
            err.contains("alignment"),
            "expected alignment error, got {err}"
        );
    }

    let huge = build(3, &[("general.alignment", Val::U64(1 << 33))], &[], None);
    let err = load_gguf_err(&huge);
    assert!(err.contains("padding") || err.contains("overflow"), "{err}");
}

#[test]
fn magic_version_and_short_buffers() {
    assert!(load_gguf_err(&[0x47, 0x47, 0x55]).contains("end of"));
    assert!(load_gguf_err(b"gguf").contains("magic"));
    let mut wrong = header(3, 0, 0);
    wrong[0] = 0x00;
    assert!(load_gguf_err(&wrong).contains("magic"));

    assert!(load_gguf_err(&header(1, 0, 0)).contains("version"));
    assert!(load_gguf_err(&header(0, 0, 0)).contains("version"));
    assert!(load_gguf_err(&header(4, 0, 0)).contains("version"));

    let mut big_endian_version = Vec::new();
    big_endian_version.extend_from_slice(&[0x47, 0x47, 0x55, 0x46]);
    big_endian_version.extend_from_slice(&3u32.to_be_bytes());
    assert!(
        load_gguf_err(&big_endian_version).contains("version")
            || load_gguf_err(&big_endian_version).contains("end of")
    );

    let tmp = TempModel::new(&[]);
    assert!(matches!(
        unwrap_gguf_err(load(tmp.path())),
        FerrumError::Gguf(_)
    ));
}

#[test]
fn missing_path_and_directory_are_errors() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ferrum-gguf-missing-{}-{}",
        std::process::id(),
        TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let err = unwrap_gguf_err(load(&path));
    assert!(matches!(err, FerrumError::Gguf(_)));
    let err = unwrap_gguf_err(load(std::env::temp_dir().as_path()));
    assert!(matches!(err, FerrumError::Gguf(_)));
}

#[test]
fn every_metadata_type_and_nested_arrays_round_trip() {
    let nested = Val::Arr(9, vec![Val::Arr(0, vec![Val::U8(9), Val::U8(8)])]);
    let words = Val::Arr(8, vec![Val::Str("a"), Val::Str("bb"), Val::Str("▁")]);
    let bytes = build(
        3,
        &[
            ("u8", Val::U8(255)),
            ("i8", Val::I8(-1)),
            ("u16", Val::U16(65535)),
            ("i16", Val::I16(-300)),
            ("u32", Val::U32(0x8000_0000)),
            ("i32", Val::I32(-2)),
            ("f32", Val::F32(-0.0)),
            ("bf", Val::Bool(0)),
            ("bt", Val::Bool(1)),
            ("s", Val::Str("héllo")),
            (
                "arr",
                Val::Arr(2, vec![Val::U16(1), Val::U16(2), Val::U16(3)]),
            ),
            ("words", words),
            ("u64", Val::U64(u64::MAX)),
            ("i64", Val::I64(i64::MIN)),
            ("f64", Val::F64(1.5)),
            ("nest", nested),
        ],
        &[],
        Some(32),
    );
    let loaded = load_ok(&bytes);
    let file = &loaded.file;
    assert_eq!(file.metadata_len(), 16);
    assert_eq!(file.metadata("u8"), Some(&MetadataValue::Uint8(255)));
    assert_eq!(file.metadata("i8"), Some(&MetadataValue::Int8(-1)));
    assert_eq!(file.metadata("u16"), Some(&MetadataValue::Uint16(65535)));
    assert_eq!(file.metadata("i16"), Some(&MetadataValue::Int16(-300)));
    assert_eq!(
        file.metadata("u32"),
        Some(&MetadataValue::Uint32(0x8000_0000))
    );
    assert_eq!(file.metadata("i32"), Some(&MetadataValue::Int32(-2)));
    match file.metadata("f32") {
        Some(MetadataValue::Float32(value)) => assert_eq!(value.to_bits(), (-0.0f32).to_bits()),
        other => panic!("f32: {other:?}"),
    }
    assert_eq!(file.metadata("bf"), Some(&MetadataValue::Bool(false)));
    assert_eq!(file.metadata("bt"), Some(&MetadataValue::Bool(true)));
    assert_eq!(
        file.metadata("s"),
        Some(&MetadataValue::String("héllo".into()))
    );
    assert_eq!(file.metadata("u64"), Some(&MetadataValue::Uint64(u64::MAX)));
    assert_eq!(file.metadata("i64"), Some(&MetadataValue::Int64(i64::MIN)));
    match file.metadata("f64") {
        Some(MetadataValue::Float64(value)) => assert_eq!(value.to_bits(), 1.5f64.to_bits()),
        other => panic!("f64: {other:?}"),
    }
    match file.metadata("arr") {
        Some(MetadataValue::Array(array)) => {
            assert_eq!(array.elem_type, ferrum_core_types::MetadataType::Uint16);
            assert_eq!(
                array.values,
                vec![
                    MetadataValue::Uint16(1),
                    MetadataValue::Uint16(2),
                    MetadataValue::Uint16(3),
                ]
            );
        }
        other => panic!("arr: {other:?}"),
    }
    match file.metadata("words") {
        Some(MetadataValue::Array(array)) => {
            assert_eq!(
                array.values,
                vec![
                    MetadataValue::String("a".into()),
                    MetadataValue::String("bb".into()),
                    MetadataValue::String("▁".into()),
                ]
            );
        }
        other => panic!("words: {other:?}"),
    }
    match file.metadata("nest") {
        Some(MetadataValue::Array(outer)) => {
            assert_eq!(outer.elem_type, ferrum_core_types::MetadataType::Array);
            match &outer.values[0] {
                MetadataValue::Array(inner) => {
                    assert_eq!(
                        inner.values,
                        vec![MetadataValue::Uint8(9), MetadataValue::Uint8(8)]
                    );
                }
                other => panic!("inner: {other:?}"),
            }
        }
        other => panic!("nest: {other:?}"),
    }
    assert_dump(file);
}

#[test]
fn invalid_bool_utf8_and_type_tag() {
    let bad_bool = build(3, &[("b", Val::Bool(2))], &[], None);
    assert!(load_gguf_err(&bad_bool).contains("bool"));
    let bad_bool = build(3, &[("b", Val::Bool(0xff))], &[], None);
    assert!(load_gguf_err(&bad_bool).contains("bool"));

    let bad_array = build(
        3,
        &[(
            "b",
            Val::Arr(7, vec![Val::Bool(0), Val::Bool(1), Val::Bool(2)]),
        )],
        &[],
        None,
    );
    assert!(load_gguf_err(&bad_array).contains("bool"));

    let mut bad_key = header(3, 0, 1);
    push_u64(&mut bad_key, 2);
    bad_key.extend_from_slice(&[0xff, 0xfe]);
    push_u32(&mut bad_key, 0);
    bad_key.push(1);
    assert!(load_gguf_err(&bad_key).contains("utf-8"));

    let bad_value = build(3, &[("a", Val::Raw(&[0xff, 0xfe]))], &[], None);
    assert!(load_gguf_err(&bad_value).contains("utf-8"));

    let nested_bad = build(
        3,
        &[("a", Val::Arr(8, vec![Val::Str("ok"), Val::Raw(&[0xff])]))],
        &[],
        None,
    );
    assert!(load_gguf_err(&nested_bad).contains("utf-8"));

    let mut unknown = header(3, 0, 1);
    push_str(&mut unknown, "a");
    push_u32(&mut unknown, 13);
    assert!(load_gguf_err(&unknown).contains("unknown metadata type"));
}

#[test]
fn claimed_counts_are_rejected_without_panicking() {
    assert!(load_gguf_err(&header(3, 0, u64::MAX)).contains("count"));
    assert!(load_gguf_err(&header(3, u64::MAX, 0)).contains("count"));

    let mut huge_string = header(3, 0, 1);
    push_u64(&mut huge_string, u64::MAX);
    assert!(
        load_gguf_err(&huge_string).contains("count")
            || load_gguf_err(&huge_string).contains("end of")
    );

    let mut huge_array = header(3, 0, 1);
    push_str(&mut huge_array, "a");
    push_u32(&mut huge_array, 9);
    push_u32(&mut huge_array, 0);
    push_u64(&mut huge_array, u64::MAX);
    assert!(load_gguf_err(&huge_array).contains("count"));

    let mut just_over = header(3, 0, 1);
    push_str(&mut just_over, "a");
    push_u32(&mut just_over, 9);
    push_u32(&mut just_over, 0);
    push_u64(&mut just_over, 11);
    just_over.extend(std::iter::repeat_n(1u8, 10));
    assert!(load_gguf_err(&just_over).contains("count"));

    let mut nested_huge = header(3, 0, 1);
    push_str(&mut nested_huge, "a");
    push_u32(&mut nested_huge, 9);
    push_u32(&mut nested_huge, 9);
    push_u64(&mut nested_huge, 1);
    push_u32(&mut nested_huge, 0);
    push_u64(&mut nested_huge, u64::MAX);
    assert!(load_gguf_err(&nested_huge).contains("count"));

    let mut rank = header(3, 1, 0);
    push_str(&mut rank, "w");
    push_u32(&mut rank, u32::MAX);
    // Enough trailing bytes that the tensor-count bound passes, so the rank
    // check itself rejects the claimed dimension count before any dim allocation.
    rank.resize(rank.len() + 64, 0);
    assert!(load_gguf_err(&rank).contains("rank"));
}

fn nested_u8(depth: usize) -> Val {
    if depth == 1 {
        Val::Arr(0, vec![Val::U8(7)])
    } else {
        Val::Arr(9, vec![nested_u8(depth - 1)])
    }
}

#[test]
fn nested_array_depth_is_bounded() {
    let ok = build(3, &[("a", nested_u8(3))], &[], Some(32));
    let loaded = load_ok(&ok);
    match loaded.file.metadata("a") {
        Some(MetadataValue::Array(level1)) => match &level1.values[0] {
            MetadataValue::Array(level2) => match &level2.values[0] {
                MetadataValue::Array(level3) => {
                    assert_eq!(level3.values, vec![MetadataValue::Uint8(7)]);
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }

    let deep_ok = build(3, &[("a", nested_u8(64))], &[], Some(32));
    assert_eq!(load_ok(&deep_ok).file.metadata_len(), 1);
    let too_deep = build(3, &[("a", nested_u8(65))], &[], None);
    assert!(load_gguf_err(&too_deep).contains("nested"));
}

#[test]
fn tensor_shape_offset_and_range_are_checked() {
    let rank0 = build(
        3,
        &[],
        &[Ten {
            name: "w".into(),
            dims: vec![],
            ggml_type: 0,
            offset: 0,
            bytes: vec![],
        }],
        None,
    );
    assert!(load_gguf_err(&rank0).contains("rank"));

    let rank5 = build(
        3,
        &[],
        &[Ten {
            name: "w".into(),
            dims: vec![1, 1, 1, 1, 1],
            ggml_type: 0,
            offset: 0,
            bytes: vec![],
        }],
        None,
    );
    assert!(load_gguf_err(&rank5).contains("rank"));

    let zero = build(3, &[], &[f32_tensor("w", &[0], 0, &[])], None);
    assert!(load_gguf_err(&zero).contains("zero"));

    let zero_inner = build(3, &[], &[f32_tensor("w", &[4, 0, 1], 0, &[])], None);
    assert!(load_gguf_err(&zero_inner).contains("zero"));

    let overflow = build(3, &[], &[f32_tensor("w", &[1 << 62], 0, &[])], None);
    assert!(load_gguf_err(&overflow).contains("overflow"));

    let unaligned = build(
        3,
        &[("general.alignment", Val::U32(32))],
        &[f32_tensor("w", &[1], 8, &[1, 2, 3, 4])],
        Some(32),
    );
    assert!(load_gguf_err(&unaligned).contains("aligned"));

    let mut exact = build(3, &[], &[f32_tensor("w", &[1], 0, &[1, 2, 3, 4])], Some(32));
    exact.pop();
    assert!(load_gguf_err(&exact).contains("past end"));

    let range = build(
        3,
        &[("general.alignment", Val::U32(8))],
        &[f32_tensor("w", &[1], u64::MAX - 7, &[1, 2, 3, 4])],
        Some(8),
    );
    let err = load_gguf_err(&range);
    assert!(
        err.contains("overflow") || err.contains("past end"),
        "{err}"
    );

    let rank4 = build(
        3,
        &[],
        &[f32_tensor("w", &[2, 3, 1, 1], 0, &[0u8; 24])],
        Some(32),
    );
    let loaded = load_ok(&rank4);
    assert_eq!(loaded.file.tensor("w").unwrap().shape, vec![2, 3, 1, 1]);
    assert_eq!(loaded.file.tensor("w").unwrap().nbytes, 24);
    assert_eq!(loaded.file.tensor_bytes("w").unwrap().len(), 24);
}

#[test]
fn q8_0_and_other_ggml_types_are_bounds_checked() {
    let q8 = build(
        3,
        &[],
        &[Ten {
            name: "q".into(),
            dims: vec![32],
            ggml_type: 8,
            offset: 0,
            bytes: vec![0x11; 34],
        }],
        Some(32),
    );
    let loaded = load_ok(&q8);
    let tensor = loaded.file.tensor("q").unwrap();
    assert_eq!(tensor.dtype, DType::Q8_0);
    assert_eq!(tensor.ggml_type, 8);
    assert_eq!(tensor.nbytes, 34);
    assert_eq!(loaded.file.tensor_bytes("q").unwrap(), &[0x11; 34]);
    assert!(loaded.file.dump().contains("q8_0"));

    let bad_q8 = build(
        3,
        &[],
        &[Ten {
            name: "q".into(),
            dims: vec![16],
            ggml_type: 8,
            offset: 0,
            bytes: vec![],
        }],
        None,
    );
    assert!(load_gguf_err(&bad_q8).contains("multiple"));

    let q4 = build(
        3,
        &[],
        &[Ten {
            name: "k".into(),
            dims: vec![256],
            ggml_type: 12,
            offset: 0,
            bytes: vec![0x22; 144],
        }],
        Some(32),
    );
    let loaded = load_ok(&q4);
    assert_eq!(loaded.file.tensor("k").unwrap().dtype, DType::Other(12));
    assert_eq!(loaded.file.tensor("k").unwrap().nbytes, 144);
    assert!(loaded.file.dump().contains("q4_K"));

    let mut short_q4_0 = build(
        3,
        &[],
        &[Ten {
            name: "q".into(),
            dims: vec![32],
            ggml_type: 2,
            offset: 0,
            bytes: vec![0x33; 18],
        }],
        Some(32),
    );
    assert_eq!(load_ok(&short_q4_0).file.tensor("q").unwrap().nbytes, 18);
    short_q4_0.pop();
    assert!(load_gguf_err(&short_q4_0).contains("past end"));

    for id in [4u32, 5, 31, 100] {
        let err = load_gguf_err(&build(
            3,
            &[],
            &[Ten {
                name: "w".into(),
                dims: vec![1],
                ggml_type: id,
                offset: 0,
                bytes: vec![],
            }],
            None,
        ));
        assert!(err.contains("no stored size"), "{id}: {err}");
    }
}

#[test]
fn every_sized_ggml_type_is_listed_and_bounded() {
    let mut tensors = Vec::new();
    let mut cursor = 0u64;
    for &(id, block, nbytes) in LAYOUTS {
        let offset = align_up(cursor, 32);
        let len = usize::try_from(nbytes).unwrap();
        tensors.push(Ten {
            name: format!("t{id}"),
            dims: vec![block],
            ggml_type: id,
            offset,
            bytes: vec![0xAB; len],
        });
        cursor = offset + nbytes;
    }
    let bytes = build(3, &[], &tensors, Some(32));
    let loaded = load_ok(&bytes);
    assert_eq!(loaded.file.tensors().len(), LAYOUTS.len());
    assert_dump(&loaded.file);
    for &(id, block, nbytes) in LAYOUTS {
        let name = format!("t{id}");
        let tensor = loaded.file.tensor(&name).unwrap();
        assert_eq!(tensor.shape, vec![block]);
        assert_eq!(tensor.nbytes, nbytes);
        assert_eq!(tensor.ggml_type, id);
        let payload = loaded.file.tensor_bytes(&name).unwrap();
        assert_eq!(payload.len() as u64, nbytes);
        assert!(payload.iter().all(|byte| *byte == 0xAB));
        let dtype_name = ggml_type_name(id).unwrap();
        assert!(
            loaded.file.dump().contains(&format!("] {dtype_name}\n")),
            "{dtype_name}"
        );
        match id {
            0 => assert_eq!(tensor.dtype, DType::F32),
            1 => assert_eq!(tensor.dtype, DType::F16),
            8 => assert_eq!(tensor.dtype, DType::Q8_0),
            other => assert_eq!(tensor.dtype, DType::Other(other)),
        }
    }
    let mut short = bytes;
    short.pop();
    assert!(load_gguf_err(&short).contains("past end"));
}

#[test]
fn duplicate_metadata_key_and_tensor_name() {
    let dup_key = build(3, &[("a", Val::U8(1)), ("a", Val::U8(2))], &[], None);
    assert!(load_gguf_err(&dup_key).contains("duplicate metadata"));

    let dup_tensor = build(
        3,
        &[],
        &[
            f32_tensor("w", &[1], 0, &[1, 2, 3, 4]),
            f32_tensor("w", &[1], 32, &[5, 6, 7, 8]),
        ],
        None,
    );
    assert!(load_gguf_err(&dup_tensor).contains("duplicate tensor"));
}

#[test]
fn llama_metadata_is_readable_by_core_types() {
    let tokens = Val::Arr(
        8,
        vec![
            Val::Str("<unk>"),
            Val::Str("<s>"),
            Val::Str("</s>"),
            Val::Str("a"),
        ],
    );
    let bytes = build(
        3,
        &[
            ("general.architecture", Val::Str("llama")),
            ("general.name", Val::Str("tiny")),
            ("llama.context_length", Val::U64(32)),
            ("llama.embedding_length", Val::U32(8)),
            ("llama.block_count", Val::U32(2)),
            ("llama.feed_forward_length", Val::U32(16)),
            ("llama.attention.head_count", Val::U32(2)),
            ("llama.attention.head_count_kv", Val::U64(1)),
            ("llama.attention.layer_norm_rms_epsilon", Val::F32(1e-5)),
            ("tokenizer.ggml.model", Val::Str("llama")),
            ("tokenizer.ggml.bos_token_id", Val::U32(1)),
            ("tokenizer.ggml.eos_token_id", Val::U32(2)),
            ("tokenizer.ggml.tokens", tokens),
        ],
        &[],
        Some(32),
    );
    let loaded = load_ok(&bytes);
    let config = llama_config(&loaded.file).expect("llama metadata");
    assert_eq!(config.n_vocab, 4);
    assert_eq!(config.n_embd, 8);
    assert_eq!(config.n_layer, 2);
    assert_eq!(config.head_dim, 4);
    assert_eq!(config.n_head_kv, 1);
    assert_eq!(config.rope_freq_base, 10_000.0);
    assert_eq!(config.bos_token_id, 1);
    assert_eq!(config.eos_token_id, 2);
    assert_eq!(config.name, "tiny");
}

#[test]
fn ferrum_model_env_is_skipped_when_unset() {
    let Some(path) = std::env::var_os("FERRUM_MODEL") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let file = load(Path::new(&path)).expect("FERRUM_MODEL should load");
    assert!(file.version() == 2 || file.version() == 3);
    assert!(file.alignment() != 0 && file.alignment().is_multiple_of(8));
    assert_dump(&file);
    for tensor in file.tensors() {
        let bytes = file.tensor_bytes(&tensor.name).expect("tensor bytes");
        assert_eq!(bytes.len() as u64, tensor.nbytes);
    }
}
