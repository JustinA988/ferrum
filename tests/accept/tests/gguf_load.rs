//! Checks that `ferrum_gguf::load` reports the same tensor names, shapes, and dtypes as the architecture contract.

#![forbid(unsafe_code)]

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrum_core_types::{format_gguf_dump, DType, FerrumError, GgufModel, MetadataValue};

const META_UINT32: u32 = 4;
const META_BOOL: u32 = 7;
const META_STRING: u32 = 8;
const META_ARRAY: u32 = 9;
const META_UINT8: u32 = 0;
const GGML_F32: u32 = 0;
const ALIGNMENT: u64 = 32;

const TENSOR_F32: [f32; 6] = [1.0, -2.5, 3.25, 0.5, 7.0, -0.125];

const EXPECTED_DUMP: &str = "\
ferrum-gguf version=3 alignment=32 metadata=4 tensors=1
token_embd.weight [2, 3] f32
";

struct Buf {
    bytes: Vec<u8>,
}

impl Buf {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn raw(&mut self, data: &[u8]) {
        self.bytes.extend_from_slice(data);
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn gguf_str(&mut self, text: &str) {
        self.u64(u64::try_from(text.len()).expect("string length"));
        self.raw(text.as_bytes());
    }

    fn header(&mut self, version: u32, tensors: u64, kvs: u64) {
        self.raw(b"GGUF");
        self.u32(version);
        self.u64(tensors);
        self.u64(kvs);
    }

    fn kv_str(&mut self, key: &str, value: &str) {
        self.gguf_str(key);
        self.u32(META_STRING);
        self.gguf_str(value);
    }

    fn kv_u32(&mut self, key: &str, value: u32) {
        self.gguf_str(key);
        self.u32(META_UINT32);
        self.u32(value);
    }

    fn kv_bool_byte(&mut self, key: &str, byte: u8) {
        self.gguf_str(key);
        self.u32(META_BOOL);
        self.u8(byte);
    }

    fn tensor(&mut self, name: &str, shape: &[u64], ggml_type: u32, offset: u64) {
        self.gguf_str(name);
        self.u32(u32::try_from(shape.len()).expect("rank"));
        for dim in shape {
            self.u64(*dim);
        }
        self.u32(ggml_type);
        self.u64(offset);
    }

    /// `offset + (alignment - (offset % alignment)) % alignment`, then zero pad.
    fn pad_to(&mut self, alignment: u64) -> usize {
        let pos = u64::try_from(self.bytes.len()).expect("offset");
        let rem = pos % alignment;
        let pad = ((alignment - rem) % alignment) as usize;
        self.bytes.extend(std::iter::repeat_n(0u8, pad));
        self.bytes.len()
    }
}

fn f32_payload(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Little-endian GGUF v3, alignment 32, one f32 tensor, architecture llama.
fn valid_tiny() -> (Vec<u8>, Vec<u8>) {
    let mut buf = Buf::new();
    buf.header(3, 1, 4);
    buf.kv_str("general.architecture", "llama");
    buf.kv_str("general.name", "accept");
    buf.kv_u32("general.alignment", 32);
    buf.kv_bool_byte("accept.flag", 0);
    buf.tensor("token_embd.weight", &[2, 3], GGML_F32, 0);
    let data_start = buf.pad_to(ALIGNMENT);
    let payload = f32_payload(&TENSOR_F32);
    buf.raw(&payload);
    assert_eq!(buf.bytes.len() - data_start, payload.len());
    assert!(buf.bytes.ends_with(&payload));
    (buf.bytes, payload)
}

fn assert_send_sync<T: Send + Sync>() {}

fn write_bytes(label: &str, bytes: &[u8]) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join("ferrum-accept-m1");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{label}-{}-{n}.gguf", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn remove(path: &Path) {
    let _ = std::fs::remove_file(path);
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn assert_gguf_err(label: &str, bytes: &[u8]) {
    let path = write_bytes(label, bytes);
    let outcome = catch_unwind(AssertUnwindSafe(|| ferrum_gguf::load(&path)));
    match outcome {
        Ok(Ok(file)) => {
            drop(file);
            remove(&path);
            panic!("{label}: expected Err, load returned a GGUF");
        }
        Ok(Err(err)) => {
            remove(&path);
            if !matches!(err, FerrumError::Gguf(_)) {
                panic!("{label}: expected FerrumError::Gguf, got {err}");
            }
        }
        Err(payload) => {
            remove(&path);
            panic!(
                "{label}: load panicked ({})",
                panic_message(payload.as_ref())
            );
        }
    }
}

#[test]
fn gguf_file_is_send_sync() {
    assert_send_sync::<ferrum_gguf::GgufFile>();
}

#[test]
fn tiny_v3_dump_and_tensor_bytes_match_the_file() {
    let (bytes, payload) = valid_tiny();
    let path = write_bytes("valid-tiny", &bytes);
    let outcome = catch_unwind(AssertUnwindSafe(|| ferrum_gguf::load(&path)));
    let file = match outcome {
        Ok(Ok(file)) => file,
        Ok(Err(err)) => {
            remove(&path);
            panic!("valid tiny GGUF returned {err}");
        }
        Err(payload) => {
            remove(&path);
            panic!(
                "valid tiny GGUF panicked ({})",
                panic_message(payload.as_ref())
            );
        }
    };

    assert_eq!(file.version(), 3);
    assert_eq!(file.alignment(), ALIGNMENT);
    assert_eq!(file.metadata_len(), 4);
    match file.metadata("general.architecture") {
        Some(MetadataValue::String(value)) => assert_eq!(value, "llama"),
        other => panic!("general.architecture: {other:?}"),
    }
    match file.metadata("general.name") {
        Some(MetadataValue::String(value)) => assert_eq!(value, "accept"),
        other => panic!("general.name: {other:?}"),
    }
    match file.metadata("general.alignment") {
        Some(MetadataValue::Uint32(value)) => assert_eq!(*value, 32),
        other => panic!("general.alignment: {other:?}"),
    }
    match file.metadata("accept.flag") {
        Some(MetadataValue::Bool(value)) => assert!(!*value),
        other => panic!("accept.flag: {other:?}"),
    }

    let tensor = file.tensor("token_embd.weight").expect("token_embd.weight");
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.dtype, DType::F32);
    assert_eq!(tensor.ggml_type, GGML_F32);
    assert_eq!(tensor.data_offset, 0);
    assert_eq!(tensor.nbytes, u64::try_from(payload.len()).unwrap());
    assert_eq!(
        file.tensor_bytes("token_embd.weight")
            .expect("tensor bytes"),
        payload.as_slice()
    );

    let dump = file.dump();
    assert_eq!(dump, EXPECTED_DUMP);
    assert_eq!(
        dump,
        format_gguf_dump(
            file.version(),
            file.alignment(),
            file.metadata_len(),
            file.tensors()
        )
    );

    drop(file);
    remove(&path);
}

#[test]
fn bad_magic_is_err() {
    let (mut bytes, _) = valid_tiny();
    bytes[0] = b'X';
    assert_gguf_err("bad-magic", &bytes);
}

#[test]
fn truncated_header_is_err() {
    let mut buf = Buf::new();
    buf.raw(b"GGUF");
    buf.u32(3);
    assert_eq!(buf.bytes.len(), 8);
    assert_gguf_err("truncated-header", &buf.bytes);
}

#[test]
fn huge_array_count_is_err() {
    let mut buf = Buf::new();
    buf.header(3, 0, 1);
    buf.gguf_str("bad.array");
    buf.u32(META_ARRAY);
    buf.u32(META_UINT8);
    buf.u64(u64::MAX);
    assert_gguf_err("huge-array-count", &buf.bytes);
}

#[test]
fn tensor_offset_past_data_section_is_err() {
    let mut buf = Buf::new();
    buf.header(3, 1, 1);
    buf.kv_str("general.architecture", "llama");
    buf.tensor("w", &[1], GGML_F32, 32);
    let data_start = buf.pad_to(ALIGNMENT);
    buf.raw(&1.0f32.to_le_bytes());
    let data_len = buf.bytes.len() - data_start;
    assert!(
        32 + 4 > data_len,
        "fixture must place the tensor past the data section"
    );
    assert_gguf_err("tensor-offset-past-data", &buf.bytes);
}

#[test]
fn alignment_not_multiple_of_8_is_err() {
    let mut buf = Buf::new();
    buf.header(3, 0, 1);
    buf.kv_u32("general.alignment", 12);
    assert_gguf_err("alignment-not-multiple-of-8", &buf.bytes);
}

#[test]
fn bool_byte_other_than_0_or_1_is_err() {
    let mut buf = Buf::new();
    buf.header(3, 0, 1);
    buf.kv_bool_byte("accept.flag", 2);
    assert_gguf_err("bool-byte-2", &buf.bytes);
}

#[test]
fn invalid_utf8_string_is_err() {
    let mut buf = Buf::new();
    buf.header(3, 0, 1);
    buf.gguf_str("general.name");
    buf.u32(META_STRING);
    buf.u64(1);
    buf.u8(0xff);
    assert_gguf_err("invalid-utf8", &buf.bytes);
}

/// Name, GGUF shape, and ggml type of every tensor match the pinned llama.cpp reader.
///
/// Default `cargo test` skips this. Run it with `--ignored` after `FERRUM_MODEL`
/// and `FERRUM_LLAMA_CPP` are set. `FERRUM_LLAMA_CPP` is the directory that holds
/// `llama-cli` from commit a01dff908b9c9311d7af316bba3c918063749bd0. The tensor
/// list comes from that commit's `gguf` Python package (`gguf-py`).
#[test]
#[ignore = "needs FERRUM_MODEL and pinned llama.cpp; run with --ignored"]
fn llama_cpp_dump_matches_ferrum_dump() {
    let model = std::env::var("FERRUM_MODEL").expect("FERRUM_MODEL");
    let pinned = std::env::var("FERRUM_LLAMA_CPP").expect("FERRUM_LLAMA_CPP");
    let cli = llama_bin(&pinned, "llama-cli");
    let version = Command::new(&cli)
        .arg("--version")
        .output()
        .unwrap_or_else(|err| panic!("run {}: {err}", cli.display()));
    let version_text = format!(
        "{}{}",
        String::from_utf8_lossy(&version.stdout),
        String::from_utf8_lossy(&version.stderr)
    );
    assert!(
        version.status.success(),
        "llama-cli --version failed: {version_text}"
    );
    assert!(
        version_text.contains("a01dff908"),
        "llama.cpp pin is not a01dff908b9c9311d7af316bba3c918063749bd0: {version_text}"
    );

    let file = ferrum_gguf::load(Path::new(&model)).expect("load FERRUM_MODEL");
    let dump = file.dump();
    assert_eq!(
        dump,
        format_gguf_dump(
            file.version(),
            file.alignment(),
            file.metadata_len(),
            file.tensors()
        )
    );
    assert!(
        dump.lines().skip(1).count() == file.tensors().len(),
        "dump line count"
    );

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("oracle/gguf_tensors.py");
    let oracle = Command::new("python")
        .arg(&script)
        .arg(&model)
        .output()
        .unwrap_or_else(|err| panic!("python {}: {err}", script.display()));
    assert!(
        oracle.status.success(),
        "gguf reader failed: {}",
        String::from_utf8_lossy(&oracle.stderr)
    );
    let rows: serde_json::Value = serde_json::from_slice(&oracle.stdout).expect("oracle json");
    let rows = rows.as_array().expect("oracle array");
    assert_eq!(rows.len(), file.tensors().len(), "tensor count");
    for (ours, theirs) in file.tensors().iter().zip(rows.iter()) {
        let name = theirs["name"].as_str().expect("name");
        let shape: Vec<u64> = theirs["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|dim| dim.as_u64().expect("dim"))
            .collect();
        let ggml_type = theirs["ggml_type"].as_u64().expect("ggml_type") as u32;
        assert_eq!(ours.name, name);
        assert_eq!(ours.shape, shape, "{name}");
        assert_eq!(ours.ggml_type, ggml_type, "{name}");
    }
}

fn llama_bin(dir: &str, name: &str) -> PathBuf {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    PathBuf::from(dir).join(exe)
}
