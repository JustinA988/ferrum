//! SentencePiece acceptance on the frozen corpus.
//!
//! The in-process round trip always runs. Loading a GGUF tokenizer and comparing
//! ids with `llama-tokenize` are ignored unless you pass `--ignored` with
//! `FERRUM_MODEL` set to the GGUF and `FERRUM_LLAMA_CPP` set to the pinned
//! llama.cpp directory.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ferrum_core_types::{
    MetadataArray, MetadataType, MetadataValue, MetadataView, Tokenizer, TOKEN_BYTE, TOKEN_CONTROL,
    TOKEN_NORMAL, TOKEN_UNKNOWN,
};
use ferrum_tokenizer::from_metadata;

struct MapMeta {
    values: HashMap<String, MetadataValue>,
}

impl MetadataView for MapMeta {
    fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.values.get(key)
    }
}

fn string_array(values: Vec<String>) -> MetadataValue {
    MetadataValue::Array(MetadataArray {
        elem_type: MetadataType::String,
        values: values.into_iter().map(MetadataValue::String).collect(),
    })
}

fn i32_array(values: &[i32]) -> MetadataValue {
    MetadataValue::Array(MetadataArray {
        elem_type: MetadataType::Int32,
        values: values.iter().copied().map(MetadataValue::Int32).collect(),
    })
}

/// Llama SentencePiece metadata with a metaspace piece and `<0xHH>` byte fallback.
///
/// This is enough for `decode(encode(text, false)) == text` without a GGUF file.
/// It is not the TinyLlama vocabulary, so it does not check llama.cpp token ids.
fn byte_fallback_meta() -> MapMeta {
    let mut tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "\u{2581}".to_string(),
    ];
    let mut kinds = vec![TOKEN_UNKNOWN, TOKEN_CONTROL, TOKEN_CONTROL, TOKEN_NORMAL];
    for byte in 0u16..=255 {
        tokens.push(format!("<0x{byte:02X}>"));
        kinds.push(TOKEN_BYTE);
    }
    let mut values = HashMap::new();
    values.insert(
        "tokenizer.ggml.model".to_string(),
        MetadataValue::String("llama".to_string()),
    );
    values.insert("tokenizer.ggml.tokens".to_string(), string_array(tokens));
    values.insert("tokenizer.ggml.token_type".to_string(), i32_array(&kinds));
    values.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        MetadataValue::Uint32(1),
    );
    values.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        MetadataValue::Uint32(2),
    );
    MapMeta { values }
}

fn corpus_lines() -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/wave1.txt");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
        .lines()
        .map(str::to_string)
        .collect()
}

fn assert_round_trip(tokenizer: &dyn Tokenizer, lines: &[String]) {
    assert_eq!(lines.len(), 200);
    assert!(lines.iter().any(String::is_empty), "corpus contains empty");
    for line in lines {
        let ids = tokenizer
            .encode(line, false)
            .unwrap_or_else(|err| panic!("encode {line:?}: {err}"));
        let text = tokenizer
            .decode(&ids)
            .unwrap_or_else(|err| panic!("decode {line:?}: {err}"));
        assert_eq!(text, *line, "round-trip {line:?} ids {ids:?}");
    }
}

#[test]
fn corpus_round_trip_with_byte_fallback_metadata() {
    let tokenizer = from_metadata(&byte_fallback_meta()).expect("from_metadata");
    assert_eq!(tokenizer.vocab_size(), 260);
    assert_eq!(tokenizer.bos_id(), Some(1));
    assert_eq!(tokenizer.eos_id(), Some(2));
    let lines = corpus_lines();
    assert_round_trip(&tokenizer, &lines);
    assert_eq!(
        tokenizer.encode("", false).expect("empty encode"),
        Vec::<u32>::new()
    );
    assert_eq!(tokenizer.decode(&[]).expect("empty decode"), "");
}

/// Real TinyLlama GGUF round-trip. Default `cargo test` skips this.
#[test]
#[ignore = "needs FERRUM_MODEL; run with --ignored"]
fn corpus_round_trip_on_gguf() {
    let model = std::env::var("FERRUM_MODEL").expect("FERRUM_MODEL");
    let file =
        ferrum_gguf::load(Path::new(&model)).unwrap_or_else(|err| panic!("load {model}: {err}"));
    let tokenizer = from_metadata(&file).expect("from_metadata");
    assert_round_trip(&tokenizer, &corpus_lines());
}

/// Token ids equal llama.cpp `llama-tokenize --no-bos --no-parse-special --no-escape`.
///
/// That is the same operation as `encode(text, false)`: no BOS, control and
/// unknown pieces stay in the raw text. Default `cargo test` skips this.
#[test]
#[ignore = "needs FERRUM_MODEL and FERRUM_LLAMA_CPP; run with --ignored"]
fn corpus_token_ids_match_llama_cpp() {
    let model = std::env::var("FERRUM_MODEL").expect("FERRUM_MODEL");
    let pinned = std::env::var("FERRUM_LLAMA_CPP").expect("FERRUM_LLAMA_CPP");
    let file =
        ferrum_gguf::load(Path::new(&model)).unwrap_or_else(|err| panic!("load {model}: {err}"));
    let tokenizer = from_metadata(&file).expect("from_metadata");
    let tokenize = llama_bin(&pinned, "llama-tokenize");
    for line in corpus_lines() {
        let ours = tokenizer
            .encode(&line, false)
            .unwrap_or_else(|err| panic!("encode {line:?}: {err}"));
        let theirs = llama_token_ids(&tokenize, Path::new(&model), &line);
        assert_eq!(ours, theirs, "token ids for {line:?}");
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

fn llama_token_ids(tokenize: &Path, model: &Path, text: &str) -> Vec<u32> {
    let mut child = Command::new(tokenize)
        .args([
            "-m",
            model.to_str().expect("model path"),
            "--stdin",
            "--ids",
            "--no-bos",
            "--no-parse-special",
            "--no-escape",
            "--log-disable",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|err| panic!("spawn {}: {err}", tokenize.display()));
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(text.as_bytes())
        .expect("write prompt");
    let output = child.wait_with_output().expect("wait llama-tokenize");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "llama-tokenize failed for {text:?}: {}\n{stdout}",
        String::from_utf8_lossy(&output.stderr)
    );
    let list = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('['))
        .unwrap_or_else(|| panic!("no id list in {stdout:?}"));
    let value: serde_json::Value = serde_json::from_str(list.trim()).unwrap_or_else(|err| {
        panic!("parse {list:?}: {err}");
    });
    value
        .as_array()
        .unwrap_or_else(|| panic!("id list {list:?}"))
        .iter()
        .map(|id| u32::try_from(id.as_u64().expect("id")).expect("id fits u32"))
        .collect()
}
