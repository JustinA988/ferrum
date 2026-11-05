//! M3 acceptance for `ferrum_tokenizer::from_metadata`.
//! The wave-1 tokenizer corpus at corpus/wave1.txt is now frozen.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::Path;

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

/// Real TinyLlama GGUF round-trip. Ignored because `FERRUM_MODEL` is unset.
#[test]
#[ignore = "FERRUM_MODEL is unset; GGUF round-trip is not run"]
fn corpus_round_trip_on_gguf() {
    let Ok(model) = std::env::var("FERRUM_MODEL") else {
        return;
    };
    let file =
        ferrum_gguf::load(Path::new(&model)).unwrap_or_else(|err| panic!("load {model}: {err}"));
    let tokenizer = from_metadata(&file).expect("from_metadata");
    assert_round_trip(&tokenizer, &corpus_lines());
}

/// Token ids equal llama.cpp on the frozen corpus. The oracle is not pinned.
#[test]
#[ignore = "FERRUM_MODEL is unset; llama.cpp token-id comparison is not run"]
fn corpus_token_ids_match_llama_cpp() {
    if std::env::var_os("FERRUM_MODEL").is_none() || std::env::var_os("FERRUM_LLAMA_CPP").is_none()
    {
        return;
    }
    panic!("llama.cpp token-id comparison has not been run; the oracle commit is not pinned");
}
