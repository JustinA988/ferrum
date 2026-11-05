//! Planned M3 tokenizer acceptance. cargo test does not compile this file.
//! The wave-1 tokenizer corpus at ../corpus/wave1.txt is now frozen.

#![forbid(unsafe_code)]

use std::path::Path;

use ferrum_core_types::{GgufModel, Tokenizer};
use ferrum_tokenizer::from_metadata;

fn corpus_lines() -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/wave1.txt");
    std::fs::read_to_string(path)
        .expect("wave1 corpus")
        .lines()
        .map(str::to_string)
        .collect()
}

/// `decode(encode(text, false)) == text` for every frozen corpus line, including empty.
#[test]
fn corpus_round_trip_without_bos() {
    let model = std::env::var("FERRUM_MODEL").expect("FERRUM_MODEL");
    let file = ferrum_gguf::load(Path::new(&model)).expect("load model");
    let tokenizer = from_metadata(&file).expect("ferrum_tokenizer::from_metadata");
    assert_eq!(
        file.metadata("tokenizer.ggml.model")
            .and_then(|value| value.as_str().ok()),
        Some("llama")
    );
    let lines = corpus_lines();
    assert_eq!(lines.len(), 200);
    for line in &lines {
        let ids = tokenizer.encode(line, false).expect("encode");
        let text = tokenizer.decode(&ids).expect("decode");
        assert_eq!(text, *line);
    }
}

/// Token ids equal llama.cpp on the frozen corpus. Not run: the pin is not chosen.
#[test]
#[ignore = "FERRUM_MODEL is unset; llama.cpp token-id comparison is not run"]
fn corpus_token_ids_match_llama_cpp() {
    let model = std::env::var("FERRUM_MODEL")
        .expect("FERRUM_MODEL is unset; llama.cpp token-id comparison is not run");
    let _pinned = std::env::var("FERRUM_LLAMA_CPP")
        .expect("FERRUM_LLAMA_CPP is unset; llama.cpp commit is not pinned");
    let file = ferrum_gguf::load(Path::new(&model)).expect("load model");
    let tokenizer = from_metadata(&file).expect("ferrum_tokenizer::from_metadata");
    for line in corpus_lines() {
        let ids = tokenizer.encode(&line, false).expect("encode");
        let _ = ids;
    }
    panic!("pinned llama.cpp token ids were not compared");
}
