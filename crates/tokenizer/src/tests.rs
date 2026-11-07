//! Encode and decode checks on a vocabulary built in the test.
//!
//! Covers a round trip, byte fallback, merge order, special tokens, and
//! metadata that must be rejected without a panic.

use std::collections::HashMap;

use ferrum_core_types::metadata::{MetadataArray, MetadataType, MetadataValue, MetadataView};
use ferrum_core_types::{
    FerrumError, Tokenizer, TOKEN_BYTE, TOKEN_CONTROL, TOKEN_NORMAL, TOKEN_UNKNOWN, TOKEN_UNUSED,
    TOKEN_USER_DEFINED,
};

use super::{from_metadata, SentencePieceTokenizer};

struct MapMeta {
    values: HashMap<String, MetadataValue>,
}

impl MetadataView for MapMeta {
    fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.values.get(key)
    }
}

struct Vocab {
    pieces: Vec<String>,
    scores: Vec<f32>,
    kinds: Vec<i32>,
    bos: u32,
    eos: u32,
}

impl Vocab {
    fn llama() -> Self {
        let mut vocab = Self {
            pieces: Vec::new(),
            scores: Vec::new(),
            kinds: Vec::new(),
            bos: 0,
            eos: 0,
        };
        vocab.push("<unk>", TOKEN_UNKNOWN, 0.0);
        vocab.bos = vocab.push("<s>", TOKEN_CONTROL, 0.0);
        vocab.eos = vocab.push("</s>", TOKEN_CONTROL, 0.0);
        vocab
    }

    fn push(&mut self, text: &str, kind: i32, score: f32) -> u32 {
        let id = u32::try_from(self.pieces.len()).unwrap();
        self.pieces.push(text.to_string());
        self.kinds.push(kind);
        self.scores.push(score);
        id
    }

    fn id(&self, text: &str) -> u32 {
        self.pieces
            .iter()
            .position(|piece| piece == text)
            .unwrap()
            .try_into()
            .unwrap()
    }

    fn meta(&self, with_scores: bool, with_types: bool) -> MapMeta {
        let mut values = HashMap::new();
        values.insert(
            "tokenizer.ggml.model".to_string(),
            MetadataValue::String("llama".to_string()),
        );
        values.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            MetadataValue::Uint32(self.bos),
        );
        values.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            MetadataValue::Uint32(self.eos),
        );
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            string_array(&self.pieces),
        );
        if with_scores {
            values.insert("tokenizer.ggml.scores".to_string(), f32_array(&self.scores));
        }
        if with_types {
            values.insert(
                "tokenizer.ggml.token_type".to_string(),
                i32_array(&self.kinds),
            );
        }
        MapMeta { values }
    }

    fn load(&self) -> SentencePieceTokenizer {
        from_metadata(&self.meta(true, true)).unwrap()
    }
}

fn string_array(values: &[String]) -> MetadataValue {
    MetadataValue::Array(MetadataArray {
        elem_type: MetadataType::String,
        values: values.iter().cloned().map(MetadataValue::String).collect(),
    })
}

fn f32_array(values: &[f32]) -> MetadataValue {
    MetadataValue::Array(MetadataArray {
        elem_type: MetadataType::Float32,
        values: values.iter().copied().map(MetadataValue::Float32).collect(),
    })
}

fn i32_array(values: &[i32]) -> MetadataValue {
    MetadataValue::Array(MetadataArray {
        elem_type: MetadataType::Int32,
        values: values.iter().copied().map(MetadataValue::Int32).collect(),
    })
}

fn expect_gguf(result: Result<SentencePieceTokenizer, FerrumError>) -> String {
    let message = result.expect_err("expected a metadata error").to_string();
    assert!(message.starts_with("gguf:"), "{message}");
    message
}

fn expect_tokenizer(result: Result<SentencePieceTokenizer, FerrumError>) -> String {
    let message = result.expect_err("expected a tokenizer error").to_string();
    assert!(message.starts_with("tokenizer:"), "{message}");
    message
}

fn mini() -> Vocab {
    let mut vocab = Vocab::llama();
    vocab.push("\u{2581}", TOKEN_NORMAL, 0.0);
    vocab.push("a", TOKEN_NORMAL, 0.0);
    vocab
}

fn round_trip_vocab() -> Vocab {
    let mut vocab = Vocab::llama();
    vocab.push("\u{2581}", TOKEN_NORMAL, 0.0);
    for ch in 'a'..='z' {
        vocab.push(&ch.to_string(), TOKEN_NORMAL, 0.0);
    }
    for ch in 'A'..='Z' {
        vocab.push(&ch.to_string(), TOKEN_NORMAL, 0.0);
    }
    vocab.push("hi", TOKEN_NORMAL, 1.0);
    for byte in 0u16..=255 {
        vocab.push(&format!("<0x{byte:02X}>"), TOKEN_BYTE, -100.0);
    }
    vocab
}

#[test]
fn rejects_missing_empty_and_non_llama_models() {
    let mut missing = mini().meta(true, true);
    missing.values.remove("tokenizer.ggml.model");
    assert!(expect_gguf(from_metadata(&missing)).contains("tokenizer.ggml.model"));

    let mut wrong_type = mini().meta(true, true);
    wrong_type
        .values
        .insert("tokenizer.ggml.model".to_string(), MetadataValue::Uint32(1));
    assert!(expect_gguf(from_metadata(&wrong_type)).contains("expected string"));

    for model in ["", "gpt2", "llama2", "LLAMA", " llama"] {
        let mut meta = mini().meta(true, true);
        meta.values.insert(
            "tokenizer.ggml.model".to_string(),
            MetadataValue::String(model.to_string()),
        );
        let message = expect_tokenizer(from_metadata(&meta));
        assert!(message.contains(model), "{message}");
    }
}

#[test]
fn rejects_bad_vocab_metadata_without_panicking() {
    let mut missing_tokens = mini().meta(true, true);
    missing_tokens.values.remove("tokenizer.ggml.tokens");
    expect_gguf(from_metadata(&missing_tokens));

    let mut missing_bos = mini().meta(true, true);
    missing_bos.values.remove("tokenizer.ggml.bos_token_id");
    let loaded = from_metadata(&missing_bos).unwrap();
    assert_eq!(loaded.bos_id(), None);
    assert_eq!(loaded.eos_id(), Some(mini().eos));
    let missing_bos_encode = loaded.encode("a", true).unwrap_err().to_string();
    assert!(
        missing_bos_encode.starts_with("tokenizer:"),
        "{missing_bos_encode}"
    );

    let mut missing_eos = mini().meta(true, true);
    missing_eos.values.remove("tokenizer.ggml.eos_token_id");
    let loaded = from_metadata(&missing_eos).unwrap();
    assert_eq!(loaded.eos_id(), None);
    assert_eq!(loaded.bos_id(), Some(mini().bos));

    let mut negative_bos = mini().meta(true, true);
    negative_bos.values.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        MetadataValue::Int32(-1),
    );
    expect_gguf(from_metadata(&negative_bos));

    let mut bos_oob = mini().meta(true, true);
    bos_oob.values.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        MetadataValue::Uint32(99),
    );
    assert!(expect_gguf(from_metadata(&bos_oob)).contains("bos_token_id"));

    let mut eos_oob = mini().meta(true, true);
    eos_oob.values.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        MetadataValue::Uint32(5),
    );
    assert!(expect_gguf(from_metadata(&eos_oob)).contains("eos_token_id"));

    let mut empty = mini().meta(true, true);
    let no_pieces: Vec<String> = Vec::new();
    empty.values.insert(
        "tokenizer.ggml.tokens".to_string(),
        string_array(&no_pieces),
    );
    assert!(expect_gguf(from_metadata(&empty)).contains("empty"));

    let mut not_strings = mini().meta(true, true);
    not_strings.values.insert(
        "tokenizer.ggml.tokens".to_string(),
        MetadataValue::Array(MetadataArray {
            elem_type: MetadataType::Uint32,
            values: vec![MetadataValue::Uint32(1)],
        }),
    );
    expect_gguf(from_metadata(&not_strings));

    let mut duplicate = mini();
    duplicate.push("a", TOKEN_NORMAL, 0.0);
    assert!(expect_gguf(from_metadata(&duplicate.meta(true, true))).contains("duplicate"));

    let mut bad_kind = mini().meta(true, true);
    let mut kinds = mini().kinds.clone();
    kinds[0] = 0;
    bad_kind
        .values
        .insert("tokenizer.ggml.token_type".to_string(), i32_array(&kinds));
    assert!(expect_gguf(from_metadata(&bad_kind)).contains("unknown token type"));

    let mut short_types = mini().meta(true, true);
    short_types.values.insert(
        "tokenizer.ggml.token_type".to_string(),
        i32_array(&[TOKEN_NORMAL]),
    );
    assert!(expect_gguf(from_metadata(&short_types)).contains("shorter"));

    let mut short_scores = mini().meta(true, true);
    short_scores
        .values
        .insert("tokenizer.ggml.scores".to_string(), f32_array(&[0.0]));
    assert!(expect_gguf(from_metadata(&short_scores)).contains("shorter"));

    let mut nan_scores = mini().meta(false, true);
    let mut scores = mini().scores.clone();
    scores[0] = f32::NAN;
    nan_scores
        .values
        .insert("tokenizer.ggml.scores".to_string(), f32_array(&scores));
    expect_gguf(from_metadata(&nan_scores));

    let mut infinite_scores = mini().meta(false, true);
    scores[0] = f32::INFINITY;
    infinite_scores
        .values
        .insert("tokenizer.ggml.scores".to_string(), f32_array(&scores));
    expect_gguf(from_metadata(&infinite_scores));

    let mut scores_wrong_type = mini().meta(false, true);
    scores_wrong_type.values.insert(
        "tokenizer.ggml.scores".to_string(),
        MetadataValue::String("0".to_string()),
    );
    expect_gguf(from_metadata(&scores_wrong_type));
}

#[test]
fn loads_when_scores_and_types_are_absent_or_longer() {
    let vocab = mini();
    let loaded = from_metadata(&vocab.meta(false, false)).unwrap();
    assert_eq!(
        loaded.vocab_size(),
        u32::try_from(vocab.pieces.len()).unwrap()
    );
    assert_eq!(loaded.bos_id(), Some(vocab.bos));
    assert_eq!(loaded.eos_id(), Some(vocab.eos));
    assert!(loaded.is_special(vocab.bos));
    assert!(loaded.is_special(vocab.eos));
    assert!(!loaded.is_special(vocab.id("a")));
    assert!(!loaded.is_special(u32::MAX));

    let mut longer = vocab.meta(true, true);
    let mut scores = vocab.scores.clone();
    scores.push(42.0);
    longer
        .values
        .insert("tokenizer.ggml.scores".to_string(), f32_array(&scores));
    let mut kinds = vocab.kinds.clone();
    kinds.push(TOKEN_NORMAL);
    longer
        .values
        .insert("tokenizer.ggml.token_type".to_string(), i32_array(&kinds));
    let loaded = from_metadata(&longer).unwrap();
    assert_eq!(
        loaded.vocab_size(),
        u32::try_from(vocab.pieces.len()).unwrap()
    );
}

#[test]
fn absent_token_types_keep_byte_spellings_as_normal_text() {
    let mut vocab = Vocab::llama();
    vocab.push("\u{2581}", TOKEN_NORMAL, 0.0);
    vocab.push("<0x0A>", TOKEN_BYTE, 0.0);
    let loaded = from_metadata(&vocab.meta(true, false)).unwrap();
    let ids = loaded.encode("\n", false).unwrap();
    assert_eq!(
        ids,
        vec![loaded_id(&vocab, "\u{2581}"), loaded_id(&vocab, "<0x0A>")]
    );
    assert_eq!(
        loaded.decode_token(loaded_id(&vocab, "<0x0A>")).unwrap(),
        "<0x0A>"
    );
    assert_eq!(loaded.decode(&ids).unwrap(), "<0x0A>");
    assert!(!loaded.is_special(loaded_id(&vocab, "<0x0A>")));
}

fn loaded_id(vocab: &Vocab, text: &str) -> u32 {
    vocab.id(text)
}

#[test]
fn empty_text_and_bos_prefix() {
    let tok = round_trip_vocab().load();
    let bos = tok.bos_id().unwrap();
    assert_eq!(tok.encode("", false).unwrap(), Vec::<u32>::new());
    assert_eq!(tok.encode("", true).unwrap(), vec![bos]);
    assert_eq!(tok.decode(&[]).unwrap(), "");
    assert_eq!(tok.decode(&[bos]).unwrap(), "<s>");
}

#[test]
fn round_trips_ordinary_strings() {
    let vocab = round_trip_vocab();
    let tok = vocab.load();
    let bos = tok.bos_id().unwrap();
    let samples = [
        "",
        "a",
        "A",
        "hi",
        "hello",
        "hello world",
        " hello",
        "  ",
        "a  b",
        "\n",
        "\t",
        "héllo",
        "🙂",
        "a🙂b",
        "line1\nline2",
        "mix\t of spaces",
        "<s>",
        "<unk>",
        "</s>",
        "<pad>",
        "<0x41>",
        "café 🙂",
        "e\u{0301}",
        "\u{feff}bom",
    ];
    for text in samples {
        let ids = tok.encode(text, false).unwrap();
        assert_eq!(
            tok.encode(text, false).unwrap(),
            ids,
            "encode was not stable for {text:?}"
        );
        assert_eq!(tok.decode(&ids).unwrap(), text, "{text:?} -> {ids:?}");
        let with_bos = tok.encode(text, true).unwrap();
        assert_eq!(with_bos.first().copied(), Some(bos));
        assert_eq!(&with_bos[1..], ids.as_slice());
    }

    let hi = tok.encode("hi", false).unwrap();
    assert_eq!(hi, vec![vocab.id("\u{2581}"), vocab.id("hi")]);
    assert_eq!(tok.decode_token(vocab.id("\u{2581}")).unwrap(), " ");
    assert_eq!(tok.decode(&hi).unwrap(), "hi");

    let letter = tok.encode("A", false).unwrap();
    assert_eq!(letter, vec![vocab.id("\u{2581}"), vocab.id("A")]);
    assert!(!letter.contains(&vocab.id("<0x41>")));

    let controls = tok.encode("<s>", false).unwrap();
    assert!(!controls.contains(&vocab.bos));
    assert!(!controls.contains(&vocab.eos));
    assert!(!controls.contains(&vocab.id("<unk>")));
}

#[test]
fn byte_fallback_uses_uppercase_hex() {
    let vocab = round_trip_vocab();
    let tok = vocab.load();
    let ids = tok.encode("€", false).unwrap();
    assert_eq!(
        ids,
        vec![
            vocab.id("\u{2581}"),
            vocab.id("<0xE2>"),
            vocab.id("<0x82>"),
            vocab.id("<0xAC>"),
        ]
    );
    assert_eq!(tok.decode(&ids).unwrap(), "€");
    assert_eq!(tok.decode_token(vocab.id("<0x0A>")).unwrap(), "\n");
    let message = tok
        .decode_token(vocab.id("<0xFF>"))
        .unwrap_err()
        .to_string();
    assert!(message.starts_with("tokenizer:"), "{message}");
    assert!(message.contains("utf-8"), "{message}");
}

#[test]
fn missing_byte_token_is_an_error() {
    let tok = mini().load();
    let err = tok.encode("€", false).unwrap_err();
    let message = err.to_string();
    assert!(message.starts_with("tokenizer:"), "{message}");
    assert!(message.contains("<0xE2>"), "{message}");
    assert!(tok.encode("a", false).is_ok());
}

#[test]
fn higher_score_merges_before_the_leftmost_pair() {
    let mut vocab = Vocab::llama();
    vocab.push("\u{2581}", TOKEN_NORMAL, 0.0);
    vocab.push("a", TOKEN_NORMAL, 0.0);
    vocab.push("b", TOKEN_NORMAL, 0.0);
    vocab.push("c", TOKEN_NORMAL, 0.0);
    let ab = vocab.push("ab", TOKEN_NORMAL, 0.0);
    let bc = vocab.push("bc", TOKEN_NORMAL, 10.0);
    let tok = vocab.load();
    let space = vocab.id("\u{2581}");
    assert_eq!(
        tok.encode("abc", false).unwrap(),
        vec![space, vocab.id("a"), bc]
    );

    vocab.scores[ab as usize] = 0.0;
    vocab.scores[bc as usize] = 0.0;
    let tok = vocab.load();
    assert_eq!(
        tok.encode("abc", false).unwrap(),
        vec![space, ab, vocab.id("c")]
    );
}

#[test]
fn successive_merges_keep_going_after_a_poor_score() {
    let mut vocab = Vocab::llama();
    let space = vocab.push("\u{2581}", TOKEN_NORMAL, 0.0);
    vocab.push("a", TOKEN_NORMAL, 0.0);
    vocab.push("b", TOKEN_NORMAL, 0.0);
    vocab.push("c", TOKEN_NORMAL, 0.0);
    vocab.push("ab", TOKEN_NORMAL, 1.0);
    let abc = vocab.push("abc", TOKEN_NORMAL, -10.0);
    let tok = vocab.load();
    assert_eq!(tok.encode("abc", false).unwrap(), vec![space, abc]);
}

#[test]
fn is_special_covers_control_unknown_and_bos_eos() {
    let mut vocab = Vocab::llama();
    let normal = vocab.push("a", TOKEN_NORMAL, 0.0);
    let byte = vocab.push("<0x41>", TOKEN_BYTE, 0.0);
    let user = vocab.push("<pad>", TOKEN_USER_DEFINED, 0.0);
    let unused = vocab.push("<unused>", TOKEN_UNUSED, 0.0);
    let tok = vocab.load();

    assert!(tok.is_special(vocab.id("<unk>")));
    assert!(tok.is_special(vocab.bos));
    assert!(tok.is_special(vocab.eos));
    assert!(!tok.is_special(unused));
    assert!(!tok.is_special(normal));
    assert!(!tok.is_special(byte));
    assert!(!tok.is_special(user));
    assert!(!tok.is_special(u32::MAX));

    assert_eq!(tok.decode_token(vocab.bos).unwrap(), "<s>");
    assert_eq!(tok.decode_token(vocab.id("<unk>")).unwrap(), "<unk>");
    assert_eq!(tok.decode_token(user).unwrap(), "<pad>");
    assert_eq!(tok.decode_token(unused).unwrap(), "");
    assert_eq!(tok.decode_token(byte).unwrap(), "A");

    vocab.kinds[vocab.bos as usize] = TOKEN_NORMAL;
    let tok = vocab.load();
    assert!(tok.is_special(vocab.bos));
    assert_eq!(tok.decode_token(vocab.bos).unwrap(), "<s>");
}

#[test]
fn user_defined_pieces_are_taken_out_before_metaspace() {
    let mut vocab = Vocab::llama();
    let space = vocab.push("\u{2581}", TOKEN_NORMAL, 0.0);
    let a = vocab.push("a", TOKEN_NORMAL, 0.0);
    let b = vocab.push("b", TOKEN_NORMAL, 0.0);
    let short = vocab.push("<p>", TOKEN_USER_DEFINED, 0.0);
    let pad = vocab.push("<pad>", TOKEN_USER_DEFINED, 0.0);
    let tok = vocab.load();

    assert_eq!(tok.encode("<pad>", false).unwrap(), vec![pad]);
    assert_eq!(
        tok.decode(&tok.encode("<pad>", false).unwrap()).unwrap(),
        "<pad>"
    );
    assert!(!tok.encode("<pad>", false).unwrap().contains(&short));

    assert_eq!(
        tok.encode("a<pad>b", false).unwrap(),
        vec![space, a, pad, space, b]
    );
    assert_eq!(
        tok.decode(&tok.encode("a<pad>b", false).unwrap()).unwrap(),
        "a<pad>b"
    );

    assert_eq!(
        tok.encode("<pad>a<pad><pad>", false).unwrap(),
        vec![pad, space, a, pad, pad]
    );
}

#[test]
fn explicit_normal_byte_text_is_not_decoded_as_a_byte() {
    let mut vocab = Vocab::llama();
    vocab.push("\u{2581}", TOKEN_NORMAL, 0.0);
    let literal = vocab.push("<0x0A>", TOKEN_NORMAL, 0.0);
    let tok = vocab.load();
    assert_eq!(tok.decode_token(literal).unwrap(), "<0x0A>");
    assert!(!tok.is_special(literal));
}

#[test]
fn out_of_range_ids_return_errors() {
    let tok = mini().load();
    let err = tok.decode_token(50).unwrap_err().to_string();
    assert!(err.starts_with("tokenizer:"), "{err}");
    let err = tok.decode(&[0, 50]).unwrap_err().to_string();
    assert!(err.starts_with("tokenizer:"), "{err}");
}
