//! Llama SentencePiece tokenizer loaded from GGUF metadata.
//!
//! Revision 1 accepts `tokenizer.ggml.model == "llama"` only. Segmentation follows
//! llama.cpp's SPM session: a leading metaspace, then greedy merges ordered by
//! piece score. This is not unigram Viterbi.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use ferrum_core_types::metadata::{meta_str, meta_u32, required};
use ferrum_core_types::{
    FerrumError, MetadataView, Tokenizer, TOKEN_BYTE, TOKEN_CONTROL, TOKEN_NORMAL, TOKEN_UNKNOWN,
    TOKEN_UNUSED, TOKEN_USER_DEFINED,
};

const KEY_MODEL: &str = "tokenizer.ggml.model";
const KEY_TOKENS: &str = "tokenizer.ggml.tokens";
const KEY_SCORES: &str = "tokenizer.ggml.scores";
const KEY_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
const KEY_BOS: &str = "tokenizer.ggml.bos_token_id";
const KEY_EOS: &str = "tokenizer.ggml.eos_token_id";

/// SentencePiece whitespace marker, U+2581.
const METASPACE: char = '\u{2581}';

const MODEL_LLAMA: &str = "llama";

/// Llama SentencePiece vocabulary loaded from GGUF metadata.
#[derive(Debug)]
pub struct SentencePieceTokenizer {
    pieces: Vec<Piece>,
    token_to_id: HashMap<String, u32>,
    /// User-defined ids, longest piece first, then smaller id.
    user_defined: Vec<u32>,
    n_vocab: u32,
    bos_id: u32,
    eos_id: u32,
}

#[derive(Debug)]
struct Piece {
    text: String,
    score: f32,
    kind: i32,
}

/// Loads a Llama SentencePiece tokenizer from GGUF metadata.
///
/// Missing keys and malformed arrays are [`FerrumError::Gguf`]. A tokenizer
/// model other than `"llama"` is [`FerrumError::Tokenizer`].
pub fn from_metadata(meta: &dyn MetadataView) -> Result<SentencePieceTokenizer, FerrumError> {
    let model = meta_str(meta, KEY_MODEL)?;
    if model != MODEL_LLAMA {
        return Err(FerrumError::tokenizer(format!(
            "unsupported tokenizer model \"{model}\""
        )));
    }

    let token_values = required(meta, KEY_TOKENS)?
        .as_str_vec()
        .map_err(|_| FerrumError::gguf(format!("{KEY_TOKENS}: expected string array")))?;
    let n_vocab = u32::try_from(token_values.len())
        .map_err(|_| FerrumError::gguf(format!("{KEY_TOKENS}: vocab does not fit u32")))?;
    if n_vocab == 0 {
        return Err(FerrumError::gguf(format!("{KEY_TOKENS}: vocab is empty")));
    }
    let n_tokens = token_values.len();

    let scores = optional_f32s(meta, KEY_SCORES, n_tokens)?;
    let kinds = optional_token_types(meta, KEY_TOKEN_TYPE, n_tokens)?;

    let bos_id = meta_u32(meta, KEY_BOS)?;
    let eos_id = meta_u32(meta, KEY_EOS)?;
    if bos_id >= n_vocab {
        return Err(FerrumError::gguf(format!(
            "{KEY_BOS}: id is outside the vocab"
        )));
    }
    if eos_id >= n_vocab {
        return Err(FerrumError::gguf(format!(
            "{KEY_EOS}: id is outside the vocab"
        )));
    }

    let mut pieces = Vec::with_capacity(n_tokens);
    let mut token_to_id = HashMap::with_capacity(n_tokens);
    let mut user_defined = Vec::new();
    for id in 0..n_vocab {
        let text = token_values[id as usize].to_string();
        if token_to_id.insert(text.clone(), id).is_some() {
            return Err(FerrumError::gguf(format!(
                "{KEY_TOKENS}: duplicate piece at index {id}"
            )));
        }
        let kind = match &kinds {
            Some(values) => values[id as usize],
            None if is_uppercase_byte_piece(&text) => TOKEN_BYTE,
            None => TOKEN_NORMAL,
        };
        let score = scores.as_ref().map_or(0.0, |values| values[id as usize]);
        if kind == TOKEN_USER_DEFINED && !text.is_empty() {
            user_defined.push(id);
        }
        pieces.push(Piece { text, score, kind });
    }
    user_defined.sort_by(|&left, &right| {
        let left_len = pieces[left as usize].text.len();
        let right_len = pieces[right as usize].text.len();
        right_len.cmp(&left_len).then(left.cmp(&right))
    });

    Ok(SentencePieceTokenizer {
        pieces,
        token_to_id,
        user_defined,
        n_vocab,
        bos_id,
        eos_id,
    })
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, FerrumError> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }
        if text.is_empty() {
            return Ok(ids);
        }

        // User-defined pieces are split out before SPM. Control and unknown
        // pieces stay in the raw text (llama.cpp `parse_special == false`).
        // Each raw fragment is prefixed with a metaspace: a leading ASCII
        // space, then every space becomes U+2581.
        for fragment in partition(text, &self.pieces, &self.user_defined) {
            match fragment {
                Fragment::Special(id) => ids.push(id),
                Fragment::Raw(raw) => {
                    let normalized = normalize_fragment(raw);
                    merge_pieces(&normalized, &self.token_to_id, &self.pieces, &mut ids)?;
                }
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String, FerrumError> {
        let mut bytes = Vec::new();
        for (index, id) in ids.iter().copied().enumerate() {
            let mut rendered = render_piece(self.piece(id)?)?;
            if index == 0 && rendered.first() == Some(&b' ') {
                rendered.remove(0);
            }
            bytes.append(&mut rendered);
        }
        String::from_utf8(bytes).map_err(|_| FerrumError::tokenizer("decoded bytes are not utf-8"))
    }

    fn decode_token(&self, id: u32) -> Result<String, FerrumError> {
        let rendered = render_piece(self.piece(id)?)?;
        String::from_utf8(rendered)
            .map_err(|_| FerrumError::tokenizer(format!("token {id} does not decode to utf-8")))
    }

    fn vocab_size(&self) -> u32 {
        self.n_vocab
    }

    fn bos_id(&self) -> Option<u32> {
        Some(self.bos_id)
    }

    fn eos_id(&self) -> Option<u32> {
        Some(self.eos_id)
    }

    fn is_special(&self, id: u32) -> bool {
        if id == self.bos_id || id == self.eos_id {
            return true;
        }
        match self.pieces.get(id as usize) {
            Some(piece) => {
                piece.kind == TOKEN_CONTROL
                    || piece.kind == TOKEN_UNKNOWN
                    || piece.kind == TOKEN_UNUSED
            }
            None => false,
        }
    }
}

impl SentencePieceTokenizer {
    fn piece(&self, id: u32) -> Result<&Piece, FerrumError> {
        self.pieces
            .get(id as usize)
            .ok_or_else(|| FerrumError::tokenizer(format!("token id {id} is outside the vocab")))
    }
}

fn optional_f32s(
    meta: &dyn MetadataView,
    key: &str,
    n_tokens: usize,
) -> Result<Option<Vec<f32>>, FerrumError> {
    let Some(value) = meta.get(key) else {
        return Ok(None);
    };
    let scores = value
        .as_f32_vec()
        .map_err(|_| FerrumError::gguf(format!("{key}: expected finite f32 array")))?;
    if scores.len() < n_tokens {
        return Err(FerrumError::gguf(format!(
            "{key}: length is shorter than the vocab"
        )));
    }
    Ok(Some(scores[..n_tokens].to_vec()))
}

fn optional_token_types(
    meta: &dyn MetadataView,
    key: &str,
    n_tokens: usize,
) -> Result<Option<Vec<i32>>, FerrumError> {
    let Some(value) = meta.get(key) else {
        return Ok(None);
    };
    let kinds = value
        .as_i32_vec()
        .map_err(|_| FerrumError::gguf(format!("{key}: expected i32 array")))?;
    if kinds.len() < n_tokens {
        return Err(FerrumError::gguf(format!(
            "{key}: length is shorter than the vocab"
        )));
    }
    for kind in kinds.iter().take(n_tokens) {
        if !is_known_token_type(*kind) {
            return Err(FerrumError::gguf(format!(
                "{key}: unknown token type {kind}"
            )));
        }
    }
    Ok(Some(kinds[..n_tokens].to_vec()))
}

fn is_known_token_type(kind: i32) -> bool {
    matches!(
        kind,
        TOKEN_NORMAL
            | TOKEN_UNKNOWN
            | TOKEN_CONTROL
            | TOKEN_USER_DEFINED
            | TOKEN_UNUSED
            | TOKEN_BYTE
    )
}

fn is_uppercase_byte_piece(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 6
        && bytes[0] == b'<'
        && bytes[1] == b'0'
        && bytes[2] == b'x'
        && bytes[5] == b'>'
        && is_uppercase_hex(bytes[3])
        && is_uppercase_hex(bytes[4])
}

fn is_uppercase_hex(byte: u8) -> bool {
    byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte)
}

fn parse_byte_piece(text: &str) -> Option<u8> {
    let bytes = text.as_bytes();
    if bytes.len() != 6
        || bytes[0] != b'<'
        || bytes[1] != b'0'
        || bytes[2] != b'x'
        || bytes[5] != b'>'
    {
        return None;
    }
    let hi = hex_value(bytes[3])?;
    let lo = hex_value(bytes[4])?;
    Some((hi << 4) | lo)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn render_piece(piece: &Piece) -> Result<Vec<u8>, FerrumError> {
    match piece.kind {
        TOKEN_UNUSED => Ok(Vec::new()),
        TOKEN_CONTROL | TOKEN_UNKNOWN | TOKEN_USER_DEFINED => Ok(piece.text.as_bytes().to_vec()),
        TOKEN_NORMAL => Ok(unescape_metaspace(&piece.text).into_bytes()),
        TOKEN_BYTE => {
            let byte = parse_byte_piece(&piece.text)
                .ok_or_else(|| FerrumError::tokenizer("byte token is not a <0xHH> piece"))?;
            Ok(vec![byte])
        }
        kind => Err(FerrumError::tokenizer(format!("unknown token type {kind}"))),
    }
}

fn unescape_metaspace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        out.push(if ch == METASPACE { ' ' } else { ch });
    }
    out
}

fn normalize_fragment(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + METASPACE.len_utf8());
    out.push(METASPACE);
    for ch in raw.chars() {
        out.push(if ch == ' ' { METASPACE } else { ch });
    }
    out
}

#[derive(Debug)]
enum Fragment<'a> {
    Raw(&'a str),
    Special(u32),
}

fn partition<'a>(text: &'a str, pieces: &[Piece], user_defined: &[u32]) -> Vec<Fragment<'a>> {
    let mut parts = vec![Fragment::Raw(text)];
    for &id in user_defined {
        let needle = &pieces[id as usize].text;
        let mut next = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                Fragment::Special(token) => next.push(Fragment::Special(token)),
                Fragment::Raw(raw) => split_user_defined(raw, needle, id, &mut next),
            }
        }
        parts = next;
    }
    parts
}

fn split_user_defined<'a>(raw: &'a str, needle: &str, id: u32, out: &mut Vec<Fragment<'a>>) {
    let mut rest = raw;
    while let Some(found) = rest.find(needle) {
        if found > 0 {
            out.push(Fragment::Raw(&rest[..found]));
        }
        out.push(Fragment::Special(id));
        rest = &rest[found + needle.len()..];
    }
    if !rest.is_empty() {
        out.push(Fragment::Raw(rest));
    }
}

#[derive(Debug)]
struct Symbol {
    prev: i32,
    next: i32,
    start: usize,
    n: usize,
}

#[derive(Debug, Clone, Copy)]
struct Bigram {
    left: i32,
    right: i32,
    score: f32,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score first. Equal scores prefer the smaller left index,
        // matching llama.cpp's `llm_bigram_spm` comparator.
        if self.score < other.score {
            return Ordering::Less;
        }
        if self.score > other.score {
            return Ordering::Greater;
        }
        if self.left != other.left {
            return other.left.cmp(&self.left);
        }
        self.right
            .cmp(&other.right)
            .then(self.size.cmp(&other.size))
    }
}

fn symbol_chain(text: &str) -> Result<Vec<Symbol>, FerrumError> {
    let mut symbols: Vec<Symbol> = Vec::new();
    let mut previous = -1i32;
    for (start, ch) in text.char_indices() {
        if symbols.len() == i32::MAX as usize {
            return Err(FerrumError::tokenizer("input is too long to tokenize"));
        }
        let index = symbols.len() as i32;
        if previous >= 0 {
            symbols[previous as usize].next = index;
        }
        symbols.push(Symbol {
            prev: previous,
            next: -1,
            start,
            n: ch.len_utf8(),
        });
        previous = index;
    }
    Ok(symbols)
}

fn merge_pieces(
    text: &str,
    token_to_id: &HashMap<String, u32>,
    pieces: &[Piece],
    ids: &mut Vec<u32>,
) -> Result<(), FerrumError> {
    let mut symbols = symbol_chain(text)?;
    if symbols.is_empty() {
        return Ok(());
    }

    let mut heap = BinaryHeap::new();
    for index in 1..symbols.len() {
        consider_bigram(
            &symbols,
            index as i32 - 1,
            index as i32,
            text,
            token_to_id,
            pieces,
            &mut heap,
        );
    }

    while let Some(bigram) = heap.pop() {
        let left = bigram.left as usize;
        let right = bigram.right as usize;
        let left_n = symbols[left].n;
        let right_n = symbols[right].n;
        let right_next = symbols[right].next;
        let Some(span) = left_n.checked_add(right_n) else {
            continue;
        };
        if left_n == 0 || right_n == 0 || span != bigram.size {
            continue;
        }

        symbols[left].n = span;
        symbols[right].n = 0;
        symbols[left].next = right_next;
        if right_next >= 0 {
            symbols[right_next as usize].prev = bigram.left;
        }
        let previous = symbols[left].prev;
        let next = symbols[left].next;
        consider_bigram(
            &symbols,
            previous,
            bigram.left,
            text,
            token_to_id,
            pieces,
            &mut heap,
        );
        consider_bigram(
            &symbols,
            bigram.left,
            next,
            text,
            token_to_id,
            pieces,
            &mut heap,
        );
    }

    let mut index = 0i32;
    while index != -1 {
        let symbol = &symbols[index as usize];
        let end = symbol
            .start
            .checked_add(symbol.n)
            .ok_or_else(|| FerrumError::tokenizer("merged piece is outside the input"))?;
        let piece = text
            .get(symbol.start..end)
            .ok_or_else(|| FerrumError::tokenizer("merged piece is outside the input"))?;
        if let Some(&id) = token_to_id.get(piece) {
            ids.push(id);
        } else {
            for &byte in piece.as_bytes() {
                ids.push(byte_token_id(byte, token_to_id)?);
            }
        }
        index = symbol.next;
    }
    Ok(())
}

fn consider_bigram(
    symbols: &[Symbol],
    left: i32,
    right: i32,
    text: &str,
    token_to_id: &HashMap<String, u32>,
    pieces: &[Piece],
    heap: &mut BinaryHeap<Bigram>,
) {
    if left < 0 || right < 0 {
        return;
    }
    let left_sym = &symbols[left as usize];
    let right_sym = &symbols[right as usize];
    if left_sym.n == 0 || right_sym.n == 0 {
        return;
    }
    let contiguous = left_sym.start.checked_add(left_sym.n) == Some(right_sym.start);
    debug_assert!(contiguous, "sentencepiece symbols are not contiguous");
    if !contiguous {
        return;
    }
    let Some(end) = right_sym.start.checked_add(right_sym.n) else {
        return;
    };
    if end > text.len() {
        return;
    }
    let merged = &text[left_sym.start..end];
    let Some(&id) = token_to_id.get(merged) else {
        return;
    };
    let Some(piece) = pieces.get(id as usize) else {
        return;
    };
    heap.push(Bigram {
        left,
        right,
        score: piece.score,
        size: merged.len(),
    });
}

fn byte_token_id(byte: u8, token_to_id: &HashMap<String, u32>) -> Result<u32, FerrumError> {
    let hex = format!("<0x{byte:02X}>");
    if let Some(&id) = token_to_id.get(&hex) {
        return Ok(id);
    }
    // llama.cpp then looks up the single raw byte. Bytes above 0x7F are not
    // one UTF-8 code unit, so only ASCII can hit this map.
    if byte < 0x80 {
        if let Ok(raw) = std::str::from_utf8(&[byte]) {
            if let Some(&id) = token_to_id.get(raw) {
                return Ok(id);
            }
        }
    }
    Err(FerrumError::tokenizer(format!("missing byte token {hex}")))
}

#[cfg(test)]
mod tests;
