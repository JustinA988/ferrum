//! Compares teacher-forced Llama logits with a pinned llama.cpp build.

#![forbid(unsafe_code)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use ferrum_core_types::{
    argmax, CausalModel, DType, GgufModel, MetadataView, Tokenizer, TOKEN_EMBD,
};
use ferrum_gguf::{load, GgufFile};
use ferrum_model::Llama;
use ferrum_tokenizer::from_metadata;

struct MetaAdapter<'a>(&'a ferrum_gguf::GgufFile);

impl MetadataView for MetaAdapter<'_> {
    fn get(&self, key: &str) -> Option<&ferrum_core_types::MetadataValue> {
        self.0.metadata(key)
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn m4_prompts() -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/m4_prompts.txt");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect()
}

/// Max allowed abs logit difference vs llama.cpp oracle (ARCHITECTURE M4).
/// Override with `FERRUM_M4_MAX_DIFF` (parseable `f32`). Otherwise inferred from
/// `token_embd.weight` dtype: F16/F32 → 0.1, Q8_0 → 2.0.
fn m4_max_logit_diff(gguf: &GgufFile) -> f32 {
    if let Ok(raw) = std::env::var("FERRUM_M4_MAX_DIFF") {
        return raw
            .parse::<f32>()
            .unwrap_or_else(|e| panic!("FERRUM_M4_MAX_DIFF={raw:?}: {e}"));
    }
    let emb = gguf.tensor(TOKEN_EMBD).expect("token_embd.weight");
    match emb.dtype {
        DType::F16 | DType::F32 => 0.1,
        DType::Q8_0 => 2.0,
        other => panic!(
            "set FERRUM_M4_MAX_DIFF or use F16, F32, or Q8_0 weights; token_embd is {other:?}"
        ),
    }
}

fn oracle_logits(llama_dir: &Path, model: &Path, tokens: &[u32]) -> Vec<Vec<f32>> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("oracle/m4_logits.py");
    let payload = serde_json::json!({
        "llama_dir": llama_dir,
        "model": model,
        "tokens": tokens,
    });
    let mut child = Command::new("python")
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|err| panic!("spawn python for {}: {err}", script.display()));
    let stdin = child.stdin.as_mut().expect("python stdin");
    write!(stdin, "{payload}").expect("write oracle payload");
    let output = child.wait_with_output().expect("wait on oracle");
    if !output.status.success() {
        panic!("oracle failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse oracle json");
    let rows = value["logits"]
        .as_array()
        .expect("logits array")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("logit row")
                .iter()
                .map(|v| v.as_f64().expect("float") as f32)
                .collect()
        })
        .collect();
    rows
}

fn compare_logits(ferrum: &[f32], oracle: &[f32], max_diff: f32) -> (f32, bool, bool) {
    assert_eq!(ferrum.len(), oracle.len());
    let mut max_abs = 0.0f32;
    for (a, b) in ferrum.iter().zip(oracle.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    let top_ferrum = argmax(ferrum).expect("ferrum argmax");
    let top_oracle = argmax(oracle).expect("oracle argmax");
    (max_abs, top_ferrum == top_oracle, max_abs <= max_diff)
}

#[test]
#[ignore = "requires FERRUM_MODEL, FERRUM_LLAMA_CPP, and python oracle"]
fn teacher_forced_logits_match_llama_cpp_q8_0() {
    let model_path = env_path("FERRUM_MODEL")
        .expect("set FERRUM_MODEL to the TinyLlama GGUF (e.g. Q8_0 or F16)");
    let llama_dir =
        env_path("FERRUM_LLAMA_CPP").expect("set FERRUM_LLAMA_CPP to pinned llama binaries");
    assert!(
        model_path.is_file(),
        "FERRUM_MODEL missing: {}",
        model_path.display()
    );

    let gguf = load(&model_path).expect("load gguf");
    let tokenizer = from_metadata(&MetaAdapter(&gguf)).expect("tokenizer");
    let n_vocab = gguf
        .metadata("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            ferrum_core_types::MetadataValue::Array(a) => Some(a.values.len() as u32),
            _ => None,
        })
        .expect("vocab size");

    let prompts = m4_prompts();
    assert!(
        prompts.len() >= 5,
        "m4_prompts.txt must contain at least five lines"
    );

    let prompt_tokens: Vec<Vec<u32>> = prompts
        .iter()
        .take(5)
        .map(|text| {
            let tokens = tokenizer.encode(text, false).expect("encode");
            assert!(
                tokens.len() >= 32,
                "prompt needs >= 32 tokens, got {}",
                tokens.len()
            );
            tokens
        })
        .collect();

    let max_seq = 512usize;
    let max_diff = m4_max_logit_diff(&gguf);
    let mut llama = Llama::load(gguf, 1, max_seq).expect("llama load");
    for (idx, tokens) in prompt_tokens.iter().enumerate() {
        let oracle_rows = oracle_logits(&llama_dir, &model_path, tokens);

        let slot = llama.alloc_slot().expect("slot");
        let mut logits = vec![0.0f32; tokens.len() * n_vocab as usize];
        llama
            .forward(tokens, 0, slot, &mut logits)
            .expect("forward");
        llama.free_slot(slot).expect("free slot");

        assert_eq!(oracle_rows.len(), tokens.len());
        let mut top_matches = 0usize;
        for (t, oracle_row) in oracle_rows.iter().enumerate() {
            let base = t * n_vocab as usize;
            let row = &logits[base..base + n_vocab as usize];
            assert_eq!(row.len(), oracle_row.len());
            let (max_abs, top_ok, within) = compare_logits(row, oracle_row, max_diff);
            assert!(
                within,
                "prompt {idx} token {t}: max abs logit diff {max_abs} > {max_diff}"
            );
            if top_ok {
                top_matches += 1;
            }
        }
        let agreement = top_matches as f64 / tokens.len() as f64;
        assert!(
            agreement >= 0.99,
            "prompt {idx}: top-1 agreement {agreement} < 0.99"
        );
    }
}
