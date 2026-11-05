// The wave-1 tokenizer corpus at corpus/wave1.txt is now frozen.

#![forbid(unsafe_code)]

use std::path::PathBuf;

fn corpus_text() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/wave1.txt");
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

#[test]
fn wave1_corpus_has_200_records() {
    let text = corpus_text();
    assert!(text.ends_with('\n'), "corpus must end with a newline");
    assert!(
        !text.ends_with("\n\n"),
        "the empty record is a line of its own, not a trailing extra break"
    );
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 200);
    assert!(lines.contains(&""), "empty string");
    assert!(
        lines
            .iter()
            .any(|line| !line.is_empty() && line.chars().all(char::is_whitespace)),
        "whitespace"
    );
    assert!(lines.iter().any(|line| line.contains('日')), "unicode");
    assert!(lines.iter().any(|line| line.contains('\u{1F600}')), "emoji");
}
