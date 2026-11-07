//! Process-level checks of `ferrum serve`.
//!
//! Spawns the binary and checks a usage error, an unknown flag, a bad bind
//! address, and the non-zero exit when `--mock` is left off.

use std::process::{Command, Output};

fn ferrum(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ferrum"))
        .args(args)
        .output()
        .expect("run ferrum")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is utf-8")
}

#[test]
fn serve_without_mock_names_revision_1() {
    let output = ferrum(&["serve"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("the model engine is not in revision 1"));
}

#[test]
fn unknown_argument_is_an_error() {
    let output = ferrum(&["serve", "--mock", "--nope"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("unknown argument: --nope"));
}

#[test]
fn missing_command_is_usage() {
    let output = ferrum(&[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("usage: ferrum serve --mock [--bind ADDR]"));
}

#[test]
fn mock_rejects_a_bad_bind_address() {
    let output = ferrum(&["serve", "--mock", "--bind", "not-a-socket"]);
    assert_eq!(output.status.code(), Some(1));
    let err = stderr(&output);
    assert!(err.contains("not-a-socket"), "{err}");
    assert!(
        !err.contains("the model engine is not in revision 1"),
        "{err}"
    );
}
