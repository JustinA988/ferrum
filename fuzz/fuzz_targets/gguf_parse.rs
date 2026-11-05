//! Calls `ferrum_gguf::load` on arbitrary bytes. The call must return a `Result`.
//!
//! This target has not been run on this Windows host.

#![no_main]

use std::path::PathBuf;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let path = fuzz_path();
    if std::fs::write(&path, data).is_err() {
        return;
    }
    let loaded = ferrum_gguf::load(&path);
    assert!(
        matches!(loaded, Ok(_) | Err(_)),
        "load did not return a Result"
    );
    drop(loaded);
    let _ = std::fs::remove_file(&path);
});

fn fuzz_path() -> PathBuf {
    std::env::temp_dir().join(format!("ferrum-fuzz-gguf-{}.gguf", std::process::id()))
}
