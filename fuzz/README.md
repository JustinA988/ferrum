`fuzz/fuzz_targets/gguf_parse.rs` calls `ferrum_gguf::load` on arbitrary bytes and asserts that the call returns a `Result`. A panic is a failure.

### Run

Needs nightly and `cargo-fuzz`. On Windows MSVC, put the VS BuildTools ASAN DLL on `PATH` (or next to the binary):

`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\clang_rt.asan_dynamic-x86_64.dll`

```text
cd fuzz
cargo +nightly fuzz run gguf_parse -- -max_total_time=600
```

Measured: see `docs/STATUS.md` (M1 fuzz).
