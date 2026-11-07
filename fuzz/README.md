# GGUF loader fuzz target

`fuzz/fuzz_targets/gguf_parse.rs` calls `ferrum_gguf::load` on arbitrary bytes.
The call must return a `Result`. A panic is a failure. `Err` is a successful
rejection of a bad file. cargo-fuzz stores new inputs under
`fuzz/corpus/gguf_parse/`. That directory is gitignored.

### Run

Needs nightly and `cargo-fuzz`. On Windows MSVC, put the VS BuildTools ASAN DLL on `PATH` (or next to the binary):

`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\clang_rt.asan_dynamic-x86_64.dll`

```text
cd fuzz
cargo +nightly fuzz run gguf_parse -- -max_total_time=600
```

Measured: see `docs/STATUS.md` (M1 fuzz).
