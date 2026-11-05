# Status

Interface revision 1 is frozen. That freeze is not a measurement.

Host for every run below: Windows, 12th Gen Intel Core i7-12650H, runtime `avx2=true` and `fma=true`, so `ferrum_kernels::select` is `Avx2Fma`. Toolchain rustc 1.99.1 / cargo 1.99.1. There is no git remote, so GitHub CI was not run. The provisional workflow was run locally on `v1-accept` and exited 0: `cargo fmt --all -- --check`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`.

Oracle llama.cpp: tag `b12128`, commit `a01dff908b9c9311d7af316bba3c918063749bd0`. `llama-cli --version` printed `version: 0.9.0-dev (build 12128, commit a01dff908)`.

Target GGUF: TheBloke `TinyLlama-1.1B-Chat-v1.0-GGUF` file `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`, 1170781568 bytes, Hugging Face repo commit `52e7645ba7c309695bec7ac98f4f005b139cf465`. Not in git.

| Milestone | Result |
| --- | --- |
| M1 GGUF loader | Dump comparison passed. cargo-fuzz not run. |
| M2 Kernels | Measured. See the trial table. |
| M3 Tokenizer | Passed against the pinned `llama-tokenize`. |
| M4 Single-sequence forward | not run |
| M5 KV cache | not run (acceptance text was truncated) |
| M6a Server skeleton | Passed against openai 3.46.0. |

`main` is still the revision 1 freeze. These results are on `v1-accept` and are not merged. cargo-fuzz has not run, and the code-quality review has one open blocking finding.

## M1

`cargo test -p ferrum-accept --test gguf_load llama_cpp_dump_matches_ferrum_dump -- --ignored --exact` passed in 2.15s. Ferrum's tensor list matched the pinned commit's `gguf` 0.19.0 reader: 201 tensors, same order, same name, same GGUF shape (`shape[0]` contiguous), same ggml type id. The printed dump matches `format_gguf_dump` for that file.

cargo-fuzz for 10 minutes was not run. This host has no WSL and no libFuzzer. That part of M1 stays not run.

## M2

Release `cargo bench --manifest-path benches/kernels/Cargo.toml` with `--sample-size 10 --warm-up-time 0.2 --measurement-time 1`. Five process trials, each running GEMV then GEMM. Kernel error count: 0. Each trial's GEMV collected about 34k iterations in one second, and each GEMM collected about 10k iterations in one second, so the timed calls did run.

Sizes from `benches/kernels/benches/throughput.rs`: GEMV is f32 256×256. Criterion bytes are the weight matrix plus `x`: 263168 bytes. GEMM is f32 128×128 by 16 tokens. Criterion elements are `2 * 128 * 128 * 16` = 524288, which is the multiply-add flop count, so Gelem/s is GFLOP/s.

The GEMV working set is 256 KiB of weights plus 1 KiB of `x`. It fits in cache. The AVX2 loop stores each 8-wide vector to a stack slot and then adds into one scalar accumulator. That is the limiter for these rates. They are not a DRAM streaming figure and not a full-model GEMV.

Point estimates are the middle criterion throughput. Median and range are over the five trials.

| Trial | GEMV GiB/s | GEMM GFLOP/s |
| --- | --- | --- |
| 1 | 8.1374 | 4.0771 |
| 2 | 8.3429 | 3.9323 |
| 3 | 8.3392 | 4.3637 |
| 4 | 7.9002 | 4.0553 |
| 5 | 8.4320 | 4.2056 |
| median | 8.3392 | 4.0771 |
| range | 7.9002–8.4320 | 3.9323–4.3637 |

SIMD versus scalar agreement is the proptest and unit tests in that same `cargo test --workspace` run (24 accept tests and 11 kernel tests, 0 failed).

## M3

`corpus_round_trip_on_gguf` passed in 0.06s: all 200 lines of `tests/accept/corpus/wave1.txt`, including the empty line, satisfy `decode(encode(text, false)) == text` on the Q8_0 GGUF.

`corpus_token_ids_match_llama_cpp` passed in 19.54s. For every corpus line, `encode(text, false)` matched `llama-tokenize -m <file> --stdin --ids --no-bos --no-parse-special --no-escape --log-disable` from the pinned build. Mismatches: 0.

## M6a

`official_openai_client_completion_and_stream` passed in 2.78s. openai 3.46.0 `completions.create` against `ferrum serve --mock` on an ephemeral `127.0.0.1` port returned choice text `mock` for a non-stream call and for the concatenated stream. The in-process mock route test also passed in the workspace test run.

## Review

Correctness review of `8847fd7..93c0f39` for the author crates found no blocking issue. One medium tokenizer case remains open: `decode` strips one leading ASCII space from a user-defined piece, so a piece that itself starts with a space does not round-trip. The frozen 200-line corpus does not cover that piece.

Code-quality review blocks on the AVX2 kernels writing each vector out through an 8-wide stack slot and rechecking the scalar formula (`crates/kernels/src/avx2.rs`). No author file in the diff is over 1000 lines. That finding is still open, so the author branches are not merged.
