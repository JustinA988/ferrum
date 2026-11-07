# Status

Interface revision 1 is frozen. Measured results only below.

Host: Windows, 12th Gen Intel Core i7-12650H, AVX2+FMA, rustc/cargo 1.99.1. No git remote; GitHub CI not run locally except the provisional workflow on `main` (fmt, workspace test, clippy). Oracle llama.cpp tag `b12128`, commit `a01dff908b9c9311d7af316bba3c918063749bd0`. Target GGUF: TheBloke TinyLlama-1.1B-Chat Q8_0 (`tinyllama-1.1b-chat-v1.0.Q8_0.gguf`, 1170781568 bytes). Not in git.

| Milestone | Result |
| --- | --- |
| M1 GGUF loader | Dump comparison passed on `v1-accept`. cargo-fuzz 10 min passed (see M1 below). |
| M2 Kernels | Measured on `v1-accept`. AVX2 merge blocker cleared on `main`. |
| M3 Tokenizer | Passed on `v1-accept`. |
| M4 Single-sequence forward | Passed on this host (see M4 below). |
| M5 KV cache | Implemented on `main`; acceptance numbers were truncated in the brief. Unit tests passed (see M5 below). |
| M6a Server skeleton | Passed on `v1-accept`. |

`main` includes wave 1 acceptance artifacts, A5 KV/scheduler, and A4 `ferrum-model` (6143d44).

## M1 fuzz (2026-11-05)

`cargo +nightly fuzz run gguf_parse -- -max_total_time=600` from `fuzz/` on this Windows host (nightly 1.101.0, cargo-fuzz 0.13.2). Required MSVC ASAN runtime `clang_rt.asan_dynamic-x86_64.dll` from VS BuildTools on `PATH` next to the target. Result: **29775 runs in 601 s**, exit 0, coverage 288 / features 395, corpus 31 inputs. No crash artifacts under `fuzz/artifacts/gguf_parse`. Panics: 0.

## M2 (from v1-accept, AVX2+FMA)

Five release criterion trials, kernel errors 0. Median GEMV 8.3392 GiB/s, GEMM 4.0771 GFLOP/s (small cached GEMV; see v1-accept notes).

## M3 (from v1-accept)

200-line corpus token ids vs pinned `llama-tokenize`, 0 mismatches.

## M6a (from v1-accept)

openai 3.46.0 non-stream and stream against `ferrum serve --mock` passed.

## M4 (2026-11-05)

`cargo test -p ferrum-accept --test m4_forward teacher_forced_logits_match_llama_cpp_q8_0 --release -- --ignored --exact` with `FERRUM_MODEL` and `FERRUM_LLAMA_CPP` set. Teacher forcing via `tests/accept/oracle/m4_logits.py` (pinned `llama.dll`, `llama_*_default_params`, f32 KV). Five prompts × 129 tokens. Q8_0 within max abs 2.0 and top-1 ≥ 0.99 per prompt. First measured run: prompt 0 token 0 max abs **0.261**; token 1 max abs **0.358**; wall ~156 s. Re-run on the post-fix tree (2026-11-07): `ok`, harness **164.75 s**, exit 0; the test does not print peak abs on success. Log: `ferrum-wt/a4-model/r1/m4-q8-rerun.log`. Scalar kernels; `GGML_ROPE_TYPE_NORMAL`; Q8_0 GEMV matches ggml (quantize activation to Q8_0, then q8·q8).

### M4 F16 (2026-11-06)

`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` has no F16 file. Weights used: `C:\Users\Justi\Downloads\ferrum-oracle\tinyllama-1.1b-chat-v1.0.f16.gguf` (2201017280 bytes; `token_embd.weight` F16), fetched from `pbatra/TinyLlama-1.1B-Chat-v1.0-GGUF`.

Exclusive single-process release run (`CARGO_TARGET_DIR=ferrum-wt/a4-model/target`, `FERRUM_LLAMA_CPP=ferrum-oracle/llama-b12128`, `llama_*_default_params`, f32 KV). Tolerance **0.1** abs and top-1 ≥ 0.99. **Pass:** `test result: ok. 1 passed`, harness **1983.78 s**, wrapper wall **1984.1 s**, exit 0. Peak abs not printed on success. Log: `ferrum-wt/a4-model/r1/m4-f16-test.log`. An earlier harness line of 1375.72 s (`ferrum-wt/a3-tokenizer/m4_f16_run.log`) is a separate pass and is not this measurement.

## M5 (2026-11-05; re-run 2026-11-06 after H1)

No numeric acceptance in the brief (cut off after "pre-allocated"). `cargo test -p ferrum-kvcache -p ferrum-scheduler --release` (2026-11-06): kvcache **3** passed (append/commit, layer order, dense pre-alloc); scheduler **7** passed (greedy/empty reject, prefill→decode finish, decode-before-prefill + cancel, one decode token per active within budget, prefill samples first token without re-forwarding prompt tail, `max_new_tokens==0` finishes without emit, admit stops when pool has no slots). Not a substitute for the missing acceptance text.

## Review

R1 thermos on AVX2 fix merged (`0bbd7bb`). R1 re-audit of the uncommitted A4/A5 tree (2026-11-07): no blocking findings. H1 closed (prefill samples the last logits row; decode does not re-forward the prompt tail). Quality re-audit clear. M1 remains open: a later decode error in the same step can leave earlier sequences advanced and drop their events (`ferrum-wt/a4-model/r1/signoff-thermo.md`). M5 still has no numeric acceptance text in the brief — unit tests only; do not invent pass/fail thresholds.
