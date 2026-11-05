# Ferrum

From-scratch CPU LLM inference server in Rust. No PyTorch, no Python at runtime, no llama.cpp linking. A smaller system that is verified beats a larger one that is not.

Interface revision **1** is frozen in `crates/core-types` and `docs/ARCHITECTURE.md`. Only the orchestrator changes those signatures, and only after Justin agrees in writing.

## Mission

Load a GGUF model via mmap and serve OpenAI-compatible `/v1/completions` with continuous batching on CPU. Correct against a pinned llama.cpp used only offline, and fast enough to benchmark honestly against it.

## Non-goals

Until every milestone in `docs/ARCHITECTURE.md` is accepted: training, LoRA, vision, speculative decoding, multi-GPU, tensor parallelism, paged attention, a web UI, and any model family other than Llama-architecture.

## Fixed decisions

- Rust 2021 workspace, stable toolchain. HTTP is actix-web on tokio. Kernel parallelism is rayon. Inference runs on dedicated OS threads, never on the async runtime.
- Primary model: TinyLlama-1.1B-Chat GGUF (SentencePiece, GQA with 4 KV heads). Compute quant formats: F32, F16, Q8_0. Q4_0 and Llama-3.2-1B (BPE) are post-MVP proofs of generality.
- CPU order: scalar reference, then AVX2+FMA, then AVX-512 (optional) and NEON (`cfg(target_arch = "aarch64")`), chosen at runtime by feature detection. CUDA is last, behind a cargo feature. Revision 1 does not select AVX-512.
- `unsafe` is allowed only in `crates/kernels` and the mmap layer of `crates/gguf`. Every block has a `// SAFETY:` comment stating the invariant.

## Ownership

- Orchestrator: `crates/core-types`, `docs/ARCHITECTURE.md`, workspace membership, merge decisions.
- A1 `crates/gguf`. A2 `crates/kernels`. A3 `crates/tokenizer`. A4 `crates/model`.
- A5 `crates/kvcache` and `crates/scheduler`. A6 `crates/server` and `crates/ferrum-cli`.
- V1 owns `tests/`, `benches/`, and `fuzz/`. Tests are written from this file, the architecture doc, and the llama.cpp oracle. V1 does not treat the implementation as the spec.
- R1 audits `unsafe`, concurrency, and error handling, and runs thermos on a branch before merge.

An agent edits only its crates. An interface change is a request to the orchestrator, not a drive-by edit. Authors do not sign off on their own work.

A branch merges only when the author reports done, V1 has signed off with evidence, R1 has no open blocking findings, and CI is green.

## Measurement

`docs/STATUS.md` gets a number only after a run. Benchmark reports follow the checklist: name the limiter, run release builds, count errors, run at least five alternating trials, report median and range, and show that the timed work happened. A comparison without those is inconclusive.

## Spec gap

The milestone brief was cut off at `M5 KV cache (A5): pre-allocated,`. Acceptance text for the rest of M5 and the CI section were not in the brief. Do not invent those acceptance numbers. The provisional CI workflow is `.github/workflows/ci.yml`.

## Learned User Preferences

- Justin works directly in code and wants correctness and measured performance over feature count.

## Learned Workspace Facts

- The ferrum workspace is `C:\Users\Justi\Downloads\ferrum`, separate from the mechinterp lab repo.
- Interface revision 1 is frozen. Operating rules above this heading are the project contract.
