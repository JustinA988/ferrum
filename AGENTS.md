# Ferrum

CPU Llama inference in Rust. No PyTorch and no llama.cpp at runtime.

Revision 1 loads a GGUF file, runs a scalar or AVX2 forward pass, and serves a mock OpenAI API. `ferrum serve` without `--mock` exits.

## Constraints

- Rust 2021, stable toolchain. HTTP is actix-web. Kernel parallelism is rayon.
- Inference runs on dedicated OS threads, never on the actix or tokio runtime.
- `unsafe` is allowed only in `crates/kernels` and the mmap layer of `crates/gguf`. Every block has a `// SAFETY:` comment stating the invariant.
- Primary model: TinyLlama-1.1B-Chat (SentencePiece, 4 KV heads). Compute formats: F32, F16, Q8_0.
- Revision 1 does not select AVX-512.
- Numbers in `docs/STATUS.md` are release measurements only.
- Released under the MIT license. See `LICENSE`.
