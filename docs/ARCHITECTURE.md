# Architecture

Interface revision **1**. Frozen on 2026-11-04.

Public items in `ferrum-core-types` are the contract. Crate entry points named in this document are also frozen: `ferrum_gguf::load`, `ferrum_kernels::scalar`, `ferrum_kernels::select`, `ferrum_tokenizer::from_metadata`, and the mock helpers in `ferrum_core_types`. A change to any of them is an orchestrator change and needs Justin's written agreement.

## System

`ferrum` loads a Llama-architecture GGUF with `mmap`, runs a scalar or SIMD forward pass on CPU, and serves OpenAI-compatible `/v1/completions` with continuous batching.

```
ferrum-cli
  └─ ferrum-server          actix-web, dedicated inference threads
       └─ ferrum-scheduler  continuous batching
            └─ ferrum-model Llama forward
                 ├─ ferrum-gguf        mmap weights
                 ├─ ferrum-kernels     scalar, then AVX2+FMA
                 ├─ ferrum-tokenizer   SentencePiece from GGUF metadata
                 └─ ferrum-kvcache     pre-allocated slot pool
```

`ferrum-core-types` is the only shared language. Higher crates depend downward. The model is generic over `GgufModel` and holds `Box<dyn Kernels>`, so it does not name the kernel crate. The scheduler is generic over `CausalModel`. The CLI is the composition root.

## Fixed stack

- Rust edition 2021, stable toolchain.
- HTTP: actix-web 4 on its tokio runtime. `Cargo.lock` has actix-web 4.18.1, tokio 1.54.1, serde_json 1.0.154, memmap2 0.9.12, and rayon 1.12.0. Confirmed against the actix-web 4.11.0 docs, which 4.18 still has to compile against: `HttpResponse::Ok() -> HttpResponseBuilder`, `HttpResponseBuilder::streaming<S, E>(&mut self, stream: S) -> HttpResponse` where `S: Stream<Item = Result<Bytes, E>> + 'static` and `E: Into<BoxError>`, `web::Json<T>` as extractor (`T: Deserialize`) and responder (`T: Serialize`), and `App::new().app_data(...).service(...)`. If a 4.18 signature differs, stop and check Context7. Do not invent one.
- Kernel parallelism: rayon, inside the kernel crate, on the inference thread.
- Inference never runs on the actix or tokio worker. The handler sends the request to an OS thread (`std::thread`) and forwards results from `std::sync::mpsc`.
- mmap: `memmap2::Mmap::map`. Confirmed signature: `unsafe fn map<T: MmapAsRawDesc>(file: T) -> std::io::Result<Mmap>`. The map derefs to `[u8]`.
- Property tests: `proptest!` with `#![proptest_config(ProptestConfig::with_cases(N))]`.
- Benchmarks: `criterion_group!`, `criterion_main!`, `BenchmarkGroup::throughput(Throughput::Bytes(n))` or `Throughput::Elements(n)` before `bench_function`.

## GGUF

Normative layout is the upstream GGUF specification (file magic, version, little-endian unless a future version grows an endian flag). Revision 1 assumes little-endian and accepts versions 2 and 3. Any other magic or version is `FerrumError::Gguf`.

Header, in order:

- magic bytes `47 47 55 46` (`GGUF`)
- `u32` version
- `u64` tensor count
- `u64` metadata kv count
- that many key/value pairs
- that many tensor infos
- zero padding to `alignment`
- tensor bytes

Strings are `u64` byte length plus UTF-8, not null-terminated. Invalid UTF-8 is an error. Metadata value tags are `MetadataType` (0 through 12). A bool byte other than 0 or 1 is an error. Arrays store a type tag, a `u64` element count, then that many values. Nested arrays are legal.

Alignment comes from `general.alignment` when present (uint32 or uint64) and must be a non-zero multiple of 8. If the key is absent, alignment is 32. Padding uses the spec formula `offset + (alignment - (offset % alignment)) % alignment`. Each tensor `data_offset` is relative to the tensor data section, is a multiple of alignment, and `data_offset + nbytes` must fit in that section with a checked add.

The reader never allocates from a claimed count until that count is bounded by the remaining bytes. Overflow or a short buffer returns `FerrumError::Gguf`. It does not panic. Rank must be 1 to 4. A zero dimension is an error.

Metadata is owned (`MetadataValue`). Tensor payloads are borrowed from the map (`tensor_bytes -> &[u8]`). `GgufFile` keeps the `File` and the `Mmap` together and is `Send + Sync`.

`// SAFETY:` for the map: the file was opened read-only, ferrum does not write it, and no other writer truncates or replaces it for the life of the map. Dropping the map is the end of that use.

Compute dtypes are F32 (4 bytes), F16 (2 bytes), and Q8_0. Q8_0 is a packed block of 32 values: little-endian binary16 scale, then 32 `i8` values, 34 bytes, no padding. Dequant is `scale_f32 * i8`. The scale conversion is `f16_to_f32` and must be bitwise identical on every backend. Other ggml type ids are stored as `DType::Other` so a dump can name them. `tensor_nbytes` rejects them. The loader still bounds-checks their declared byte length.

`GgufFile::dump` equals `format_gguf_dump`: one header line, then one tensor per line in file order, trailing newline.

```
ferrum-gguf version=3 alignment=32 metadata=12 tensors=201
token_embd.weight [2048, 32000] q8_0
```

Shape order is GGUF order (`shape[0]` contiguous). Dtype names come from `ggml_type_name`. K-quants use a capital `K`.

## Llama weights

`llama_config` requires `general.architecture == "llama"`. Counts accept uint32 or uint64. Missing required keys are errors. Defaults:

| Key | Default |
| --- | --- |
| `llama.attention.head_count_kv` | `head_count` |
| `llama.attention.key_length` | `embedding_length / head_count` |
| `llama.rope.freq_base` | `10000` |
| `llama.rope.dimension_count` | `head_dim` |

`layer_norm_rms_epsilon` is required. `head_dim` is even, `n_embd == n_head * head_dim`, and `n_head % n_head_kv == 0`. `rope_dim` is even and `<= head_dim`. A present `value_length` must equal `head_dim`. Rope scaling other than absent or `"none"`, `use_parallel_residual = true`, and a non-zero `expert_count` are errors. Vocab length is `tokenizer.ggml.tokens`. BOS and EOS are required and in range.

Tensor names, GGUF dimension order:

| Name | Shape (`ne[0]`, `ne[1]`) |
| --- | --- |
| `token_embd.weight` | `n_embd`, `n_vocab` |
| `output.weight` | `n_embd`, `n_vocab` (optional; else tied to the embedding) |
| `output_norm.weight` | `n_embd` |
| `blk.{i}.attn_norm.weight` | `n_embd` |
| `blk.{i}.attn_q.weight` | `n_embd`, `n_heads * head_dim` |
| `blk.{i}.attn_k.weight` | `n_embd`, `n_kv_heads * head_dim` |
| `blk.{i}.attn_v.weight` | same as K |
| `blk.{i}.attn_output.weight` | `n_heads * head_dim`, `n_embd` |
| `blk.{i}.ffn_norm.weight` | `n_embd` |
| `blk.{i}.ffn_gate.weight` | `n_embd`, `n_ff` |
| `blk.{i}.ffn_up.weight` | `n_embd`, `n_ff` |
| `blk.{i}.ffn_down.weight` | `n_ff`, `n_embd` |

`llama_tensor(i, "attn_q")` builds `blk.{i}.attn_q.weight`. A bias tensor, or any unexpected weight, is a model error. Silent ignore is not allowed.

For a linear `y[m] = dot(W[m], x)`, GGUF stores K as `shape[0]` and M as `shape[1]`. Embedding lookup for token `t` is the contiguous `n_embd` values at index `t`.

## Forward pass

Reference algorithm, one token at a time. A fused GEMM prefill is allowed only when it matches this algorithm within the logit tolerance.

```
h = x + Attn(RMSNorm(x))
y = h + Down(SiLU(Gate(RMSNorm(h))) * Up(RMSNorm(h)))
```

The same normalized vector feeds gate and up. RMSNorm is

```
inv = 1 / sqrt(mean(x^2) + eps)
out[i] = x[i] * inv * weight[i]
```

with the mean over `x.len()`. SiLU is `x * sigmoid(x)`, sigmoid `1 / (1 + exp(-x))`.

RoPE matches llama.cpp `GGML_ROPE_TYPE_NORMAL`: consecutive pairs within each head, not NeoX half-split. `Kernels::rope` takes `RopeArgs`. `Kernels::gemm` takes `GemmArgs` (`n_tokens == 0` writes nothing). For pair index `i` in `0 .. rope_dim/2`:

```
freq = pos / theta ^ (2i / rope_dim)
q'[2i]     = q[2i] * cos(freq) - q[2i+1] * sin(freq)
q'[2i+1]   = q[2i] * sin(freq) + q[2i+1] * cos(freq)
```

Components from `rope_dim` to `head_dim` are unchanged. The same transform applies independently to each query head and each key head. `pos` is the absolute token position.

Attention is causal GQA. Query head `h` reads KV head `h / (n_heads / n_kv_heads)`. Score is `dot(q, k) / sqrt(head_dim)` over positions `0..=pos` only. Softmax is stable. A non-finite max writes zeros. Cached K is post-RoPE. Cached V is post-projection and not rotated. Both are f32, head-major `[n_kv_heads, head_dim]`.

`forward` checks `position0 == seq_len(slot)` and `logits_out.len() == token_ids.len() * n_vocab`. Logits are raw, row-major `[token][vocab]`. No final softmax.

`argmax` is greedy sampling. The lowest index wins a tie. NaN is skipped.

## Kernel tolerance

Scalar is the reference. AVX2+FMA must use unaligned loads.

| Op | SIMD versus scalar |
| --- | --- |
| `add`, `mul` | bitwise identical |
| F16 and Q8_0 dequant | bitwise identical (`f16_to_f32`) |
| `silu`, `scale` | abs diff `<= 1e-5 * max(1, abs(scalar))` and `<= 1e-6` floor on the abs term's companion: `max(1e-6, 1e-5 * max(1, abs(scalar)))` |
| `dot`, `rms_norm`, `softmax`, `rope`, `gemv`, `gemm` | `max(1e-6, 1e-4 * max(1, abs(scalar)))` when the reduced axis is `<= 4096`; `max(1e-6, 1e-3 * max(1, abs(scalar)))` above that |

Both backends return the same `Err` for a length mismatch, a non-positive `theta`, an odd `rope_dim`, or a Q8_0 K that is not a multiple of 32. F32 and F16 GEMV accept odd sizes. proptest covers those odd sizes. Empty `dot` is an error. `softmax` of one element is `1`. `gemm` with zero tokens writes nothing and returns `Ok`.

## KV and scheduling

`KvPool` allocates `max_slots * n_layers * max_seq_len * n_kv_heads * head_dim * 2` f32s in its constructor. `append_layer` and `commit_token` do not allocate. `append_layer` does not advance `seq_len`. `commit_token` advances it by one and errors if any layer is missing.

`ContinuousBatcher<M: CausalModel>`:

- Decode (one token per active sequence) is admitted before new prefill.
- A step's token budget is `max_batch_tokens`.
- Prefill may be chunked across steps.
- A step emits at most one new token per active sequence.
- `temperature != 0` is `invalid_request` on the real engine. `top_p` is stored and ignored by greedy.

The acceptance text for M5 was cut off after "pre-allocated". Do not invent pass/fail numbers for it.

## Tokenizer

`from_metadata` reads `tokenizer.ggml.tokens`, optional `scores`, optional `token_type`, BOS, and EOS. Token type codes are the `TOKEN_*` constants (1 normal through 6 byte). Revision 1 supports `tokenizer.ggml.model == "llama"` only.

Observable behavior, matching llama.cpp's llama SentencePiece path:

- `encode(text, add_bos)` prepends a metaspace (`▁`, U+2581) for non-empty user text, then BOS when `add_bos` is true.
- Byte fallback uses the `<0xHH>` pieces.
- `decode(encode(text, false)) == text` on the acceptance corpus, including the empty string.
- Token ids equal llama.cpp on that corpus.

V1 commits the 200 strings (unicode, whitespace, emoji, empty) before measuring. Changing the corpus is an interface change.

## HTTP

`ferrum serve --mock [--bind 127.0.0.1:8080]` serves the mock. Without `--mock`, the process exits with a message that the model engine is not in revision 1. Argument parsing is manual. No clap.

| Method and path | Result |
| --- | --- |
| `GET /health` | 200 `{"status":"ok"}` |
| `GET /v1/models` | 200 `ModelList::mock()` |
| `POST /v1/completions` | validated `CompletionRequestBody` |

Unknown JSON fields are ignored so the official OpenAI client can send them. JSON extract failures use actix `JsonConfig::error_handler` and still return the OpenAI envelope: `{"error":{"message","type","param","code"}}`. `code` is null. Status and `type` come from `FerrumError::http_status` and `api_error`.

Validation already implemented in `CompletionRequestBody::validate`:

- missing `model` or `prompt`: 400 `invalid_request_error`
- `n` present and not 1: 400, param `n`
- `max_tokens == 0`: 400, param `max_tokens`
- bad `temperature` or `top_p`: 400
- prompt string array, or empty token array: 400, param `prompt`
- unknown model id: 404 `not_found_error`, param `model`, message contains the id
- defaults: `max_tokens` 16, `temperature` 1, `top_p` 1, `stream` false, `seed` 0

The mock accepts the client's default temperature. It does not sample. A non-stream call on `ferrum-mock` returns `mock_completion`. A stream call writes two SSE events from `mock_stream_chunks`, then `data: [DONE]\n\n`. Each event is `data: {json}\n\n` with `Content-Type: text/event-stream`. `created` is 0 and the id is `cmpl-mock`.

`echo: true` uses `mock_completion_text`: `"hi"` becomes `"hi mock"`, tokens `[1, 2]` become `"1 2 mock"`, empty text stays `"mock"`.

Prompt token accounting on the mock is not real tokenization: a non-empty string counts as 1, a token array counts as its length, completion tokens are 1.

## Threads and errors

- Library paths return `FerrumError`. Panics are bugs.
- The HTTP boundary maps that error through `api_error`.
- Slot exhaustion is `FerrumError::unavailable` (503), not a panic.
- `unsafe` only in `crates/kernels` and the mmap layer of `crates/gguf`.

## Numeric acceptance for M4

Five prompts, each at least 32 tokens, committed by V1 before the run. Teacher-forced logits against llama.cpp:

- F16 weights, f32 accumulation: per prompt, top-1 agreement `>= 99%` and max abs logit diff `<= 0.1`.
- Q8_0: per prompt, top-1 agreement `>= 99%` and max abs logit diff `<= 2.0`.

A prompt under 100 tokens therefore needs a perfect top-1 to clear 99%. These bounds are the contract. Tightening them later is allowed. Loosening them is an interface change.

## Benchmarks

Correctness tests may run in debug. Anything reported in `docs/STATUS.md` is a release build (`cargo bench` or `cargo test --release`). GEMV throughput is `Throughput::Bytes` of the weight bytes read plus the f32 vector bytes, reported as GB/s. GEMM sets `Throughput::Elements(2 * rows * cols * n_tokens)` and reports that elements/s figure as FLOP/s (one multiply and one add).

Before a number is written into `STATUS.md`: name the limiter, count errors, run at least five alternating trials, report median and range, and show the timed work ran. A comparison that misses those is inconclusive and is not a result.

## Milestones

Wave 1, parallel, after this freeze:

| Id | Owner | Acceptance |
| --- | --- | --- |
| M1 | A1 | `load` of the target GGUF. Dump name, shape, and dtype match llama.cpp. `cargo-fuzz` on corrupt files for 10 minutes with zero panics. |
| M2 | A2 | Scalar for every `Kernels` method, then AVX2+FMA. proptest odd sizes inside the tolerance table. Criterion reports GEMV GB/s and GEMM FLOP/s. |
| M3 | A3 | Token ids identical to llama.cpp on the 200-string corpus. Decode round-trips. |
| M6a | A6 | `/v1/completions`, `/v1/models`, `/health`, SSE, OpenAI error shape, against the mock. The official OpenAI client completes a non-stream and a stream call. |

Wave 2:

| Id | Owner | Acceptance |
| --- | --- | --- |
| M4 | A4 | Uses M1 and the M2 scalar backend. Teacher-forced logits meet the M4 bounds on five prompts, for F16 and Q8_0. |
| M5 | A5 | Pre-allocated KV pool. Further acceptance text was not in the brief. |

Fuzz runs on Linux. It is not a Windows host gate. Oracle tests skip unless `FERRUM_MODEL` is set, and they are `#[ignore]` so CI stays green without the file. Sign-off is an explicit `--ignored` run recorded in `STATUS.md`.

## Oracle

llama.cpp is pinned when V1 builds it. The commit is written into `docs/STATUS.md` at that moment. The pin is not chosen yet. The oracle is offline: tests and benches only. The server does not link it.

`FERRUM_MODEL` is the TinyLlama-1.1B-Chat GGUF, F16 or Q8_0, SentencePiece, 4 KV heads. `FERRUM_LLAMA_CPP` is the pinned binary directory.

## Provisional CI

The brief's CI section was missing. Until it is restored, green means `.github/workflows/ci.yml`: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` on stable.

## Crate map

| Path | Package | unsafe |
| --- | --- | --- |
| `crates/core-types` | `ferrum-core-types` | forbidden |
| `crates/gguf` | `ferrum-gguf` | mmap only |
| `crates/kernels` | `ferrum-kernels` | SIMD only |
| `crates/tokenizer` | `ferrum-tokenizer` | forbidden |
| `crates/model` | `ferrum-model` | forbidden |
| `crates/kvcache` | `ferrum-kvcache` | forbidden |
| `crates/scheduler` | `ferrum-scheduler` | forbidden |
| `crates/server` | `ferrum-server` | forbidden |
| `crates/ferrum-cli` | `ferrum-cli` | forbidden |
| `tests/accept` | `ferrum-accept` | forbidden, owned by V1 |
