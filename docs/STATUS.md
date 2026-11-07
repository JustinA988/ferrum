# Status

Measured on a 12th Gen Intel Core i7-12650H with AVX2+FMA, rustc/cargo 1.99.1.

Oracle: llama.cpp tag `b12128`, commit `a01dff908b9c9311d7af316bba3c918063749bd0`.

Weights are not in git.

| Weights | Source | Bytes |
| --- | --- | --- |
| TinyLlama-1.1B-Chat Q8_0 | TheBloke | 1170781568 |
| TinyLlama-1.1B-Chat F16 | pbatra | 2201017280 |

| Check | Result |
| --- | --- |
| GGUF load | Dump name, shape, and dtype match. |
| GGUF fuzz | 29775 runs in 601 s, exit 0, panics 0. nightly 1.101.0, cargo-fuzz 0.13.2. |
| Kernels | Five release trials, 0 errors. Median GEMV 8.3392 GiB/s, GEMM 4.0771 GFLOP/s. |
| Tokenizer | 200-line corpus, 0 mismatches against `llama-tokenize`. |
| Q8_0 logits | Five prompts, 129 tokens each. Max abs ≤ 2.0 and top-1 ≥ 0.99. Prompt 0 token 0 max abs 0.261, token 1 max abs 0.358. Re-run 2026-11-07: pass, 164.75 s. |
| F16 logits | Max abs ≤ 0.1 and top-1 ≥ 0.99. Pass, harness 1983.78 s, wall 1984.1 s, exit 0. |
| KV cache and scheduler | Release tests 2026-11-07: kvcache 4 passed, scheduler 8 passed. |
| Mock HTTP | OpenAI client non-stream and stream against `ferrum serve --mock` passed. |
