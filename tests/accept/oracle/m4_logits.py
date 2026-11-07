#!/usr/bin/env python3
"""Offline oracle for teacher-forced logits.

Loads the pinned llama.cpp shared library, runs the token ids from stdin, and
prints one logit row per position. The Rust acceptance test compares those
rows with its own forward pass. This script is not part of the server.

Stdin JSON: {"llama_dir": "...", "model": "...", "tokens": [u32, ...]}
Stdout JSON: {"logits": [[float, ...], ...]}
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32


class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int),
        ("load_mode", ctypes.c_int),
        ("lazy_mode", ctypes.c_int),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("moe_cache_size", ctypes.c_size_t),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


def _load_dlls(llama_dir: Path) -> tuple[ctypes.WinDLL, ctypes.WinDLL, ctypes.WinDLL]:
    os.add_dll_directory(str(llama_dir))
    for name in llama_dir.glob("*.dll"):
        try:
            ctypes.WinDLL(str(name))
        except OSError:
            pass
    ggml = ctypes.WinDLL(str(llama_dir / "ggml.dll"))
    llama = ctypes.WinDLL(str(llama_dir / "llama.dll"))
    return ggml, llama, llama


def teacher_forced_logits(llama_dir: Path, model_path: Path, tokens: list[int]) -> list[list[float]]:
    if not tokens:
        return []

    ggml, llama, _ = _load_dlls(llama_dir)

    ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    ggml.ggml_backend_load_all_from_path.restype = None
    ggml.ggml_backend_load_all_from_path(str(llama_dir).encode("utf-8"))

    llama.llama_backend_init.argtypes = []
    llama.llama_backend_init.restype = None
    llama.llama_backend_free.argtypes = []
    llama.llama_backend_free.restype = None

    llama.llama_model_default_params.argtypes = []
    llama.llama_model_default_params.restype = llama_model_params
    llama.llama_context_default_params.argtypes = []
    llama.llama_context_default_params.restype = llama_context_params

    llama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
    llama.llama_model_load_from_file.restype = ctypes.c_void_p
    llama.llama_model_free.argtypes = [ctypes.c_void_p]
    llama.llama_model_free.restype = None

    llama.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
    llama.llama_init_from_model.restype = ctypes.c_void_p
    llama.llama_free.argtypes = [ctypes.c_void_p]
    llama.llama_free.restype = None

    llama.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    llama.llama_model_get_vocab.restype = ctypes.c_void_p
    llama.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    llama.llama_vocab_n_tokens.restype = ctypes.c_int32

    llama.llama_n_vocab.argtypes = [ctypes.c_void_p]
    llama.llama_n_vocab.restype = ctypes.c_int32

    llama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    llama.llama_batch_init.restype = llama_batch
    llama.llama_batch_free.argtypes = [llama_batch]
    llama.llama_batch_free.restype = None

    llama.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    llama.llama_decode.restype = ctypes.c_int32

    llama.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    llama.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    llama.llama_get_memory.argtypes = [ctypes.c_void_p]
    llama.llama_get_memory.restype = ctypes.c_void_p
    llama.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    llama.llama_memory_clear.restype = None

    llama.llama_backend_init()
    try:
        # Must use default_params — a zeroed ctypes Structure leaves
        # attention_type/flash/type_k at 0 and changes decode vs llama.cpp CLI.
        mp = llama.llama_model_default_params()
        mp.n_gpu_layers = 0
        model = llama.llama_model_load_from_file(str(model_path).encode("utf-8"), mp)
        if not model:
            raise RuntimeError(f"failed to load model {model_path}")

        cp = llama.llama_context_default_params()
        cp.n_ctx = max(len(tokens) + 16, 512)
        cp.n_batch = max(len(tokens), 512)
        cp.n_ubatch = cp.n_batch
        # Match Ferrum's f32 KV for a fair teacher-forced comparison.
        cp.type_k = 0  # GGML_TYPE_F32
        cp.type_v = 0
        ctx = llama.llama_init_from_model(model, cp)
        if not ctx:
            llama.llama_model_free(model)
            raise RuntimeError("failed to create llama context")

        vocab_obj = llama.llama_model_get_vocab(model)
        vocab = int(llama.llama_vocab_n_tokens(vocab_obj))
        if vocab <= 0:
            raise RuntimeError("llama_vocab_n_tokens returned non-positive value")
        mem = llama.llama_get_memory(ctx)
        if mem:
            llama.llama_memory_clear(mem, True)

        batch = llama.llama_batch_init(1, 0, 1)
        rows: list[list[float]] = []
        try:
            for pos, tok in enumerate(tokens):
                batch.n_tokens = 1
                batch.token[0] = llama_token(tok)
                batch.pos[0] = llama_pos(pos)
                batch.n_seq_id[0] = 1
                batch.seq_id[0][0] = llama_seq_id(0)
                batch.logits[0] = 1
                if llama.llama_decode(ctx, batch) != 0:
                    raise RuntimeError(f"llama_decode failed at position {pos}")
                ptr = llama.llama_get_logits_ith(ctx, 0)
                if not ptr:
                    raise RuntimeError(f"no logits at position {pos}")
                rows.append([float(ptr[i]) for i in range(vocab)])
        finally:
            llama.llama_batch_free(batch)

        llama.llama_free(ctx)
        llama.llama_model_free(model)
        return rows
    finally:
        llama.llama_backend_free()


def main() -> int:
    payload = json.load(sys.stdin)
    llama_dir = Path(payload["llama_dir"])
    model = Path(payload["model"])
    tokens = [int(t) for t in payload["tokens"]]
    logits = teacher_forced_logits(llama_dir, model, tokens)
    json.dump({"logits": logits}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
