#!/usr/bin/env python3
"""Create a synthetic llama-architecture GGUF for offline benchmarking.

The default shape matches the public stories15M checkpoint (vocab 32000,
embd 288, 6 layers, 6 heads, ff 768, tied embeddings); shape flags let you
mimic larger models, e.g. a TinyLlama-1.1B-like GQA shape:

    python3 tests/make_bench_model.py /tmp/b1.gguf Q4_0 \\
        --embd 2048 --ff 5632 --layers 22 --heads 32 --kv 4 --untied

Weights are deterministic random values, so prefill/decode cost and
weight-memory behaviour match a real model of the same shape without
needing network access. The output is gibberish; only use it for
performance measurements.

usage: python3 tests/make_bench_model.py out.gguf [F32|Q8_0|Q4_0] [shape flags]

NumPy is used to quantize quickly when available; the pure fallback works
but takes minutes per 15M parameters.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpacca import gguf, quants  # noqa: E402

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

N_VOCAB = 32000
N_EMBD = 288
N_HEAD = 6
N_LAYER = 6
N_FF = 768
N_CTX = 2048


def _np_quantize_q8_0(vals) -> bytes:
    v = vals.reshape(-1, 32)
    amax = _np.abs(v).max(axis=1)
    d = (amax / 127.0).astype(_np.float32)
    inv = _np.where(d > 0, 1.0 / _np.where(d > 0, d, 1), 0.0)
    qs = _np.clip(_np.round(v * inv[:, None]), -128, 127).astype(_np.int8)
    out = _np.empty((len(v), 34), dtype=_np.uint8)
    out[:, 0:2] = d.astype("<f2").view(_np.uint8).reshape(-1, 2)
    out[:, 2:] = qs.view(_np.uint8)
    return out.tobytes()


def _np_quantize_q4_0(vals) -> bytes:
    v = vals.reshape(-1, 32)
    vmax = v[_np.arange(len(v)), _np.abs(v).argmax(axis=1)]
    d = (vmax / -8.0).astype(_np.float32)
    inv = _np.where(d != 0, 1.0 / _np.where(d != 0, d, 1), 0.0)
    q = _np.clip(_np.floor(v * inv[:, None] + 8.5), 0, 15).astype(_np.uint8)
    out = _np.empty((len(v), 18), dtype=_np.uint8)
    out[:, 0:2] = d.astype("<f2").view(_np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.tobytes()


def main(path: str, dtype: str = "Q4_0", *, n_vocab: int = N_VOCAB,
         n_embd: int = N_EMBD, n_head: int = N_HEAD, n_kv: int = 0,
         n_layer: int = N_LAYER, n_ff: int = N_FF, n_ctx: int = N_CTX,
         tied: bool = True) -> None:
    if dtype not in ("F32", "Q8_0", "Q4_0"):
        raise SystemExit(f"unsupported bench dtype {dtype}")
    n_kv = n_kv or n_head
    head_dim = n_embd // n_head
    kv_dim = n_kv * head_dim

    tokens: list[str] = ["<unk>", "<s>", "</s>"]
    scores: list[float] = [0.0, 0.0, 0.0]
    types: list[int] = [2, 3, 3]
    for b in range(256):
        tokens.append(f"<0x{b:02X}>")
        scores.append(-1000.0)
        types.append(6)
    for i in range(n_vocab - len(tokens)):
        tokens.append(f"▁w{i:05d}")
        scores.append(-float(i + 1))
        types.append(1)

    w = gguf.GGUFWriter(path, "llama")
    w.add("general.name", gguf.T_STRING, "alpacca-bench-synthetic")
    w.add("llama.context_length", gguf.T_UINT32, n_ctx)
    w.add("llama.embedding_length", gguf.T_UINT32, n_embd)
    w.add("llama.block_count", gguf.T_UINT32, n_layer)
    w.add("llama.feed_forward_length", gguf.T_UINT32, n_ff)
    w.add("llama.attention.head_count", gguf.T_UINT32, n_head)
    w.add("llama.attention.head_count_kv", gguf.T_UINT32, n_kv)
    w.add("llama.attention.layer_norm_rms_epsilon", gguf.T_FLOAT32, 1e-5)
    w.add("llama.rope.dimension_count", gguf.T_UINT32, head_dim)
    w.add("llama.vocab_size", gguf.T_UINT32, n_vocab)
    w.add("tokenizer.ggml.model", gguf.T_STRING, "llama")
    w.add_array("tokenizer.ggml.tokens", gguf.T_STRING, tokens)
    w.add_array("tokenizer.ggml.scores", gguf.T_FLOAT32, scores)
    w.add_array("tokenizer.ggml.token_type", gguf.T_INT32, types)
    w.add("tokenizer.ggml.bos_token_id", gguf.T_UINT32, 1)
    w.add("tokenizer.ggml.eos_token_id", gguf.T_UINT32, 2)
    w.add("tokenizer.ggml.unknown_token_id", gguf.T_UINT32, 0)
    w.add("tokenizer.ggml.add_bos_token", gguf.T_BOOL, True)

    if _np is not None:
        rng = _np.random.default_rng(42)

        def add_matrix(name: str, shape: tuple[int, ...]) -> None:
            n = 1
            for dim in shape:
                n *= dim
            vals = (rng.standard_normal(n) * 0.02).astype(_np.float32)
            if dtype == "F32":
                w.add_raw_tensor(name, shape, "F32", vals.astype("<f4").tobytes())
            elif dtype == "Q8_0":
                w.add_raw_tensor(name, shape, "Q8_0", _np_quantize_q8_0(vals))
            else:
                w.add_raw_tensor(name, shape, "Q4_0", _np_quantize_q4_0(vals))
    else:
        import random
        prng = random.Random(42)

        def add_matrix(name: str, shape: tuple[int, ...]) -> None:
            n = 1
            for dim in shape:
                n *= dim
            vals = [prng.gauss(0.0, 0.02) for _ in range(n)]
            if dtype == "F32":
                w.add_tensor(name, shape, vals, "F32")
            elif dtype == "Q8_0":
                w.add_raw_tensor(name, shape, "Q8_0", quants.quantize_q8_0(vals))
            else:
                w.add_raw_tensor(name, shape, "Q4_0", quants.quantize_q4_0(vals))

    def add_norm(name: str) -> None:
        w.add_tensor(name, (n_embd,), [1.0] * n_embd, "F32")

    # tied embeddings by default (no output.weight), like stories15M;
    # --untied adds a separate output matrix, like llama-3-class models
    add_matrix("token_embd.weight", (n_embd, n_vocab))
    for i in range(n_layer):
        p = f"blk.{i}."
        add_norm(p + "attn_norm.weight")
        add_matrix(p + "attn_q.weight", (n_embd, n_embd))
        add_matrix(p + "attn_k.weight", (n_embd, kv_dim))
        add_matrix(p + "attn_v.weight", (n_embd, kv_dim))
        add_matrix(p + "attn_output.weight", (n_embd, n_embd))
        add_norm(p + "ffn_norm.weight")
        add_matrix(p + "ffn_gate.weight", (n_embd, n_ff))
        add_matrix(p + "ffn_up.weight", (n_embd, n_ff))
        add_matrix(p + "ffn_down.weight", (n_ff, n_embd))
    add_norm("output_norm.weight")
    if not tied:
        add_matrix("output.weight", (n_embd, n_vocab))

    w.write()
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"wrote {path} ({dtype}, embd {n_embd} ff {n_ff} layers {n_layer} "
          f"heads {n_head}/{n_kv} vocab {n_vocab} "
          f"{'tied' if tied else 'untied'}, {size_mb:.1f} MiB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("dtype", nargs="?", default="Q4_0")
    ap.add_argument("--vocab", type=int, default=N_VOCAB)
    ap.add_argument("--embd", type=int, default=N_EMBD)
    ap.add_argument("--heads", type=int, default=N_HEAD)
    ap.add_argument("--kv", type=int, default=0, help="kv heads (default: --heads)")
    ap.add_argument("--layers", type=int, default=N_LAYER)
    ap.add_argument("--ff", type=int, default=N_FF)
    ap.add_argument("--ctx", type=int, default=N_CTX)
    ap.add_argument("--untied", action="store_true",
                    help="write a separate output.weight matrix")
    args = ap.parse_args()
    main(args.out, args.dtype, n_vocab=args.vocab, n_embd=args.embd,
         n_head=args.heads, n_kv=args.kv, n_layer=args.layers, n_ff=args.ff,
         n_ctx=args.ctx, tied=not args.untied)
