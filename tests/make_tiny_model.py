#!/usr/bin/env python3
"""Create a tiny random-weight llama-architecture GGUF using Alpacca's own
GGUF writer - no third-party packages needed.

The model is gibberish but loads and generates, which is what the tests
need. usage: python3 tests/make_tiny_model.py out.gguf [dtype]
(dtype: F32 (default), F16, Q8_0 or Q4_0 - quantized variants exercise the
dequantizers.)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpacca import gguf  # noqa: E402

N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
N_FF = 128
N_CTX = 256
N_EXTRA = 48  # normal word-piece tokens on top of specials + bytes


def main(path: str, dtype: str = "F32") -> None:
    rng = random.Random(42)

    tokens: list[str] = []
    scores: list[float] = []
    types: list[int] = []

    def add(text: str, score: float, ttype: int) -> None:
        tokens.append(text)
        scores.append(score)
        types.append(ttype)

    add("<unk>", 0.0, 2)
    add("<s>", 0.0, 3)
    add("</s>", 0.0, 3)
    for b in range(256):
        add(f"<0x{b:02X}>", -1000.0, 6)
    # a tiny "vocabulary" so the SPM tokenizer has real pieces to work with
    words = ["\u2581the", "\u2581a", "\u2581and", "\u2581to", "\u2581of", "\u2581in", "\u2581is", "\u2581it",
             "\u2581hello", "\u2581world", "\u2581test", "\u2581ok", "he", "llo", "wor", "ld",
             "ing", "ed", "er", "es", "\u2581s", "\u2581b", "an", "at", "on", "or",
             "\u2581c", "\u2581d", "\u2581f", "\u2581g", "\u2581h", "\u2581l", "\u2581m", "\u2581n", "\u2581p", "\u2581r",
             "\u2581t", "\u2581w", "th", "en", "re", "nd", "st", "ar", "ou", "le",
             "\u2581I", "\u2581you"]
    for i, w in enumerate(words[:N_EXTRA]):
        add(w, -float(i + 1), 1)

    n_vocab = len(tokens)
    w = gguf.GGUFWriter(path, "llama")
    w.add("general.name", gguf.T_STRING, "alpacca-tiny-test")
    w.add("llama.context_length", gguf.T_UINT32, N_CTX)
    w.add("llama.embedding_length", gguf.T_UINT32, N_EMBD)
    w.add("llama.block_count", gguf.T_UINT32, N_LAYER)
    w.add("llama.feed_forward_length", gguf.T_UINT32, N_FF)
    w.add("llama.attention.head_count", gguf.T_UINT32, N_HEAD)
    w.add("llama.attention.head_count_kv", gguf.T_UINT32, N_HEAD)
    w.add("llama.attention.layer_norm_rms_epsilon", gguf.T_FLOAT32, 1e-5)
    w.add("llama.rope.dimension_count", gguf.T_UINT32, N_EMBD // N_HEAD)
    w.add("llama.vocab_size", gguf.T_UINT32, n_vocab)

    w.add("tokenizer.ggml.model", gguf.T_STRING, "llama")
    w.add_array("tokenizer.ggml.tokens", gguf.T_STRING, tokens)
    w.add_array("tokenizer.ggml.scores", gguf.T_FLOAT32, scores)
    w.add_array("tokenizer.ggml.token_type", gguf.T_INT32, types)
    w.add("tokenizer.ggml.bos_token_id", gguf.T_UINT32, 1)
    w.add("tokenizer.ggml.eos_token_id", gguf.T_UINT32, 2)
    w.add("tokenizer.ggml.unknown_token_id", gguf.T_UINT32, 0)
    w.add("tokenizer.ggml.add_bos_token", gguf.T_BOOL, True)

    def rand(n: int) -> list[float]:
        return [rng.gauss(0.0, 0.05) for _ in range(n)]

    def ones(n: int) -> list[float]:
        return [1.0] * n

    # note: GGUF shape order is (cols, rows) - shape[0] is the input dim
    w.add_tensor("token_embd.weight", (N_EMBD, n_vocab), rand(n_vocab * N_EMBD), dtype)
    for i in range(N_LAYER):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", (N_EMBD,), ones(N_EMBD))
        w.add_tensor(p + "attn_q.weight", (N_EMBD, N_EMBD), rand(N_EMBD * N_EMBD), dtype)
        w.add_tensor(p + "attn_k.weight", (N_EMBD, N_EMBD), rand(N_EMBD * N_EMBD), dtype)
        w.add_tensor(p + "attn_v.weight", (N_EMBD, N_EMBD), rand(N_EMBD * N_EMBD), dtype)
        w.add_tensor(p + "attn_output.weight", (N_EMBD, N_EMBD), rand(N_EMBD * N_EMBD), dtype)
        w.add_tensor(p + "ffn_norm.weight", (N_EMBD,), ones(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", (N_EMBD, N_FF), rand(N_EMBD * N_FF), dtype)
        w.add_tensor(p + "ffn_up.weight", (N_EMBD, N_FF), rand(N_EMBD * N_FF), dtype)
        w.add_tensor(p + "ffn_down.weight", (N_FF, N_EMBD), rand(N_FF * N_EMBD), dtype)
    w.add_tensor("output_norm.weight", (N_EMBD,), ones(N_EMBD))
    w.add_tensor("output.weight", (N_EMBD, n_vocab), rand(n_vocab * N_EMBD), dtype)

    w.write()
    print(f"wrote {path} (vocab={n_vocab}, dtype={dtype})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny.gguf",
         sys.argv[2] if len(sys.argv) > 2 else "F32")
