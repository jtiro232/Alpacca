#!/usr/bin/env python3
"""Create a tiny random-weight llama-architecture GGUF using Alpacca's own
GGUF writer - no third-party packages needed.

The model is gibberish but loads and generates, which is what the tests
need. usage: python3 tests/make_tiny_model.py out.gguf [dtype]
(dtype: F32 (default), F16, Q8_0, Q4_0, or raw Q4_1/Q5_0/Q5_1/Q2_K/Q4_K/
Q5_K/Q6_K - quantized variants exercise the dequantizers and quantized
matvec loader paths. Raw classic variants keep their norm vectors F32.)
"""
from __future__ import annotations

import random
import struct
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
    n_embd = 256 if dtype in ("Q2_K", "Q4_K", "Q5_K", "Q6_K") else N_EMBD
    n_ff = 256 if dtype in ("Q2_K", "Q4_K", "Q5_K", "Q6_K") else N_FF

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
    w.add("llama.embedding_length", gguf.T_UINT32, n_embd)
    w.add("llama.block_count", gguf.T_UINT32, N_LAYER)
    w.add("llama.feed_forward_length", gguf.T_UINT32, n_ff)
    w.add("llama.attention.head_count", gguf.T_UINT32, N_HEAD)
    w.add("llama.attention.head_count_kv", gguf.T_UINT32, N_HEAD)
    w.add("llama.attention.layer_norm_rms_epsilon", gguf.T_FLOAT32, 1e-5)
    w.add("llama.rope.dimension_count", gguf.T_UINT32, n_embd // N_HEAD)
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

    def raw_q2_k(n: int) -> bytes:
        if n % 256:
            raise ValueError("Q2_K needs a multiple of 256 values")
        out = bytearray()
        for block in range(n // 256):
            scales = bytes(rng.randrange(256) for _ in range(16))
            qs = bytes(rng.randrange(256) for _ in range(64))
            d = 0.00390625 + (block % 7) * 0.00048828125
            dmin = 0.001953125 + (block % 5) * 0.000244140625
            out += scales + qs + struct.pack("<ee", d, dmin)
        return bytes(out)

    def raw_q4_k(n: int) -> bytes:
        if n % 256:
            raise ValueError("Q4_K needs a multiple of 256 values")
        out = bytearray()
        for block in range(n // 256):
            d = 0.015625 + (block % 7) * 0.001953125
            dmin = 0.00390625 + (block % 5) * 0.0009765625
            scales = bytes(rng.randrange(256) for _ in range(12))
            qs = bytes(rng.randrange(256) for _ in range(128))
            out += struct.pack("<ee", d, dmin) + scales + qs
        return bytes(out)

    def raw_q5_k(n: int) -> bytes:
        if n % 256:
            raise ValueError("Q5_K needs a multiple of 256 values")
        out = bytearray()
        for block in range(n // 256):
            d = 0.015625 + (block % 7) * 0.001953125
            dmin = 0.00390625 + (block % 5) * 0.0009765625
            scales = bytes(rng.randrange(256) for _ in range(12))
            qh = bytes(rng.randrange(256) for _ in range(32))
            ql = bytes(rng.randrange(256) for _ in range(128))
            out += struct.pack("<ee", d, dmin) + scales + qh + ql
        return bytes(out)

    def raw_q6_k(n: int) -> bytes:
        if n % 256:
            raise ValueError("Q6_K needs a multiple of 256 values")
        out = bytearray()
        for block in range(n // 256):
            ql = bytes(rng.randrange(256) for _ in range(128))
            qh = bytes(rng.randrange(256) for _ in range(64))
            scales = [rng.randrange(-32, 32) for _ in range(16)]
            d = 0.001953125 + (block % 5) * 0.000244140625
            out += ql + qh + struct.pack("<16b", *scales) + struct.pack("<e", d)
        return bytes(out)

    def raw_q4_1(n: int) -> bytes:
        if n % 32:
            raise ValueError("Q4_1 needs a multiple of 32 values")
        out = bytearray()
        for block in range(n // 32):
            d = 0.0078125 + (block % 7) * 0.0009765625
            m = -0.0625 + (block % 5) * 0.03125
            qs = bytes(rng.randrange(256) for _ in range(16))
            out += struct.pack("<ee", d, m) + qs
        return bytes(out)

    def raw_q5_0(n: int) -> bytes:
        if n % 32:
            raise ValueError("Q5_0 needs a multiple of 32 values")
        out = bytearray()
        for block in range(n // 32):
            d = 0.0078125 + (block % 7) * 0.0009765625
            qh = bytes(rng.randrange(256) for _ in range(4))
            qs = bytes(rng.randrange(256) for _ in range(16))
            out += struct.pack("<e", d) + qh + qs
        return bytes(out)

    def raw_q5_1(n: int) -> bytes:
        if n % 32:
            raise ValueError("Q5_1 needs a multiple of 32 values")
        out = bytearray()
        for block in range(n // 32):
            d = 0.0078125 + (block % 7) * 0.0009765625
            m = -0.0625 + (block % 5) * 0.03125
            qh = bytes(rng.randrange(256) for _ in range(4))
            qs = bytes(rng.randrange(256) for _ in range(16))
            out += struct.pack("<ee", d, m) + qh + qs
        return bytes(out)

    raw_makers = {"Q2_K": raw_q2_k, "Q4_K": raw_q4_k, "Q5_K": raw_q5_k,
                  "Q6_K": raw_q6_k, "Q4_1": raw_q4_1, "Q5_0": raw_q5_0,
                  "Q5_1": raw_q5_1}

    def add_weight(name: str, shape: tuple[int, ...], values: list[float],
                   tdtype: str) -> None:
        maker = raw_makers.get(tdtype)
        if maker is not None:
            if tdtype in ("Q4_1", "Q5_0", "Q5_1") and len(shape) == 1:
                w.add_tensor(name, shape, values, "F32")  # keep norms sane
                return
            n = 1
            for dim in shape:
                n *= dim
            w.add_raw_tensor(name, shape, tdtype, maker(n))
        else:
            w.add_tensor(name, shape, values, tdtype)

    # note: GGUF shape order is (cols, rows) - shape[0] is the input dim
    add_weight("token_embd.weight", (n_embd, n_vocab),
               rand(n_vocab * n_embd), dtype)
    for i in range(N_LAYER):
        p = f"blk.{i}."
        add_weight(p + "attn_norm.weight", (n_embd,), ones(n_embd), dtype)
        add_weight(p + "attn_q.weight", (n_embd, n_embd),
                   rand(n_embd * n_embd), dtype)
        add_weight(p + "attn_k.weight", (n_embd, n_embd),
                   rand(n_embd * n_embd), dtype)
        add_weight(p + "attn_v.weight", (n_embd, n_embd),
                   rand(n_embd * n_embd), dtype)
        add_weight(p + "attn_output.weight", (n_embd, n_embd),
                   rand(n_embd * n_embd), dtype)
        add_weight(p + "ffn_norm.weight", (n_embd,), ones(n_embd), dtype)
        add_weight(p + "ffn_gate.weight", (n_embd, n_ff),
                   rand(n_embd * n_ff), dtype)
        add_weight(p + "ffn_up.weight", (n_embd, n_ff),
                   rand(n_embd * n_ff), dtype)
        add_weight(p + "ffn_down.weight", (n_ff, n_embd),
                   rand(n_ff * n_embd), dtype)
    add_weight("output_norm.weight", (n_embd,), ones(n_embd), dtype)
    add_weight("output.weight", (n_embd, n_vocab),
               rand(n_vocab * n_embd), dtype)

    w.write()
    print(f"wrote {path} (vocab={n_vocab}, dtype={dtype})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny.gguf",
         sys.argv[2] if len(sys.argv) > 2 else "F32")
