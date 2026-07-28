#!/usr/bin/env python3
"""Create a tiny random-weight llama-architecture GGUF using Alpacca's own
GGUF writer - no third-party packages needed.

The model is gibberish but loads and generates, which is what the tests
need. usage: python3 tests/make_tiny_model.py out.gguf [dtype] [--arch ARCH]
(dtype: F32 (default), F16, Q8_0, Q4_0, or raw Q4_1/Q5_0/Q5_1/Q2_K/Q4_K/
Q5_K/Q6_K - quantized variants exercise the dequantizers and quantized
matvec loader paths. Raw classic variants keep their norm vectors F32.)
"""
from __future__ import annotations

import argparse
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

# The word-frequency corpus the fixture vocabulary is trained on. Real SPM
# vocabularies are built by BPE, so every multi-character piece is reachable by
# merging two shorter pieces that are themselves in the vocabulary; a
# hand-written word list is not, and the tokenizer can then never produce its
# longest pieces. Whole words carry most of the weight, as in real text.
FIXTURE_WORDS = {
    "▁hello": 9, "▁world": 8, "▁test": 7, "▁the": 6, "▁ok": 5,
    "▁and": 4, "▁is": 4, "▁it": 4, "▁to": 3, "▁of": 3, "▁in": 3,
    "▁a": 3, "▁I": 2, "▁you": 2, "▁can": 2, "▁not": 2, "▁one": 2,
    "hello": 2, "world": 2, "testing": 2, "there": 1, "later": 1,
}


def bpe_vocab(words: dict[str, int], budget: int) -> list[str]:
    """Train a miniature BPE vocabulary: the alphabet, then merges in order.

    Returns pieces ordered by merge priority, which is what the SPM tokenizer
    reads out of `tokenizer.ggml.scores` as a rank.
    """
    seqs = [(list(w), n) for w, n in words.items()]
    pieces = sorted({ch for w in words for ch in w})
    while len(pieces) < budget:
        counts: dict[tuple[str, str], int] = {}
        for seq, n in seqs:
            for pair in zip(seq, seq[1:]):
                counts[pair] = counts.get(pair, 0) + n
        if not counts:
            break
        # deterministic: most frequent pair, ties broken alphabetically
        best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        merged = best[0] + best[1]
        pieces.append(merged)
        for seq, _ in seqs:
            i = 0
            while i < len(seq) - 1:
                if (seq[i], seq[i + 1]) == best:
                    seq[i:i + 2] = [merged]
                else:
                    i += 1
    return pieces[:budget]


GEMMA_TEMPLATE = (
    "{{ bos_token }}"
    "{%- for message in messages -%}"
    "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim }}"
    "{{ '<end_of_turn>\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{'<start_of_turn>model\n'}}{%- endif -%}"
)


def main(path: str, dtype: str = "F32", arch: str = "llama",
         minimal: bool = False, layers: int = 0,
         swa_pattern: int | None = None, corrupt: str = "") -> None:
    """`minimal` omits every gemma3 metadata key that no real Gemma 3 GGUF
    carries, so the fixture takes the same fallback paths production does.
    `swa_pattern` writes the key as a scalar period instead of a bool array.
    `corrupt` writes one deliberately invalid value so the loader's
    validation can be tested: a value the engine would otherwise accept and
    then fail on much later with an unrelated error."""
    rng = random.Random(42)
    if arch == "gemma3":
        n_embd = 64
        n_head = 2
        n_kv = 1
        head_dim = 16
        n_layer = 6
        n_ff = 128
        n_ctx = 32
    else:
        n_embd = 256 if dtype in ("Q2_K", "Q4_K", "Q5_K", "Q6_K") else N_EMBD
        n_head = N_HEAD
        n_kv = N_HEAD
        head_dim = n_embd // n_head
        n_layer = N_LAYER
        n_ff = 256 if dtype in ("Q2_K", "Q4_K", "Q5_K", "Q6_K") else N_FF
        n_ctx = N_CTX
    if layers:
        n_layer = layers

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
    if arch == "gemma3":
        # real Gemma 3 files carry these as control tokens and reference them
        # from the chat template
        add("<start_of_turn>", 0.0, 3)
        add("<end_of_turn>", 0.0, 3)
    for b in range(256):
        add(f"<0x{b:02X}>", -1000.0, 6)
    # a tiny BPE "vocabulary" so the SPM tokenizer has real pieces to work
    # with. Scores are merge ranks, the way Gemma 3 and other BPE-trained SPM
    # vocabularies store them - not unigram log-probabilities.
    words = bpe_vocab(FIXTURE_WORDS, N_EXTRA)
    for i, w in enumerate(words):
        add(w, -float(i + 1), 1)

    n_vocab = len(tokens)
    w = gguf.GGUFWriter(path, arch)
    w.add("general.name", gguf.T_STRING, f"alpacca-tiny-{arch}-test")
    w.add(f"{arch}.context_length", gguf.T_UINT32, n_ctx)
    w.add(f"{arch}.embedding_length", gguf.T_UINT32, n_embd)
    w.add(f"{arch}.block_count", gguf.T_UINT32, n_layer)
    w.add(f"{arch}.feed_forward_length", gguf.T_UINT32, n_ff)
    w.add(f"{arch}.attention.head_count", gguf.T_UINT32, n_head)
    w.add(f"{arch}.attention.head_count_kv", gguf.T_UINT32, n_kv)
    w.add(f"{arch}.attention.layer_norm_rms_epsilon", gguf.T_FLOAT32,
          1e-6 if arch == "gemma3" else 1e-5)
    if not minimal:
        w.add(f"{arch}.rope.dimension_count", gguf.T_UINT32, head_dim)
        w.add(f"{arch}.vocab_size", gguf.T_UINT32, n_vocab)
    if arch == "gemma3":
        w.add("gemma3.attention.key_length", gguf.T_UINT32, head_dim)
        w.add("gemma3.attention.value_length", gguf.T_UINT32, head_dim)
        w.add("gemma3.attention.sliding_window", gguf.T_UINT32, 3)
        w.add("gemma3.rope.freq_base", gguf.T_FLOAT32,
              0.0 if corrupt == "rope_base" else 1000000.0)
        if corrupt == "embed_scale":
            w.add("gemma3.embedding_scale", gguf.T_FLOAT32, 0.0)
        w.add("gemma3.rope.freq_base_swa", gguf.T_FLOAT32, 10000.0)
        # None of these exist in a real Gemma 3 GGUF. The minimal variant
        # leaves them out so the fallbacks production takes get exercised:
        # the full_attention_period=6 default, the 1/sqrt(head_dim) attention
        # scale (and the n_layer==62 27B rule), unscaled RoPE, and no softcap.
        if swa_pattern is not None:
            w.add("gemma3.attention.sliding_window_pattern", gguf.T_UINT32,
                  swa_pattern)
        if not minimal:
            w.add_array("gemma3.attention.sliding_window_pattern", gguf.T_BOOL,
                        [(i + 1) % 6 != 0 for i in range(n_layer)])
            # deliberately NOT 1/sqrt(head_dim): if the fixture used the same
            # value as the fallback, nothing would prove the key is read
            w.add("gemma3.attention.scale", gguf.T_FLOAT32, 0.125)
            w.add("gemma3.rope.scaling.type", gguf.T_STRING, "linear")
            w.add("gemma3.rope.scaling.factor", gguf.T_FLOAT32, 2.0)
            w.add("gemma3.final_logit_softcapping", gguf.T_FLOAT32, 20.0)
        w.add("tokenizer.chat_template", gguf.T_STRING, GEMMA_TEMPLATE)

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

    def normish(n: int, base: float = 1.0) -> list[float]:
        return [base + ((i % 7) - 3) * 0.03125 for i in range(n)]

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
    q_dim = n_head * head_dim
    kv_dim = n_kv * head_dim
    add_weight("token_embd.weight", (n_embd, n_vocab),
               rand(n_vocab * n_embd), dtype)
    for i in range(n_layer):
        p = f"blk.{i}."
        # a wrong-length norm vector loads cleanly without validation and
        # then fails mid-generation with a cryptic broadcast error
        _norm_n = n_embd - 1 if corrupt == "norm_size" else n_embd
        add_weight(p + "attn_norm.weight", (_norm_n,), normish(_norm_n), "F32")
        add_weight(p + "attn_q.weight", (n_embd, q_dim),
                   rand(n_embd * q_dim), dtype)
        add_weight(p + "attn_k.weight", (n_embd, kv_dim),
                   rand(n_embd * kv_dim), dtype)
        add_weight(p + "attn_v.weight", (n_embd, kv_dim),
                   rand(n_embd * kv_dim), dtype)
        add_weight(p + "attn_output.weight", (q_dim, n_embd),
                   rand(q_dim * n_embd), dtype)
        if arch == "gemma3":
            add_weight(p + "attn_q_norm.weight", (head_dim,),
                       normish(head_dim, 0.875), "F32")
            add_weight(p + "attn_k_norm.weight", (head_dim,),
                       normish(head_dim, 1.125), "F32")
            add_weight(p + "post_attention_norm.weight", (n_embd,),
                       normish(n_embd, 0.75), "F32")
        add_weight(p + "ffn_norm.weight", (n_embd,), normish(n_embd, 1.25), "F32")
        add_weight(p + "ffn_gate.weight", (n_embd, n_ff),
                   rand(n_embd * n_ff), dtype)
        add_weight(p + "ffn_up.weight", (n_embd, n_ff),
                   rand(n_embd * n_ff), dtype)
        add_weight(p + "ffn_down.weight", (n_ff, n_embd),
                   rand(n_ff * n_embd), dtype)
        if arch == "gemma3":
            add_weight(p + "post_ffw_norm.weight", (n_embd,),
                       normish(n_embd, 0.625), "F32")
    add_weight("output_norm.weight", (n_embd,), normish(n_embd, 1.5), "F32")
    if arch not in ("gemma", "gemma3"):
        # the Gemma family ties the output head to the token embedding
        add_weight("output.weight", (n_embd, n_vocab),
                   rand(n_vocab * n_embd), dtype)

    w.write()
    print(f"wrote {path} (arch={arch}, vocab={n_vocab}, dtype={dtype}, "
          f"layers={n_layer}{', minimal metadata' if minimal else ''})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="tiny.gguf")
    ap.add_argument("dtype", nargs="?", default="F32")
    ap.add_argument("--arch", default="llama",
                    choices=["llama", "mistral", "qwen2", "qwen3",
                             "stablelm", "gemma", "gemma3"])
    ap.add_argument("--minimal", action="store_true",
                    help="omit every key no real Gemma 3 GGUF carries")
    ap.add_argument("--layers", type=int, default=0,
                    help="override the block count (62 selects the 27B rules)")
    ap.add_argument("--swa-pattern", type=int, default=None,
                    help="write sliding_window_pattern as a scalar period")
    ap.add_argument("--corrupt", default="",
                    choices=["", "rope_base", "embed_scale", "norm_size"],
                    help="write one invalid value, to test load validation")
    args = ap.parse_args()
    main(args.path, args.dtype, arch=args.arch, minimal=args.minimal,
         layers=args.layers, swa_pattern=args.swa_pattern,
         corrupt=args.corrupt)
