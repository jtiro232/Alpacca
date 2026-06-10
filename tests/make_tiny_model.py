#!/usr/bin/env python3
"""Create a tiny random-weight llama-architecture GGUF.

The output loads and generates (gibberish) with llama.cpp, which is all the
smoke test needs. Requires the vendored gguf-py:

    pip install numpy && pip install --no-deps ./vendor/llama.cpp/gguf-py
    python3 tests/make_tiny_model.py /tmp/tiny.gguf
"""
import sys

import numpy as np
import gguf

N_VOCAB_EXTRA = 16   # normal tokens on top of specials + byte tokens
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
N_FF = 128
N_CTX = 256


def main(path: str) -> None:
    rng = np.random.default_rng(42)

    tokens, scores, types = [], [], []

    def add(text, score, ttype):
        tokens.append(text)
        scores.append(score)
        types.append(ttype)

    add("<unk>", 0.0, gguf.TokenType.UNKNOWN)
    add("<s>", 0.0, gguf.TokenType.CONTROL)
    add("</s>", 0.0, gguf.TokenType.CONTROL)
    for b in range(256):
        add(f"<0x{b:02X}>", -1000.0, gguf.TokenType.BYTE)
    for i in range(N_VOCAB_EXTRA):
        add(f"▁t{i}", -float(i), gguf.TokenType.NORMAL)

    n_vocab = len(tokens)

    w = gguf.GGUFWriter(path, "llama")
    w.add_name("alpacca-tiny-test")
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(N_EMBD // N_HEAD)
    w.add_vocab_size(n_vocab)

    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(True)

    def t(name, *shape):
        w.add_tensor(name, (rng.standard_normal(shape, dtype=np.float32) * 0.02))

    t("token_embd.weight", n_vocab, N_EMBD)
    for i in range(N_LAYER):
        t(f"blk.{i}.attn_norm.weight", N_EMBD)
        t(f"blk.{i}.attn_q.weight", N_EMBD, N_EMBD)
        t(f"blk.{i}.attn_k.weight", N_EMBD, N_EMBD)
        t(f"blk.{i}.attn_v.weight", N_EMBD, N_EMBD)
        t(f"blk.{i}.attn_output.weight", N_EMBD, N_EMBD)
        t(f"blk.{i}.ffn_norm.weight", N_EMBD)
        t(f"blk.{i}.ffn_gate.weight", N_FF, N_EMBD)
        t(f"blk.{i}.ffn_down.weight", N_EMBD, N_FF)
        t(f"blk.{i}.ffn_up.weight", N_FF, N_EMBD)
    t("output_norm.weight", N_EMBD)
    t("output.weight", n_vocab, N_EMBD)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path} (vocab={n_vocab})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny.gguf")
