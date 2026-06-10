# Alpacca - the transformer, implemented from scratch in Python.
# Llama-class decoder: RMSNorm, rotary embeddings, grouped-query attention,
# SwiGLU MLP, KV cache. Runs on NumPy when available, pure Python otherwise.
# MIT License. See LICENSE.
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from . import tensor as T
from .gguf import GGUFFile
from .quants import dequantize
from .tokenizer import Tokenizer

if T.HAS_NUMPY:
    import numpy as np

# rope style per architecture: "norm" rotates adjacent pairs (llama/mistral),
# "neox" rotates split halves (qwen2 & friends)
SUPPORTED_ARCHES = {
    "llama": "norm",
    "mistral": "norm",
    "qwen2": "neox",
    "qwen3": "neox",
    "stablelm": "neox",
    "gemma": "neox",
}


@dataclass
class Hyperparams:
    arch: str
    n_layer: int
    n_embd: int
    n_head: int
    n_kv: int
    n_ff: int
    n_vocab: int
    n_ctx_train: int
    head_dim: int
    n_rot: int
    rms_eps: float
    rope_base: float
    rope_style: str


class Layer:
    __slots__ = ("attn_norm", "wq", "wk", "wv", "wo", "bq", "bk", "bv",
                 "ffn_norm", "w_gate", "w_up", "w_down")


class Model:
    def __init__(self, hp: Hyperparams, tokenizer: Tokenizer):
        self.hp = hp
        self.tok = tokenizer
        self.layers: list[Layer] = []
        self.tok_embd = None
        self.out_norm = None
        self.output = None
        self.metadata: dict = {}

    # ---- loading --------------------------------------------------------

    @classmethod
    def load(cls, path: str, n_ctx: int = 0, progress: bool = True) -> "Model":
        t0 = time.time()
        gf = GGUFFile.open(path)
        try:
            if int(gf.get("split.count", 1) or 1) > 1:
                raise ValueError(
                    "multi-part (split) GGUFs are not supported by the python "
                    "engine yet - pick a single-file quantization")
            arch = gf.architecture
            if arch not in SUPPORTED_ARCHES:
                raise ValueError(
                    f"architecture '{arch}' is not supported by the alpacca engine yet "
                    f"(supported: {', '.join(sorted(SUPPORTED_ARCHES))})")

            def meta(key, default=None):
                return gf.get(f"{arch}.{key}", default)

            n_embd = int(meta("embedding_length"))
            n_head = int(meta("attention.head_count"))
            n_kv = int(meta("attention.head_count_kv", n_head) or n_head)
            head_dim = int(meta("attention.key_length", n_embd // n_head) or n_embd // n_head)
            hp = Hyperparams(
                arch=arch,
                n_layer=int(meta("block_count")),
                n_embd=n_embd,
                n_head=n_head,
                n_kv=n_kv,
                n_ff=int(meta("feed_forward_length")),
                n_vocab=int(gf.get(f"{arch}.vocab_size",
                                   len(gf.get("tokenizer.ggml.tokens", [])))),
                n_ctx_train=int(meta("context_length", 4096)),
                head_dim=head_dim,
                n_rot=int(meta("rope.dimension_count", head_dim) or head_dim),
                rms_eps=float(meta("attention.layer_norm_rms_epsilon", 1e-5)),
                rope_base=float(meta("rope.freq_base", 10000.0)),
                rope_style=SUPPORTED_ARCHES[arch],
            )

            tokenizer = Tokenizer.from_gguf(gf.metadata)
            m = cls(hp, tokenizer)
            m.metadata = {k: v for k, v in gf.metadata.items()
                          if not isinstance(v, list) or len(v) < 64}

            def tensor_mat(name, rows, cols, required=True):
                info = gf.tensors.get(name)
                if info is None:
                    if required:
                        raise ValueError(f"missing tensor {name} in {path}")
                    return None
                vals = dequantize(gf.tensor_bytes(name), info.n_elements, info.dtype)
                return T.matrix(vals, rows, cols)

            def tensor_vec(name, required=True):
                info = gf.tensors.get(name)
                if info is None:
                    if required:
                        raise ValueError(f"missing tensor {name} in {path}")
                    return None
                vals = dequantize(gf.tensor_bytes(name), info.n_elements, info.dtype)
                return T.vector(vals)

            kv_dim = hp.n_kv * hp.head_dim
            q_dim = hp.n_head * hp.head_dim
            m.tok_embd = tensor_mat("token_embd.weight", hp.n_vocab, hp.n_embd)
            import sys
            for i in range(hp.n_layer):
                if progress:
                    print(f"\rloading layers {i + 1}/{hp.n_layer}", end="",
                          flush=True, file=sys.stderr)
                p = f"blk.{i}."
                ly = Layer()
                ly.attn_norm = tensor_vec(p + "attn_norm.weight")
                ly.wq = tensor_mat(p + "attn_q.weight", q_dim, hp.n_embd)
                ly.wk = tensor_mat(p + "attn_k.weight", kv_dim, hp.n_embd)
                ly.wv = tensor_mat(p + "attn_v.weight", kv_dim, hp.n_embd)
                ly.wo = tensor_mat(p + "attn_output.weight", hp.n_embd, q_dim)
                ly.bq = tensor_vec(p + "attn_q.bias", required=False)
                ly.bk = tensor_vec(p + "attn_k.bias", required=False)
                ly.bv = tensor_vec(p + "attn_v.bias", required=False)
                ly.ffn_norm = tensor_vec(p + "ffn_norm.weight")
                ly.w_gate = tensor_mat(p + "ffn_gate.weight", hp.n_ff, hp.n_embd)
                ly.w_up = tensor_mat(p + "ffn_up.weight", hp.n_ff, hp.n_embd)
                ly.w_down = tensor_mat(p + "ffn_down.weight", hp.n_embd, hp.n_ff)
                m.layers.append(ly)
            if progress:
                print("\r" + " " * 40 + "\r", end="", flush=True, file=sys.stderr)

            m.out_norm = tensor_vec("output_norm.weight")
            m.output = tensor_mat("output.weight", hp.n_vocab, hp.n_embd, required=False)
            if m.output is None:
                m.output = m.tok_embd  # tied embeddings

            m.n_ctx = min(n_ctx, hp.n_ctx_train) if n_ctx else min(hp.n_ctx_train, 4096)
            m._init_cache()
            m.load_seconds = time.time() - t0
            return m
        finally:
            gf.close()

    # ---- KV cache -------------------------------------------------------

    def _init_cache(self):
        hp = self.hp
        if T.HAS_NUMPY:
            self.cache_k = [np.zeros((self.n_ctx, hp.n_kv, hp.head_dim), dtype=np.float32)
                            for _ in range(hp.n_layer)]
            self.cache_v = [np.zeros((self.n_ctx, hp.n_kv, hp.head_dim), dtype=np.float32)
                            for _ in range(hp.n_layer)]
        else:
            self.cache_k = [[] for _ in range(hp.n_layer)]
            self.cache_v = [[] for _ in range(hp.n_layer)]
        self.n_past = 0

    def reset(self):
        self._init_cache()

    # ---- rotary embeddings ----------------------------------------------

    def _rope_pure(self, vec: list, n_heads: int, pos: int) -> list:
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        out = list(vec)
        half = n_rot // 2
        for h in range(n_heads):
            base = h * hd
            for i in range(half):
                theta = pos * hp.rope_base ** (-2.0 * i / n_rot)
                c, s = math.cos(theta), math.sin(theta)
                if hp.rope_style == "norm":
                    a, b = base + 2 * i, base + 2 * i + 1
                else:  # neox
                    a, b = base + i, base + half + i
                x0, x1 = out[a], out[b]
                out[a] = x0 * c - x1 * s
                out[b] = x0 * s + x1 * c
        return out

    def _rope_np(self, vec, n_heads: int, pos: int):
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        half = n_rot // 2
        v = vec.reshape(n_heads, hd).copy()
        inv = hp.rope_base ** (-2.0 * np.arange(half, dtype=np.float32) / n_rot)
        theta = pos * inv
        c, s = np.cos(theta), np.sin(theta)
        if hp.rope_style == "norm":
            x0 = v[:, 0:n_rot:2].copy()
            x1 = v[:, 1:n_rot:2].copy()
            v[:, 0:n_rot:2] = x0 * c - x1 * s
            v[:, 1:n_rot:2] = x0 * s + x1 * c
        else:
            x0 = v[:, :half].copy()
            x1 = v[:, half:n_rot].copy()
            v[:, :half] = x0 * c - x1 * s
            v[:, half:n_rot] = x0 * s + x1 * c
        return v.reshape(-1)

    # ---- forward pass ----------------------------------------------------

    def forward(self, token: int) -> "object":
        """Process one token at the current position; returns logits."""
        if self.n_past >= self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")
        return self._forward_np(token) if T.HAS_NUMPY else self._forward_pure(token)

    def _forward_np(self, token: int):
        hp = self.hp
        pos = self.n_past
        x = self.tok_embd[token].astype(np.float32).copy()
        inv_sqrt = 1.0 / math.sqrt(hp.head_dim)
        group = hp.n_head // hp.n_kv

        for li, ly in enumerate(self.layers):
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            q = ly.wq @ h
            k = ly.wk @ h
            v = ly.wv @ h
            if ly.bq is not None:
                q = q + ly.bq
            if ly.bk is not None:
                k = k + ly.bk
            if ly.bv is not None:
                v = v + ly.bv
            q = self._rope_np(q, hp.n_head, pos).reshape(hp.n_head, hp.head_dim)
            k = self._rope_np(k, hp.n_kv, pos).reshape(hp.n_kv, hp.head_dim)
            self.cache_k[li][pos] = k
            self.cache_v[li][pos] = v.reshape(hp.n_kv, hp.head_dim)

            K = self.cache_k[li][:pos + 1]            # (t, n_kv, hd)
            V = self.cache_v[li][:pos + 1]
            att_out = np.empty((hp.n_head, hp.head_dim), dtype=np.float32)
            for hh in range(hp.n_head):
                kvh = hh // group
                scores = K[:, kvh, :] @ q[hh] * inv_sqrt        # (t,)
                scores -= scores.max()
                w = np.exp(scores)
                w /= w.sum()
                att_out[hh] = w @ V[:, kvh, :]
            x = x + ly.wo @ att_out.reshape(-1)

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            gate = ly.w_gate @ h
            up = ly.w_up @ h
            act = gate / (1.0 + np.exp(-gate)) * up
            x = x + ly.w_down @ act

        self.n_past += 1
        return self.output @ T.rmsnorm(x, self.out_norm, hp.rms_eps)

    def _forward_pure(self, token: int):
        hp = self.hp
        pos = self.n_past
        x = T.matrix_row(self.tok_embd, token)
        inv_sqrt = 1.0 / math.sqrt(hp.head_dim)
        group = hp.n_head // hp.n_kv
        hd = hp.head_dim

        for li, ly in enumerate(self.layers):
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            q = T.matvec(ly.wq, h)
            k = T.matvec(ly.wk, h)
            v = T.matvec(ly.wv, h)
            if ly.bq is not None:
                q = T.add(q, ly.bq)
            if ly.bk is not None:
                k = T.add(k, ly.bk)
            if ly.bv is not None:
                v = T.add(v, ly.bv)
            q = self._rope_pure(q, hp.n_head, pos)
            k = self._rope_pure(k, hp.n_kv, pos)
            self.cache_k[li].append(k)
            self.cache_v[li].append(v)

            att_out = [0.0] * (hp.n_head * hd)
            t_len = pos + 1
            for hh in range(hp.n_head):
                kvh = hh // group
                qh = q[hh * hd:(hh + 1) * hd]
                scores = []
                for t in range(t_len):
                    kt = self.cache_k[li][t][kvh * hd:(kvh + 1) * hd]
                    scores.append(T.dot(qh, kt) * inv_sqrt)
                w = T.softmax(scores)
                acc = [0.0] * hd
                for t, wt in enumerate(w):
                    if wt == 0.0:
                        continue
                    vt = self.cache_v[li][t][kvh * hd:(kvh + 1) * hd]
                    for d in range(hd):
                        acc[d] += wt * vt[d]
                att_out[hh * hd:(hh + 1) * hd] = acc
            x = T.add(x, T.matvec(ly.wo, att_out))

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            act = T.mul(T.silu(T.matvec(ly.w_gate, h)), T.matvec(ly.w_up, h))
            x = T.add(x, T.matvec(ly.w_down, act))

        self.n_past += 1
        return T.matvec(self.output, T.rmsnorm(x, self.out_norm, hp.rms_eps))

    # ---- convenience -----------------------------------------------------

    def prefill(self, tokens: list[int]):
        """Feed prompt tokens; returns logits of the last one."""
        logits = None
        for t in tokens:
            logits = self.forward(t)
        return logits

    def describe(self) -> str:
        hp = self.hp
        params = hp.n_vocab * hp.n_embd
        for ly in range(hp.n_layer):
            params += 2 * hp.n_embd  # norms
            params += hp.n_embd * hp.n_head * hp.head_dim * 2  # wq, wo
            params += hp.n_embd * hp.n_kv * hp.head_dim * 2    # wk, wv
            params += 3 * hp.n_embd * hp.n_ff
        return (f"{hp.arch} | {hp.n_layer} layers | embd {hp.n_embd} | "
                f"heads {hp.n_head}/{hp.n_kv} | ff {hp.n_ff} | vocab {hp.n_vocab} | "
                f"~{params / 1e6:.0f}M params | ctx {self.n_ctx} | "
                f"backend {T.backend_name()}")
