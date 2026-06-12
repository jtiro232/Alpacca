# Alpacca - the transformer, implemented from scratch in Python.
# Llama-class decoder: RMSNorm, rotary embeddings, grouped-query attention,
# SwiGLU MLP, KV cache. Runs on NumPy when available, pure Python otherwise.
# MIT License. See LICENSE.
from __future__ import annotations

import math
import os
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

_KNOWN_QUANT_DTYPES = {
    "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q5_1", "Q5_K",
    "Q6_K", "Q8_0", "Q8_1", "Q8_K",
    "IQ1_M", "IQ1_S", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S",
    "IQ3_XXS", "IQ4_NL", "IQ4_XS",
}

# ALPACCA_DENSE_WEIGHT_MB densification order: NumPy's quantized matvec has
# no BLAS-class kernel, so spending RAM on dense float32 buys decode speed
# roughly in proportion to how much of a token's matvec work a matrix does.
# FFN projections dominate llama-class decode, attention q/output come next,
# k/v are smaller (GQA), and the output projection is amortized by the
# last-token-only prefill. The token embedding is only ever row-gathered, so
# it is densified solely when it doubles as a tied output projection.
_DENSIFY_TIERS: tuple[tuple[str, ...], ...] = (
    ("ffn_gate", "ffn_up", "ffn_down"),
    ("attn_q", "attn_output"),
    ("attn_k", "attn_v"),
    ("output",),
)


def _dense_budget_bytes() -> int:
    raw = os.environ.get("ALPACCA_DENSE_WEIGHT_MB")
    if raw is None or not raw.strip():
        return 0
    try:
        mb = float(raw.strip())
    except ValueError:
        return 0
    if not math.isfinite(mb) or mb <= 0.0:
        return 0
    return int(mb * 1024 * 1024)


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
        self.weight_storage: dict = {"dense": 0, "quantized": {}, "fallback": {},
                                     "densified": [], "densified_bytes": 0}
        self.cached_ids: list[int] = []
        self.last_prefill_forwarded = 0
        self._rope_inv_freq = None
        self._rope_cos = None
        self._rope_sin = None
        if T.HAS_NUMPY:
            half = hp.n_rot // 2
            self._rope_inv_freq = (
                hp.rope_base ** (-2.0 * np.arange(half, dtype=np.float32) / hp.n_rot)
            )

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
            dense_matrices = 0
            quantized_matrices: dict[str, int] = {}
            fallback_matrices: dict[str, int] = {}
            densified_names: list[str] = []

            # ALPACCA_DENSE_WEIGHT_MB: pick which quantizable matrices to
            # expand to dense float32 at load (BLAS-speed decode), spending
            # the budget tier by tier; everything else stays quantized.
            densify_plan: set[str] = set()
            densified_bytes = 0
            budget = 0
            if T.HAS_NUMPY and not os.environ.get("ALPACCA_F32"):
                budget = _dense_budget_bytes()
            if budget > 0:
                tied_output = "output.weight" not in gf.tensors
                for tier in _DENSIFY_TIERS:
                    for role in tier:
                        if role == "output":
                            names = ["token_embd.weight" if tied_output
                                     else "output.weight"]
                        else:
                            names = [f"blk.{i}.{role}.weight"
                                     for i in range(hp.n_layer)]
                        for nm in names:
                            info = gf.tensors.get(nm)
                            if info is None or len(info.shape) < 2:
                                continue
                            if not T.can_quantized_matvec(
                                    info.dtype, int(info.shape[0])):
                                continue  # loads dense anyway, costs no budget
                            nbytes = info.n_elements * 4
                            if densified_bytes + nbytes <= budget:
                                densify_plan.add(nm)
                                densified_bytes += nbytes

            def tensor_mat(name, rows, cols, required=True):
                nonlocal dense_matrices
                info = gf.tensors.get(name)
                if info is None:
                    if required:
                        raise ValueError(f"missing tensor {name} in {path}")
                    return None
                if info.n_elements != rows * cols:
                    raise ValueError(
                        f"tensor {name} has {info.n_elements} elements, "
                        f"expected {rows * cols}")
                if (T.HAS_NUMPY and not os.environ.get("ALPACCA_F32") and
                        name not in densify_plan and
                        T.can_quantized_matvec(info.dtype, cols)):
                    quantized_matrices[info.dtype] = (
                        quantized_matrices.get(info.dtype, 0) + 1)
                    return T.quantized_matrix(gf.tensor_bytes(name), info.dtype, rows, cols)
                if name in densify_plan:
                    densified_names.append(name)
                elif info.dtype in _KNOWN_QUANT_DTYPES:
                    fallback_matrices[info.dtype] = (
                        fallback_matrices.get(info.dtype, 0) + 1)
                dense_matrices += 1
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

            m.weight_storage = {
                "dense": dense_matrices,
                "quantized": dict(sorted(quantized_matrices.items())),
                "fallback": dict(sorted(fallback_matrices.items())),
                "densified": sorted(densified_names),
                "densified_bytes": densified_bytes,
            }

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
            if self._rope_cos is None or len(self._rope_cos) != self.n_ctx:
                theta = (np.arange(self.n_ctx, dtype=np.float32)[:, None] *
                         self._rope_inv_freq[None, :])
                self._rope_cos = np.cos(theta)
                self._rope_sin = np.sin(theta)
        else:
            self.cache_k = [[] for _ in range(hp.n_layer)]
            self.cache_v = [[] for _ in range(hp.n_layer)]
        self.n_past = 0
        self.cached_ids = []
        self.last_prefill_forwarded = 0

    def reset(self):
        self._init_cache()

    def _truncate_cache(self, n_tokens: int) -> None:
        """Keep only the first `n_tokens` KV entries."""
        n_tokens = max(0, min(n_tokens, self.n_past))
        if T.HAS_NUMPY:
            self.n_past = n_tokens
        else:
            for li in range(self.hp.n_layer):
                del self.cache_k[li][n_tokens:]
                del self.cache_v[li][n_tokens:]
            self.n_past = n_tokens
        del self.cached_ids[n_tokens:]

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
        c, s = self._rope_cos[pos], self._rope_sin[pos]
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

    def _rope_batch_np(self, vecs, n_heads: int, positions):
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        half = n_rot // 2
        v = vecs.reshape(len(vecs), n_heads, hd).copy()
        c = self._rope_cos[positions][:, None, :]
        s = self._rope_sin[positions][:, None, :]
        if hp.rope_style == "norm":
            x0 = v[:, :, 0:n_rot:2].copy()
            x1 = v[:, :, 1:n_rot:2].copy()
            v[:, :, 0:n_rot:2] = x0 * c - x1 * s
            v[:, :, 1:n_rot:2] = x0 * s + x1 * c
        else:
            x0 = v[:, :, :half].copy()
            x1 = v[:, :, half:n_rot].copy()
            v[:, :, :half] = x0 * c - x1 * s
            v[:, :, half:n_rot] = x0 * s + x1 * c
        return v.reshape(len(vecs), -1)

    # ---- forward pass ----------------------------------------------------

    def forward(self, token: int) -> "object":
        """Process one token at the current position; returns logits."""
        if self.n_past >= self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")
        logits = self._forward_np(token) if T.HAS_NUMPY else self._forward_pure(token)
        self.cached_ids.append(token)
        return logits

    def _attention_np(self, q, K, V, group: int, inv_sqrt: float):
        hp = self.hp
        qg = q.reshape(hp.n_kv, group, hp.head_dim)
        scores = np.matmul(qg, K.transpose(1, 2, 0)) * inv_sqrt
        scores -= scores.max(axis=2, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=2, keepdims=True)
        att_out = np.matmul(w, V.transpose(1, 0, 2))
        return att_out.reshape(hp.n_head, hp.head_dim)

    def _attention_batch_np(self, q, K, V, positions, group: int, inv_sqrt: float):
        hp = self.hp
        qg = q.reshape(len(q), hp.n_kv, group, hp.head_dim)
        scores = np.einsum("tkgh,skh->tkgs", qg, K, optimize=True) * inv_sqrt
        allowed = np.arange(K.shape[0], dtype=np.int32)[None, :] <= positions[:, None]
        scores = np.where(allowed[:, None, None, :], scores, -1.0e30)
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        out = np.einsum("tkgs,skh->tkgh", w, V, optimize=True)
        return out.reshape(len(q), hp.n_head * hp.head_dim)

    def forward_batch(self, tokens: list[int]):
        """Process a NumPy batch at the current position; returns last-token logits."""
        if not T.HAS_NUMPY:
            raise RuntimeError("forward_batch requires the NumPy backend")
        if not tokens:
            return None
        if self.n_past + len(tokens) > self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")

        hp = self.hp
        pos0 = self.n_past
        positions = np.arange(pos0, pos0 + len(tokens), dtype=np.int32)
        # matrix_rows returns a fresh float32 array for dense and quantized
        x = T.matrix_rows(self.tok_embd, tokens)
        inv_sqrt = 1.0 / math.sqrt(hp.head_dim)
        group = hp.n_head // hp.n_kv

        for li, ly in enumerate(self.layers):
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            q = T.matmul_t(h, ly.wq)
            k = T.matmul_t(h, ly.wk)
            v = T.matmul_t(h, ly.wv)
            if ly.bq is not None:
                q = q + ly.bq
            if ly.bk is not None:
                k = k + ly.bk
            if ly.bv is not None:
                v = v + ly.bv
            q = self._rope_batch_np(q, hp.n_head, positions).reshape(
                len(tokens), hp.n_head, hp.head_dim)
            k = self._rope_batch_np(k, hp.n_kv, positions).reshape(
                len(tokens), hp.n_kv, hp.head_dim)
            self.cache_k[li][pos0:pos0 + len(tokens)] = k
            self.cache_v[li][pos0:pos0 + len(tokens)] = v.reshape(
                len(tokens), hp.n_kv, hp.head_dim)

            K = self.cache_k[li][:pos0 + len(tokens)]
            V = self.cache_v[li][:pos0 + len(tokens)]
            att_out = self._attention_batch_np(q, K, V, positions, group, inv_sqrt)
            x = x + T.matmul_t(att_out, ly.wo)

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            gate = T.matmul_t(h, ly.w_gate)
            up = T.matmul_t(h, ly.w_up)
            act = gate / (1.0 + np.exp(-gate)) * up
            x = x + T.matmul_t(act, ly.w_down)

        self.n_past += len(tokens)
        self.cached_ids.extend(tokens)
        return T.matvec(self.output, T.rmsnorm(x[-1], self.out_norm, hp.rms_eps))

    def _forward_np(self, token: int):
        hp = self.hp
        pos = self.n_past
        x = T.matrix_row(self.tok_embd, token)  # fresh float32 copy
        inv_sqrt = 1.0 / math.sqrt(hp.head_dim)
        group = hp.n_head // hp.n_kv

        for li, ly in enumerate(self.layers):
            h = T.rmsnorm(x, ly.attn_norm, hp.rms_eps)
            q = T.matvec(ly.wq, h)
            k = T.matvec(ly.wk, h)
            v = T.matvec(ly.wv, h)
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
            att_out = self._attention_np(q, K, V, group, inv_sqrt)
            x = x + T.matvec(ly.wo, att_out.reshape(-1))

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            gate = T.matvec(ly.w_gate, h)
            up = T.matvec(ly.w_up, h)
            act = gate / (1.0 + np.exp(-gate)) * up
            x = x + T.matvec(ly.w_down, act)

        self.n_past += 1
        return T.matvec(self.output, T.rmsnorm(x, self.out_norm, hp.rms_eps))

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
        self.last_prefill_forwarded = 0
        if not tokens:
            return None
        if len(tokens) > self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")

        n = 0
        max_prefix = min(len(tokens), len(self.cached_ids))
        while n < max_prefix and tokens[n] == self.cached_ids[n]:
            n += 1
        if n == len(tokens):
            n = max(0, len(tokens) - 1)
        if n != self.n_past:
            self._truncate_cache(n)

        suffix = tokens[n:]
        self.last_prefill_forwarded = len(suffix)
        logits = None
        if T.HAS_NUMPY:
            raw = os.environ.get("ALPACCA_PREFILL_CHUNK", "256")
            try:
                chunk = max(1, int(raw))
            except ValueError:
                chunk = 256
            for i in range(0, len(suffix), chunk):
                logits = self.forward_batch(suffix[i:i + chunk])
        else:
            for t in suffix:
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
        storage = self._storage_description()
        return (f"{hp.arch} | {hp.n_layer} layers | embd {hp.n_embd} | "
                f"heads {hp.n_head}/{hp.n_kv} | ff {hp.n_ff} | vocab {hp.n_vocab} | "
                f"~{params / 1e6:.0f}M params | ctx {self.n_ctx} | "
                f"backend {T.backend_name()} | {storage}")

    def _storage_description(self) -> str:
        q = self.weight_storage.get("quantized", {})
        dense = int(self.weight_storage.get("dense", 0) or 0)
        fallback = self.weight_storage.get("fallback", {})
        densified = self.weight_storage.get("densified") or []
        if q:
            q_desc = "/".join(q.keys())
            total_q = sum(q.values())
            parts = [f"weights quantized {q_desc} ({total_q} matrices)"]
            if dense:
                parts.append(f"dense {dense}")
        else:
            parts = [f"weights dense ({dense} matrices)"]
        if fallback:
            fb_desc = "/".join(fallback.keys())
            parts.append(f"dense fallback {fb_desc}")
        if densified:
            mb = self.weight_storage.get("densified_bytes", 0) / (1024 * 1024)
            size = f"{mb / 1024:.1f} GiB" if mb >= 1024 else f"{mb:.1f} MiB"
            parts.append(f"dense budget {len(densified)} matrices ({size})")
        return ", ".join(parts)
