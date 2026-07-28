# Alpacca - the transformer, implemented from scratch in Python.
# Llama-class decoder: RMSNorm, rotary embeddings, grouped-query attention,
# SwiGLU MLP, KV cache. Runs on NumPy when available, pure Python otherwise.
# MIT License. See LICENSE.
from __future__ import annotations

import math
import os
import sys
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
    "gemma3": "neox",
}

# The Gemma family scales the embedding by sqrt(n_embd) on the way in and uses
# GELU rather than SiLU in the MLP. Gemma 1 otherwise runs the llama-class
# decoder; Gemma 3 has its own forward pass for the extra norms and dual RoPE.
_GEMMA_ARCHES = ("gemma", "gemma3")

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
    try:
        return int(mb * 1024 * 1024)
    except (OverflowError, ValueError):
        return 0


def auto_budget_fit_mb(path: str, n_ctx: int = 0):
    """Exact sizing inputs for the CLI's auto dense budget, from the GGUF
    header alone: (eligible_mb, fixed_mb) where eligible_mb is the dense
    float32 size of every matrix the densify plan could select, and
    fixed_mb covers what stays resident regardless - residual quantized
    storage (~1.3 B/weight upper bound, counted for the token embedding
    whether or not it is tied), the KV cache at the effective context, and a
    runtime baseline. Matrices with unsupported dtypes
    load dense float32 regardless of any budget and are counted in
    neither term, so mixed-format files understate fixed memory (same
    blind spot as the fallback formula). Returns None if the header
    cannot be read or the architecture is unsupported."""
    try:
        gf = GGUFFile.open(path)
    except Exception:
        return None
    try:
        arch = gf.architecture
        if arch not in SUPPORTED_ARCHES:
            return None

        def meta(key, default=None):
            return gf.get(f"{arch}.{key}", default)

        n_layer = int(meta("block_count", 0) or 0)
        n_embd = int(meta("embedding_length", 0) or 0)
        n_head = int(meta("attention.head_count", 1) or 1)
        n_kv = int(meta("attention.head_count_kv", n_head) or n_head)
        head_dim = int(meta("attention.key_length", n_embd // max(n_head, 1))
                       or n_embd // max(n_head, 1))
        train_ctx = int(meta("context_length", 4096) or 4096)
        if n_layer <= 0 or n_embd <= 0:
            return None

        tied = "output.weight" not in gf.tensors
        eligible = 0
        residual = 0
        for tier in _DENSIFY_TIERS:
            for role in tier:
                if role == "output":
                    names = ["token_embd.weight" if tied else "output.weight"]
                else:
                    names = [f"blk.{i}.{role}.weight" for i in range(n_layer)]
                for nm in names:
                    info = gf.tensors.get(nm)
                    if info is None or len(info.shape) < 2:
                        continue
                    if T.can_quantized_matvec(info.dtype, int(info.shape[0])):
                        eligible += info.n_elements * 4
        # The token embedding is resident either way: quantized if the budget
        # does not reach it, dense (already in `eligible`) if it does. Counting
        # the quantized form unconditionally makes `fixed` an upper bound
        # rather than leaving a *tied* embedding out of both terms - on a
        # Gemma 3 1B that gap was 374 MiB of unaccounted memory.
        embd = gf.tensors.get("token_embd.weight")
        if embd is not None and len(embd.shape) >= 2 and \
                T.can_quantized_matvec(embd.dtype, int(embd.shape[0])):
            residual += int(embd.n_elements * 1.3)

        ctx_eff = min(train_ctx, n_ctx) if n_ctx else min(train_ctx, 4096)
        kv_bytes = 2 * n_layer * max(ctx_eff, 0) * n_kv * head_dim * 4
        mib = 1024.0 * 1024.0
        fixed_mb = residual / mib + kv_bytes / mib + 512.0
        return eligible / mib, fixed_mb
    except Exception:
        return None
    finally:
        gf.close()


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
    rope_freq_scale: float = 1.0
    rope_base_swa: float = 0.0
    sliding_window: int = 0
    sliding_layers: tuple[bool, ...] = ()
    attention_scale: float = 0.0
    final_logit_softcap: float = 0.0
    embed_scale: float = 1.0
    full_attention_period: int = 0
    # which rule decided the sliding-window layout, so a mis-detection is
    # visible rather than silently changing every layer's attention mask
    swa_rule: str = ""
    attention_scale_from_metadata: bool = False


class Layer:
    __slots__ = ("attn_norm", "wq", "wk", "wv", "wo", "bq", "bk", "bv",
                 "q_norm", "k_norm", "post_attn_norm",
                 "ffn_norm", "w_gate", "w_up", "w_down", "post_ffw_norm")


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
        self._rope_inv_freq_swa = None
        self._rope_cos = None
        self._rope_sin = None
        self._rope_cos_swa = None
        self._rope_sin_swa = None
        if T.HAS_NUMPY:
            half = hp.n_rot // 2
            self._rope_inv_freq = (
                hp.rope_freq_scale *
                hp.rope_base ** (-2.0 * np.arange(half, dtype=np.float32) / hp.n_rot)
            )
            if hp.rope_base_swa > 0.0:
                self._rope_inv_freq_swa = (
                    hp.rope_base_swa **
                    (-2.0 * np.arange(half, dtype=np.float32) / hp.n_rot)
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

            def meta_required(key) -> int:
                """A core dimension. Missing or non-positive means the file is
                unusable; say so instead of dying in int(None) or NumPy."""
                raw = meta(key)
                try:
                    value = int(raw)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{path}: required metadata '{arch}.{key}' is "
                        f"{'missing' if raw is None else repr(raw)}") from None
                if value <= 0:
                    raise ValueError(
                        f"{path}: metadata '{arch}.{key}' must be positive, got {value}")
                return value

            n_embd = meta_required("embedding_length")
            n_head = meta_required("attention.head_count")
            n_kv = int(meta("attention.head_count_kv", n_head) or n_head)
            head_dim = int(meta("attention.key_length", n_embd // n_head) or n_embd // n_head)
            n_layer = meta_required("block_count")
            # not `or 10000.0`: that idiom coerces a stored 0.0 to the default
            # and the check below could then never fire for the one value the
            # check exists for. Absent key -> default; present key -> validated.
            _rope_base_md = meta("rope.freq_base", None)
            rope_base = 10000.0 if _rope_base_md is None else float(_rope_base_md)
            if not math.isfinite(rope_base) or rope_base <= 0.0:
                raise ValueError(
                    f"{path}: {arch}.rope.freq_base must be positive and "
                    f"finite, got {rope_base}")
            rope_base_swa = float(meta("rope.freq_base_swa", 0.0) or 0.0)
            sliding_window = int(meta("attention.sliding_window", 0) or 0)
            if arch == "gemma3" and sliding_window > 0 and rope_base_swa <= 0.0:
                rope_base_swa = 10000.0

            rope_scale_legacy = meta("rope.scale_linear", None)
            rope_scaling_type = str(meta("rope.scaling.type", "") or "").lower()
            rope_scaling_factor = float(
                meta("rope.scaling.factor",
                     rope_scale_legacy if rope_scale_legacy is not None else 1.0) or 1.0)
            rope_freq_scale = 1.0
            # llama.cpp treats a bare rope.scaling.factor as linear, so a file
            # that omits the type still scales. Applies to every architecture,
            # which is what upstream does - yarn/longrope are not implemented
            # and are deliberately left unscaled rather than scaled wrongly.
            if rope_scaling_factor > 0.0 and rope_scaling_type in ("", "linear"):
                rope_freq_scale = 1.0 / rope_scaling_factor
            elif rope_scaling_type not in ("", "linear", "none"):
                print(f"warning: {path}: rope scaling type "
                      f"'{rope_scaling_type}' is not implemented; "
                      f"running unscaled", file=sys.stderr)

            fallback_attn_scale = 1.0 / math.sqrt(head_dim)
            # Gemma 3 27B alone scales by the per-head width rather than the
            # key length. llama.cpp keys this on the layer count alone
            # (src/models/gemma3.cpp: case 62 -> LLM_TYPE_27B), so keep the
            # same rule rather than inventing a stricter one.
            if arch == "gemma3" and n_layer == 62:
                fallback_attn_scale = 1.0 / math.sqrt(n_embd / n_head)
            attention_scale = float(meta("attention.scale", fallback_attn_scale)
                                    or fallback_attn_scale)
            # llama.cpp never reads this key - it always computes the scale -
            # so anything writing it is third-party. A real scale is a
            # reciprocal square root and so lies in (0, 1]; a converter using
            # query_pre_attn_scalar semantics would write 256 instead of
            # 0.0625 and every softmax would be catastrophically mis-scaled.
            if not math.isfinite(attention_scale) or not 0.0 < attention_scale <= 1.0:
                print(f"warning: {path}: {arch}.attention.scale is "
                      f"{attention_scale}, which is not a reciprocal square "
                      f"root; using {fallback_attn_scale:.6g} instead",
                      file=sys.stderr)
                attention_scale = fallback_attn_scale

            sliding_layers: tuple[bool, ...] = ()
            full_attention_period = 0
            swa_rule = ""
            if arch == "gemma3":
                pattern = meta("attention.sliding_window_pattern", None)
                if isinstance(pattern, list):
                    if len(pattern) != n_layer:
                        raise ValueError(
                            "gemma3.attention.sliding_window_pattern has "
                            f"{len(pattern)} entries, expected {n_layer}")
                    sliding_layers = tuple(bool(v) for v in pattern)
                    swa_rule = "per-layer pattern"
                elif pattern is not None:
                    # a scalar is a period, llama.cpp's set_swa_pattern:
                    #   is_swa[i] = n == 0 or (i % n < n - 1)
                    # so 1 means "no sliding window at all" and 0 means "every
                    # layer slides". A converter writes 1 for exactly the
                    # models that have no SWA, so do not second-guess it.
                    full_attention_period = int(pattern)
                    swa_rule = "period from metadata"
                else:
                    full_attention_period = int(meta("full_attention_interval", 6) or 6)
                    swa_rule = ("period from full_attention_interval"
                                if meta("full_attention_interval") is not None
                                else "default period 6")

            # a stored 0 here silently makes every logit token-independent,
            # so validate it the same way rope.freq_base is validated above
            if arch in _GEMMA_ARCHES:
                _embed_scale_md = meta("embedding_scale", None)
                embed_scale = (math.sqrt(n_embd) if _embed_scale_md is None
                               else float(_embed_scale_md))
                if not math.isfinite(embed_scale) or embed_scale <= 0.0:
                    raise ValueError(
                        f"{path}: {arch}.embedding_scale must be positive and "
                        f"finite, got {embed_scale}")
            else:
                embed_scale = 1.0

            hp = Hyperparams(
                arch=arch,
                n_layer=n_layer,
                n_embd=n_embd,
                n_head=n_head,
                n_kv=n_kv,
                n_ff=meta_required("feed_forward_length"),
                n_vocab=int(gf.get(f"{arch}.vocab_size",
                                   len(gf.get("tokenizer.ggml.tokens", [])))),
                n_ctx_train=int(meta("context_length", 4096)),
                head_dim=head_dim,
                n_rot=int(meta("rope.dimension_count", head_dim) or head_dim),
                rms_eps=float(meta("attention.layer_norm_rms_epsilon", 1e-5)),
                rope_base=rope_base,
                rope_style=SUPPORTED_ARCHES[arch],
                rope_freq_scale=rope_freq_scale,
                rope_base_swa=rope_base_swa,
                sliding_window=sliding_window,
                sliding_layers=sliding_layers,
                attention_scale=attention_scale,
                final_logit_softcap=float(meta("final_logit_softcapping", 0.0) or 0.0),
                embed_scale=embed_scale,
                full_attention_period=full_attention_period,
                swa_rule=swa_rule,
                attention_scale_from_metadata=meta("attention.scale") is not None,
            )

            tokenizer = Tokenizer.from_gguf(gf.metadata)
            m = cls(hp, tokenizer)
            m.metadata = {k: v for k, v in gf.metadata.items()
                          if not isinstance(v, list) or len(v) < 64}
            dense_matrices = 0
            dense_bytes = 0
            quantized_bytes = 0
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
                nonlocal dense_matrices, dense_bytes, quantized_bytes
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
                    quantized_bytes += info.n_bytes
                    return T.quantized_matrix(gf.tensor_bytes(name), info.dtype, rows, cols)
                if name in densify_plan:
                    densified_names.append(name)
                elif info.dtype in _KNOWN_QUANT_DTYPES:
                    fallback_matrices[info.dtype] = (
                        fallback_matrices.get(info.dtype, 0) + 1)
                dense_matrices += 1
                dense_bytes += info.n_elements * 4
                vals = dequantize(gf.tensor_bytes(name), info.n_elements, info.dtype)
                return T.matrix(vals, rows, cols)

            def tensor_vec(name, required=True, size=None):
                info = gf.tensors.get(name)
                if info is None:
                    if required:
                        raise ValueError(f"missing tensor {name} in {path}")
                    return None
                if size is not None and info.n_elements != size:
                    raise ValueError(
                        f"tensor {name} has {info.n_elements} elements, "
                        f"expected {size}")
                vals = dequantize(gf.tensor_bytes(name), info.n_elements, info.dtype)
                return T.vector(vals)

            kv_dim = hp.n_kv * hp.head_dim
            q_dim = hp.n_head * hp.head_dim
            m.tok_embd = tensor_mat("token_embd.weight", hp.n_vocab, hp.n_embd)
            for i in range(hp.n_layer):
                if progress:
                    print(f"\rloading layers {i + 1}/{hp.n_layer}", end="",
                          flush=True, file=sys.stderr)
                p = f"blk.{i}."
                ly = Layer()
                ly.q_norm = None
                ly.k_norm = None
                ly.post_attn_norm = None
                ly.post_ffw_norm = None
                ly.attn_norm = tensor_vec(p + "attn_norm.weight", size=hp.n_embd)
                ly.wq = tensor_mat(p + "attn_q.weight", q_dim, hp.n_embd)
                ly.wk = tensor_mat(p + "attn_k.weight", kv_dim, hp.n_embd)
                ly.wv = tensor_mat(p + "attn_v.weight", kv_dim, hp.n_embd)
                ly.wo = tensor_mat(p + "attn_output.weight", hp.n_embd, q_dim)
                ly.bq = tensor_vec(p + "attn_q.bias", required=False, size=q_dim)
                ly.bk = tensor_vec(p + "attn_k.bias", required=False, size=kv_dim)
                ly.bv = tensor_vec(p + "attn_v.bias", required=False, size=kv_dim)
                if arch == "gemma3":
                    # q/k-norm are per-head vectors shared across heads
                    ly.q_norm = tensor_vec(p + "attn_q_norm.weight",
                                           size=hp.head_dim)
                    ly.k_norm = tensor_vec(p + "attn_k_norm.weight",
                                           size=hp.head_dim)
                    ly.post_attn_norm = tensor_vec(p + "post_attention_norm.weight",
                                                   size=hp.n_embd)
                ly.ffn_norm = tensor_vec(p + "ffn_norm.weight", size=hp.n_embd)
                ly.w_gate = tensor_mat(p + "ffn_gate.weight", hp.n_ff, hp.n_embd)
                ly.w_up = tensor_mat(p + "ffn_up.weight", hp.n_ff, hp.n_embd)
                ly.w_down = tensor_mat(p + "ffn_down.weight", hp.n_embd, hp.n_ff)
                if arch == "gemma3":
                    ly.post_ffw_norm = tensor_vec(p + "post_ffw_norm.weight",
                                                  size=hp.n_embd)
                m.layers.append(ly)
            if progress:
                print("\r" + " " * 40 + "\r", end="", flush=True, file=sys.stderr)

            m.out_norm = tensor_vec("output_norm.weight", size=hp.n_embd)
            m.output = tensor_mat("output.weight", hp.n_vocab, hp.n_embd, required=False)
            if m.output is None:
                m.output = m.tok_embd  # tied embeddings

            m.weight_storage = {
                "dense": dense_matrices,
                "dense_bytes": dense_bytes,
                "quantized_bytes": quantized_bytes,
                "quantized": dict(sorted(quantized_matrices.items())),
                "fallback": dict(sorted(fallback_matrices.items())),
                "densified": sorted(densified_names),
                "densified_bytes": densified_bytes,
            }

            # A quantization the engine cannot matvec is dequantized to dense
            # float32 at load - Q2_K weights become 8x their file size, which
            # neither budget formula accounts for. Say so before the RAM goes.
            if fallback_matrices and progress:
                fb_bytes = sum(gf.tensors[nm].n_elements * 4
                               for nm in gf.tensors
                               if gf.tensors[nm].dtype in fallback_matrices
                               and len(gf.tensors[nm].shape) >= 2)
                fb_mb = fb_bytes / (1024 * 1024)
                size = f"{fb_mb / 1024:.1f} GiB" if fb_mb >= 1024 else f"{fb_mb:.0f} MiB"
                print(f"warning: alpacca has no quantized matvec for "
                      f"{'/'.join(sorted(fallback_matrices))}, so "
                      f"{sum(fallback_matrices.values())} matrices load as dense "
                      f"float32 ({size}) - no memory budget accounts for this",
                      file=sys.stderr)

            m.n_ctx = min(n_ctx, hp.n_ctx_train) if n_ctx else min(hp.n_ctx_train, 4096)
            m._init_cache()
            if quantized_matrices and T.HAS_NUMPY:
                from . import kernels
                if kernels.available():
                    kernels.warmup()  # JIT compile/cache-load counts as load
            m.load_seconds = time.time() - t0
            return m
        finally:
            gf.close()

    # ---- KV cache -------------------------------------------------------

    def _init_cache(self):
        """Allocate the KV cache at the full context for every layer.

        Sliding-window layers only ever read the last `sliding_window` rows, so
        this over-allocates - 1.35 GiB at a 32k context on Gemma 3, though the
        default 4096 clamp bounds it to ~154 MiB and np.zeros is lazily mapped,
        so nothing is touched until it is written.

        A ring buffer indexed by `pos % window` would reclaim that, and it
        would be WRONG here. `_truncate_cache` plus `prefill`'s prefix reuse can
        restart at position 100 after the cache has reached 3000, and slot
        `100 % window` would still hold position 2659's K. Any fix has to keep
        absolute-position semantics or invalidate the cache on truncation. Note
        also that the batch path writes up to `window + chunk - 1` rows, so a
        window-sized buffer is too small at the default chunk of 256.
        """
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
            if self._rope_inv_freq_swa is not None and (
                    self._rope_cos_swa is None or
                    len(self._rope_cos_swa) != self.n_ctx):
                theta = (np.arange(self.n_ctx, dtype=np.float32)[:, None] *
                         self._rope_inv_freq_swa[None, :])
                self._rope_cos_swa = np.cos(theta)
                self._rope_sin_swa = np.sin(theta)
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
        return self._rope_pure_base(vec, n_heads, pos, self.hp.rope_base,
                                    self.hp.rope_freq_scale)

    def _rope_pure_base(self, vec: list, n_heads: int, pos: int,
                        rope_base: float, freq_scale: float = 1.0) -> list:
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        out = list(vec)
        half = n_rot // 2
        for h in range(n_heads):
            base = h * hd
            for i in range(half):
                theta = pos * freq_scale * rope_base ** (-2.0 * i / n_rot)
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
        if self.hp.arch == "gemma3":
            logits = (self._forward_gemma3_np(token) if T.HAS_NUMPY
                      else self._forward_gemma3_pure(token))
        else:
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

    def forward_batch(self, tokens: list[int], want_logits: bool = True):
        """Process a NumPy batch at the current position; returns last-token
        logits, or None when `want_logits` is False - prefill discards every
        chunk's logits but the last, and on a tied 262144-row head that
        projection is a large share of the chunk's work."""
        if not T.HAS_NUMPY:
            raise RuntimeError("forward_batch requires the NumPy backend")
        if self.hp.arch == "gemma3":
            return self._forward_batch_gemma3_np(tokens, want_logits)
        if not tokens:
            return None
        if self.n_past + len(tokens) > self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")

        hp = self.hp
        pos0 = self.n_past
        positions = np.arange(pos0, pos0 + len(tokens), dtype=np.int32)
        # matrix_rows returns a fresh float32 array for dense and quantized
        x = T.matrix_rows(self.tok_embd, tokens)
        if hp.embed_scale != 1.0:
            x = x * hp.embed_scale
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
            act = (T.gelu_pytorch_tanh(gate) if hp.arch in _GEMMA_ARCHES
                   else gate / (1.0 + np.exp(-gate))) * up
            x = x + T.matmul_t(act, ly.w_down)

        self.n_past += len(tokens)
        self.cached_ids.extend(tokens)
        if not want_logits:
            return None
        return T.matvec(self.output, T.rmsnorm(x[-1], self.out_norm, hp.rms_eps))

    def _forward_np(self, token: int):
        hp = self.hp
        pos = self.n_past
        x = T.matrix_row(self.tok_embd, token)  # fresh float32 copy
        if hp.embed_scale != 1.0:
            x = x * hp.embed_scale
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
            act = (T.gelu_pytorch_tanh(gate) if hp.arch in _GEMMA_ARCHES
                   else gate / (1.0 + np.exp(-gate))) * up
            x = x + T.matvec(ly.w_down, act)

        self.n_past += 1
        return T.matvec(self.output, T.rmsnorm(x, self.out_norm, hp.rms_eps))

    def _gemma3_layer_is_sliding(self, layer_index: int) -> bool:
        hp = self.hp
        if hp.sliding_window <= 0:
            return False
        if hp.sliding_layers:
            return hp.sliding_layers[layer_index]
        if hp.full_attention_period <= 0:
            # llama.cpp reads a period of 0 as "every layer slides"
            return hp.arch == "gemma3"
        return (layer_index + 1) % hp.full_attention_period != 0

    def _rope_np_gemma3(self, vec, n_heads: int, pos: int, sliding: bool):
        cos = self._rope_cos_swa if sliding else self._rope_cos
        sin = self._rope_sin_swa if sliding else self._rope_sin
        # a sliding layer only exists when sliding_window > 0, which forces
        # rope_base_swa > 0, which builds this table - so falling back to the
        # global table here would silently rotate sliding layers with the 1e6
        # base and there would be no way to notice
        assert cos is not None and sin is not None, (
            "sliding-window RoPE table missing for a sliding layer")
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        half = n_rot // 2
        v = vec.reshape(n_heads, hd).copy()
        c, s = cos[pos], sin[pos]
        x0 = v[:, :half].copy()
        x1 = v[:, half:n_rot].copy()
        v[:, :half] = x0 * c - x1 * s
        v[:, half:n_rot] = x0 * s + x1 * c
        return v.reshape(-1)

    def _rope_batch_np_gemma3(self, vecs, n_heads: int, positions, sliding: bool):
        cos = self._rope_cos_swa if sliding else self._rope_cos
        sin = self._rope_sin_swa if sliding else self._rope_sin
        # a sliding layer only exists when sliding_window > 0, which forces
        # rope_base_swa > 0, which builds this table - so falling back to the
        # global table here would silently rotate sliding layers with the 1e6
        # base and there would be no way to notice
        assert cos is not None and sin is not None, (
            "sliding-window RoPE table missing for a sliding layer")
        hp = self.hp
        hd, n_rot = hp.head_dim, hp.n_rot
        half = n_rot // 2
        v = vecs.reshape(len(vecs), n_heads, hd).copy()
        c = cos[positions][:, None, :]
        s = sin[positions][:, None, :]
        x0 = v[:, :, :half].copy()
        x1 = v[:, :, half:n_rot].copy()
        v[:, :, :half] = x0 * c - x1 * s
        v[:, :, half:n_rot] = x0 * s + x1 * c
        return v.reshape(len(vecs), -1)

    def _rmsnorm_heads_np(self, vec, n_heads: int, weight):
        hp = self.hp
        return T.rmsnorm(vec.reshape(n_heads, hp.head_dim), weight,
                         hp.rms_eps).reshape(-1)

    def _rmsnorm_heads_batch_np(self, vecs, n_heads: int, weight):
        hp = self.hp
        return T.rmsnorm(vecs.reshape(len(vecs), n_heads, hp.head_dim), weight,
                         hp.rms_eps).reshape(len(vecs), -1)

    def _rmsnorm_heads_pure(self, vec: list, n_heads: int, weight) -> list:
        hp = self.hp
        out: list[float] = []
        for h in range(n_heads):
            start = h * hp.head_dim
            out.extend(T.rmsnorm(vec[start:start + hp.head_dim], weight,
                                 hp.rms_eps))
        return out

    def _attention_batch_window_np(self, q, K, V, positions, group: int,
                                   inv_sqrt: float, window: int = 0,
                                   kv_start: int = 0):
        hp = self.hp
        qg = q.reshape(len(q), hp.n_kv, group, hp.head_dim)
        scores = np.einsum("tkgh,skh->tkgs", qg, K, optimize=True) * inv_sqrt
        kv_pos = np.arange(kv_start, kv_start + K.shape[0], dtype=np.int32)
        allowed = kv_pos[None, :] <= positions[:, None]
        if window > 0:
            allowed &= kv_pos[None, :] > (positions[:, None] - window)
        scores = np.where(allowed[:, None, None, :], scores, -1.0e30)
        scores -= scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=-1, keepdims=True)
        out = np.einsum("tkgs,skh->tkgh", w, V, optimize=True)
        return out.reshape(len(q), hp.n_head * hp.head_dim)

    def _softcap_logits(self, logits):
        cap = self.hp.final_logit_softcap
        if cap <= 0.0:
            return logits
        if T.HAS_NUMPY:
            return np.tanh(logits / cap) * cap
        return [math.tanh(v / cap) * cap for v in logits]

    def _forward_batch_gemma3_np(self, tokens: list[int], want_logits: bool = True):
        if not tokens:
            return None
        if self.n_past + len(tokens) > self.n_ctx:
            raise RuntimeError(f"context window full ({self.n_ctx} tokens)")

        hp = self.hp
        pos0 = self.n_past
        positions = np.arange(pos0, pos0 + len(tokens), dtype=np.int32)
        x = T.matrix_rows(self.tok_embd, tokens) * hp.embed_scale
        inv_sqrt = hp.attention_scale
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
            q = self._rmsnorm_heads_batch_np(q, hp.n_head, ly.q_norm)
            k = self._rmsnorm_heads_batch_np(k, hp.n_kv, ly.k_norm)
            sliding = self._gemma3_layer_is_sliding(li)
            q = self._rope_batch_np_gemma3(q, hp.n_head, positions, sliding).reshape(
                len(tokens), hp.n_head, hp.head_dim)
            k = self._rope_batch_np_gemma3(k, hp.n_kv, positions, sliding).reshape(
                len(tokens), hp.n_kv, hp.head_dim)
            self.cache_k[li][pos0:pos0 + len(tokens)] = k
            self.cache_v[li][pos0:pos0 + len(tokens)] = v.reshape(
                len(tokens), hp.n_kv, hp.head_dim)

            window = hp.sliding_window if sliding else 0
            kv_start = max(0, int(positions[0]) - window + 1) if window > 0 else 0
            K = self.cache_k[li][kv_start:pos0 + len(tokens)]
            V = self.cache_v[li][kv_start:pos0 + len(tokens)]
            att_out = self._attention_batch_window_np(q, K, V, positions,
                                                       group, inv_sqrt, window,
                                                       kv_start)
            att_proj = T.matmul_t(att_out, ly.wo)
            x = x + T.rmsnorm(att_proj, ly.post_attn_norm, hp.rms_eps)

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            gate = T.matmul_t(h, ly.w_gate)
            up = T.matmul_t(h, ly.w_up)
            act = T.gelu_pytorch_tanh(gate) * up
            ffn_out = T.matmul_t(act, ly.w_down)
            x = x + T.rmsnorm(ffn_out, ly.post_ffw_norm, hp.rms_eps)

        self.n_past += len(tokens)
        self.cached_ids.extend(tokens)
        if not want_logits:
            return None
        logits = T.matvec(self.output, T.rmsnorm(x[-1], self.out_norm, hp.rms_eps))
        return self._softcap_logits(logits)

    def _forward_gemma3_np(self, token: int):
        hp = self.hp
        pos = self.n_past
        x = T.matrix_row(self.tok_embd, token) * hp.embed_scale
        inv_sqrt = hp.attention_scale
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
            q = self._rmsnorm_heads_np(q, hp.n_head, ly.q_norm)
            k = self._rmsnorm_heads_np(k, hp.n_kv, ly.k_norm)
            sliding = self._gemma3_layer_is_sliding(li)
            q = self._rope_np_gemma3(q, hp.n_head, pos, sliding).reshape(
                hp.n_head, hp.head_dim)
            k = self._rope_np_gemma3(k, hp.n_kv, pos, sliding).reshape(
                hp.n_kv, hp.head_dim)
            self.cache_k[li][pos] = k
            self.cache_v[li][pos] = v.reshape(hp.n_kv, hp.head_dim)

            start = max(0, pos - hp.sliding_window + 1) if sliding else 0
            K = self.cache_k[li][start:pos + 1]
            V = self.cache_v[li][start:pos + 1]
            att_out = self._attention_np(q, K, V, group, inv_sqrt)
            att_proj = T.matvec(ly.wo, att_out.reshape(-1))
            x = x + T.rmsnorm(att_proj, ly.post_attn_norm, hp.rms_eps)

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            act = T.gelu_pytorch_tanh(T.matvec(ly.w_gate, h)) * T.matvec(ly.w_up, h)
            ffn_out = T.matvec(ly.w_down, act)
            x = x + T.rmsnorm(ffn_out, ly.post_ffw_norm, hp.rms_eps)

        self.n_past += 1
        logits = T.matvec(self.output, T.rmsnorm(x, self.out_norm, hp.rms_eps))
        return self._softcap_logits(logits)

    def _forward_gemma3_pure(self, token: int):
        hp = self.hp
        pos = self.n_past
        x = T.scale(T.matrix_row(self.tok_embd, token), hp.embed_scale)
        inv_sqrt = hp.attention_scale
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
            q = self._rmsnorm_heads_pure(q, hp.n_head, ly.q_norm)
            k = self._rmsnorm_heads_pure(k, hp.n_kv, ly.k_norm)
            sliding = self._gemma3_layer_is_sliding(li)
            base = hp.rope_base_swa if sliding else hp.rope_base
            freq_scale = 1.0 if sliding else hp.rope_freq_scale
            q = self._rope_pure_base(q, hp.n_head, pos, base, freq_scale)
            k = self._rope_pure_base(k, hp.n_kv, pos, base, freq_scale)
            self.cache_k[li].append(k)
            self.cache_v[li].append(v)

            start = max(0, pos - hp.sliding_window + 1) if sliding else 0
            att_out = [0.0] * (hp.n_head * hd)
            for hh in range(hp.n_head):
                kvh = hh // group
                qh = q[hh * hd:(hh + 1) * hd]
                scores = []
                for t in range(start, pos + 1):
                    kt = self.cache_k[li][t][kvh * hd:(kvh + 1) * hd]
                    scores.append(T.dot(qh, kt) * inv_sqrt)
                w = T.softmax(scores)
                acc = [0.0] * hd
                for offs, wt in enumerate(w):
                    if wt == 0.0:
                        continue
                    vt = self.cache_v[li][start + offs][kvh * hd:(kvh + 1) * hd]
                    for d in range(hd):
                        acc[d] += wt * vt[d]
                att_out[hh * hd:(hh + 1) * hd] = acc
            att_proj = T.matvec(ly.wo, att_out)
            x = T.add(x, T.rmsnorm(att_proj, ly.post_attn_norm, hp.rms_eps))

            h = T.rmsnorm(x, ly.ffn_norm, hp.rms_eps)
            act = T.mul(T.gelu_pytorch_tanh(T.matvec(ly.w_gate, h)),
                        T.matvec(ly.w_up, h))
            ffn_out = T.matvec(ly.w_down, act)
            x = T.add(x, T.rmsnorm(ffn_out, ly.post_ffw_norm, hp.rms_eps))

        self.n_past += 1
        logits = T.matvec(self.output, T.rmsnorm(x, self.out_norm, hp.rms_eps))
        return self._softcap_logits(logits)

    def _forward_pure(self, token: int):
        hp = self.hp
        pos = self.n_past
        x = T.matrix_row(self.tok_embd, token)
        if hp.embed_scale != 1.0:
            x = T.scale(x, hp.embed_scale)
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
            gate = T.matvec(ly.w_gate, h)
            act = T.mul(T.gelu_pytorch_tanh(gate) if hp.arch in _GEMMA_ARCHES
                        else T.silu(gate), T.matvec(ly.w_up, h))
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
                logits = self.forward_batch(suffix[i:i + chunk],
                                            want_logits=i + chunk >= len(suffix))
        else:
            for t in suffix:
                logits = self.forward(t)
        return logits

    def describe(self) -> str:
        hp = self.hp
        params = hp.n_vocab * hp.n_embd
        if self.output is not self.tok_embd:
            params += hp.n_vocab * hp.n_embd  # untied output projection
        params += hp.n_embd  # output_norm
        for ly in range(hp.n_layer):
            params += 2 * hp.n_embd  # norms
            if hp.arch == "gemma3":
                params += 2 * hp.head_dim  # q_norm/k_norm are shared per head
                params += 2 * hp.n_embd  # post-attention/post-ffw norms
            params += hp.n_embd * hp.n_head * hp.head_dim * 2  # wq, wo
            params += hp.n_embd * hp.n_kv * hp.head_dim * 2    # wk, wv
            params += 3 * hp.n_embd * hp.n_ff
        storage = self._storage_description()
        # the default clamps a 32768-token Gemma 3 down to 4096; say so rather
        # than letting the model look like it has a quarter of its real window
        ctx = (f"ctx {self.n_ctx}" if self.n_ctx >= hp.n_ctx_train
               else f"ctx {self.n_ctx} of {hp.n_ctx_train}")
        attn = ""
        if hp.arch == "gemma3" and hp.sliding_window > 0:
            layout = (f"pattern {sum(hp.sliding_layers)}/{hp.n_layer}"
                      if hp.sliding_layers else f"every {hp.full_attention_period}")
            attn = (f" | swa {hp.sliding_window} ({layout}, {hp.swa_rule}) | "
                    f"attn scale {hp.attention_scale:.4g}"
                    f"{' (metadata)' if hp.attention_scale_from_metadata else ''}")
        return (f"{hp.arch} | {hp.n_layer} layers | embd {hp.n_embd} | "
                f"heads {hp.n_head}/{hp.n_kv} | ff {hp.n_ff} | vocab {hp.n_vocab} | "
                f"~{params / 1e6:.0f}M params | {ctx}{attn} | "
                f"backend {T.backend_name()} | {storage}")

    @staticmethod
    def _size(nbytes: float) -> str:
        mb = nbytes / (1024 * 1024)
        return f"{mb / 1024:.1f} GiB" if mb >= 1024 else f"{mb:.0f} MiB"

    def _storage_description(self) -> str:
        q = self.weight_storage.get("quantized", {})
        dense = int(self.weight_storage.get("dense", 0) or 0)
        fallback = self.weight_storage.get("fallback", {})
        densified = self.weight_storage.get("densified") or []
        q_bytes = self.weight_storage.get("quantized_bytes", 0)
        d_bytes = self.weight_storage.get("dense_bytes", 0)
        # matrix counts alone say nothing about the RAM this actually costs
        if q:
            q_desc = "/".join(q.keys())
            total_q = sum(q.values())
            parts = [f"weights quantized {q_desc} "
                     f"({total_q} matrices, {self._size(q_bytes)})"]
            if dense:
                parts.append(f"dense {dense} ({self._size(d_bytes)})")
        else:
            parts = [f"weights dense ({dense} matrices, {self._size(d_bytes)})"]
        if fallback:
            fb_desc = "/".join(fallback.keys())
            parts.append(f"dense fallback {fb_desc}")
        if densified:
            size = self._size(self.weight_storage.get("densified_bytes", 0))
            parts.append(f"dense budget {len(densified)} matrices ({size})")
        return ", ".join(parts)
