# Alpacca - token sampling: greedy, temperature, top-k, top-p, repeat
# penalty. Deterministic for a given seed. MIT License. See LICENSE.
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import tensor as T

if T.HAS_NUMPY:
    import numpy as _np


@dataclass
class SamplerParams:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    seed: int = -1  # -1 -> random


def _topk_order(arr, k: int):
    """Indices of the k largest entries of a float64 array, descending,
    ties broken by lower index. Returns None when the array contains NaN;
    the caller falls back to the stable Python sort whose NaN behaviour
    the tests pin down. Everything here is O(n) or O(k log k) NumPy - the
    old list-based path converted 128k-262k logits to Python floats every
    token, which cost more than the whole top-k selection."""
    n = arr.size
    if _np.isnan(arr).any():
        # NaN sorts as the largest value for argpartition but compares
        # False against everything, so the masks below would select
        # nothing and hand the caller an empty list. Infinities are fine.
        return None
    if k >= n:
        # full descending order; the secondary arange key reproduces the
        # stable sort's ascending-index tie-break
        return _np.lexsort((_np.arange(n), -arr))
    # argpartition alone is not enough: when logits tie across the cut it
    # keeps an arbitrary one, where the stable sort keeps the lowest index.
    # So take everything strictly above the k-th value, then fill from the
    # tied indices in ascending order.
    thr = arr[_np.argpartition(arr, n - k)[n - k:]].min()
    above = _np.flatnonzero(arr > thr)
    ties = _np.flatnonzero(arr == thr)[:k - above.size]
    sel = _np.concatenate((above, ties))
    order = _np.lexsort((sel, -arr[sel]))  # last key is primary
    return sel[order]


def _topk_indices(logits: list, k: int) -> list[int]:
    """Indices of the k largest logits, descending, ties broken by lower index.

    Identical output to ``sorted(range(n), key=logits.__getitem__,
    reverse=True)[:k]`` - which is a stable sort, so equal logits keep
    ascending index order - but O(n) instead of O(n log n). Gemma 3's 262144
    -entry vocabulary makes the full sort a measurable share of every token.
    """
    n = len(logits)
    if T.HAS_NUMPY:
        order = _topk_order(_np.asarray(logits, dtype=_np.float64), min(k, n))
        if order is None:
            return sorted(range(n), key=logits.__getitem__, reverse=True)[:k]
        return [int(i) for i in order]
    if k >= n:
        return sorted(range(n), key=logits.__getitem__, reverse=True)
    import heapq

    top = heapq.nlargest(k, range(n), key=lambda i: (logits[i], -i))
    return top


@dataclass
class Sampler:
    params: SamplerParams = field(default_factory=SamplerParams)

    def __post_init__(self):
        seed = self.params.seed
        self.rng = random.Random(seed if seed >= 0 else None)
        self.recent: list[int] = []

    def accept(self, token: int) -> None:
        self.recent.append(token)
        if len(self.recent) > max(self.params.repeat_last_n, 1):
            self.recent.pop(0)

    def sample(self, logits) -> int:
        if T.HAS_NUMPY:
            # np.array always copies, so the penalty below cannot mutate the
            # caller's logits; float64 matches the Python-float arithmetic of
            # the list path bit for bit
            arr = _np.array(logits, dtype=_np.float64)
            # a NaN anywhere, or a +/-inf *maximum*, drives the softmax
            # through inf-inf=NaN and the two paths' NaN comparisons differ;
            # np.max propagates NaN, so one finiteness test covers both and
            # the degenerate vectors keep the historical list behaviour.
            # -inf among finite logits stays on the fast path: exp(-inf)=0.
            if arr.size and bool(_np.isfinite(arr.max())):
                return self._sample_array(arr)
        return self._sample_list(T.to_list(logits))

    def _sample_array(self, arr) -> int:
        """NumPy sampling path. Same arithmetic as _sample_list in the same
        order (cumsum accumulates sequentially, like the running Python
        sums), so a given seed picks the same token; measured 0.5 ms vs
        6.6 ms per token on a 128256 vocabulary, and a request asking for
        top_k=0 costs a lexsort instead of a full-vocab Python sort."""
        p = self.params

        if p.repeat_penalty and p.repeat_penalty != 1.0 and self.recent:
            for t in set(self.recent):
                v = arr[t]
                arr[t] = v / p.repeat_penalty if v > 0 else v * p.repeat_penalty

        if p.temperature <= 0:
            return int(_np.argmax(arr))

        k = p.top_k if p.top_k and p.top_k > 0 else arr.size
        idx = _topk_order(arr, min(k, arr.size))

        maxl = arr[idx[0]]
        weights = _np.exp((arr[idx] - maxl) / p.temperature)
        cum = _np.cumsum(weights)
        probs = weights / cum[-1]

        if 0.0 < p.top_p < 1.0:
            cp = _np.cumsum(probs)
            # first index whose running sum reaches top_p, inclusive - and
            # like the list path, renormalize even when nothing was cut
            cut = min(int(_np.searchsorted(cp, p.top_p, side="left")) + 1,
                      idx.size)
            idx = idx[:cut]
            probs = probs[:cut] / cp[cut - 1]

        r = self.rng.random()
        j = int(_np.searchsorted(_np.cumsum(probs), r, side="left"))
        if j >= idx.size:
            j = idx.size - 1  # r beyond the last running sum: keep last
        return int(idx[j])

    def _sample_list(self, logits: list) -> int:
        """Reference implementation over Python lists: the pure-stdlib path,
        and the fallback for NaN logits (whose comparison quirks the tests
        pin down)."""
        p = self.params

        if p.repeat_penalty and p.repeat_penalty != 1.0 and self.recent:
            for t in set(self.recent):
                v = logits[t]
                logits[t] = v / p.repeat_penalty if v > 0 else v * p.repeat_penalty

        if p.temperature <= 0:
            return max(range(len(logits)), key=logits.__getitem__)

        # work on the top-k slice only (huge speedup for big vocabs)
        k = p.top_k if p.top_k and p.top_k > 0 else len(logits)
        k = min(k, len(logits))
        idx = _topk_indices(logits, k)

        maxl = logits[idx[0]]
        weights = [math.exp((logits[i] - maxl) / p.temperature) for i in idx]
        total = sum(weights)
        probs = [w / total for w in weights]

        if 0.0 < p.top_p < 1.0:
            acc = 0.0
            cut = len(probs)
            for n, pr in enumerate(probs):
                acc += pr
                if acc >= p.top_p:
                    cut = n + 1
                    break
            idx, probs = idx[:cut], probs[:cut]
            total = sum(probs)
            probs = [pr / total for pr in probs]

        r = self.rng.random()
        acc = 0.0
        for i, pr in zip(idx, probs):
            acc += pr
            if r <= acc:
                return i
        return idx[-1]
