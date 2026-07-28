# Alpacca - token sampling: greedy, temperature, top-k, top-p, repeat
# penalty. Deterministic for a given seed. MIT License. See LICENSE.
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import tensor as T


@dataclass
class SamplerParams:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    seed: int = -1  # -1 -> random


def _topk_indices(logits: list, k: int) -> list[int]:
    """Indices of the k largest logits, descending, ties broken by lower index.

    Identical output to ``sorted(range(n), key=logits.__getitem__,
    reverse=True)[:k]`` - which is a stable sort, so equal logits keep
    ascending index order - but O(n) instead of O(n log n). Gemma 3's 262144
    -entry vocabulary makes the full sort a measurable share of every token.
    """
    n = len(logits)
    if k >= n:
        return sorted(range(n), key=logits.__getitem__, reverse=True)
    if T.HAS_NUMPY:
        import numpy as np

        arr = np.asarray(logits, dtype=np.float64)
        if np.isnan(arr).any():
            # NaN sorts as the largest value for argpartition but compares
            # False against everything, so the masks below would select
            # nothing and hand the caller an empty list. Infinities are fine.
            return sorted(range(n), key=logits.__getitem__, reverse=True)[:k]
        # argpartition alone is not enough: when logits tie across the cut it
        # keeps an arbitrary one, where the stable sort keeps the lowest index.
        # So take everything strictly above the k-th value, then fill from the
        # tied indices in ascending order.
        thr = arr[np.argpartition(arr, n - k)[n - k:]].min()
        above = np.flatnonzero(arr > thr)
        ties = np.flatnonzero(arr == thr)[:k - above.size]
        sel = np.concatenate((above, ties))
        order = np.lexsort((sel, -arr[sel]))  # last key is primary
        return [int(i) for i in sel[order]]
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
        p = self.params
        logits = T.to_list(logits)

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
