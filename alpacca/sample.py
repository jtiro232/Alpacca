# Alpacca — token sampling: greedy, temperature, top-k, top-p, repeat
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
    seed: int = -1  # -1 → random


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
        idx = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:k]

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
