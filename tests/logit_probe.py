#!/usr/bin/env python3
"""Print the logits after one prefill, as numbers rather than as the token
they happen to produce.

Written to pin down the open defect in `docs/PERFORMANCE.md` section 7.0: a
cold Numba cache and a warm one disagree. Comparing generated text cannot
tell "the arithmetic changed" from "a near-tie broke the other way", and
greedy decoding turns a single flipped logit into a completely different
sentence. The checksum does tell them apart.

usage:
    python3 tests/logit_probe.py <model.gguf> [--ctx N]

    rm -f alpaccaroo/__pycache__/*.nbi alpaccaroo/__pycache__/*.nbc
    python3 tests/logit_probe.py model.gguf   # cold
    python3 tests/logit_probe.py model.gguf   # warm - should match, does not
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# A fixed token list, not an encoded prompt: this probe compares one build
# against another on identical input, and hardcoding the ids keeps the
# tokenizer out of the comparison entirely.
FIXED_IDS = [74785, 697, 259, 90724, 13, 3555, 374, 432, 1075, 30]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=512)
    args = ap.parse_args()

    from alpaccaroo.model import Model
    from alpaccaroo import tensor as T
    if not T.HAS_NUMPY:
        print("needs NumPy: the pure path has nothing to compare against",
              file=sys.stderr)
        return 2
    import numpy as np

    m = Model.load(args.model, n_ctx=args.ctx, progress=False)
    logits = np.asarray(m.prefill(FIXED_IDS), dtype=np.float64)
    top = np.argsort(logits)[::-1][:5]

    print(f"backend         {T.backend_detail()}")
    print(f"prefill tokens  {len(FIXED_IDS)}   vocab {logits.shape[0]}")
    for i, t in enumerate(top):
        print(f"  {i}: id={int(t):6d} logit={logits[t]:.9f}")
    # the gap is what decides a greedy token; the checksum is what tells you
    # the arithmetic moved even when the gap did not
    print(f"gap(top1-top2)  {logits[top[0]] - logits[top[1]]:.9f}")
    print(f"checksum(f64)   {logits.sum():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
