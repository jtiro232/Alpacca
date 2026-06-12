#!/usr/bin/env python3
"""Small repeatable benchmark for Alpacca inference.

The benchmark intentionally uses only the standard library plus this repo. It
loads a local GGUF path or Alpacca model reference, builds a deterministic
synthetic prompt, then times prompt prefill and greedy decode separately.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _resolve_model(ref: str) -> Path:
    p = Path(ref).expanduser()
    if p.exists():
        return p
    from alpacca.pull import pull_model
    from alpacca.store import find_local, parse_model_ref

    model_ref = parse_model_ref(ref)
    local = find_local(model_ref)
    if local is None:
        local = pull_model(model_ref)
    return local.model_path


def _rss_mb() -> float | None:
    if os.name == "nt":
        return None
    try:
        import resource
    except Exception:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _synthetic_prompt(model, n_tokens: int) -> list[int]:
    seed = (
        "Once upon a time there was a small alpaca who liked clear Python. "
        "The model reads the same sentence again for a deterministic prompt. "
    )
    ids = model.tok.encode(seed, add_bos=True)
    if not ids and model.tok.bos_id >= 0:
        ids = [model.tok.bos_id]
    if not ids:
        raise ValueError("synthetic prompt produced no tokens")
    reps = (n_tokens + len(ids) - 1) // len(ids)
    return (ids * reps)[:n_tokens]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local GGUF path or Alpacca model ref")
    ap.add_argument("--prefill", type=int, default=512)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from alpacca import tensor as T
    from alpacca.model import Model
    from alpacca.sample import Sampler, SamplerParams

    path = _resolve_model(args.model)
    model = Model.load(str(path), n_ctx=args.ctx, progress=False)
    prompt = _synthetic_prompt(model, args.prefill)

    sampler = Sampler(SamplerParams(temperature=0.0, seed=args.seed))
    for tid in prompt:
        sampler.accept(tid)

    t0 = time.perf_counter()
    logits = model.prefill(prompt)
    prefill_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded = 0
    while decoded < args.decode and model.n_past < model.n_ctx:
        tid = sampler.sample(logits)
        sampler.accept(tid)
        if model.tok.is_eog(tid):
            break
        logits = model.forward(tid)
        decoded += 1
    decode_s = time.perf_counter() - t0

    prefill_tps = len(prompt) / prefill_s if prefill_s > 0 else 0.0
    decode_tps = decoded / decode_s if decode_s > 0 else 0.0
    rss = _rss_mb()
    rss_text = "n/a" if rss is None else f"{rss:.1f}"

    print(f"model: {path}")
    print(f"backend: {T.backend_name()}")
    print(f"load seconds: {getattr(model, 'load_seconds', 0.0):.3f}")
    print(f"prefill: {len(prompt)} tokens in {prefill_s:.3f}s = {prefill_tps:.2f} tok/s")
    print(f"decode: {decoded} tokens in {decode_s:.3f}s = {decode_tps:.2f} tok/s")
    print(f"rss mb: {rss_text}")
    print(
        "BENCH "
        f"model={path.name} backend={T.backend_name()} "
        f"load_s={getattr(model, 'load_seconds', 0.0):.3f} "
        f"prefill_tps={prefill_tps:.3f} decode_tps={decode_tps:.3f} "
        f"rss_mb={rss_text}"
    )


if __name__ == "__main__":
    main()
