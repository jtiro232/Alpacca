"""End-to-end paired A/B for decode, in one process and one KV state.

This is round 4's method (prompts/04-RESULTS.md NEGATIVE 1), reused rather
than reinvented: ABBA round-robin so a wandering clock cancels, a sign test
on rounds won rather than the median alone, and the greedy token ids
compared as well as the times so a "win" that changed the output is caught.

The timed region is the DECODE loop only. Each round resets the model and
re-prefills the same prompt untimed, so every round does identical work and
a growing KV cache cannot drift the comparison.

    python e2e_ab.py <model> <experiment> [rounds] [decode_tokens]
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/home/ubuntu/alpaccaroo")

from alpaccaroo import bench as B          # noqa: E402
from alpaccaroo import kernels as K        # noqa: E402
from alpaccaroo.sample import Sampler, SamplerParams   # noqa: E402


def _apply(env: dict, threads=None) -> None:
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    if threads is not None:
        K.set_threads(threads)


def one_round(model, prompt, n_decode, seed=1234):
    """Untimed reset+prefill, then a timed greedy decode. Returns (s, ids)."""
    model.reset()
    if getattr(model, "_prefix_slots", None):
        model._prefix_slots.clear()
        model._prefix_bytes = 0
    sampler = Sampler(SamplerParams(temperature=0.0, seed=seed))
    for tid in prompt:
        sampler.accept(tid)
    logits = model.prefill(prompt)

    ids = []
    t0 = time.perf_counter()
    for _ in range(n_decode):
        tid = sampler.sample(logits)
        sampler.accept(tid)
        ids.append(int(tid))
        logits = model.forward(tid)
    return time.perf_counter() - t0, ids


def paired(model, variants, rounds, n_decode, prefill_tokens):
    """variants: [(label, env_dict, threads_or_None)]. First is the baseline."""
    prompt = B.synthetic_prompt(model, prefill_tokens)
    labels = [v[0] for v in variants]
    samples = {l: [] for l in labels}
    tokens = {}

    for _lbl, env, thr in variants:          # warmup, one round each
        _apply(env, thr)
        one_round(model, prompt, min(4, n_decode))

    order = list(range(len(variants)))
    for r in range(rounds):
        for i in (order if r % 2 == 0 else order[::-1]):
            lbl, env, thr = variants[i]
            _apply(env, thr)
            s, ids = one_round(model, prompt, n_decode)
            samples[lbl].append(s)
            tokens.setdefault(lbl, ids)
        print(f"  round {r + 1}/{rounds}: "
              + "  ".join(f"{l} {samples[l][-1] * 1e3:8.1f} ms"
                          for l in labels), flush=True)

    def stats(xs):
        s = sorted(xs)
        return {"min": s[0], "median": s[len(s) // 2], "max": s[-1],
                "mean": sum(s) / len(s)}

    base = labels[0]
    out = {"rounds": rounds, "decode_tokens": n_decode,
           "prefill_tokens": prefill_tokens,
           "variants": {l: stats(samples[l]) for l in labels},
           "baseline": base, "ratios": {}}
    for lbl in labels[1:]:
        pr = [b / a for a, b in zip(samples[base], samples[lbl])]
        srt = sorted(pr)
        out["ratios"][lbl] = {
            "median_vs_baseline": srt[len(srt) // 2],
            "ratio_of_medians": (stats(samples[lbl])["median"]
                                 / stats(samples[base])["median"]),
            "best_vs_baseline": srt[0], "worst_vs_baseline": srt[-1],
            "rounds_won": sum(1 for v in pr if v < 1.0), "rounds": len(pr),
        }
        out["ratios"][lbl]["tokens_identical"] = (tokens[lbl] == tokens[base])
    out["tokens_identical_all"] = all(tokens[l] == tokens[base]
                                      for l in labels)
    out["token_ids"] = tokens[base][:24]
    return out


EXPERIMENTS = {
    # Package J: does the narrow-matrix dispatch pay end to end on a model
    # whose shapes actually cross the threshold?
    "narrow": lambda: [
        ("narrow-off", {"ALPACCAROO_SERIAL_MATVEC_ELEMS": "0"}, None),
        ("narrow-on-131072", {"ALPACCAROO_SERIAL_MATVEC_ELEMS": "131072"}, None),
    ],
    # Package H1: the knob that won 6-10% per call and lost 22/25 end to end.
    "quantize": lambda: [
        ("quantize-off", {"ALPACCAROO_SERIAL_QUANTIZE_COLS": "0"}, None),
        ("quantize-8192", {"ALPACCAROO_SERIAL_QUANTIZE_COLS": "8192"}, None),
    ],
    # Package N: the tuner picks 16 logical threads over 8 physical cores.
    "threads": lambda: [
        ("threads-8-physical", {}, 8),
        ("threads-16-smt", {}, 16),
    ],
    "threads4": lambda: [
        ("threads-8-physical", {}, 8),
        ("threads-4", {}, 4),
    ],
    # Package K / D: the grouped kernel, on whichever pairing the file has.
    "group": lambda: [
        ("group-off", {"ALPACCAROO_GROUP_KERNEL": "0"}, None),
        ("group-on", {"ALPACCAROO_GROUP_KERNEL": None}, None),
    ],
}


def main() -> int:
    ref = sys.argv[1]
    exp = sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    n_decode = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    prefill = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    out_path = sys.argv[6] if len(sys.argv) > 6 else None

    from alpaccaroo.model import Model
    ctx = int(os.environ.get("AB_CTX", "4096"))
    path, display = B.resolve_model(ref)
    model = Model.load(str(path), n_ctx=ctx, progress=False)
    print(f"model {display}  ctx {ctx}  kernels threads={K.threads()}  "
          f"experiment={exp}  rounds={rounds}  decode={n_decode}",
          flush=True)
    res = paired(model, EXPERIMENTS[exp](), rounds, n_decode, prefill)
    res["model"] = ref
    res["experiment"] = exp

    print("\n" + "=" * 72)
    for lbl, st in res["variants"].items():
        print(f"{lbl:<24} median {st['median'] * 1e3:9.1f} ms  "
              f"min {st['min'] * 1e3:9.1f}  max {st['max'] * 1e3:9.1f}"
              f"   ({n_decode / st['median']:6.2f} tok/s)")
    for lbl, r in res["ratios"].items():
        print(f"\n{lbl} vs {res['baseline']}:")
        print(f"  median of per-round ratios : {r['median_vs_baseline']:.4f}"
              "   (<1 means faster)")
        print(f"  ratio of medians           : {r['ratio_of_medians']:.4f}")
        print(f"  rounds won                 : {r['rounds_won']}/{r['rounds']}"
              "   (near half = noise)")
        print(f"  greedy tokens identical    : {r['tokens_identical']}")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
