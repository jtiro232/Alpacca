#!/usr/bin/env python3
"""What the stdlib-only tier costs, in bytes per weight and mul-adds per second.

Round 6 Step 6 cites four numbers about the pure-Python tier. This script is
where they come from, so they can be re-derived on another machine rather
than trusted.

    python3 pure_tier_cost.py                       # inner-loop bench only
    python3 pure_tier_cost.py --model some.gguf     # + load cost of a real file
    python3 pure_tier_cost.py --model m.gguf --json out.json

Forces the pure path with ALPACCAROO_PURE=1, so it reports the same thing on
a machine that has NumPy installed as on one that does not. Stdlib only, in
keeping with the tier it measures.

What it answers:

1. Bytes per weight at load. The pure tier dequantizes every quantized
   tensor into Python float lists (model.py builds a QuantMatrix only when
   T.HAS_NUMPY), so a Q4_0 file at 0.655 B/w on disk expands by ~63x. This
   is the number that decides which model classes fit in RAM at all.

2. The interpreter's mul-add ceiling. Decode at this tier is compute-bound
   on the interpreter, not bandwidth-bound, so tokens/s is (mul-adds/s) /
   (weights per token) and a model's feasibility can be predicted from one
   throughput number.

3. What math.sumprod is worth. The inner loop is
   `sum(w * xv for w, xv in zip(row, x))`; math.sumprod (3.12+) is the same
   contraction in C.

4. Whether sumprod is drop-in. It accumulates more carefully than naive
   left-to-right summation, so it is NOT bit-identical. The cross-tier gate
   at smoke.py:2656 is a 1e-3 tolerance rather than exact equality, which is
   why it can still be adopted - but the divergence is real and is counted
   here rather than hand-waved.

MIT License. See LICENSE.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time
from array import array
from pathlib import Path

os.environ["ALPACCAROO_PURE"] = "1"   # must precede the alpaccaroo import

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

HAS_SUMPROD = hasattr(math, "sumprod")


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def inner_loop_bench(cols: int, rows: int, rep: int) -> dict:
    """Matvec inner loop, three ways, on the same data."""
    random.seed(1)
    x = [random.random() for _ in range(cols)]
    W = [[random.random() for _ in range(cols)] for _ in range(rows)]
    xa = array("f", x)
    Wa = [array("f", r) for r in W]
    work = rows * cols

    def timed(fn) -> tuple[float, float]:
        fn()                                    # warm any lazy machinery
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        dt = (time.perf_counter() - t0) / rep
        return dt, work / dt / 1e6

    out: dict = {"cols": cols, "rows": rows, "rep": rep}

    dt, mops = timed(lambda: [sum(w * xv for w, xv in zip(row, x)) for row in W])
    out["current_list_genexpr"] = {"seconds": dt, "m_muladd_per_s": mops}

    if HAS_SUMPROD:
        dt, mops = timed(lambda: [math.sumprod(row, x) for row in W])
        out["sumprod_list"] = {"seconds": dt, "m_muladd_per_s": mops}
        dt, mops = timed(lambda: [math.sumprod(row, xa) for row in Wa])
        out["sumprod_array_f"] = {"seconds": dt, "m_muladd_per_s": mops}
        base = out["current_list_genexpr"]["m_muladd_per_s"]
        out["speedup_sumprod_list"] = out["sumprod_list"]["m_muladd_per_s"] / base
        out["speedup_sumprod_array_f"] = (
            out["sumprod_array_f"]["m_muladd_per_s"] / base)
    else:
        out["sumprod_list"] = None
        out["note"] = f"math.sumprod needs 3.12+; running {sys.version.split()[0]}"
    return out


def sumprod_divergence(trials: int = 2000) -> dict:
    """How often sumprod disagrees with naive summation.

    Not a defect in either - sumprod is the more accurate of the two. It is
    counted because the project compares greedy token ids, and a change in
    summation order can move one.
    """
    if not HAS_SUMPROD:
        return {"available": False}
    random.seed(7)
    differ = 0
    for _ in range(trials):
        n = random.choice([256, 896, 4096])
        a = [random.uniform(-1, 1) * 10 ** random.randint(-3, 3) for _ in range(n)]
        b = [random.uniform(-1, 1) * 10 ** random.randint(-3, 3) for _ in range(n)]
        if sum(p * q for p, q in zip(a, b)) != math.sumprod(a, b):
            differ += 1
    return {"available": True, "trials": trials, "differ": differ,
            "fraction": differ / trials}


def model_load_cost(path: str) -> dict:
    """Bytes per weight the pure tier actually holds for a real GGUF."""
    from alpaccaroo import tensor as T
    from alpaccaroo import profiling as P
    from alpaccaroo.model import Model

    assert not T.HAS_NUMPY, "ALPACCAROO_PURE did not take effect"
    base = _rss_mb()
    t0 = time.perf_counter()
    model = Model.load(path)
    load_s = time.perf_counter() - t0
    peak = _rss_mb()

    paths = P.model_paths(model)
    weights = paths.get("total_weights") or 0
    file_bytes = Path(path).stat().st_size
    resident = (peak - base) * 1024 * 1024
    return {
        "path": path,
        "file_bytes": file_bytes,
        "total_weights": weights,
        "rss_baseline_mb": base,
        "rss_peak_mb": peak,
        "resident_bytes_attributable": resident,
        "bytes_per_weight_resident": (resident / weights) if weights else None,
        "bytes_per_weight_on_disk": (file_bytes / weights) if weights else None,
        "expansion_x": (resident / file_bytes) if file_bytes else None,
        "load_seconds": load_s,
        "backend_summary": paths.get("summary"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="GGUF to load and measure (optional)")
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--rep", type=int, default=20)
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args()

    result: dict = {
        "python": sys.version.split()[0],
        "has_sumprod": HAS_SUMPROD,
        "inner_loop": inner_loop_bench(args.cols, args.rows, args.rep),
        "sumprod_divergence": sumprod_divergence(),
    }

    il = result["inner_loop"]
    print(f"python {result['python']}  math.sumprod: {HAS_SUMPROD}")
    print(f"\ninner loop ({args.rows} rows x {args.cols} cols, {args.rep} reps)")
    print(f"  {'current: sum(genexpr) on lists':<32}"
          f"{il['current_list_genexpr']['m_muladd_per_s']:8.1f} M mul-add/s")
    if HAS_SUMPROD:
        print(f"  {'math.sumprod on lists':<32}"
              f"{il['sumprod_list']['m_muladd_per_s']:8.1f} M mul-add/s"
              f"   ({il['speedup_sumprod_list']:.2f}x)")
        print(f"  {'math.sumprod on array(f)':<32}"
              f"{il['sumprod_array_f']['m_muladd_per_s']:8.1f} M mul-add/s"
              f"   ({il['speedup_sumprod_array_f']:.2f}x)")
        d = result["sumprod_divergence"]
        print(f"\nsumprod vs naive summation: {d['differ']}/{d['trials']} rows differ"
              f"  ({d['fraction']*100:.1f}%) - more accurate, not identical")

    if args.model:
        m = model_load_cost(args.model)
        result["model"] = m
        print(f"\nmodel {m['path']}")
        print(f"  backend            {m['backend_summary']}")
        print(f"  weights            {m['total_weights']:,}")
        print(f"  on disk            {m['bytes_per_weight_on_disk']:.3f} B/weight")
        print(f"  resident (pure)    {m['bytes_per_weight_resident']:.2f} B/weight"
              f"   ({m['expansion_x']:.1f}x expansion)")
        print(f"  load               {m['load_seconds']:.2f} s")
        mops = il["current_list_genexpr"]["m_muladd_per_s"] * 1e6
        print("\n  predicted decode at this tier (weights/token / mul-adds per s):")
        for name, n in (("1B", 1e9), ("3B", 3e9), ("8B", 8.03e9)):
            gb = n * m["bytes_per_weight_resident"] / 1e9
            print(f"    {name:>3}: {n/mops:8.1f} s/token, {gb:8.1f} GB resident")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
