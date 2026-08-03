"""Condense `alpaccaroo bench --json` files into the markdown the log wants.

Cold and warm are never merged: they are different workloads and the log
quotes them separately.
"""
import json
import sys
from statistics import median


def load_records(paths):
    rows = []
    env = None
    for p in paths:
        d = json.loads(open(p, encoding="utf-8").read())
        env = env or d.get("runtime")
        for r in d["records"]:
            rows.append(r)
    return rows, env


def main():
    rows, env = load_records(sys.argv[1:])
    key = lambda r: (r["model"], r["n_ctx"], r["shape"], r["phase"])
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    print(f"{'model':<26} {'ctx':>5} {'shape':<12} {'phase':<5} "
          f"{'prefill t/s':>11} {'decode t/s':>10} {'p50 ms':>8} "
          f"{'p95 ms':>8} {'TTFT s':>7} {'load s':>7} {'RSS MiB':>8}")
    for k in sorted(groups):
        g = groups[k]
        m, ctx, shape, phase = k
        pf = median([r["prefill_tok_per_s"] for r in g])
        dc = median([r["decode_tok_per_s"] for r in g])
        p50 = median([r["token_latency_seconds"]["p50"] for r in g]) * 1e3
        p95 = median([r["token_latency_seconds"]["p95"] for r in g]) * 1e3
        ttft = median([r["time_to_first_token_seconds"] for r in g])
        load_s = g[0].get("load_seconds")
        rss = median([r["peak_rss_mb"] for r in g])
        print(f"{m[:26]:<26} {ctx:>5} {shape:<12} {phase:<5} "
              f"{pf:>11.2f} {dc:>10.2f} {p50:>8.1f} {p95:>8.1f} "
              f"{ttft:>7.2f} {('-' if load_s is None else f'{load_s:.2f}'):>7} "
              f"{rss:>8.0f}")
    if rows:
        print(f"\nstorage: {rows[0]['storage']}")
        print(f"paths:   {rows[0]['paths']}")
    if env:
        k = env.get("kernels", {})
        print(f"kernels: {k}")


if __name__ == "__main__":
    main()
