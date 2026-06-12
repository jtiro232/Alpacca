#!/usr/bin/env python3
"""Alpacca acceptance test - pull a real instruct model, ask it a factual
question headlessly, and check the answer.

    python3 tests/acceptance.py                       # llama3.2:1b (default)
    python3 tests/acceptance.py --model NousResearch/Hermes-3-Llama-3.1-8B

The default is a 1B model (~770 MB download, ~2 GB RAM with NumPy now
that quantized weights stay quantized in RAM; ~6 GB with ALPACCA_F32=1).
The 8B Hermes model works with NumPy at roughly 13 GB RAM quantized
(35+ GB with ALPACCA_F32=1). Without NumPy, weights become Python float
objects at ~38 bytes each (measured), so 1B-class models need ~45 GB and
8B-class ~300 GB - impractical; the script checks RAM and warns before
committing. Needs network access to the model source on first run.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QUESTION = "What is Abraham Lincoln's birthday?"
EXPECT = ("february", "1809")


def available_ram_gb() -> float:
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return -1.0  # unknown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--yes", action="store_true", help="skip the RAM confirmation")
    args = ap.parse_args()

    from alpacca import chat, tensor
    from alpacca.model import Model
    from alpacca.pull import pull_model
    from alpacca.sample import SamplerParams
    from alpacca.store import find_local, parse_model_ref

    big = "8b" in args.model.lower() or "7b" in args.model.lower()
    ram = available_ram_gb()
    if tensor.HAS_NUMPY and not os.environ.get("ALPACCA_F32"):
        need = 13 if big else 2    # quantized int8 weight storage
    elif tensor.HAS_NUMPY:
        need = 35 if big else 6    # ALPACCA_F32=1 dense float32 expansion
    else:
        need = 300 if big else 45  # pure python: ~38 bytes per list weight
    if ram >= 0 and ram < need:
        print(f"warning: ~{need} GB RAM recommended for {args.model}, "
              f"only {ram:.1f} GB available", file=sys.stderr)
        if not args.yes:
            print("rerun with --yes to proceed anyway, or use the default 1B model",
                  file=sys.stderr)
            sys.exit(2)
    if not tensor.HAS_NUMPY:
        print("warning: NumPy not installed - generation will be very slow "
              "(pip install numpy)", file=sys.stderr)

    ref = parse_model_ref(args.model)
    local = find_local(ref) or pull_model(ref)
    model = Model.load(str(local.model_path), n_ctx=512)
    print(model.describe(), file=sys.stderr)

    print(f"\nQ: {QUESTION}")
    print("A: ", end="", flush=True)
    res = chat.chat_once(
        model, [{"role": "user", "content": QUESTION}],
        SamplerParams(temperature=0.0), n_predict=96,
        stream=lambda s: print(s, end="", flush=True))
    print(f"\n\n[{res.tokens} tokens, {res.tok_per_sec:.2f} tok/s]", file=sys.stderr)

    answer = res.text.lower()
    if any(e in answer for e in EXPECT):
        print("PASS: the answer mentions Lincoln's birthday (February 12, 1809)")
    else:
        print("FAIL: expected the answer to mention February 12 / 1809")
        sys.exit(1)


if __name__ == "__main__":
    main()
