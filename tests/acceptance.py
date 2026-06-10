#!/usr/bin/env python3
"""Alpacca acceptance test — pull a real instruct model, ask it a factual
question headlessly, and check the answer.

    python3 tests/acceptance.py                       # llama3.2:1b (default)
    python3 tests/acceptance.py --model NousResearch/Hermes-3-Llama-3.1-8B

The default is a 1B model (~770 MB download, ~6 GB RAM with NumPy).
The 8B Hermes model works but the pure-Python engine keeps weights in
float32: expect a ~4.6 GB download, 35+ GB of RAM, and slow generation
without NumPy — the script checks RAM and warns before committing.
Needs network access to the model source on first run.
"""
from __future__ import annotations

import argparse
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
    need = 35 if big else 6
    if ram >= 0 and ram < need:
        print(f"warning: ~{need} GB RAM recommended for {args.model}, "
              f"only {ram:.1f} GB available", file=sys.stderr)
        if not args.yes:
            print("rerun with --yes to proceed anyway, or use the default 1B model",
                  file=sys.stderr)
            sys.exit(2)
    if not tensor.HAS_NUMPY:
        print("warning: NumPy not installed — generation will be very slow "
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
