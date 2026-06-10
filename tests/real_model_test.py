#!/usr/bin/env python3
"""Real-model correctness gate (needs network to huggingface.co).

Downloads TinyLlama-stories (stories15M, ~19 MB Q4_0 GGUF) through
alpacca's own Hugging Face pull path, then runs greedy generation with the
alpacca engine and checks the output is coherent English — which exercises
the GGUF parser, Q4_0 dequantizer, SPM tokenizer, transformer and sampler
against weights trained by someone else.

usage: python3 tests/real_model_test.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL_REF = "hf:ggml-org/models:stories15M-q4_0.gguf"


def main() -> None:
    os.environ.setdefault("ALPACCA_HOME", str(REPO / ".smoke-home"))

    from alpacca import chat
    from alpacca.model import Model
    from alpacca.pull import pull_model
    from alpacca.sample import SamplerParams
    from alpacca.store import parse_model_ref

    local = pull_model(parse_model_ref(MODEL_REF))
    model = Model.load(str(local.model_path), progress=False)
    print(model.describe())

    ids = model.tok.encode("Once upon a time")
    res = chat.generate(model, ids, SamplerParams(temperature=0.0), n_predict=60)
    text = res.text
    print(f"---\nOnce upon a time{text}\n---")
    print(f"{res.tokens} tokens at {res.tok_per_sec:.1f} tok/s")

    words = re.findall(r"[a-zA-Z']+", text)
    common = {"the", "a", "and", "to", "of", "was", "she", "he", "it", "they",
              "her", "his", "in", "there", "day", "little", "wanted", "with"}
    hits = sum(1 for w in words if w.lower() in common)
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")

    assert len(words) >= 15, f"too few words generated: {words}"
    assert hits >= 5, f"output does not look like English: {text!r}"
    assert printable >= len(text) * 0.95, "output contains junk bytes"
    print("PASS: real model generates coherent English through the alpacca engine")


if __name__ == "__main__":
    main()
