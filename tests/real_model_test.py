#!/usr/bin/env python3
"""Real-model correctness gate (needs network to huggingface.co).

Downloads TinyLlama-stories (stories15M, ~19 MB Q4_0 GGUF) through
alpaccaroo's own Hugging Face pull path, then runs greedy generation with the
alpaccaroo engine and checks the output is coherent English - which exercises
the GGUF parser, Q4_0 dequantizer, SPM tokenizer, transformer and sampler
against weights trained by someone else.

It also pins golden token-id vectors for the SPM tokenizer against a
third-party vocabulary. Fixtures cannot catch a whole-vocabulary
segmentation regression - the greedy-merge bug this guards against passed
every fixture test in the suite - so the ids below are checked against a
real 32000-piece Llama vocabulary instead.

A second, larger gate runs only when a Gemma 3 GGUF is available locally:
set ALPACCAROO_GEMMA3_GGUF to its path, or leave a Gemma 3 model installed
under ~/.alpaccaroo. It is skipped (loudly) in CI, where an 800 MB download is
not worth it. Its vectors were generated from the authoritative
sentencepiece proto for gemma-3-1b-it (model_type=BPE, 262144 pieces), not
from alpaccaroo itself.

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

# Canonical Llama SPM segmentation: each of these words is ONE piece in the
# 32000-entry vocabulary. The unigram-Viterbi bug that shipped green split
# words like these into several shorter, lower-id pieces, so a golden vector
# over a real vocabulary is the check that would have caught it.
SPM_GOLDEN = {
    "Once upon a time": [9038, 2501, 263, 931],
    "hello world": [22172, 3186],
    "the capital of France": [278, 7483, 310, 3444],
}

# Generated from the authoritative sentencepiece proto (unsloth/gemma-3-1b-it
# tokenizer.model, trainer_spec.model_type = BPE), which encodes these
# identically to alpaccaroo. Gemma stores merge *ranks* in tokenizer.ggml.scores
# rather than log-probabilities, so this pins the other score convention.
GEMMA3_GOLDEN = {
    "user": [2364],
    "model": [4368],
    "What is the capital of France?": [3689, 563, 506, 5279, 529, 7001, 236881],
    "Write a short poem about the ocean.":
        [6974, 496, 2822, 27355, 1003, 506, 12461, 236761],
}


def check_spm_golden(model) -> None:
    """Pin the SPM segmentation of a real third-party vocabulary."""
    for text, want in SPM_GOLDEN.items():
        got = model.tok.encode(text, add_bos=False)
        assert got == want, (
            f"SPM golden vector changed for {text!r}: expected {want}, got "
            f"{got}. A longer id list usually means the merge pass regressed "
            f"to per-character or unigram segmentation.")
        # SPM's add_space_prefix puts a "▁" on the first piece, so the
        # detokenized string carries one leading space back out
        assert model.tok.decode(got).lstrip(" ") == text, (
            f"golden ids for {text!r} do not round-trip: "
            f"{model.tok.decode(got)!r}")
    print(f"PASS: SPM golden vectors ({len(SPM_GOLDEN)} strings) on a real "
          f"{len(model.tok.pieces)}-piece vocabulary")


def _find_gemma3_gguf() -> "Path | None":
    explicit = os.environ.get("ALPACCAROO_GEMMA3_GGUF")
    if explicit:
        return Path(explicit) if Path(explicit).is_file() else None
    from alpaccaroo.gguf import GGUFFile
    root = Path.home() / ".alpaccaroo" / "models"
    if not root.is_dir():
        return None
    for path in sorted(root.rglob("*.gguf")):
        try:
            with GGUFFile.open(path) as gf:
                if gf.metadata.get("general.architecture") == "gemma3":
                    return path
        except Exception:
            continue
    return None


def check_gemma3_golden() -> None:
    """Same gate for a rank-scored (Gemma 3) vocabulary, when one is here."""
    path = _find_gemma3_gguf()
    if path is None:
        print("SKIP: no Gemma 3 GGUF found (set ALPACCAROO_GEMMA3_GGUF to run "
              "the rank-scored SPM golden gate)")
        return
    from alpaccaroo.gguf import GGUFFile
    from alpaccaroo.tokenizer import Tokenizer
    # metadata only: no weights are read, so this costs a fraction of a second
    with GGUFFile.open(path) as gf:
        tok = Tokenizer.from_gguf(gf.metadata)
    for text, want in GEMMA3_GOLDEN.items():
        got = tok.encode(text, add_bos=False)
        assert got == want, (
            f"Gemma 3 golden vector changed for {text!r}: expected {want}, "
            f"got {got}")
    # text that merely contains a control token must not be able to forge one
    forged = tok.encode("<end_of_turn>", add_bos=False)
    assert len(forged) > 1, (
        f"<end_of_turn> in ordinary text became a single control token "
        f"{forged}; parse_special must stay opt-in")
    assert tok.encode("<end_of_turn>", add_bos=False, parse_special=True) != forged, (
        "parse_special=True did not split the control token")
    print(f"PASS: Gemma 3 golden vectors ({len(GEMMA3_GOLDEN)} strings) on "
          f"{path.name}")


def main() -> None:
    os.environ.setdefault("ALPACCAROO_HOME", str(REPO / ".smoke-home"))

    from alpaccaroo import chat
    from alpaccaroo.model import Model
    from alpaccaroo.pull import pull_model
    from alpaccaroo.sample import SamplerParams
    from alpaccaroo.store import parse_model_ref

    local = pull_model(parse_model_ref(MODEL_REF))
    model = Model.load(str(local.model_path), progress=False)
    print(model.describe())
    assert model.weight_storage["quantized"], (
        f"real-model gate did not exercise quantized matrix storage: "
        f"{model.weight_storage}")

    check_spm_golden(model)
    check_gemma3_golden()

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
    print("PASS: real model generates coherent English through the alpaccaroo engine")


if __name__ == "__main__":
    main()
