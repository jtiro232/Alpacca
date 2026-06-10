#!/usr/bin/env bash
# Alpacca acceptance test — real model, real question, headless.
#
#   1. pulls NousResearch/Hermes-3-Llama-3.1-8B
#      (alpacca falls back to the GGUF sibling repo
#       NousResearch/Hermes-3-Llama-3.1-8B-GGUF and picks Q4_K_M, ≈4.7 GB)
#   2. asks it Abraham Lincoln's birthday in a one-shot, non-interactive run
#   3. passes if the answer mentions February 12 / 1809
#
# Needs network access to huggingface.co, ~5 GB disk, and enough RAM for an
# 8B Q4_K_M model (≈6 GB).
#
# usage: tests/hermes_acceptance.sh [path-to-bin-dir]   (default: build/bin)
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bin_dir=${1:-"$repo_root/build/bin"}
alpacca="$bin_dir/alpacca"

model="NousResearch/Hermes-3-Llama-3.1-8B"
question="What is Abraham Lincoln's birthday?"

[ -x "$alpacca" ] || { echo "FAIL: $alpacca not built"; exit 1; }

echo "== pulling $model =="
"$alpacca" pull "$model"

echo
echo "== headless one-shot: \"$question\" =="
out=$("$alpacca" run "$model" "$question" --no-warmup < /dev/null 2>&1 | tee /dev/stderr)

echo
if printf '%s' "$out" | grep -qiE 'february[^0-9]{0,3}12|1809'; then
    echo "PASS: the model answered with Lincoln's birthday (February 12, 1809)"
else
    echo "FAIL: expected the answer to mention February 12 / 1809"
    exit 1
fi
