#!/bin/bash
# Package G benchmark matrix, run STRICTLY serially - two benchmarks sharing
# this CPU would measure each other.
set -u
cd /home/ubuntu/alpaccaroo
A=prompts/05-artifacts/ryzen7-7730u-ubuntu
BIN=/home/ubuntu/alpaccaroo-venv/bin/alpaccaroo
ALL=short-short,short-long,long-short,long-long,default

echo "######## qwen05bs (0.5B Q4_K_S) - full shape x ctx sweep"
time $BIN bench --model qwen05bs --shapes $ALL --ctx 2048,4096,8192 \
    --repeat 3 --quiet --json $A/G-qwen05bs.json

echo "######## llama1b (1B Q8_0) - shape axis at ctx 4096"
time $BIN bench --model llama1b --shapes $ALL --ctx 4096 \
    --repeat 2 --quiet --json $A/G-llama1b-shapes.json

echo "######## llama1b - context axis at long-long"
time $BIN bench --model llama1b --shapes long-long --ctx 2048,4096,8192 \
    --repeat 2 --quiet --json $A/G-llama1b-ctx.json

echo "######## qwen3bmed (3B Q4_K_M) - shape axis at ctx 4096"
time $BIN bench --model qwen3bmed --shapes $ALL --ctx 4096 \
    --repeat 2 --quiet --json $A/G-qwen3bmed-shapes.json

echo "######## qwen3bmed - context axis at long-long"
time $BIN bench --model qwen3bmed --shapes long-long --ctx 2048,4096,8192 \
    --repeat 2 --quiet --json $A/G-qwen3bmed-ctx.json

echo "######## qwen3bmed - decode profile (round-4 comparable)"
time $BIN bench --model qwen3bmed --shapes default --ctx 4096 --repeat 1 \
    --quiet --profile --profile-json $A/G-qwen3bmed-profile.json

echo "ALL DONE"
