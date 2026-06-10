#!/usr/bin/env bash
# Alpacca offline smoke test.
#
# Exercises the full pull→list→show→run→rm cycle against a local mock of
# the Ollama registry and the Hugging Face API, using a tiny random-weight
# GGUF — no network, no real model downloads.
#
# usage: tests/smoke.sh [path-to-bin-dir]   (default: build/bin)
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
bin_dir=${1:-"$repo_root/build/bin"}
alpacca="$bin_dir/alpacca"
[ -x "$alpacca.exe" ] && alpacca="$alpacca.exe"   # Windows (git-bash)
python=$(command -v python3 || command -v python)

[ -x "$alpacca" ] || { echo "FAIL: $alpacca not built"; exit 1; }

tmp=$(mktemp -d)
server_pid=""
cleanup() {
    [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null || true
    rm -rf "$tmp"
}
trap cleanup EXIT

pass=0
check() { # check <label> <command...>
    local label=$1
    shift
    if "$@" > "$tmp/out.log" 2>&1 < /dev/null; then
        echo "ok   $label"
        pass=$((pass + 1))
    else
        echo "FAIL $label"
        sed 's/^/     | /' "$tmp/out.log"
        exit 1
    fi
}

echo "== preparing tiny model and mock registry =="
mkdir -p "$tmp/srv"
"$python" "$repo_root/tests/make_tiny_model.py" "$tmp/srv/model.gguf" > /dev/null
printf '{"temperature": 0.7, "num_ctx": 256, "stop": ["</s>"]}' > "$tmp/srv/params.json"
printf 'You are a smoke test.' > "$tmp/srv/system.txt"
printf 'test license — MIT' > "$tmp/srv/license.txt"

"$python" "$repo_root/tests/mock_registry.py" "$tmp/srv" "$tmp/port" &
server_pid=$!
for _ in $(seq 1 50); do
    [ -s "$tmp/port" ] && break
    sleep 0.1
done
port=$(cat "$tmp/port")
echo "mock registry on 127.0.0.1:$port"

# Env vars are not path-converted by git-bash on Windows; do it ourselves.
winpath() { if command -v cygpath > /dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

export ALPACCA_HOME="$(winpath "$tmp/home")"
export ALPACCA_OLLAMA_REGISTRY="http://127.0.0.1:$port"
export ALPACCA_HF_ENDPOINT="http://127.0.0.1:$port"
export ALPACCA_LLAMA_BIN_DIR="$(winpath "$bin_dir")"

echo "== basics =="
check "help"    "$alpacca" help
check "version" "$alpacca" version
check "doctor"  "$alpacca" doctor

echo "== ollama-registry path =="
check "pull tiny"            "$alpacca" pull tiny
check "pull is idempotent"   "$alpacca" pull tiny
check "list shows tiny"      grep -q '^tiny ' <("$alpacca" list)
check "show has params"      grep -q '"temperature"' <("$alpacca" show tiny)
check "show has system"      grep -q 'smoke test' <("$alpacca" show tiny)
check "license stored"       test -f "$ALPACCA_HOME/models/ollama/library/tiny/latest/license.txt"
check "digest verified"      grep -q '"digest": "sha256:' <("$alpacca" show tiny)

# The tiny model has random weights; a grammar pins its output to clean
# ASCII so llama-cli's response parsing always succeeds.
grammar='root ::= "ok"'

echo "== inference through llama-cli (real exec, tiny weights) =="
check "run one-shot" "$alpacca" run tiny "hello there" --grammar "$grammar" --no-warmup
check "run by path"  "$alpacca" run "$ALPACCA_HOME/models/ollama/library/tiny/latest/model.gguf" \
                     "hi" --grammar "$grammar" --no-warmup

echo "== hugging-face path (incl. -GGUF fallback) =="
check "pull hf:test/tiny"    "$alpacca" pull hf:test/tiny       # falls back to tiny-GGUF
check "pull hf exact quant"  "$alpacca" pull hf:test/tiny-GGUF:model.gguf
check "list shows hf models" grep -q 'hf:test/tiny' <("$alpacca" list)
check "run hf model"         "$alpacca" run hf:test/tiny "hi" --grammar "$grammar" --no-warmup

echo "== tool passthrough with model-name resolution =="
check "tokenize via name" "$alpacca" tokenize -m tiny -p "hello"

echo "== removal =="
check "rm tiny"     "$alpacca" rm tiny
check "rm hf both"  "$alpacca" rm hf:test/tiny hf:test/tiny-GGUF:model.gguf
check "store empty" grep -q 'no models installed' <("$alpacca" list)

echo
echo "all $pass checks passed"
