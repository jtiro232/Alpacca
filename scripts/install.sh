#!/usr/bin/env sh
# Alpacca installer (Linux / macOS).
#
# Run from a clone:        scripts/install.sh
# Or bootstrap directly:   curl -fsSL https://raw.githubusercontent.com/jtiro232/Alpacca/main/scripts/install.sh | sh
#
# What it does: fetch sources (when bootstrapping), build llama.cpp +
# alpacca, install to $PREFIX (default ~/.local), and make sure `alpacca`
# is on your PATH so you can just type `alpacca` in a new terminal.
#
# knobs:
#   PREFIX=/usr/local            install location (default ~/.local)
#   CMAKE_FLAGS="-DGGML_CUDA=ON" extra llama.cpp/ggml build flags
#   NO_MODIFY_PATH=1             don't touch shell rc files
set -eu

repo_url="https://github.com/jtiro232/Alpacca"
prefix=${PREFIX:-"$HOME/.local"}
jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

for tool in git cmake; do
    command -v "$tool" > /dev/null 2>&1 || {
        echo "error: '$tool' is required. install it first (apt/dnf/brew install $tool)." >&2
        exit 1
    }
done

# Find the sources: next to this script when run from a clone, otherwise
# clone into ~/.alpacca/src (bootstrap mode, e.g. curl | sh).
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd) || script_dir=""
if [ -n "$script_dir" ] && [ -f "$script_dir/../CMakeLists.txt" ]; then
    src=$(CDPATH= cd -- "$script_dir/.." && pwd)
else
    src="$HOME/.alpacca/src"
    if [ -d "$src/.git" ]; then
        echo "==> updating sources in $src"
        git -C "$src" pull --ff-only
    else
        echo "==> cloning $repo_url into $src"
        mkdir -p "$(dirname "$src")"
        git clone --depth 1 "$repo_url" "$src"
    fi
fi

echo "==> fetching pinned llama.cpp submodule"
git -C "$src" submodule update --init --depth 1

echo "==> configuring (prefix: $prefix)"
# shellcheck disable=SC2086  # CMAKE_FLAGS is intentionally word-split
cmake -S "$src" -B "$src/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    ${CMAKE_FLAGS:-}

echo "==> building with $jobs jobs (llama.cpp takes a few minutes)"
cmake --build "$src/build" --parallel "$jobs"

echo "==> installing to $prefix"
cmake --install "$src/build"

# Make `alpacca` reachable from a fresh terminal.
bin_dir="$prefix/bin"
case ":$PATH:" in
    *":$bin_dir:"*) on_path=1 ;;
    *)              on_path=0 ;;
esac
if [ "$on_path" -eq 0 ] && [ -z "${NO_MODIFY_PATH:-}" ]; then
    path_line="export PATH=\"$bin_dir:\$PATH\""
    updated=""
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        [ -f "$rc" ] || continue
        grep -qF "$path_line" "$rc" 2>/dev/null && continue
        printf '\n# added by the alpacca installer\n%s\n' "$path_line" >> "$rc"
        updated="$updated $rc"
    done
    if [ -n "$updated" ]; then
        echo "==> added $bin_dir to PATH in:$updated"
    else
        echo "==> add this to your shell profile to finish:"
        echo "      $path_line"
    fi
fi

echo
echo "done. open a NEW terminal and try:"
echo "  alpacca doctor"
echo "  alpacca run llama3.2:1b \"hello!\""
