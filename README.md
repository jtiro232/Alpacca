# Alpacca 🦙

**llama.cpp in your terminal, with Ollama-style model management — and zero
overhead on inference.**

Alpacca is a terminal LLM tool built directly on
[llama.cpp](https://github.com/ggml-org/llama.cpp). It gives you the
conveniences people love from [Ollama](https://github.com/ollama/ollama) —
`pull` a model by name, `run` it, `list` what you have, auto-download on
first use — while inference itself is the stock llama.cpp binaries,
untouched and at full speed.

```text
$ alpacca pull llama3.2:1b           # straight from the Ollama registry
$ alpacca run llama3.2:1b            # interactive chat
$ alpacca run llama3.2:1b "why is the sky blue?"
$ alpacca serve llama3.2:1b          # OpenAI-compatible API on :8080
```

## How it stays fast

Alpacca never sits between you and the model. The `alpacca` binary resolves
the model name, builds the right command line, then **execs the real
llama.cpp binary** (`llama-cli`, `llama-server`, …) — the wrapper process is
*replaced*, so by the time tokens flow there is nothing of Alpacca left in
the process. Model management (downloads, manifests, integrity checks)
happens strictly before inference starts.

```text
alpacca run llama3.2:1b
   │  resolve name → ~/.alpacca/models/…/model.gguf   (alpacca)
   │  map saved params → llama.cpp flags              (alpacca)
   └─ exec llama-cli -m model.gguf …                  (100% stock llama.cpp)
```

The llama.cpp sources are vendored **unmodified** as a git submodule pinned
to an upstream release tag, so every upstream capability and optimization is
here: CPU (AVX/NEON), CUDA, Metal, Vulkan, ROCm/HIP, quantization,
speculative decoding, multimodal, the full tool suite.

## Install

After installing, open a **new** terminal and just type `alpacca`.

**Linux / macOS** — one-liner (or run `scripts/install.sh` from a clone):

```sh
curl -fsSL https://raw.githubusercontent.com/jtiro232/Alpacca/main/scripts/install.sh | sh
```

Requirements: git, CMake ≥ 3.14, a C++17 compiler (`apt install build-essential
cmake git` / `xcode-select --install`). Installs to `~/.local/bin` (override
with `PREFIX=`) and adds it to your PATH (opt out with `NO_MODIFY_PATH=1`).

**Windows (PowerShell)** — one-liner (or run `scripts\install.ps1` from a clone):

```powershell
irm https://raw.githubusercontent.com/jtiro232/Alpacca/main/scripts/install.ps1 | iex
```

One-time requirements: `winget install Git.Git Kitware.CMake` and the
Visual Studio 2022 Build Tools with the C++ workload. Installs to
`%LOCALAPPDATA%\Alpacca\bin` and adds it to your user PATH.

**By hand:**

```sh
git clone https://github.com/jtiro232/Alpacca && cd Alpacca
git submodule update --init --depth 1
cmake -B build && cmake --build build --parallel
build/bin/alpacca doctor
```

GPU backends are llama.cpp's own — pass the flags straight through:

```sh
CMAKE_FLAGS="-DGGML_CUDA=ON"   scripts/install.sh   # NVIDIA
CMAKE_FLAGS="-DGGML_VULKAN=ON" scripts/install.sh   # Vulkan (AMD/Intel/NVIDIA)
# Metal is on by default on Apple Silicon; same flags work with install.ps1
```

## Getting models

Models live in `~/.alpacca/models` (override with `$ALPACCA_HOME`). Three
ways to reference a model:

| Reference                        | Source                                  |
| -------------------------------- | --------------------------------------- |
| `llama3.2:1b`, `qwen3:4b`        | Ollama registry (`registry.ollama.ai`)  |
| `ollama:user/model:tag`          | Ollama registry, user namespace         |
| `hf:org/repo` or `org/repo`      | Hugging Face — picks the best GGUF quant |
| `hf:org/repo:Q5_K_M`             | Hugging Face — specific quant or file    |
| `./path/to/model.gguf`           | any local GGUF file                      |

```sh
alpacca pull llama3.2:1b                         # Ollama library model
alpacca pull hf:ggml-org/gemma-3-4b-it-GGUF      # best quant from HF
alpacca pull hf:bartowski/Qwen2.5-7B-Instruct-GGUF:Q5_K_M
alpacca list
alpacca show llama3.2:1b
alpacca rm llama3.2:1b
```

Downloads resume if interrupted and are verified against the publisher's
SHA-256 digests. Gated Hugging Face repos work with `HF_TOKEN` set.
Multi-part (split) GGUFs and multimodal projector files (`mmproj`) are
detected and handled automatically. `alpacca run` auto-pulls a model that
isn't installed yet — just like Ollama.

## Running models

```sh
alpacca run llama3.2:1b                          # interactive chat
alpacca run llama3.2:1b "summarize: ..."         # one-shot, then exit
alpacca run llama3.2:1b --temp 0.2 -c 8192       # any llama.cpp flag works
alpacca run ./local-model.gguf
```

Everything after the model name (and optional prompt) is passed verbatim to
`llama-cli`, so the *entire* llama.cpp option surface is available. Saved
model parameters (temperature, context size, stop sequences, system prompt —
e.g. from an Ollama-published model) are applied as defaults; your flags win.

### Serving an API

```sh
alpacca serve llama3.2:1b                 # OpenAI-compatible, 127.0.0.1:8080
alpacca serve llama3.2:1b --port 11434 --api-key secret
```

`alpacca serve` execs `llama-server` — chat completions, embeddings,
parallel requests, continuous batching, all upstream features.

### The whole llama.cpp toolbox

Any other subcommand is dispatched to the matching `llama-*` tool, with
model names resolved for `-m`:

```sh
alpacca bench -m llama3.2:1b              # llama-bench
alpacca quantize in.gguf out.gguf Q4_K_M  # llama-quantize
alpacca tokenize -m llama3.2:1b -p "hi"   # llama-tokenize
alpacca perplexity -m llama3.2:1b -f txt  # llama-perplexity
alpacca cli --help                        # raw llama-cli
alpacca server --help                     # raw llama-server
```

## Environment

| Variable                 | Meaning                                        |
| ------------------------ | ---------------------------------------------- |
| `ALPACCA_HOME`           | data dir (default `~/.alpacca`)                |
| `ALPACCA_HOST` / `ALPACCA_PORT` | defaults for `alpacca serve`            |
| `ALPACCA_LLAMA_BIN_DIR`  | where to find `llama-*` binaries               |
| `HF_TOKEN`               | Hugging Face token for gated repos             |
| `ALPACCA_OLLAMA_REGISTRY`| alternate Ollama-compatible registry           |
| `ALPACCA_HF_ENDPOINT`    | alternate Hugging Face endpoint (mirrors)      |

## Testing

```sh
tests/smoke.sh               # offline: mock registry + tiny GGUF, full
                             # pull/list/show/run/rm cycle (CI runs this on
                             # Linux, macOS and Windows)
tests/hermes_acceptance.sh   # real 8B model from Hugging Face, headless
                             # one-shot factual question (needs network,
                             # ~5 GB disk, ~6 GB RAM)
```

## Updating llama.cpp

The submodule is pinned to a release tag for reproducible builds. To move
to a newer upstream release:

```sh
cd vendor/llama.cpp
git fetch --depth 1 origin tag b<NEW>     # pick a tag from upstream releases
git checkout b<NEW>
cd ../.. && cmake --build build --parallel
git add vendor/llama.cpp && git commit -m "bump llama.cpp to b<NEW>"
```

## Credits & licensing

Alpacca is MIT licensed (see [LICENSE](LICENSE)) and is built **totally
free and proper** on:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT, © The ggml
  authors) — vendored unmodified; it does all the actual inference. Thank
  you, ggml community.
- **[Ollama](https://github.com/ollama/ollama)** (MIT) — inspiration for
  the model-management UX, and Alpacca speaks its public registry protocol
  (independent implementation, no Ollama code included).

Details in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Model weights
you download have their own licenses — `alpacca show <model>` keeps the
publisher's license text next to the weights.
