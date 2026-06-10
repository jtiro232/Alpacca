# Alpacca

**LLMs in your terminal - a from-scratch, 100% Python inference engine for
GGUF models, with Ollama-style model management. Zero dependencies.**

Alpacca is not a wrapper around llama.cpp, PyTorch, or anything else. The
entire stack is implemented in this repository, in Python, on the standard
library alone:

| Layer | Where | What's implemented |
| --- | --- | --- |
| GGUF file format | `alpacca/gguf.py` | reader (mmap) + writer, metadata, tensor table |
| Quantization | `alpacca/quants.py` | F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K |
| Tokenizers | `alpacca/tokenizer.py` | SentencePiece-style (Viterbi + byte fallback) and byte-level BPE with a GPT-2/llama-3 pre-tokenizer |
| Transformer | `alpacca/model.py` | RMSNorm, RoPE (llama & neox styles), grouped-query attention, SwiGLU, KV cache |
| Sampling | `alpacca/sample.py` | greedy, temperature, top-k, top-p, repeat penalty |
| Chat | `alpacca/chat.py` | llama3 / chatml / gemma / llama2 / zephyr templates, streaming, interactive REPL |
| API server | `alpacca/serve.py` | OpenAI-compatible `/v1/chat/completions` (incl. SSE streaming) on `http.server` |
| Model manager | `alpacca/store.py`, `alpacca/pull.py` | Ollama-registry protocol + Hugging Face pulls via `urllib`, resumable, SHA-256 verified |

If NumPy happens to be installed it is auto-detected and used as a math
accelerator (10-100x faster); without it everything still runs, just slowly.
Set `ALPACCA_PURE=1` to force the stdlib path. Both backends produce
identical results and verify each other in CI.

```text
$ alpacca pull llama3.2:1b            # straight from the Ollama registry
$ alpacca run llama3.2:1b             # interactive chat
$ alpacca run llama3.2:1b "why is the sky blue?"
$ alpacca serve llama3.2:1b           # OpenAI-compatible API on :8080
```

## Install - offline by design

There is nothing to compile and nothing to download beyond this repository
itself. Get the code (git clone, or a release tarball verified against its
published SHA-256), then either:

```sh
# 1. no install at all:
python3 -m alpacca doctor

# 2. or put an `alpacca` launcher on your PATH (offline, creates one file):
scripts/install.sh          # Linux/macOS   (PREFIX=... to relocate)
.\scripts\install.ps1     # Windows PowerShell
```

Requires Python >= 3.10. Optional: `pip install numpy` for fast generation -
that is the only thing that would ever touch a package index, it's off by
default, and Alpacca works without it. `pip install .` also works if you
prefer a normal Python install.

## Getting models

Models live in `~/.alpacca/models` (override: `$ALPACCA_HOME`). Reference
them three ways:

| Reference | Source |
| --- | --- |
| `llama3.2:1b`, `qwen2.5:0.5b` | Ollama registry (`registry.ollama.ai`) |
| `ollama:user/model:tag` | Ollama registry, user namespace |
| `hf:org/repo` or `org/repo` | Hugging Face - picks the best GGUF quant |
| `hf:org/repo:Q4_K_M` (or a filename) | Hugging Face - specific quant/file |
| `./path/to/model.gguf` | any local GGUF |

`alpacca pull` speaks the Ollama registry protocol directly (manifest +
content-addressed layers - weights, parameters, system prompt, license) and
the Hugging Face API (quant selection, `-GGUF` sibling-repo fallback,
`HF_TOKEN` for gated repos). Downloads resume after interruption and are
verified against the publisher's SHA-256 digests. `alpacca run` auto-pulls
on first use.

```sh
alpacca list
alpacca show llama3.2:1b --metadata
alpacca rm llama3.2:1b
alpacca tokenize -m llama3.2:1b -p "hello world"
```

## Running models

```sh
alpacca run llama3.2:1b                          # interactive (/exit, /clear)
alpacca run llama3.2:1b "one-shot question"      # answers and exits
alpacca run ./model.gguf --temp 0.2 -n 256 -c 4096 --seed 1
alpacca serve llama3.2:1b --port 8080
```

The server is OpenAI-compatible - point any OpenAI client at
`http://127.0.0.1:8080/v1` (chat completions, streaming included), or use
the llama.cpp-style `POST /completion`.

Supported architectures: llama (1/2/3, TinyLlama, Mistral-family), qwen2/3,
stablelm, gemma. Chat templates are detected from the model's metadata.

### Honest performance expectations

This engine values clarity, auditability, and zero dependencies over raw
speed. Weights are held in float32. Rules of thumb:

- **with NumPy**: 1B-class models chat comfortably; 7-8B models work but
  need ~35 GB RAM and patience.
- **stdlib only**: tiny models (stories15M-class) are fine; 1B is slow.
  Good for air-gapped checks, not long conversations.

If you need llama.cpp-class speed, you need llama.cpp-class native kernels -
that's a different project (and an explicit non-goal here).

Recent local acceptance coverage includes `NousResearch/Hermes-3-Llama-3.1-8B`
via its single-file `Q4_K_M` GGUF. On a 16-core Ryzen 9 7950X with NumPy, it
loaded at ~36 GB RAM and answered the Lincoln birthday check at about 1.75
tokens/sec. Treat that as a correctness and compatibility signal, not an
Ollama-speed performance target.

## Roadmap

Alpacca's near-term goal is to stay inspectable and Python-first while making
the fast path more practical. Planned work:

- **Quantized matvec backend**: keep GGUF weights in their native quantized
  representation during inference instead of expanding everything to float32.
  This is the main path toward large speed and memory wins.
- **Backend selection**: keep the current stdlib and NumPy paths, then add
  optional accelerated backends behind clear flags. Candidate backends include
  Numba, CuPy, PyTorch, Triton, or small native kernels exposed through Python.
- **Kernel fusion**: fuse common transformer operations such as QKV projection,
  SwiGLU gate/up projection, RoPE, and attention reductions where the selected
  backend can do so without excessive memory duplication.
- **Model compatibility gates**: add recurring real-model tests for Q4_K,
  Q5_K/Q6_K, qwen, gemma, mistral, and larger llama-family GGUFs.
- **Operational UX**: improve Windows launchers, model selection, and clearer
  warnings for split GGUFs, unsupported architectures, and memory-heavy runs.
- **Server compatibility**: broaden OpenAI-compatible API behavior while keeping
  the standard-library server usable in offline environments.

## Testing

```sh
python3 tests/smoke.py            # offline: 33 checks - mock registry pulls,
                                  # real inference, API server, store mgmt
python3 tests/real_model_test.py  # downloads a 19 MB real model (network),
                                  # asserts it generates coherent English
python3 tests/acceptance.py       # pulls llama3.2:1b and asks it Lincoln's
                                  # birthday; --model ... for bigger models
```

CI runs the offline suite on Linux/macOS/Windows, with and without NumPy,
plus the real-model gate.

## Security & supply chain

- **No install-time network access**: the repo is the program. No package
  index, no build step, no binary artifacts, no submodules.
- Release archives ship with SHA-256 checksums.
- Model downloads (`alpacca pull`) are the only network feature, are
  explicit, and verify the publisher's digests. Models carry their own
  licenses - when the publisher provides one, it is stored next to the
  weights.

## Credits & licensing

Alpacca is MIT licensed (see [LICENSE](LICENSE)). All code here is written
from scratch in Python. It interoperates with formats and protocols designed
by others, with thanks - see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md): the GGUF format and
quantization schemes (ggml/llama.cpp project), the Ollama registry protocol,
and the SentencePiece/BPE tokenization algorithms.
