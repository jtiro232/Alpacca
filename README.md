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
speed. The NumPy path batches prompt prefill, reuses the KV cache for shared
prompt prefixes, and keeps Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, and
Q6_K matrix weights quantized in RAM: at load each matrix is unpacked once
into int8 quant codes plus per-sub-block float32 scales (about 1.1-1.3
bytes per weight instead of 4), and decode/prefill kernels consume that
form directly - nothing is re-dequantized per token. F16/BF16 and the
remaining quant formats load as dense float32 (measured: NumPy's
float16-to-float32 conversion is far slower than the BLAS GEMV it would
feed, so wrapping F16 would only slow decode down).

Measured on a 4-core Intel Xeon 2.80 GHz Linux container, Python 3.11,
NumPy 2.4.6 (OpenBLAS), with a stories15M-shaped synthetic model from
`tests/make_bench_model.py` (same architecture dimensions as the real
stories15M; CI runs the real one). Decode and RSS are medians of 3 runs;
prefill on the 64-token rows is a single ~50 ms window and jitters
+-40% on this shared machine, the 256-token rows are steadier:

| Run (`tests/bench.py`) | Mode | Load | Prefill | Decode | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4_0, 64 prompt / 32 decode, ctx 128 | quantized weights | 0.084 s | ~1,100 tok/s | 63.2 tok/s | 65.5 MB |
| Q4_0, same run | `ALPACCA_F32=1` dense | 0.130 s | 3,192 tok/s | 126.7 tok/s | 107.1 MB |
| Q4_0, 256 prompt / 128 decode, ctx 512 | quantized weights | 0.083 s | 2,998 tok/s | 60.3 tok/s | 71.3 MB |
| Q4_0, same run | `ALPACCA_F32=1` dense | 0.128 s | 4,393 tok/s | 131.3 tok/s | 115.7 MB |
| Q8_0, 64 prompt / 32 decode, ctx 128 | quantized weights | 0.078 s | ~930 tok/s | 58.6 tok/s | 72.6 MB |
| F32 GGUF, 64 prompt / 32 decode | native dense | 0.096 s | 3,638 tok/s | 152.0 tok/s | 156.7 MB |

(The previous revision of this engine decoded the same quantized model at
19.4 tok/s on this machine: the int8 unpacked storage is a ~2.9x decode
improvement, plus another ~8-15% from decode-overhead trims - grouped
attention, precomputed RoPE tables, BLAS-dot rmsnorm - that also sped the
float32 paths up. But read the next paragraph before expecting quantized
to beat float32.)

What quantized storage does and does not buy here, measured honestly:

- **Memory**: weight storage is 17.1 MB vs 60.8 MB float32 for the same
  model (0.28x, measured). Whole-process RSS at this tiny model size is
  dominated by the ~50 MB Python+NumPy baseline, so it shows 65 MB vs
  107 MB (0.61x); the ratio approaches the 0.28x storage ratio as models
  grow. RSS is reported by `tests/bench.py` on Linux/macOS and is `n/a`
  on Windows (no `resource` module).
- **Load time**: 0.084 s vs 0.130 s for float32 expansion (writes ~1.1
  bytes per weight instead of 4; the advantage grows with model size).
- **Decode speed**: quantized decode remains ~0.5x of `ALPACCA_F32=1`
  dense decode. This is a measured NumPy ceiling, not a missing
  optimization in this codebase: OpenBLAS SGEMV runs multithreaded at
  memory bandwidth (0.53 ms for the dominant 32000x288 output projection),
  while NumPy has no mixed int8xf32 GEMV primitive - every strategy
  (einsum, astype+GEMV, integer matmul) pays a single-threaded conversion
  pass that costs 4.5-9 ms on the same matrix. Per-token profile of the
  quantized path after the overhead trims: ~76% in the int8 matvec
  kernels (~54% just the output projection), ~9% residual Python
  overhead, the rest attention/normalization. Beating BLAS by 2x with
  quantized weights needs native SIMD dot-product kernels
  (llama.cpp-class), which is an explicit non-goal here.

### Spending RAM for speed: the dense-weight budget

Because dense BLAS is the fast path and quantized storage is the small
path, the practical dial for 1B-8B models is `ALPACCA_DENSE_WEIGHT_MB=N`:
at load time Alpacca expands up to `N` MiB of the most decode-critical
matrices to dense float32 and keeps the rest quantized. Matrices are
picked in measured-benefit order - FFN projections first (they dominate
llama-class decode), then attention q/output, then k/v, then the output
projection; a token embedding is only densified when it doubles as a tied
output matrix. Chosen matrices never keep their quantized copy, so unlike
`ALPACCA_HOT_WEIGHT_MB` nothing is stored twice.

Measured on the same Linux container with a 1.1B-parameter
TinyLlama-shaped synthetic Q4_0 model (GQA 32/4 heads, untied output;
`tests/make_bench_model.py --embd 2048 --ff 5632 --layers 22 --heads 32
--kv 4 --untied`), 32-token prompt / 16-token decode:

| `ALPACCA_DENSE_WEIGHT_MB` | Storage | Prefill | Decode | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| unset (all quantized) | 156 quant | 18.1 tok/s | 1.29 tok/s | 1.85 GB |
| `3100` (FFN stack dense) | 76 quant + 80 dense | 34.6 tok/s | 3.81 tok/s | 4.07 GB |
| `4000` (all but embedding) | 1 quant + 155 dense | 49.3 tok/s | 7.34 tok/s | 4.79 GB |
| `ALPACCA_F32=1` (everything) | 156 dense | 53.4 tok/s | 7.37 tok/s | 4.97 GB |

Decode scales almost linearly with how much of the per-token matvec work
runs through BLAS: the FFN-only budget buys 3.0x decode for ~2.2 GB, and
the everything-but-embedding budget matches full float32 speed while the
embedding stays quantized. For an 8B model (e.g. Hermes-3-Llama-3.1-8B
Q4), the FFN stack is ~22.5 GB (21.0 GiB) dense, so
`ALPACCA_DENSE_WEIGHT_MB=24000` is the "fast decode if you have ~35 GB
total RAM" setting, and smaller budgets degrade gracefully - every MiB
goes to the highest-impact matrices first. `tests/bench.py` prints the
resulting storage split per run.

**Fast is the default for `alpacca run` and `alpacca serve`**: unless
`ALPACCA_DENSE_WEIGHT_MB` is set, the CLI sizes the budget automatically
from detected available RAM (it reserves the quantized residue plus a
KV/runtime allowance, spends 85% of the rest, and prints the chosen value
at load). Set `ALPACCA_DENSE_WEIGHT_MB=0` for the low-RAM all-quantized
mode, or an explicit MiB value to pin the budget. RAM detection uses
`/proc/meminfo` on Linux, `GlobalMemoryStatusEx` on Windows, and a
conservative half-of-physical heuristic on macOS; if detection fails the
CLI says so and stays quantized. Library use (`Model.load`) keeps the
explicit opt-in semantics so embedders and tests get deterministic
storage.

Rules of thumb:

- **with NumPy**: tiny and 1B-class models are the practical target. Use
  quantized weights when RAM is the constraint, `ALPACCA_DENSE_WEIGHT_MB`
  to spend whatever RAM you can spare on decode speed, and `ALPACCA_F32=1`
  when the full float32 expansion fits comfortably anyway.
- **stdlib only**: tiny models (stories15M-class) are fine; 1B is slow. Good
  for air-gapped checks, not long conversations.
- **chat/server reuse**: repeated turns or requests with a shared prompt prefix
  skip already-cached K/V work automatically.

Useful environment knobs:

- `ALPACCA_PURE=1`: force the standard-library backend.
- `ALPACCA_DENSE_WEIGHT_MB=N`: densify up to `N` MiB of the most
  decode-critical matrices at load time (FFN first) and keep the rest
  quantized - the main RAM-for-speed dial. The CLI auto-sizes this from
  available RAM when unset; `0` disables densification; library callers
  opt in explicitly. See the table above.
- `ALPACCA_F32=1`: force the NumPy loader to expand all quantized matrices
  to float32, useful for A/B checks and small models where BLAS wins.
- `ALPACCA_PREFILL_CHUNK=N`: prompt batch size for NumPy prefill; default 256.
- `ALPACCA_HOT_WEIGHT_MB=N`: optional lazy dense float32 cache for quantized
  matrices, capped at `N` MiB. Unlike the dense budget this caches at first
  use and keeps the quantized copy too; prefer `ALPACCA_DENSE_WEIGHT_MB`
  unless you specifically want runtime-populated caching.
  (`ALPACCA_UNPACKED_WEIGHT_MB` is gone; the int8 unpacked form is now the
  default storage and needs no budget.)

## Roadmap

Alpacca's near-term goal is to stay inspectable and Python-first while making
the fast path more practical. Current status:

- **Done**: batched NumPy prefill with last-token-only vocab projection.
- **Done**: prefix-aware KV-cache reuse across chat turns and serialized server
  requests.
- **Done**: quantized matrix storage (int8 codes + per-sub-block scales,
  unpacked once at load) with matvec/matmul/row dispatch for
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K; other formats fall back to dense
  float32.
- **Done**: documented the NumPy quantized-decode ceiling with kernel-level
  measurements (see "Honest performance expectations"); pushing past it
  requires a native/accelerated backend, below.
- **Backend selection**: keep the current stdlib and NumPy paths, then add
  optional accelerated backends behind clear flags. Candidate backends include
  Numba, CuPy, PyTorch, Triton, or small native kernels exposed through Python.
- **Kernel fusion**: continue fusing common transformer operations where the
  selected backend can do so without excessive memory duplication.
- **Model compatibility gates**: add recurring real-model tests for Q5_K/Q6_K,
  qwen, gemma, mistral, and larger llama-family GGUFs.
- **Operational UX**: improve Windows launchers, model selection, and clearer
  warnings for split GGUFs, unsupported architectures, and memory-heavy runs.
- **Server compatibility**: broaden OpenAI-compatible API behavior while keeping
  the standard-library server usable in offline environments.

## Testing

```sh
python3 tests/smoke.py            # offline suite: mock registry pulls,
                                  # real inference, API server, store mgmt
python3 tests/real_model_test.py  # downloads a 19 MB real model (network),
                                  # asserts it generates coherent English
python3 tests/bench.py --model hf:ggml-org/models:stories15M-q4_0.gguf \
  --prefill 64 --decode 32 --ctx 128
python3 tests/make_bench_model.py /tmp/s15m-q4.gguf Q4_0   # offline bench
python3 tests/bench.py --model /tmp/s15m-q4.gguf \
  --prefill 64 --decode 32 --ctx 128
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
