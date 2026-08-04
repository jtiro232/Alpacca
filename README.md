# Alpaccaroo

**LLMs in your terminal - a from-scratch, 100% Python inference engine for
GGUF models, with Ollama-style model management. Zero dependencies.**

## Mission

Alpaccaroo stands on three commitments:

1. **Totally our own software.** Every layer is implemented from scratch
   in this repository: the GGUF parser, the quantization codecs, the
   tokenizers, the transformer, the sampler, the chat templates, the
   OpenAI-compatible server, and the registry clients. It is not a wrapper
   around llama.cpp, PyTorch, or anything else - no vendored code, no
   binaries, no submodules. You can read the whole engine in an afternoon.
2. **Pure Python.** The engine runs on the standard library alone, on any
   Python >= 3.10, and every algorithm in this repository - including the
   fast kernels - is written in Python. Acceleration is optional and
   tiered: NumPy when installed (10-100x faster math), and Alpaccaroo's own
   kernels in `alpaccaroo/kernels.py` - our Python source, JIT-compiled to
   native SIMD at runtime by a *pinned* Numba (`python -m pip install
   ".[kernels]"` from this checkout; the pin is never bumped implicitly).
   No C, no Rust, no compiled files in the repo. `ALPACCAROO_PURE=1` forces the stdlib
   path; the pure and NumPy paths are the reference implementations that
   everything else must match in CI.
3. **Fast and reliable - honestly.** Speed is engineered as far as Python
   plus NumPy can go: quantized int8 weight storage, a hybrid
   dense/quantized policy that auto-tunes to your machine's RAM, batched
   prefill, KV-cache reuse. Every number in this README is measured, and
   the ceilings are documented next to the wins. Reliability means a CI
   matrix across Linux/macOS/Windows with and without NumPy (hundreds of checks)
   plus a real-model generation gate on every push.

## Structure

| Layer | Where | What's implemented |
| --- | --- | --- |
| GGUF file format | `alpaccaroo/gguf.py` | reader (mmap) + writer, metadata, tensor table |
| Quantization codecs | `alpaccaroo/quants.py` | decode/encode support for F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K |
| Quantized weight storage | `alpaccaroo/qmatrix.py` | fast in-RAM matvec/matmul storage for Q2_K/Q3_K/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K; unsupported matrix formats fall back to dense, and say so at load |
| Fast kernels | `alpaccaroo/kernels.py` | our fused quantized-matvec algorithms in Python source, JIT-compiled by pinned optional Numba |
| Tokenizers | `alpaccaroo/tokenizer.py` | SentencePiece (llama.cpp's greedy highest-score-first merge, special-token pre-split, byte fallback) and byte-level BPE with a GPT-2/llama-3 pre-tokenizer |
| Transformer | `alpaccaroo/model.py` | RMSNorm, RoPE (llama & neox styles), grouped-query attention, SwiGLU, KV cache, dense-budget loader |
| Sampling | `alpaccaroo/sample.py` | greedy, temperature, top-k, top-p, repeat penalty |
| Chat | `alpaccaroo/chat.py` | llama3 / chatml / gemma / llama2 / zephyr templates, streaming, Esc-to-menu interactive REPL with saved history |
| History | `alpaccaroo/history.py` | local JSON chat sessions, delete one/delete all controls, saved-chat statistics |
| API server | `alpaccaroo/serve.py` | OpenAI-compatible `/v1/chat/completions` (incl. SSE streaming) on `http.server` |
| Model manager | `alpaccaroo/store.py`, `alpaccaroo/pull.py` | Ollama-registry protocol + Hugging Face pulls via `urllib`, resumable, SHA-256 verified |
| CLI / terminal app | `alpaccaroo/cli.py` | repo-owned `alpaccaroo menu`; pull/run/serve/list/show/rm/tokenize/history/hist/doctor; default model persistence; auto RAM-aware speed defaults |
| Tests | `tests/` | offline smoke (mock registry, kernel parity, menu/history, both backends), real-model gate, benchmarks, synthetic bench-model builders |
| Tooling | `scripts/` | offline installers for Linux/macOS/Windows, generating thin launchers into this checkout |

```text
$ alpaccaroo pull llama3.2:1b            # straight from the Ollama registry
$ alpaccaroo menu                        # local terminal app menu
$ alpaccaroo run llama3.2:1b             # interactive chat
$ alpaccaroo run llama3.2:1b "why is the sky blue?"
$ alpaccaroo serve llama3.2:1b           # OpenAI-compatible API on :8080
```

## Install - offline by design

There is nothing to compile and nothing to download beyond this repository
itself. Get the code (git clone, or a release tarball verified against its
published SHA-256), then either:

```sh
# 1. no install at all:
python3 -m alpaccaroo doctor

# 2. or put an `alpaccaroo` launcher on your PATH (offline, creates one file):
scripts/install.sh          # Linux/macOS   (PREFIX=... to relocate)
.\scripts\install.ps1     # Windows PowerShell
```

Requires Python >= 3.10. The generated launchers are deliberately thin:
they set `PYTHONPATH` to this checkout and dispatch to `python -m alpaccaroo`.
Running `alpaccaroo` with no arguments opens the repo-owned terminal menu in an
interactive terminal; `alpaccaroo menu` opens it explicitly; all normal commands
still work (`alpaccaroo doctor`, `alpaccaroo run ...`, `alpaccaroo history stats`).
Installers use the normal store at `~/.alpaccaroo` unless you set
`ALPACCAROO_HOME` yourself.

Optional: `python -m pip install numpy` for fast generation, or
`python -m pip install ".[kernels]"` from this checkout for Alpaccaroo's pinned
Numba kernel tier. Those are the only commands here that would touch a
package index for Python packages, they are opt-in, and Alpaccaroo works without
them. `python -m pip install .` also works if you prefer a normal Python
install.

## Getting models

Models live in `~/.alpaccaroo/models` (override: `$ALPACCAROO_HOME`). Reference
them three ways:

| Reference | Source |
| --- | --- |
| `llama3.2:1b`, `qwen2.5:0.5b` | Ollama registry (`registry.ollama.ai`) |
| `ollama:user/model:tag` | Ollama registry, user namespace |
| `hf:org/repo` or `org/repo` | Hugging Face - picks the best GGUF quant |
| `hf:org/repo:Q4_K_M` (or a filename) | Hugging Face - specific quant/file |
| `./path/to/model.gguf` | any local GGUF |

`alpaccaroo pull` speaks the Ollama registry protocol directly (manifest +
content-addressed layers - weights, parameters, system prompt, license) and
the Hugging Face API (quant selection, `-GGUF` sibling-repo fallback,
`HF_TOKEN` for gated repos). Downloads resume after interruption and are
verified against the publisher's SHA-256 digests. `alpaccaroo run` auto-pulls
on first use. Model nicknames are stored as local aliases under
`$ALPACCAROO_HOME` and do not rename the downloaded model directory or manifest.
Use `alpaccaroo nickname <model> <nickname>` or the Model manager menu to set one;
`alpaccaroo list` shows the `NICKNAME` column, and `alpaccaroo nickname --list`
enumerates every alias - including any left orphaned by a removed model.

```sh
alpaccaroo list
alpaccaroo nickname llama3.2:1b "quick llama"
alpaccaroo nickname --list
alpaccaroo show "quick llama" --metadata
alpaccaroo rm llama3.2:1b
alpaccaroo tokenize -m "quick llama" -p "hello world"
alpaccaroo history list
alpaccaroo history stats
```

## Running models

```sh
alpaccaroo run llama3.2:1b                          # interactive (Esc or /exit returns, /clear resets)
alpaccaroo run "quick llama" "one-shot question"    # nicknames work for model selectors
alpaccaroo run ./model.gguf --temp 0.2 -n 256 -c 4096 --seed 1
alpaccaroo serve llama3.2:1b --port 8080
```

The terminal app has a repo-owned menu:

```sh
alpaccaroo menu       # or just `alpaccaroo` in an interactive terminal
```

The menu lists installed models, opens chat, switches the default chat
model by name or nickname, sets model nicknames, shows model details, exposes
history/statistics, and links back to the normal commands. The default chat
model is stored locally in `~/.alpaccaroo/default-model.txt` (or
`$ALPACCAROO_HOME/default-model.txt`) and is only a UI convenience; model selector
commands accept an explicit model reference or a nickname.

An interactive chat stays inside the model's context window on its own: when the
conversation no longer leaves room for a reply, the oldest turns are dropped
(the system message is kept) and the REPL prints how many it dropped. `/clear`
resets the conversation outright. The effective context length is printed at
load next to the model's trained maximum, because `alpaccaroo run` defaults to a
smaller window than most models advertise - pass `-c` to raise it.

Interactive chats are saved as local JSON files under `~/.alpaccaroo/history`
(or `$ALPACCAROO_HOME/history`). One-shot `alpaccaroo run MODEL "prompt"` calls and
server/API requests are not logged. Use:

```sh
alpaccaroo history list            # list saved interactive chats
alpaccaroo history show <chat>     # <chat> = list number, full ID, or unique prefix
alpaccaroo history stats           # read-only saved-chat token/s summary
alpaccaroo history rm <chat>       # delete one saved chat
alpaccaroo history clear --yes     # delete all saved chat history
```

`history stats` also lists installed models with zero saved chats so you can
distinguish "installed but unused" from "used and deleted". History remains
offline and zero-dependency.

The server is OpenAI-compatible - point any OpenAI client at
`http://127.0.0.1:8080/v1` (chat completions, streaming included), or use
the llama.cpp-style `POST /completion`.

Supported architectures: llama (1/2/3, TinyLlama, Mistral-family), qwen2/3,
stablelm, gemma, and Gemma 3 text GGUFs. Gemma 3 support includes Q/K
normalization, local sliding-window attention patterns, dual RoPE bases, linear
RoPE scaling, metadata-driven attention scale, GELU FFNs, post-attention/post-FFN
norms, tied output embeddings, and optional final logit softcapping. Gemma 3
support is text-only; multimodal projector/vision support is outside Alpaccaroo's
current engine scope.

What has actually been exercised, and what has not: Gemma 3 1B has been run end
to end against a real GGUF. The 4B/12B/27B variants have not - the RoPE-scaling
path and the 27B attention-scale rule are implemented from llama.cpp's rules and
covered by fixtures, but no such checkpoint has been loaded. Gemma 1 (`gemma`)
is likewise implemented from the architecture specification - the sqrt(n_embd)
embedding scale and GELU FFN - and validated against an independent float64
reference on a fixture, not against a released checkpoint.

Chat templates are *detected* from the model's metadata, and the renderer emits
the format's real control tokens. It is not a Jinja interpreter: the template is
matched to one of five built-in renderers (llama3, chatml, gemma, llama2,
zephyr). The Gemma renderer follows the shipped template where it matters: a
leading system message is folded into the first user turn as a prefix, because
the template has no system turn and refuses two user turns in a row. Measured on
Gemma 3 1B over 36 greedy generations, that rendering and a separate-system-turn
rendering are not distinguishable in how well the model obeys a system prompt.
Two divergences remain: every non-assistant role is mapped to `user`, and the
template's `raise_exception` on non-alternating roles is not ported - alpaccaroo
renders such a conversation instead of rejecting it.

### Honest performance expectations

> **Measuring your own machine:** `docs/PERFORMANCE.md` is the reference for
> the profiler (`alpaccaroo run <model> --profile`), the benchmark harness
> (`alpaccaroo bench`), the opt-in autotuner (`alpaccaroo tune`), the
> codegen inspector (`alpaccaroo tune --asm`), the resident-server
> reconnect (`alpaccaroo run <model> --connect`), every environment knob,
> and the measured results behind them. Start there before changing
> anything - it also documents *why* two separate timing runs on a
> power-limited laptop cannot be compared, which is the mistake that costs
> the most time.

This engine values clarity, auditability, and zero dependencies over raw
speed. The NumPy path batches prompt prefill, reuses the KV cache for shared
prompt prefixes, and keeps Q2_K, Q3_K, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K,
Q5_K, and Q6_K matrix weights quantized in RAM: at load each matrix is unpacked once
into int8 quant codes plus per-sub-block float32 scales (about 1.1-1.3
bytes per weight instead of 4), and decode/prefill kernels consume that
form directly - nothing is re-dequantized per token. F16/BF16 and the
remaining quant formats load as dense float32 (measured: NumPy's
float16-to-float32 conversion is far slower than the BLAS GEMV it would
feed, so wrapping F16 would only slow decode down).

#### Where the time goes on a large model

Single-stream decode is memory-bandwidth-bound: on a Ryzen 5 7640HS
(6 cores, DDR5) streaming reads top out at ~59-60 GB/s no matter how many
threads ask (measured at 4/6/8/12), so tokens per second is bytes per
weight divided into that wall. With the pinned Numba kernels, Q4_K and
Q6_K weights now stay in the *file's own* block fields - 4-bit split-nibble
codes, 6-bit integer sub-scales, and float16 super-scales, 0.578 bytes per
weight for Q4_K and 1.066 for Q6_K - and the decode kernel dots them
against int8-quantized activations (per-256-block absmax/127 scale, our
own Q8-style scheme) using whatever integer contraction LLVM can emit for
the host. On a part with AVX-512 VNNI that is `vpdpwssd`; on one without
it - measured on a Zen 3 Ryzen 7 7730U, which has no VNNI in any form -
LLVM falls back to 256-bit `vpmaddwd` and the kernels still run at the
machine's bandwidth. What matters portably is that **neither emits
`vpmuldq`**: the `np.int32(acc + i32*i32)` re-cast idiom keeps the
contraction in 32-bit lanes on both, and without it Numba's int64
promotion collapses the loop to 8-lane 64-bit multiplies. The idiom is
not a VNNI trick, and `alpaccaroo tune --asm` will tell you which of the
two your CPU got. Llama-3.1-8B-Instruct decodes at
**9.9 tok/s (Q4_K_M, 5.0 GiB weight RAM) and 10.8 tok/s (Q4_K_S,
4.6 GiB)** against 5.2 tok/s and 9.35 GiB before this work, and 11.9
tok/s for Ollama/llama.cpp on the same machine - 90% of parity, with
the remainder bounded by measured hardware ceilings (see the notes for
future engineers below). Q5_K matrices get the same native integer
treatment (0.70 B/weight, 1.5x their old path), which is what makes
Q4_K_S files - whose attention-V is stored Q5_K - the faster pick. An earlier
attempt at 4-bit packed storage measured *slower*; the culprit was found
this pass - Numba promotes scalar integer arithmetic to int64, which had
been hiding every 8/16-bit SIMD pattern from LLVM - and the fix (int32
re-cast accumulators) is load-bearing in every integer kernel.

The integer dot changes numerics: activations are rounded to int8 before
the weight dot, the same trade llama.cpp makes. Measured end to end on the
8B model, 48 greedy-decoded tokens agree 48/48 with the exact float32
path, and the kernels are bit-exact against an integer simulation of
their own algebra (a smoke check enforces <=1e-5 and was verified to fail
under a mutated kernel). `ALPACCAROO_INT_DOT=0` restores the previous exact
storage and kernels. Weights themselves are represented exactly - the
codes and scales are the file's own bits.

Long-prompt prefill (batch > 64) dequantizes tiles with a JIT kernel and
hands them to BLAS, whose GEMM is the floor (~240 GFLOP/s measured):
911 tokens prefill at 7-8 tok/s vs 5.9 for the previous storage measured
the same day. Prefill numbers on this machine jitter +-15% with
background load; compare like with like.

What did not help, measured: transparent huge pages (madvise accepted but
never materialized on this kernel; 4K-page streaming already sits on the
DRAM wall), fusing attn_q+attn_k and gate+up launches (kept for structure,
but 226 -> 162 launches was worth 0.0 ms here), and thread counts above
the physical-core count (SMT contention costs 9-16%; kernels now default
to physical cores, override with ALPACCAROO_THREADS).

#### Notes for future engineers (human or AI) on the kernels

Read `prompts/03-RESULTS.md` first: it is the experiment log with every
number, and several expensive lessons are recorded there so they are not
repeated. Then `prompts/04-RESULTS.md`, which logs the portable-performance
round on very different hardware (a bandwidth-starved laptop rather than
this one's 60 GB/s wall) and carries the continuation plan; it also records
which of 03's verdicts reopen when the ALU/bandwidth ratio inverts. The
short version:

- **Numba promotes scalar integer arithmetic to int64.** Every integer
  kernel here depends on the re-cast idiom `acc = np.int32(acc + ...)`
  to keep the IR in i32; without it LLVM sees i64 lanes and emits no
  8/16-bit SIMD at all. This single fact once hid AVX-512 VNNI from a
  whole engineering pass.
- **The winning kernel shape** is: j-loop outermost over a 32-byte
  stride, all streams of a 256-weight block unrolled inline, 6-bit
  sub-scales premultiplied into int16 codes in-register (products stay
  under 2^15 - check the bound for any new format), ONE vector reduce
  per block. Per-sub-block reduces cost ~2x; manual multi-accumulator
  "optimizations" have regressed 2-7x every time they were tried.
- **Never put scalar per-block work inside the hot loop** (in-kernel
  6-bit scale unpack collapsed a 110 Gw/s kernel to 12.7), and **never
  let a constant trip count reach the vectorizer** (it const-unrolls and
  the VNNI pattern dies; derive trip counts from runtime shapes).
- **Offsets fold into integer min-terms**: sum(sc*(code-K)) =
  sum(sc*code) - K*sc*bsum. Subtracting K per element broke
  vectorization for Q6_K; the fold rescued it.
- **Never call threaded BLAS from the decode loop.** OpenBLAS fans out
  its own pool past a size threshold and thrashes the numba omp pool
  (measured 8-20x). Decode-side ops are JIT kernels gated on
  `Model._use_kernel_attention` (true only for models that run the
  quantized kernels - dense models stay single-pool on BLAS, and moving
  only part of their work to numba recreates the thrash).
- **Closed doors, with the measurements that closed them** (do not
  reopen without new evidence): Q6_K 6-bit packing is ALU-bound at
  exactly its bandwidth saving (52 Gw/s vs the 72 needed); speculative
  decoding is capped at ~1.10x because batched integer matmuls cost
  linearly - the single-token kernel already uses ~all vpdpwssd
  throughput; LLVM does not auto-emit vpdpbusd from any loop shape
  tried, TESTED THROUGH LLVM 22 (numba 0.66), and LLVM 22 also stops
  folding vpmaddwd+add into vpdpwssd for our premultiplied shape, so a
  numba pin bump would REGRESS decode - re-test before ever bumping.
  What reopens all three at once: a toolchain that emits vpdpbusd
  (4 int8 MACs/lane) from Python-authored loops.
- **Measurement discipline**: warm the JIT before timing, benchmark on
  a quiet machine (background load once produced a 3 tok/s reading that
  was pure noise), pair alpaccaroo and Ollama numbers same-day, and treat
  multi-minute prefill windows as +-15%. Sustained decode runs this
  hardware at its thermal ceiling with clocks intact - verify clocks
  and DRAM bandwidth before blaming code.
- **Every numerics change needs a differential test against the old
  path loaded from git** (see the sampler/tokenizer harnesses in the
  results log: zero-mismatch over 19k trials / 352 real-vocab encodes),
  and every new kernel needs an integer-simulation exactness check in
  `tests/smoke.py` that was SHOWN to fail under a mutated kernel.

Two bigger avenues were measured shut (details and numbers in
prompts/03-RESULTS.md): packing Q6_K to its native 0.82 bytes/weight is
a wash - the 6-bit unpack costs exactly the ALU that the bandwidth
saving buys back - and speculative decoding is capped at ~1.10x here
regardless of draft acceptance, because batched integer matmuls cost
linearly in batch size: the single-token kernel already uses ~all of
the machine's vpdpwssd throughput, so there is no headroom to verify
drafts in. Both would reopen if LLVM's auto-vectorizer learns to emit
vpdpbusd (4 int8 MACs per lane instead of vpdpwssd's 2).

Batched work was a different story. `matmul_t` used to dequantize the whole
matrix to float32 before handing it to BLAS, a cost proportional to the
*weights* rather than to the batch, so prefilling a single new token cost a
full-model dequantize - 5.0 s on that 8B model, 25x a whole decode step, and
paid again on every chat turn. Two fused kernels now stream the codes
directly for batches up to `ALPACCAROO_FUSED_MATMUL_MAX_BATCH` (96 by default),
one laid out for narrow batches and one for wide ones; past that the tiled
BLAS path still wins and is still used. Prefill of N new tokens, 8B Q4_K_M,
model already loaded:

| new tokens | 1 | 4 | 8 | 16 | 32 | 64 | 96 | 128+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 5.02 s | 6.96 s | 7.14 s | 7.26 s | 8.06 s | 9.36 s | 10.68 s | unchanged |
| after | 0.30 s | 0.56 s | 1.27 s | 2.72 s | 3.55 s | 5.72 s | 8.46 s | unchanged |
| speedup | 16.6x | 12.5x | 5.6x | 2.7x | 2.3x | 1.6x | 1.3x | 1.0x |

This is time-to-first-token for a chat turn, which prefills only the tokens
prefix reuse did not already cover. Decode speed and long-prompt prefill
(>=128 tokens, which stays on the BLAS path) are deliberately unchanged -
this work removed a fixed overhead, it did not raise the bandwidth ceiling.
The kernels are architecture-agnostic: they operate on quant codes, so every
supported model benefits without a per-architecture path.

Measured on a 4-core Intel Xeon 2.80 GHz Linux container, Python 3.11,
NumPy 2.4.6 (OpenBLAS), with a stories15M-shaped synthetic model from
`tests/make_bench_model.py` (same architecture dimensions as the real
stories15M; CI runs the real one). Decode and RSS are medians of 3 runs;
prefill on the 64-token rows is a single ~50 ms window and jitters
+-40% on this shared machine, the 256-token rows are steadier:

| Run (`tests/bench.py`) | Mode | Load | Prefill | Decode | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4_0, 64 prompt / 32 decode, ctx 128 | quantized weights | 0.084 s | ~1,100 tok/s | 63.2 tok/s | 65.5 MB |
| Q4_0, same run | `ALPACCAROO_F32=1` dense | 0.130 s | 3,192 tok/s | 126.7 tok/s | 107.1 MB |
| Q4_0, 256 prompt / 128 decode, ctx 512 | quantized weights | 0.083 s | 2,998 tok/s | 60.3 tok/s | 71.3 MB |
| Q4_0, same run | `ALPACCAROO_F32=1` dense | 0.128 s | 4,393 tok/s | 131.3 tok/s | 115.7 MB |
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
- **Decode speed**: quantized decode remains ~0.5x of `ALPACCAROO_F32=1`
  dense decode. This is a measured NumPy ceiling, not a missing
  optimization in this codebase: OpenBLAS SGEMV runs multithreaded at
  memory bandwidth (0.53 ms for the dominant 32000x288 output projection),
  while NumPy has no mixed int8xf32 GEMV primitive - every strategy
  (einsum, astype+GEMV, integer matmul) pays a single-threaded conversion
  pass that costs 4.5-9 ms on the same matrix. Per-token profile of the
  quantized path after the overhead trims: ~76% in the int8 matvec
  kernels (~54% just the output projection), ~9% residual Python
  overhead, the rest attention/normalization. Closing that gap needs
  native SIMD dot-product kernels - which is exactly what the optional
  pinned kernel tier below provides, while staying our own Python
  source.

#### Three machines, and what generalises between them

A tokens-per-second figure without its machine is not a measurement. Every
number in this project now comes with one of these:

| | machine A | machine B | machine C |
|---|---|---|---|
| CPU | Ryzen 5 7640HS (Zen 4) 6C/12T | Intel Tiger Lake 4C/8T | Ryzen 7 7730U (Zen 3) 8C/16T |
| memory | DDR5 | LPDDR4x | DDR4 |
| sustained bandwidth | **59-60 GB/s** | **2.2-4.0 GB/s** | **25.4-26.8 GB/s** |
| clock stability (spread) | flat | 1.4-1.8x | **1.06x** |
| AVX-512 / VNNI | yes | yes | **no** |
| 3B Q4_K_M decode | - | 0.79 tok/s | **11.2-12.3 tok/s** |
| 8B Q4_K_M decode | 9.9 tok/s | - | (does not fit) |

Machine A is ALU-bound at a 60 GB/s wall, B is bandwidth-starved and
thermally unstable, C sits between them and **holds its clocks**. If a
change wins on only one of these, it belongs behind a flag or in the
`alpaccaroo tune` cache, not in a default.

Two results that transfer, both from `prompts/05-RESULTS.md`:

- **A per-call win is not a per-token win, and the gap is worst exactly at
  a threshold.** Two separate knobs have now been measured to win 6-14% in
  isolation and *lose* end to end (`ALPACCAROO_SERIAL_QUANTIZE_COLS`, 22 of
  25 rounds lost; `ALPACCAROO_SERIAL_MATVEC_ELEMS`, 21 of 25 lost). Both
  were calibrated on per-call data at the crossover, which is the one
  region where per-call and end-to-end measurements diverge.
- **The native integer kernels cover less of the model population than they
  look like they do.** They handle Q4_K/Q5_K/Q6_K. A Q8_0 file - what
  `alpaccaroo pull llama3.2:1b` gives you - runs 100% on the generic
  `numba-codes-f32` path, and a model whose embedding width is not a
  multiple of 256 (Qwen2.5-0.5B's 896, for instance) cannot be K-quantized
  at all by llama.cpp and lands 83% on the same fallback. Check with
  `alpaccaroo run <model> --profile` before assuming which kernel you are
  benchmarking.

### Spending RAM for speed: the dense-weight budget

Because dense BLAS is the fast path and quantized storage is the small
path, the practical dial for 1B-8B models is `ALPACCAROO_DENSE_WEIGHT_MB=N`:
at load time Alpaccaroo expands up to `N` MiB of the most decode-critical
matrices to dense float32 and keeps the rest quantized. Matrices are
picked in measured-benefit order - FFN projections first (they dominate
llama-class decode), then attention q/output, then k/v, then the output
projection; a token embedding is only densified when it doubles as a tied
output matrix. Chosen matrices never keep their quantized copy, so unlike
`ALPACCAROO_HOT_WEIGHT_MB` nothing is stored twice.

Measured on the same Linux container with a 1.1B-parameter
TinyLlama-shaped synthetic Q4_0 model (GQA 32/4 heads, untied output;
`tests/make_bench_model.py --embd 2048 --ff 5632 --layers 22 --heads 32
--kv 4 --untied`), 32-token prompt / 16-token decode:

| `ALPACCAROO_DENSE_WEIGHT_MB` | Storage | Prefill | Decode | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| unset (all quantized) | 156 quant | 18.1 tok/s | 1.29 tok/s | 1.85 GB |
| `3100` (FFN stack dense) | 76 quant + 80 dense | 34.6 tok/s | 3.81 tok/s | 4.07 GB |
| `4000` (all but embedding) | 1 quant + 155 dense | 49.3 tok/s | 7.34 tok/s | 4.79 GB |
| `ALPACCAROO_F32=1` (everything) | 156 dense | 53.4 tok/s | 7.37 tok/s | 4.97 GB |

Decode scales almost linearly with how much of the per-token matvec work
runs through BLAS: the FFN-only budget buys 3.0x decode for ~2.2 GB, and
the everything-but-embedding budget matches full float32 speed while the
embedding stays quantized. For an 8B model (e.g. Hermes-3-Llama-3.1-8B
Q4), the FFN stack is ~22.5 GB (21.0 GiB) dense, so
`ALPACCAROO_DENSE_WEIGHT_MB=24000` is the "fast decode if you have ~35 GB
total RAM" setting, and smaller budgets degrade gracefully - every MiB
goes to the highest-impact matrices first. `tests/bench.py` prints the
resulting storage split per run.

### Our own kernels: native speed, still our Python

`python -m pip install ".[kernels]"` from this checkout adds the third tier:
the fused quantized-matvec algorithms in `alpaccaroo/kernels.py` - written and
maintained as ordinary Python in this repository - get JIT-compiled to
native SIMD machine code at runtime by Numba, **pinned at
`numba==0.65.1`** (a validated pair with this code; the pin is never
updated implicitly, and a different installed version deactivates the
kernels rather than running unvalidated). Weights stay quantized in RAM
(~1.1-1.3 bytes/weight) and the kernel reads them exactly once per
token, fused dequant-and-dot, multithreaded.

Microbenchmarks on the 4-core container, 8B-class (Hermes) matrix
shapes, int8 codes + scales vs the same matvec through NumPy:

| Matrix | dense BLAS f32 | NumPy einsum (quant) | our kernel (quant) |
| --- | ---: | ---: | ---: |
| 4096x4096 attn | 0.69 ms | 6.20 ms | **0.74 ms** |
| 14336x4096 ffn | 2.34 ms | 22.4 ms | **2.23 ms** |
| 4096x14336 ffn | 2.74 ms | 22.4 ms | **1.86 ms** |
| 128256x4096 output | 39.9 ms | 278 ms | **48.0 ms** |

~10x the NumPy quantized path, and dense-BLAS-class speed while reading
3.5x fewer bytes - which is exactly why, when the kernels are active,
`alpaccaroo run` keeps weights quantized instead of densifying: fastest
path and lowest RAM at the same time. Without Numba nothing changes;
the NumPy and pure paths remain the reference and the fallback
(`ALPACCAROO_KERNELS=0` disables; `ALPACCAROO_KERNELS=force` accepts an
unpinned Numba at your own risk). First use compiles the kernels once
(~1 s, cached on disk).

**Fast is the default for `alpaccaroo run` and `alpaccaroo serve`**: unless
`ALPACCAROO_DENSE_WEIGHT_MB` is set, the CLI sizes the budget automatically
from detected available RAM (cgroup-aware inside containers). When the
machine can hold every densifiable matrix - checked exactly from the
GGUF header against the residual quantized storage and the KV cache at
the requested context - it spends exactly that; otherwise it reserves
the quantized residue plus a context-scaled KV/runtime allowance and
spends 85% of the remainder. The chosen value is printed at load. Set
`ALPACCAROO_DENSE_WEIGHT_MB=0` for the low-RAM all-quantized mode, or an
explicit MiB value to pin the budget - any set value pins the budget,
and unparseable values fall back to all-quantized. RAM detection uses
`/proc/meminfo` on Linux, `GlobalMemoryStatusEx` on Windows, and a
conservative half-of-physical heuristic on macOS; if detection fails the
CLI says so and stays quantized. Library use (`Model.load`) keeps the
explicit opt-in semantics so embedders and tests get deterministic
storage.

Rules of thumb:

- **with the pinned kernels** (`python -m pip install ".[kernels]"` from this
  checkout): quantized weights become the fastest *and* smallest path;
  1B-class models decode at several tok/s and 8B-class models become usable.
  Prefer this tier whenever you can install the pinned Numba.
- **with NumPy**: tiny and 1B-class models are the practical target. Use
  quantized weights when RAM is the constraint, `ALPACCAROO_DENSE_WEIGHT_MB`
  to spend whatever RAM you can spare on decode speed, and `ALPACCAROO_F32=1`
  when the full float32 expansion fits comfortably anyway.
- **stdlib only**: tiny models (stories15M-class) are fine; 1B is slow. Good
  for air-gapped checks, not long conversations.
- **chat/server reuse**: repeated turns or requests with a shared prompt prefix
  skip already-cached K/V work automatically, and interleaved conversations
  restore each other's discarded contexts through the multi-slot prefix
  cache (`ALPACCAROO_PREFIX_CACHE_MB` below) instead of re-prefilling.

Useful environment knobs:

- `ALPACCAROO_PURE=1`: force the standard-library backend.
- `ALPACCAROO_KERNELS=0`: disable the optional JIT kernels;
  `ALPACCAROO_KERNELS=force` accepts a non-pinned Numba at your own risk.
- `ALPACCAROO_DENSE_WEIGHT_MB=N`: densify up to `N` MiB of the most
  decode-critical matrices at load time (FFN first) and keep the rest
  quantized - the main RAM-for-speed dial. The CLI auto-sizes this from
  available RAM when unset; `0` disables densification; library callers
  opt in explicitly. See the table above.
- `ALPACCAROO_F32=1`: force the NumPy loader to expand all quantized matrices
  to float32, useful for A/B checks and small models where BLAS wins.
- `ALPACCAROO_PREFILL_CHUNK=N`: prompt batch size for NumPy prefill; default 256.
- `ALPACCAROO_PREFIX_CACHE_MB=N`: MiB budget (default 1024, `0` disables) for
  the multi-slot prefix cache. When `prefill` switches away from a live
  context about to lose 256+ tokens of K/V work, those rows are
  snapshotted, keyed by their exact token ids; switching back restores
  the slot instead of re-prefilling, so interleaved conversations - even
  behind a long shared system prompt - each pay their unique prefill once.
  Slots are LRU-evicted against the budget, a mid-run shrink drains the
  store on the next call, and a restore copies every shared row, so a
  warm prefill is byte-identical to a cold one on the same tier path.
  `0` is exactly the pre-feature single-cache behavior; the stdlib tier
  never uses slots. `Model.prefix_cache_stats()` counts
  slots/bytes/hits/misses/saves/evictions and `describe()` shows a
  `prefix cache` segment while slots exist.
- `ALPACCAROO_SMALL_MATVEC_ELEMS=N`: matrices below `N` elements use the
  batched-matmul quantized matvec instead of the einsum one. Default 0 (off) -
  einsum measured faster at every shape a llama- or Gemma-class model uses, on
  two different machines. Re-measure before raising it.
- `ALPACCAROO_HOT_WEIGHT_MB=N`: optional lazy dense float32 cache for quantized
  matrices, capped at `N` MiB. Unlike the dense budget this caches at first
  use and keeps the quantized copy too; prefer `ALPACCAROO_DENSE_WEIGHT_MB`
  unless you specifically want runtime-populated caching.
  (`ALPACCAROO_UNPACKED_WEIGHT_MB` is gone; the int8 unpacked form is now the
  default storage and needs no budget.)

### The GPU tier: CUDA kernels, still our Python

`python -m pip install ".[gpu]"` from this checkout adds a fourth optional
tier: the quantized matvec/matmul kernels in `alpaccaroo/cuda.py` - written
and maintained as ordinary Python in this repository - get JIT-compiled
for the local NVIDIA GPU at runtime by **numba-cuda, pinned at `0.30.4`**
together with its exact CUDA wheel set (same pin policy as `[kernels]`:
the pins are a validated combination, never bumped implicitly, and a
different installed version deactivates the tier rather than running
unvalidated; `ALPACCAROO_GPU=force` overrides at your own risk). Weight
matrices upload once at load in the same int8-codes + float32-scales
layout the NumPy backend unpacks and stay quantized in VRAM (~1.25
bytes/weight). The KV cache stays host-side and authoritative - prefix
reuse and truncation are untouched - but the hot math now runs on the
device: prefill's batched causal attention and its fused
GEMM-silu-GEMM FFN, and (when every chain matrix of a llama-class model
is resident) whole-token decode as a device-resident chain with one
synchronization per token. Every piece degrades per call or per
instance to the exact CPU paths, so every other tier keeps working
unchanged and `ALPACCAROO_GPU=0` restores them exactly. Like everything
else here the wheels install once and run offline; keep copies
(`pip download ".[gpu]"`) if the machine will be air-gapped later.

Measured on this machine (RTX 4090 24 GB, driver 591.86, Windows 11)
with `tests/bench.py` (512-token prefill, greedy decode), GPU tier vs
the CPU kernels tier:

| Model | tier | prefill tok/s | decode tok/s |
| --- | --- | ---: | ---: |
| llama3.2:1b Q8_0 | cpu kernels | 84.5 | 33.2 |
| llama3.2:1b Q8_0 | **gpu** | **1250** | **85.6** (87.8 steady) |
| Hermes-3 8B Q4_K_M | cpu kernels | 7.7 | 9.3 |
| Hermes-3 8B Q4_K_M | **gpu** | **279.8** | **52.1** (64.5 steady) |

The bench decode column includes the decode chain's one-time lazy build
(K/V-mirror allocation and upload) on the first token, amortized over
only 29-43 tokens before the model hits end-of-generation; "steady" is
the same greedy loop measured over 100 tokens after warmup.

Prefill: the tiled shared-memory GEMM from v1 (10.0 TFLOP/s on the 8B's
fused FFN matrix, batch 256) plus two stage-2 device paths - the batched
causal GQA attention that used to be an O(n^2) NumPy einsum, and a fused
GEMM-silu-GEMM FFN that keeps the (batch, 2*n_ff) gate|up block in VRAM.
The einsum was why prefill *degraded with length*; the 8B ladder, same
machine, end to end:

| prompt tokens | v1 (einsum attention) | stage 2 |
| ---: | ---: | ---: |
| 512 | 11.67 s | 2.08 s |
| 1024 | 27.0 s | 3.70 s |
| 2048 | 71.2 s | 7.64 s |
| 4096 | 210.6 s | 16.21 s |

At 4096 the remaining 16.2 s split into ~2.8 s of GPU attention (of
which 1.2 s is re-uploading the host-authoritative K/V slices - 10.5
GiB over the run at 8.6 GiB/s), ~2.8 s of GEMM calls, and ~10.6 s of
host-side batch work (rope, rmsnorm, residuals) that still runs on
NumPy between kernels. A 4,400-token prompt (`-c 6144`) prefills in
17.8 s.

Decode runs as a device-resident chain when every chain matrix of a
llama-class model is in VRAM: rmsnorm, the fused qkv matvecs, rope,
single-token GQA over a device K/V mirror (split across
512-position-chunk blocks - one block per head measured 26.1 tok/s at a
4400-token context, the split form 53.0), silu*up and the
residual-fused projections, ~354 queued launches and ONE
synchronization per token, the logits download. The host cache stays
authoritative: each token's new K/V rows are async-copied back inside
that same sync, and every host mutation path (prefill writes,
truncation, reset, CPU decode) invalidates the mirror so it re-uploads
exactly the changed rows. Any failure - build or runtime - parks the
chain for that model instance and the token is recomputed on the
existing per-matvec path. Where a steady-state 8B token goes now
(15.5 ms, 64.5 tok/s, up from 27.5-31 in v1): 9.9 ms of Python launch
queueing overlapped under 15.4 ms of GPU work (the sync waits the 5.5 ms
difference), 0.1 ms of host tail. The tied-embedding 1b pays its
128256-row output head on the CPU inside the chain's one sync (5.1 of
its 11.4 ms), because the token embedding still never uploads.

The 8B keeps 8.7 GiB of weights resident in VRAM (161 of its 226
matrices) plus a 1.1 GiB K/V mirror at the default 4096 context; the
token embedding never uploads - row gathers are the one workload these
kernels are wrong for. Placement is per-matrix, so running out of VRAM
mid-load just leaves the remaining matrices on the CPU tiers with one
warning line - a capped run (`ALPACCAROO_GPU_VRAM_MB=500` on the 1b: 44
matrices on GPU, the rest on CPU) decodes token-identically to the
uncapped one, with the chain declining mixed placement.

Correctness, measured: GpuMatrix matches QuantMatrix within 9.9e-7
worst relative error across all ten quant formats (bar 2e-5), row access
is bit-exact, and the batch attention kernel matches the NumPy einsum
within 1.9e-6 worst relative error over randomized ragged shapes
(batch==t, batch<t continuation, GQA groups 1-8, sliding windows).
48-token greedy decode through the full chain is token-identical to the
exact f32 reference path on both test models (llama3.2:1b vs the CPU
tiers directly, 48/48; the 8B vs `ALPACCAROO_INT_DOT=0`, 48/48 - the
int-dot CPU tier is approximate by design, so greedy text can drift
from *it* on Q4_K_M while both stay glued to the reference), and a
two-prompt shared-prefix sequence (prefill, decode, re-prefill with a
different suffix, decode - the truncation/mirror-invalidation path) is
token-identical chain-on vs chain-off.

GPU knobs:

- `ALPACCAROO_GPU=0`: disable the tier; `ALPACCAROO_GPU=force` accepts a
  non-pinned numba-cuda at your own risk.
- `ALPACCAROO_GPU_CHAIN=0`: keep the tier but disable the device-resident
  decode chain (decode falls back to the v1 per-matvec dispatch).
- `ALPACCAROO_GPU_VRAM_MB=N`: cap uploaded weight bytes (the
  mixed-placement test hook; free VRAM minus a 512 MiB reserve decides
  otherwise).
- `ALPACCAROO_KV_F16=1`: opt-in half-precision K/V mirror for the decode
  chain (stores cast f32 -> f16 on the device, kernels read back up to
  f32; the host cache stays f32 and authoritative). Halves the mirror's
  VRAM - 1.1 GiB -> 0.55 GiB for the 8B at ctx 8192 - which is the
  measured benefit; decode at depth 8000 is unchanged (49.2 vs 49.4
  tok/s) and prefill pays ~4% for the converts, so treat it as a VRAM
  dial, not a speed dial. The numerics legitimately shift (~1e-3): no
  byte- or token-parity claim is made for this mode, and the default
  stays f32 so every parity guarantee above holds untouched.
- `ALPACCAROO_GPU_WIDE_MATMUL_ELEMS` is retired: the tiled GEMM handles
  every matrix size in one launch, so there are no longer two batched
  kernels to choose between. The variable is accepted and silently
  ignored so existing environments keep working.

One installation trap the `[gpu]` pins exist to prevent: with
`nvidia-nvjitlink` missing, cuda.core's DLL search can silently find a
torch wheel's bundled stale nvJitLink and every kernel dies with
`ERROR_OUTDATED_LIBRARY(14)`. The availability probe compiles and runs a
real kernel before the tier may activate and reports the actual cause
through `alpaccaroo doctor`, which shows the device, free VRAM and tier
status either way.

### Ollama-native API

`alpaccaroo serve` also speaks the Ollama REST protocol on the same port, so
the official `ollama` Python client works unmodified - point it at the
server and `client.chat(...)`, `client.generate(...)`, `ollama.list()`,
`show()`, and `ps()` behave as they do against Ollama itself (verified
end to end against `ollama` 0.6.1, streaming and non-streaming):

```python
import ollama
client = ollama.Client(host="http://127.0.0.1:8080")
client.chat(model="llama3.2:1b",
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 64, "seed": 1})
```

Implemented: `POST /api/chat` and `POST /api/generate` (newline-delimited
JSON streaming by default, exactly like Ollama; `stream: false` for a
single response), `GET /api/tags` (the local model store, with the served
model always present), `POST /api/show` (the loaded model's real GGUF
metadata), `GET /api/ps`, and `GET /api/version`. Token counts in the
responses are exact - counted by the generation loop, not inferred.
Durations are honest wall-clock nanoseconds: `eval_duration` is the decode
loop's own clock, `total_duration` is the request's, and
`prompt_eval_duration` is the difference (genuinely prefill plus
render/lock overhead - no fabricated split). Requests naming a model this
process is not serving get Ollama's 404 error shape; `name` and
`name:latest` are the same model.

Deliberate divergences, stated plainly: one model is resident per server,
so `keep_alive` is accepted and ignored and `/api/ps` reports a far-future
expiry; `options.num_ctx` is accepted but the context window was fixed at
load time - an oversize ask proceeds at the loaded window instead of
erroring; `format` as a JSON-schema dict is honoured as `format: "json"`
(the output is valid JSON, the schema itself is not enforced); and
`/api/generate` returns `context: []` rather than a resumable token list.

### Guaranteed-valid JSON output

`chat.generate(...)` and `chat.chat_once(...)` take a keyword-only
`json_only=True`. With it, every fragment the call streams or returns is a
prefix of one syntactically valid JSON value, and generation stops
(`stop_reason "stop"`) the moment the top-level value closes. This is a hard
guarantee at any temperature, on any supported model, on both the NumPy and
pure-Python backends - not a prompt trick.

The mechanism is a byte-level JSON grammar guard (`alpaccaroo/jsonform.py`)
driving a candidate-rejection loop: each position is sampled normally, the
candidate token's raw bytes are tested against the guard, and a token that
would break the JSON is masked to -inf and the position resampled. The guard
works on bytes rather than text because byte-level BPE tokens can split a
UTF-8 character; end-of-generation tokens are banned until the value is
complete, so a model that gives up mid-object is pushed to close it instead.
After 512 rejections at one position the engine falls back to a ranked scan
of the whole vocabulary, which is the guarantee rather than the fast path:
measured over 10 llama3.2:1b replies at temperature 0.9 (807 positions), the
loop rejected 33 candidates in total and never more than 4 at one position,
and all 10 replies parsed with `json.loads`. A 24-seed sweep on the tiny test
fixture at temperature 2.0 produced only valid values or valid prefixes on
both backends.

Measured cost on llama3.2:1b, 64-token decode, temperature 0.9: 27.3 ms/token
plain vs 27.4 ms/token with `json_only` (+0.2%), plus a one-time 83 ms
token-bytes table build per tokenizer (cached on the model).

Honest limits: the guarantee is syntax, not schema - the model still chooses
the keys and values, so keep prompting for the JSON you want (the prompt is
never modified). A reply that hits the `n_predict` or context budget is cut
mid-value; it is still a valid prefix, and `stop_reason` says `"length"` or
`"context"` so the caller can tell. A reply that is a bare top-level number
ends at the model's EOG (`stop_reason "eog"`), because a number has no
closing character. Nesting is capped at 256 levels. If no token in the
vocabulary could continue the JSON - impossible for any tokenizer that
covers ASCII - the call raises `ValueError` rather than emit broken output.

<!-- suggested bullet for "Landed recently": -->
- Guaranteed-valid JSON decoding (`json_only=True` on `chat.generate` /
  `chat.chat_once`): a byte-level grammar guard plus candidate-rejection
  sampling makes every reply parseable JSON at any temperature, for +0.2%
  measured decode overhead on llama3.2:1b.

## Roadmap

The mission is fixed - pure Python, fast and reliable, all our own code -
and the roadmap orders the work that serves it.

**Landed recently**

- A third machine's worth of cross-device measurement (`prompts/05-RESULTS.md`):
  the benchmark matrix for 0.5B/1B/3B across five prompt shapes and three
  context windows, a grouped **Q4_K+Q5_K** kernel so Q4_K_S files get the
  same one-dispatch treatment Q4_K_M files already had, and two defect
  fixes it exposed - the first decode token of a Q4_K_M model was paying a
  **1865 ms in-token JIT compile** because the round-4 pair kernel was
  never added to `warmup()`, and the `alpacca` -> `alpaccaroo` rebrand had
  moved the model store so an upgrading user's models went invisible and
  silently re-downloaded. Both now have regression tests; the warmup one
  pins the *property* (every compiled kernel must have a signature after
  `warmup()`), so the next kernel cannot reintroduce it.
  It also produced the round's most useful negative: the narrow-matrix
  dispatch that has shipped **enabled** since round 4 was validated
  end-to-end for the first time and **lost 21 of 25 rounds**.
- A measurement architecture for performance work: a decode profiler that
  names which kernel ran each matrix (`alpaccaroo run --profile`), a
  benchmark harness with cold/warm separation and a clock-stability probe
  (`alpaccaroo bench`), an opt-in autotuner (`alpaccaroo tune`), a codegen
  census (`alpaccaroo tune --asm`), and CLI reconnect to a resident server
  (`alpaccaroo run --connect`). It produced **no end-to-end speedup** on the
  machine it was built against - which it says plainly - but it made the
  next question decidable: decode there is 94.2% weight-matrix products and
  0.05% Python. Measured negatives, including one of its own changes that
  was reverted after losing 22 of 25 end-to-end rounds, are in
  `prompts/04-RESULTS.md`; the tooling reference is `docs/PERFORMANCE.md`.
- The GPU tier: `alpaccaroo/cuda.py`, our own Python kernels JIT-compiled
  for CUDA, with device-resident prefill, batched GQA attention, a
  device-resident decode chain, and an opt-in half-precision K/V mirror
  (`ALPACCAROO_KV_F16`) as a VRAM dial.
- Ollama-native API, so the official client works unmodified, beside the
  OpenAI-compatible surface.
- Guaranteed-JSON decoding (`json_only`): every emitted fragment is a prefix
  of one syntactically valid JSON value, on any backend at any temperature.
- Multi-slot prefix cache, so interleaved conversations keep their K/V
  instead of re-prefilling whenever the server switches between them.
- Fused quantized *matmul* kernels, so a batched forward pass costs what the
  batch asks for instead of a full-model dequantize. Time-to-first-token on
  Llama-3.1-8B Q4_K_M drops 16.6x for a 1-token prefill and 2.3-2.7x for the
  16-32 token turns a chat actually produces (see "Where the time goes on a
  large model"). Architecture-agnostic: it works on quant codes.
- Repo-owned terminal app menu (`alpaccaroo menu`, or no-arg `alpaccaroo` in an
  interactive terminal), with model switching, history navigation, deletion
  controls, saved-chat statistics, and Esc-to-menu chat return.
- Interactive chat history: local JSON sessions under `ALPACCAROO_HOME`,
  `history list/show/stats/rm/clear --yes`, and stats rows for installed
  models even when they have no saved chats.
- Optional pinned kernels: our Python-source fused quantized matvecs
  JIT-compiled by `numba==0.65.1`; when active, the CLI
  keeps weights quantized by default because that is the fastest and
  smallest path.
- Quantized weight storage with nothing re-dequantized per token (2.9x
  decode over the previous engine). Q4_K/Q5_K/Q6_K keep the file's own block
  fields and stream at **0.578 / 0.703 / 1.070** bytes per weight;
  Q2_K/Q3_K/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 take the int8-codes path at ~1.125,
  and the IQ family still falls back to dense float32 and says so at load.
  Note the inversion this leaves, since decode is bandwidth-bound: a Q4_0
  file streams roughly twice the bytes of a Q4_K one at the same nominal
  precision, and a Q2_K file is smaller on disk but no cheaper to decode.
- The dense-weight budget (`ALPACCAROO_DENSE_WEIGHT_MB`): spend RAM on BLAS
  speed exactly where it pays, FFN projections first - and the CLI sizes
  it automatically from available RAM (cgroup-aware in containers,
  scales its reserve with the requested context), so `alpaccaroo run` is as
  fast as the machine affords by default.
- Batched prefill with last-token-only vocab projection, prefix-aware
  KV-cache reuse across chat turns and server requests, decode-overhead
  trims (grouped attention, precomputed RoPE tables, BLAS-dot rmsnorm).
- A kernel-level study documenting the NumPy quantized-decode ceiling
  (see "Honest performance expectations") so the speed story stays
  honest.

**Next**

- *Product UX:* keep the repo-owned menu and installer launchers in lockstep;
  add history search/export/resume; add richer saved-chat statistics that
  separate assistant messages from timed responses; normalize model identity
  so raw GGUF paths and installed model refs merge in stats; keep default
  model config portable and explicit.
- *Reliability:* broaden the real-model CI gates beyond stories15M
  (K-quant files, qwen2/3, gemma, mistral, 1B-class llama); harden the
  GGUF parser against malformed files; add a server soak test.
- *Pure-path memory:* move stdlib-only weight and KV storage from Python
  lists (~38 bytes per weight, measured) to `array('f')` (~4), with
  optional raw-quant storage, so air-gapped stdlib mode can hold
  1B-class models in normal RAM.
- *Performance within the constraint:* refine the densify ranking with
  per-matrix measurements (now one `alpaccaroo bench` invocation rather
  than a guess), keep shaving prefill and per-token overhead, and extend
  the opt-in f16 K/V option from VRAM to the **host** cache, which is still
  float32 and whose cost grows with context.
- *Operational UX:* clearer RAM/budget messaging, Windows polish, better
  guidance for split GGUFs and unsupported architectures.
- *Server:* broaden OpenAI-compatible behavior while staying on
  `http.server` and usable fully offline.
- *Architecture for speed:* on hardware with bandwidth the kernels already
  sit close to the memory wall, so what remains is structural rather than
  more kernel tuning. Each of these is recorded with its evidence in
  `prompts/03-RESULTS.md`, `prompts/04-RESULTS.md` and
  `docs/PERFORMANCE.md`, and none should be started without re-reading them:
  - *Concurrent generation.* `serve` holds one lock for the whole process.
    A token reads the entire model however many sequences are decoding, so
    batching concurrent requests costs almost no extra bandwidth and is the
    largest throughput lever in the project. It does nothing for
    single-user latency, and that trade should be made explicitly.
  - *Native integer-dot kernels for the non-K quants*, to close the
    inversion noted above; the pattern already exists three times over for
    Q4_K/Q5_K/Q6_K.
  - *Speculative decoding, re-evaluated where bandwidth binds.* Round 3
    measured it capped at ~1.10x, but that was an ALU-saturation argument
    on a 60 GB/s machine; a bandwidth-starved one has idle ALU to verify
    drafts with.
  - *Q6_K 6-bit packing*, on the same conditional reasoning: a measured
    wash at 60 GB/s, worth re-measuring where bandwidth is the constraint.
  - *Fusing the per-token non-weight work* (attention, rope, norms,
    sampling, activation quantization) that costs ~8-9 ms on a fast machine
    against llama.cpp's ~1-2 ms - a large share of the token there, and
    almost nothing on a slow one.

**Non-goals**

- Wrapping llama.cpp, Ollama, PyTorch, or any third-party inference
  runtime - that would be someone else's software.
- Shipping precompiled/native extension modules or required compiled
  dependencies. Optional runtime JIT of our Python-source kernels is allowed
  only behind an explicit, pinned dependency, and the engine must always run,
  and stay readable, as pure Python.
- Marketing numbers. When a performance gate is not met, this README
  says so.

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

CI runs the offline suite on Linux/macOS/Windows for the stdlib and NumPy
paths, adds a pinned-numba kernel job on Ubuntu, and runs the real-model
generation gate on every push.

## Security & supply chain

- **No install-time network access**: the repo is the program. No package
  index, no build step, no binary artifacts, no submodules.
- Release archives ship with SHA-256 checksums.
- Model downloads (`alpaccaroo pull`) are the only network feature, are
  explicit, and verify the publisher's digests. Models carry their own
  licenses - when the publisher provides one, it is stored next to the
  weights.

## Credits

All code here is written from scratch in Python by the Alpaccaroo project.
It interoperates with formats and protocols designed by others, with
thanks - see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md): the GGUF format and
quantization schemes (ggml/llama.cpp project), the Ollama registry protocol,
and the SentencePiece/BPE tokenization algorithms.
