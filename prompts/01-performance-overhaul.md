# Alpacca performance overhaul: bandwidth-aware inference, batched prefill, KV reuse

You are working on **Alpacca** (this repo): a from-scratch, 100% Python GGUF inference
engine with Ollama-style model management. Zero runtime dependencies; NumPy is an
*optional* accelerator auto-detected at import (`alpacca/tensor.py`), and a pure-stdlib
path must always keep working (`ALPACCA_PURE=1` forces it).

Your task is to make the NumPy path dramatically faster **without changing the project's
identity**: no new dependencies, no native code, no GPU, pure Python + optional NumPy only.

## Read these files first, in this order

1. `README.md` — project philosophy, "Honest performance expectations", roadmap
2. `alpacca/model.py` — Model.load, KV cache, `_forward_np`, `_forward_pure`, `prefill`
3. `alpacca/tensor.py` — the NumPy/pure dual backend
4. `alpacca/quants.py` — pure + vectorized NumPy dequantizers (you will reuse these)
5. `alpacca/gguf.py` — mmap reader; `tensor_bytes()` returns a memoryview into the mmap
6. `alpacca/chat.py` — `generate()`, `interactive()` (note `model.reset()` at line ~216)
7. `alpacca/serve.py` — ThreadingHTTPServer + lock; `model.reset()` per request
8. `tests/smoke.py`, `tests/make_tiny_model.py`, `tests/real_model_test.py`

## The diagnosis (do not re-derive it; optimize against it)

Single-stream LLM decode is **memory-bandwidth-bound**: every generated token streams
every weight byte through the CPU once. Alpacca currently dequantizes ALL weights to
float32 at load time (`model.py` `tensor_mat` → `quants.dequantize`), so an 8B Q4_K_M
model (4.9 GB on disk) becomes ~32 GB of f32 in RAM, and the measured 1.75 tok/s on a
DDR5 desktop is ~90% of the theoretical ceiling (60 GB/s ÷ 32 GB/token ≈ 1.9 tok/s).
BLAS is already doing its job — the f32 inflation is the bottleneck. llama.cpp streams
the same model at ~4.9 GB/token, which is the entire ~6.5x decode gap.

Separately, prompt processing (`Model.prefill`) feeds tokens **one at a time** through
`forward()`, so prefill runs at decode speed (GEMV, bandwidth-bound) instead of as
batched GEMM (compute-bound, near-peak BLAS). And every chat turn / server request
calls `model.reset()` and re-prefills the entire conversation from scratch.

Therefore the work, in priority order: batch the prefill (GEMM), reuse the KV cache
across turns, keep weights quantized in RAM with a tiled dequant-matvec, then shave
per-token Python overhead. Decode parity with llama.cpp is explicitly NOT the goal
(that needs native kernels — out of scope); the goals below are measurable multiples.

## Hard constraints (violating any of these = failure)

1. **Zero new runtime dependencies.** stdlib only; NumPy remains optional. It is fine
   to `pip install numpy` in your dev environment for testing the fast path.
2. **The pure-stdlib path keeps working and keeps passing tests.** `ALPACCA_PURE=1`
   must exercise it end-to-end. Pure-path performance/memory may stay as-is; do not
   regress its correctness. CI runs the suite both with and without NumPy installed.
3. **Backends keep verifying each other.** `tests/smoke.py` asserts numpy-vs-pure
   logits agree (diff < 1e-3 on a tiny model). That check must keep passing unchanged.
4. **Public interfaces unchanged**: CLI commands/flags, the OpenAI-compatible API,
   `/completion`, GGUF reader/writer formats, model store layout.
5. **Cross-platform**: Linux/macOS/Windows CI must stay green. No POSIX-only code on
   required paths (POSIX-only niceties like RSS reporting must degrade gracefully).
6. **Keep the house style**: MIT header comments, stdlib-only modules, small focused
   functions, the existing docstring tone. Match `quants.py`'s pure/NumPy dual-impl
   pattern for anything new.
7. **Honesty**: update README performance claims with *measured* numbers; never claim
   llama.cpp/Ollama parity.

Numerical note: reordering float summation (tiling, batching) legitimately changes
outputs in the last bits; acceptance tests check semantics (coherent English, correct
facts), not exact strings. Keep parity tolerances tight: batched-vs-sequential logits
on a tiny F32 model < 1e-4 max-abs-diff; quantized-matvec vs dequantize-then-matvec
< 1e-3 (same dequantized values, only summation order differs).

## Phase 0 — benchmark harness (do this first, commit separately)

Create `tests/bench.py` (stdlib + the repo only):

- Args: `--model PATH_OR_REF`, `--prefill N` (default 512), `--decode N` (default 128),
  `--ctx`, `--seed`. Greedy sampling. Synthetic deterministic prompt is fine (e.g. a
  repeated sentence tokenized to ≥ N tokens, truncated to exactly N).
- Reports: load seconds (`model.load_seconds`), prefill tok/s, decode tok/s, backend
  name, and peak RSS where available (`resource.getrusage` on POSIX — note Linux
  reports KB and macOS bytes; print "rss n/a" on Windows). Print one machine-readable
  summary line, e.g. `BENCH model=... prefill_tps=... decode_tps=... rss_mb=...`.
- Run it on the tiny smoke model (`tests/make_tiny_model.py out.gguf F32` / `Q8_0` /
  `Q4_0`) and, if network is available, the real 19 MB model
  (`hf:ggml-org/models:stories15M-q4_0.gguf`, see `tests/real_model_test.py`).
- **Capture baseline numbers at the current HEAD before any optimization and include
  before/after numbers in every subsequent commit message.**

## Phase 1 — batched prefill (GEMM) with last-token-only logits

The single biggest UX win. In `alpacca/model.py`:

- Add a NumPy-only `Model.forward_batch(tokens: list[int]) -> logits` that processes a
  chunk of T tokens at positions `n_past .. n_past+T-1` in one pass:
  - Hidden states as `(T, n_embd)` f32. RMSNorm vectorized over rows.
  - Projections as GEMM: with weights stored `(out, in)`, use `H @ W.T` (or
    `(W @ H.T).T` — pick one, be consistent). Biases broadcast.
  - RoPE vectorized over positions: precompute the inverse-frequency vector once in
    `__init__` (it is currently recomputed inside `_rope_np` on every call); apply
    cos/sin for the whole position range at once. Both "norm" and "neox" styles.
  - Write K/V for all T positions into the cache, then attend: per layer compute
    grouped-query attention with a causal mask — query at local index i attends to
    cache positions `0 .. n_past+i`. Use reshape/einsum over `(n_kv, group, ...)`
    so there is no per-head Python loop. Keep f32 tiles; do NOT attempt int8 einsum.
  - **Compute the output-vocab projection only for the final token of the final
    chunk** (the vocab matmul is ~0.5–2 GB of weights per token on real models —
    skipping it for prompt tokens is a large win on its own).
- Chunk the prompt (default 256 tokens per chunk, env override `ALPACCA_PREFILL_CHUNK`)
  to bound the `(T, n_ff)` activation and `(T, S)` score memory.
- `Model.prefill()` uses chunked `forward_batch` when NumPy is available; the pure
  path keeps the existing token-at-a-time loop. Same `RuntimeError` when the prompt
  exceeds `n_ctx`.
- **Parity test (add to `tests/smoke.py`)**: on a tiny F32 model, prefill the same
  ~40-token prompt via forward_batch and via sequential forward into two fresh models;
  assert max-abs logits diff < 1e-4 and identical KV cache contents within 1e-5.
- Expected effect: prefill tok/s improves by well over an order of magnitude vs
  baseline on the stories15M model. Gate: **≥ 10x measured prefill speedup** (it
  should comfortably exceed this).

## Phase 2 — KV-cache reuse across turns and requests (prefix caching)

- `Model` tracks `self.cached_ids: list[int]` with the invariant
  `len(cached_ids) == n_past` — exactly the tokens whose K/V are in the cache.
  Updated only where forward passes happen (`forward`, `forward_batch`); `reset()`
  clears it.
- Make `prefill(ids)` prefix-aware: compute the longest common prefix between `ids`
  and `cached_ids`; truncate the cache to that length (NumPy: set `n_past`; pure:
  `del cache[li][n:]`) and forward only the suffix. Edge case: if the prefix equals
  the *entire* `ids` (e.g. a regenerate request), truncate to `len(ids) - 1` and
  re-forward the last token so logits are produced.
- `chat.generate()` continues to feed ALL prompt ids to the sampler (repeat-penalty
  context must not change) — only the model-side forwarding is skipped.
- Remove the now-redundant resets: `interactive()` in `chat.py` (the
  `model.reset()  # re-prefill whole conversation` line) and the three
  `model.reset()` calls in `serve.py` (`completion`, streaming and non-streaming
  `chat_completions`). The LCP logic keeps unrelated conversations correct
  automatically (small prefix → mostly re-prefilled). The server's existing
  generation lock already serializes access; keep it.
- Expose a counter for tests (e.g. `model.last_prefill_forwarded: int`).
- **Tests**: simulate two chat turns; assert the second turn forwards only the suffix
  (counter, not wall-time). Same for two sequential `/v1/chat/completions` requests
  sharing a conversation prefix. Assert a conversation still produces correct output
  after truncation (turn 2 diverging from a longer cached turn-1+answer).

## Phase 3 — keep weights quantized in RAM (tiled dequant-matvec)

The big decode + memory win. Design:

- New module `alpacca/qmatrix.py` (NumPy-only feature): class `QuantMatrix` holding
  the tensor's **raw GGUF block bytes copied out of the mmap** (the mmap closes in
  `Model.load`'s `finally`, so views must be copies — `np.frombuffer(bytes(mv), ...)`
  pattern as in `quants._np_blocks`), plus `dtype, rows, cols`.
  - GGUF layout fact: quant blocks run along the innermost (cols) dimension, so the
    byte stride of one row is `(cols // block_elems) * block_bytes`; any row range is
    a contiguous byte slice. Reuse the existing verified decoders in `quants.py`
    (`_np_deq_q8_0`, `_np_deq_q4_0`, `_np_deq_q4_k`, `_np_deq_q5_k`, `_np_deq_q6_k`,
    plus straightforward NumPy decoders you add for Q4_1/Q5_0/Q5_1, and F16 via
    `.astype`) on those row-slices. Refactor for reuse; do not duplicate decode logic.
  - `matvec(x)`: iterate row tiles sized so the dequantized f32 tile stays cache-
    resident (target ~2–4 MB, i.e. `tile_rows ≈ max(32, (2<<20) // (cols*4))`);
    per tile: dequant slice → `(tile, cols)` f32 → `y[tile] = tile_f32 @ x`. Total
    DRAM traffic ≈ the quantized bytes, which is the entire point.
  - `matmul_t(X)` for batched prefill: same tiling, `Y[:, tile] = X @ tile_f32.T` —
    dequant cost amortized over the whole chunk (synergy with Phase 1).
  - `row(i)`: dequantize a single row (token-embedding gather; also `rows(list)` for
    batched embedding lookup).
  - Implement `__matmul__` for the 1-D case so `_forward_np`'s `ly.wq @ h` keeps
    working unchanged; `forward_batch` calls `matmul_t` via a small dispatch helper.
- `Model.load`: when NumPy is active, the dtype is supported, `cols` is a multiple of
  the block size, and the tensor is a weight **matrix**, wrap it in `QuantMatrix`
  instead of dequantizing. Vectors (norms, biases) stay dequantized f32. Unsupported
  dtypes (Q2_K, Q3_K, BF16) and the pure path keep the existing f32 expansion.
  Tied output (`output = tok_embd`) must keep working.
- Escape hatch: `ALPACCA_F32=1` env var forces the old full-dequant behavior. All
  A/B benchmarks in this phase are quant-path vs `ALPACCA_F32=1` on the same commit.
- **Tests**:
  - Block-level parity per format: build random valid tensors (use
    `quantize_q8_0`/`quantize_q4_0` for those; for K-quants, take bytes from a real
    tiny tensor or craft random blocks with the f16 scale fields patched to small
    finite values), assert `QuantMatrix.matvec(x)` ≈ `dequantize(...) reshaped @ x`
    within 1e-3, and `matmul_t` consistent with stacked matvecs.
  - End-to-end: smoke already writes tiny Q8_0/Q4_0 models — run generation with the
    quant path and with `ALPACCA_F32=1`; assert greedy argmax token sequences match
    on a short generation, and logits stay within 1e-2.
  - Real-model gate: `tests/real_model_test.py` (stories15M-q4_0) must pass on the
    quant path.
- **Gates**: on a quantized real model, decode tok/s ≥ 2x vs `ALPACCA_F32=1`; peak
  RSS ≤ 1/3 of the f32 path; load time should drop substantially (no full dequant) —
  report it. RAM for an 8B Q4_K_M should land well under 10 GB (vs ~36 GB today);
  state the measured value in the commit message if you can test one.

## Phase 4 — decode-path Python-overhead cleanup

Only after Phase 3 (at 570 ms/token this is noise; at <100 ms/token it matters):

- Replace the per-head Python loop in `_forward_np` (~n_head small NumPy calls per
  layer per token) with one grouped einsum/matmul over `(n_kv, group, head_dim)`,
  mirroring the Phase 1 attention shapes with T=1.
- Use the precomputed RoPE inverse-frequency (and optionally a lazily-grown cos/sin
  table up to `n_ctx` — it is ~2 MB at 4k ctx); remove the redundant
  `x.astype(np.float32)` copy inside `rmsnorm` for already-f32 inputs.
- Optional stretch, flag-gated `ALPACCA_KV=f16`: store the KV cache as float16 and
  upcast per-use (halves cache traffic at long contexts). Default stays f32. Skip if
  it threatens the parity checks.
- Gate: measurable decode improvement on a quantized 1B-class model (target ≥ 15%);
  if profiling shows it is already negligible post-Phase-3, say so in the commit
  message with numbers instead of forcing it.

## Phase 5 — documentation and honest numbers

- Update README "Honest performance expectations" with a measured before/after table
  (machine described, model named, prefill and decode tok/s, RSS) and rewrite the
  rules of thumb (e.g. 8B Q4_K_M now needs ~6 GB instead of ~35 GB). Keep the honest
  framing: llama.cpp-class decode still requires native kernels; that remains a
  non-goal.
- Tick the roadmap items now done (quantized matvec backend, parts of kernel fusion);
  reword what remains.
- Document new env vars (`ALPACCA_F32`, `ALPACCA_PREFILL_CHUNK`, `ALPACCA_KV` if
  shipped) in the README, and `tests/bench.py` usage under Testing.

## Verification protocol (every phase, before its commit)

```sh
python3 tests/smoke.py                       # all checks, must pass 100%
ALPACCA_PURE=1 python3 -m alpacca --help     # pure path still imports/runs
python3 tests/real_model_test.py             # network; run when available
python3 tests/bench.py --model <tiny + real> # record numbers for the commit message
```
Also run smoke in an environment without NumPy if you can (CI does); at minimum,
audit that every new code path is either NumPy-gated or has a pure twin.

## Git protocol

- Work on the branch this session designates; create it if needed.
- One commit per phase, message format `perf: <what>` with measured before/after
  numbers in the body. Run the verification protocol before each commit.
- Push with `git push -u origin <branch>` when done. Do NOT open a PR unless asked.

## Mandatory closure protocol

Before ending the session, read `C:\Users\Jon\Desktop\MANDATORY-CLOSURE-PROTOCOL.md`
and execute every step it contains. If the session is not running on the machine
where that path exists (e.g. a remote Linux container), state explicitly that the
protocol file was not accessible and complete the Git protocol above in full instead.

## Explicitly out of scope (do not do these)

Native/C/Cython/Numba kernels; GPU backends; new dependencies (hard or optional
beyond NumPy); threading changes; cross-request batching or parallel decode in the
server; speculative decoding; new model architectures; multi-part GGUFs; tokenizer
or sampler changes; changing pure-path memory behavior.

## Definition of done

- [ ] Phases 0–3 landed and gated as specified (Phase 4 landed or justified-skipped
      with numbers; Phase 5 landed)
- [ ] Prefill ≥ 10x faster (measured); second-turn chat forwards only the suffix
      (counter-verified)
- [ ] Quantized models decode ≥ 2x faster than `ALPACCA_F32=1` with ≤ 1/3 the RSS
- [ ] `tests/smoke.py` fully green with and without NumPy paths exercised; real-model
      test green; numpy-vs-pure parity check untouched and passing
- [ ] README updated with measured numbers; all work committed and pushed
- [ ] `C:\Users\Jon\Desktop\MANDATORY-CLOSURE-PROTOCOL.md` read and executed (or its
      inaccessibility explicitly reported)
