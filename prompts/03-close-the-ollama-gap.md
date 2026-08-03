# Alpaccaroo: close the decode-throughput gap with llama.cpp, in pure Python

> **This file is the source of truth for this task, not the chat message that
> pointed you here.** It is committed to the repo at
> `prompts/03-close-the-ollama-gap.md`. This is long work and your context will
> almost certainly be compacted before it is done. When that happens — or any
> time you are unsure of a number, a constraint, or whether something was
> already tried — **re-read this file from disk** rather than relying on your
> memory of it or on a summary. Nothing here is superseded by a summary.
>
> Keep a running log at `prompts/03-RESULTS.md` as you work: one line per
> avenue with what you tried, what you expected, what you measured, and whether
> you kept it. Write to it *as you go*, not at the end. If you lose context,
> that file plus this one is enough to resume without repeating work.

You are working on **Alpaccaroo** (`/home/hoopbot/alpaccaroo`), my own project: a
from-scratch GGUF inference engine written entirely in Python. NumPy is an
optional accelerator; an optional, pinned Numba JIT compiles Alpaccaroo's *own
Python source* kernels to native SIMD at runtime. No C, no C++, no shipped
binaries, no GPU. I own this codebase and I'm asking you to modify it freely.

**Goal: raise Alpaccaroo's 8B decode throughput toward llama.cpp's (measured via
Ollama) on the same machine, without leaving pure Python.**

A prior engineering pass concluded that parity is **not reachable** and that the
engine is ~2× short, permanently, absent hand-written SIMD. Its measurements are
reproduced below so you don't repeat its dead ends. **Treat that conclusion as a
hypothesis to test, not a settled result.** It may be wrong; parts of its
reasoning were later shown to be wrong (§5). Your job is to settle the question
with evidence and to move the number if it can be moved.

Don't open by agreeing or disagreeing. Open by reproducing.

---

## 1. Success criteria

The only metric that decides success:

```
8B Q4_K_M, single stream, CPU only, warm JIT, model already loaded:
    tokens/second of decode, Alpaccaroo vs Ollama, same machine.
```

| | Ollama (llama.cpp) | Alpaccaroo today | target |
|---|---:|---:|---:|
| Decode | 11.0–12.2 tok/s | 5.0–5.15 tok/s | **≥ 11** |
| Prefill (911 tok) | 32.0 tok/s | ~10 tok/s | stretch goal |
| Weights in RAM | ~4.6 GiB | 9.35 GiB | lower is better |

Partial credit is real: 5.0 → 8 tok/s would be a major result. Report honestly
what you achieved. Don't round up, and don't quote a microbenchmark as if it
were end-to-end throughput.

**Hard constraints.** Violating any of these makes the work useless to me:

- No C/C++/Rust/assembly source, no compiled extension modules, no GPU. Numba
  JIT of Alpaccaroo's own Python source is the intended vehicle. NumPy is fine.
- No new *required* runtime dependencies. NumPy stays optional; the pure-stdlib
  path (`ALPACCAROO_PURE=1`) must keep working and stay correct.
- **Write original code.** Alpaccaroo's premise is that every algorithm in it is
  ours. Where llama.cpp is referenced below, it is as a description of a
  published technique (quantization block layouts, fused dequantize-dot, VNNI
  integer dot products) — implement from the specification and from first
  principles. Do not copy source from llama.cpp or any other project into this
  repo. Any third-party material that legitimately needs attribution goes in
  `THIRD-PARTY-NOTICES.md`.
- Numerical parity: outputs must stay correct. Quantify any change in numerics
  rather than hand-waving it.
- Both test suites stay green and their check counts must not drop.

---

## 2. The machine (every number below is from it)

```
AMD Ryzen 5 7640HS, 6 physical cores / 12 threads, DDR5, 25 GB RAM
Radeon 760M iGPU present but UNUSED by both engines (verified)
Python 3.12.3, NumPy 1.26.4, Numba 0.65.1 (the pinned version)
```

Measured memory bandwidth ceiling: **~59 GB/s**. Three independent probes agree
(BLAS f32 GEMV 58.6, plain Numba f32 loop 59.7, Alpaccaroo's quantized kernel
58.9 GB/s). Treat ~59 GB/s as the wall until you disprove it — and do try to
disprove it (§6.1).

**Verify the Ollama baseline is CPU-only before trusting it.** It was on the
prior run: the server log showed `GPULayers:[]` and Vulkan disabled. If your
run has GPU offload enabled, the comparison is void.

Ollama also ran with `NumThreads:6` and an **f16 KV cache**. Alpaccaroo defaults to
12 Numba threads and an **f32 KV cache**. Those differences are unexplored and
are live leads in §6.

---

## 3. Reproduce the baseline first

Don't write optimization code until these reproduce.

```sh
cd /home/hoopbot/alpaccaroo

# Alpaccaroo test suites (must stay green throughout)
python3 tests/smoke.py                                              # 402 checks
env PYTHONDONTWRITEBYTECODE=1 ALPACCAROO_PURE=1 python3 tests/smoke.py # 305 checks
# with Numba installed the suite gains one kernels-only check       # 403 checks

# The 8B model under test (already downloaded)
/home/hoopbot/.alpaccaroo/models/hf/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/\
Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Ollama reference (8B Q4_K_M, CPU only). It is often not running.
ollama serve &
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"nemotron:latest",
  "prompt":"Write a short paragraph about the sea.","stream":false,
  "options":{"num_predict":64,"temperature":0,"num_ctx":512}}'
# decode tok/s = eval_count / eval_duration
```

Numba is **not** installed system-wide and must not become a runtime
dependency. Use a throwaway venv for kernel work:

```sh
python3 -m venv /tmp/kvenv && /tmp/kvenv/bin/pip install "numpy==1.26.4" "numba==0.65.1"
```

Decode harness (warm the JIT first — see §5.1):

```python
import os, time
os.environ['ALPACCAROO_DENSE_WEIGHT_MB'] = '0'      # keep weights fully quantized
from alpaccaroo.model import Model
m = Model.load(PATH, progress=False)
m.forward(1); m.reset()                           # WARM THE JIT
t = time.perf_counter()
for i in range(8): m.forward(1000 + i)
print((time.perf_counter() - t) / 8 * 1000, "ms/token")
```

---

## 4. Where the time goes (established — but re-verify)

Alpaccaroo stores each quantized weight as an **int8 code plus per-sub-block f32
scales**: 1.25 bytes/weight for Q4_K. llama.cpp keeps the native packed block,
~0.56 bytes/weight. For an 8B Q4_K_M model that is **9.35 GiB vs ~4.6 GiB**
streamed per token.

Decode is memory-bandwidth-bound. Alpaccaroo reads 9.35 GiB at ~50 GB/s = 200
ms/token = 5.0 tok/s, i.e. **85% of the 59 GB/s ceiling**. The decode gap
(2.2–2.4×) and the storage ratio (1.9–2.0×) are the same number. That is the
core finding: the gap is bytes per weight, not kernel quality.

Decode profile (8B, warm, per token): ~80% of wall time is inside the fused
matvec kernel across 226 matrices; ~20% is everything else (Python dispatch,
RMSNorm, RoPE, attention, sampling).

---

## 5. Mistakes the prior pass made — please don't repeat them

Each of these cost real time or produced a wrong belief.

**5.1 Timing a cold JIT.** First measurements showed Numba kernels giving *zero*
speedup (1.4 tok/s, identical to NumPy). The kernels were fine; ~16 s of
one-time JIT compilation was amortized over 24 tokens. Warm decode was actually
**29 ms/token (34 tok/s)** on a 1B model — 25× off the reported figure. Always
warm the JIT, then time. Numba's `cache=True` persists compilation to disk, so
the second run of a process is representative and the first never is.

**5.2 Hand-unrolling to "fix" a dependency chain.** Hypothesis: the inner
reduction used one accumulator, so it was FMA-latency-bound and four independent
accumulators should give up to 4×. Result: **8.6 Gw/s vs 57.1 — a 7×
regression.** `fastmath=True` was already reassociating, and manual unrolling
destroyed the pattern LLVM's vectorizer recognized. Don't restructure loops on
theory; measure, and read the emitted assembly.

**5.3 Reasoning about vectorization instead of reading the assembly.** The
packing failure was first blamed on "no SIMD". `njit_fn.inspect_asm()` showed
`vpmovzx`, `vpand`, `vpsrl`, `vfmadd` on `zmm` registers — it *was* vectorized,
just not efficiently enough. Use `inspect_asm()` early.

**5.4 Dismissing an avenue analytically without measuring.** NumPy-level
unpacking with cache tiling was ruled out on a back-of-envelope estimate and
never benchmarked. The estimate may be right, but it isn't evidence.

**5.5 Never checking the thread count.** Alpaccaroo doesn't set
`NUMBA_NUM_THREADS`, so it used all 12 SMT threads; Ollama used 6. **Measured: 6
threads = 5.15 tok/s, 12 threads = 4.71 tok/s — 9% free**, on a bandwidth-bound
loop where SMT siblings contend. This surfaced only at the very end. Check the
obvious environment knobs first.

**5.6 Writing a test that compares a code path against itself.** A parity test
for the fused kernel silently compared the fallback path with itself whenever
Numba was absent, so it would have passed even if the kernel were deleted. Prove
a new test can fail: mutate the kernel and confirm the test catches it.

**5.7 A related earlier trap.** A commit changed a storage-description string
and updated the assertions that ran locally, but missed one behind
`if has_pinned_numba:`. Numba is absent locally, so that branch never ran, and
the commits hadn't been pushed, so CI never ran either. **The only job that
exercises the kernels is `smoke (ubuntu-latest, kernels)` in CI.** If you touch
kernels, run the suite inside the Numba venv locally *and* push so CI sees it.

---

## 6. Avenues never explored — start here

Ranked by expected value. The prior pass touched none of these.

### 6.1 Thread count, threading layer, and the ceiling itself
- `NUMBA_NUM_THREADS=6` already measured **+9%**. Sweep 4/6/8/12. Consider
  defaulting Alpaccaroo to physical-core count, with a knob.
- The Numba threading layer is `default`. Try `tbb`, `omp`, `workqueue` via
  `NUMBA_THREADING_LAYER`. Layer choice changes barrier cost per `prange`, and
  there are 226 kernel launches per token.
- Is 59 GB/s really the ceiling? Try many threads reading disjoint regions, and
  check whether it's a per-thread or prefetcher limit rather than a DRAM limit.
  If the true ceiling is higher, everything downstream moves.

### 6.2 Transparent huge pages
THP is `madvise` on this machine, so a 9.35 GiB streaming working set runs on
4 KB pages — ~2.3M TLB entries, far beyond any TLB. **Size the prize first** by
testing with THP set to `always` system-wide; that needs no code change. If it
proves large, then discuss how to request it per-allocation. One route is
stdlib `ctypes` calling `madvise` — arguably fine since it's an OS call and
ships no native code, but it's a judgement call about the project's identity, so
raise it with me rather than assuming. Completely untested; plausibly
several percent to much more.

### 6.3 Per-token work that isn't weight streaming
- **226 separate kernel launches per token.** Fuse Q/K/V into one matrix (they
  share an input) and gate/up in the FFN: fewer launches, better streaming,
  fewer parallel barriers.
- **~20% of decode is outside the matvec kernel** — a 1.25× ceiling on its own.
  Profile it and attack the top entries.
- **The output head is 525M params (0.66 GB) read every token** for a 128256-row
  vocab. Worth asking whether the full logit vector is needed for the active
  sampler. Only pursue this if you can preserve exact equivalence for every
  sampler the engine supports; a shortcut that changes sampling behaviour is a
  correctness bug, not an optimization.

### 6.4 KV cache
Alpaccaroo uses **f32**; Ollama uses **f16** (verified in its logs). At long
context this traffic is real, and it's exactly where users feel it. Also check
whether Alpaccaroo's attention re-reads more of the cache than it needs.

### 6.5 Cheaper scales (the one near-certain win)
Q4_K stores `d` and `m` as f32 per 32 weights = 0.25 of the 1.25 B/weight.
Storing them as bf16/f16 gives **1.25 → 1.125 B/weight ≈ 11% decode**, and the
int8 kernel's ALU ceiling (104.9 Gw/s, §7) is far above the resulting
memory-limited rate, so the saving should convert fully into speed. Numba can't
take f16 *array arguments* — a real limitation to work around — but scales are
read once per 32 weights, so a 65536-entry uint16→f32 lookup table (256 KB,
L2-resident) decodes them at negligible cost. Low risk.

### 6.6 Prefill (secondary; 3.2× behind)
Alpaccaroo dequantizes to f32 and calls BLAS (~240 GFLOP/s measured). llama.cpp
uses int8 GEMM, which NumPy/BLAS don't expose. Cheap things first: check whether
OpenBLAS is actually threading across all cores, and raise the prefill chunk
size so the per-chunk dequantize amortizes over more columns.

### 6.7 The question that would actually settle it: VNNI
llama.cpp's decode advantage comes from a kernel where 4-bit unpack is 2–3
instructions per 32 weights and the dot product uses `vpdpbusd` (AVX-512 VNNI),
doing 4 int8 MACs per lane per instruction. **Numba/LLVM did not emit `vpdpbusd`
in any prior experiment.** If you can get LLVM to emit VNNI from Python source —
via target flags, `numba.config` options, or restructuring an int8→int32
accumulation loop into a shape LLVM's pattern matcher recognizes — the analysis
in §7 collapses and parity becomes reachable. This is the highest-value open
question. Investigate whether Numba exposes `-mcpu`/`target-features`.

---

## 7. Already tried and failed — with numbers

Don't re-run these hoping for a different result. Do challenge the framing.

The idea: store codes packed at 4 bits instead of expanded to int8, halving
bytes/weight. Measured on a 14336×4096 Q4_K matrix (a real Llama-3.1-8B FFN
shape). "Gw/s" = billions of weights processed per second.

Measuring from L3 cache isolates ALU throughput from memory:

| decode loop | ALU ceiling (L3-resident) |
|---|---:|
| int8 codes | **104.9 Gw/s** |
| 4-bit, best variant | **62.0 Gw/s** |

To beat int8 in RAM, a 4-bit kernel must exceed **78.7 Gw/s** (59 GB/s ÷ 0.75
B/weight). Nine variants, none reached it:

| variant | Gw/s |
|---|---:|
| 2-pass expand → int8 buffer | **62.0** (best) |
| nibble, single accumulator | 57.1 |
| multi-row ×4 for ILP | 53.0 |
| 2-pass expand → f32 buffer | 51.1 |
| 4-bit + int8-quantized activations (no VNNI emitted) | 46.7 |
| naive interleaved nibble | 41.5 |
| 64-bit word extraction | 17.4 |
| whole-row f32 expand | 12.0 |
| four manual accumulators | 8.6 |

**Conclusion drawn:** unpacking costs ~40% of ALU throughput and 4-bit packing
saves ~40% of bandwidth, so they cancel. In RAM the best 4-bit variant measured
58.3 Gw/s vs int8's 56.4 — a 1.03× wash.

Two things that follow from this and were never pursued:

1. Packed 4-bit is **1.67× less RAM at no speed cost** (9.35 → ~5.6 GiB for 8B).
   That alone decides whether an 8B model fits on a 16 GB laptop.
2. It should **win more on slower machines**. The packed path is ALU-bound at
   ~62 Gw/s regardless of memory; int8 is memory-bound. Solving
   `min(62, B/0.75) > B/1.125` suggests packed wins on any machine below
   ~70 GB/s — most consumer hardware. This Ryzen at ~59 GB/s sits near the top
   of that range, which is why the gain measured only ~3% *here*; a 30–40 GB/s
   laptop projects to ~1.5×. **This projection was never validated on real
   hardware and should be.**

Also ruled out, with reasons:

- **Speculative decoding** (n-gram/prompt-lookup, no draft model). It relies on
  batching being nearly free. Alpaccaroo's batched kernel is ALU-bound at ~42
  GMAC/s, so a k-token draft costs ~k×. llama.cpp gets this win because its
  kernel has compute headroom under the bandwidth wall; Alpaccaroo doesn't. **If
  you raise the batched kernel's compute efficiency, revisit this** — it becomes
  viable the moment batching stops costing linearly.
- **Micro-optimizing the existing int8 kernel.** Already at ~100% of achievable
  bandwidth for that format.

---

## 8. Codebase traps

- `tests/smoke.py`'s `check()` calls `sys.exit(1)` on the **first** failure, so
  one early failure hides every later test. If a mutation "isn't caught",
  confirm the test actually ran.
- Run fully quantized with `ALPACCAROO_DENSE_WEIGHT_MB=0`. Without it the CLI may
  densify weights to f32 for BLAS, changing what you're measuring.
- `ALPACCAROO_PURE=1` forces the stdlib path. `ALPACCAROO_KERNELS=0` disables Numba.
  `ALPACCAROO_FUSED_MATMUL_MAX_BATCH` sets the fused/BLAS crossover for batched
  matmul (default 96).
- Numba is pinned to 0.65.1; another version deactivates the kernels unless
  `ALPACCAROO_KERNELS=force`.
- Recent relevant work is on branch `perf/fused-quantized-matmul`: fused batched
  matmul kernels that made small-batch prefill 2.3–16.6× faster. Read
  `alpaccaroo/kernels.py` and `alpaccaroo/qmatrix.py` first — they are the hot path.
- `README.md` has an "Honest performance expectations" section that is meant to
  stay honest. If you change performance, update it with measured numbers,
  including anything that got worse.

---

## 9. How to work

1. **Reproduce before optimizing.** Confirm §3 and the Ollama reference. If your
   numbers differ materially from this document, say so plainly and trust your
   own measurements — the environment may have changed.
2. **Cheap knobs before deep engineering.** §6.1 and §6.2 are hours of work with
   real upside. Exhaust them before touching storage formats.
3. **Measure end-to-end, not just microbenchmarks.** A kernel 2× faster in
   isolation and 1.0× in `Model.forward` is not a win. Report both.
4. **Read the assembly** when a kernel underperforms (`inspect_asm()`).
5. **Report negative results as first-class output.** A well-measured dead end
   is valuable and stops the next person repeating it; §7 exists for that reason.
6. **Never claim a speedup you haven't measured end-to-end on the 8B model.**
7. Keep both suites green with non-decreasing counts. If you add a test, prove
   it can fail.
8. Commit incrementally with measurements in the commit message. Don't merge to
   `main` without asking.

## 10. Deliverables

- The decode tok/s achieved on 8B Q4_K_M vs Ollama on the same machine, measured
  the same way, with the commands used.
- A table of every avenue tried: what, expected, measured, kept or discarded.
- Working, tested, committed code for whatever landed.
- An honest verdict on the central question: **can pure Python reach llama.cpp's
  decode throughput on this hardware?** If yes, prove it with the number. If no,
  state precisely what the binding constraint is and what evidence would change
  the answer.

A negative answer, well evidenced, is a perfectly acceptable outcome. An
unsupported positive one is not.
