# Alpaccaroo performance: measuring, reading, and tuning

This is the reference for the measurement tools and the knobs they inform.
It exists because the honest answer to "why is this slow?" is almost never
a guess, and this engine now has the instruments to answer it.

Nothing here promises a speed. Every number below is reported with the
machine, OS, model, quantization, context and dependency versions that
produced it, because without those a tokens-per-second figure is not a
measurement.

---

## 1. What ran? Backend and execution-path labels

`backend numpy` was too coarse to answer the only question that matters
when decode is slow: *are the native quantized kernels actually running?*
Three levels of detail are now available.

**Tier**, from `alpaccaroo doctor` or `T.backend_detail()`:

```
path:        numpy + alpaccaroo-kernels numba==0.65.1 (native integer-dot)
```

| tier string | meaning |
|---|---|
| `pure-python` | standard library only; no NumPy |
| `numpy (no kernels: einsum quantized matvec, BLAS dense)` | NumPy present, JIT absent or disabled |
| `numpy + alpaccaroo-kernels numba==X (fused f32-scale)` | JIT active, `ALPACCAROO_INT_DOT=0` |
| `numpy + alpaccaroo-kernels numba==X (native integer-dot)` | JIT active, the fast path |
| `gpu-cuda (host fallback: ...)` | weights resident in VRAM |

**Per matrix**, from `--profile` or `profiling.model_paths(model)`. Every
weight matrix reports which kernel its matvec runs through:

| label | kernel |
|---|---|
| `pure-python-dense` | list-of-lists matvec |
| `pure-python-quant` | per-row block decode |
| `numpy-dense` | float32 ndarray, BLAS GEMV |
| `numpy-dense-hotcache` | quantized weights expanded by `ALPACCAROO_HOT_WEIGHT_MB` |
| `numpy-quant-einsum` | NumPy quantized fallback, large matrices |
| `numpy-quant-batched` | NumPy quantized fallback, small matrices |
| `numba-codes-f32` | fused f32-scale kernel over int8 codes |
| `numba-int-q4k` / `numba-int-q5k` / `numba-int-q6k` | native integer-dot kernels |
| `gpu-cuda` | our CUDA kernels |

A role whose layers disagree - a partial VRAM upload, or a Q4_K_M file
whose `ffn_down` is Q4_K in some layers and Q6_K in others - reports the
`+`-joined set rather than the first layer's answer:

```
ffn_down               numba-int-q4k+numba-int-q6k   36 mat     2048x11008     811.6 Mw
```

---

## 2. Where did the time go? The profiler

```powershell
alpaccaroo run qwen3bmed --profile
alpaccaroo run qwen3bmed --profile-json perf.json
alpaccaroo bench --model qwen3bmed --profile-json perf.json
```

The profiler is **off by default and costs one module attribute read per
forward pass when off**. When on, it measures its own overhead and prints
it, so the report discloses what share of itself it is.

Turning it on does not change what the model emits; `tests/smoke.py` pins
that greedy output is character-identical with the profiler on and off.

### Decode buckets

Non-overlapping, so they sum to less than the decode wall clock:

| bucket | covers |
|---|---|
| `embedding` | token embedding row gather |
| `norm` | every RMSNorm (attention, FFN, final, and Gemma 3's per-head q/k norms) |
| `attn_qkv` | q/k/v projection, including the shared-activation group dispatch |
| `rope` | rotary embedding and the KV cache writes |
| `attention` | scores, softmax, weighted V |
| `attn_out` | attention output projection |
| `ffn_gate_up` | FFN gate + up projection |
| `ffn_act` | SwiGLU / GELU |
| `ffn_down` | FFN down projection |
| `output_proj` | vocabulary output projection |
| `orchestration` | **residue**: decode wall clock minus every bucket above |

`orchestration` is a residue, not a measurement. It is Python dispatch,
allocation, and the profiler's own instrumentation, and the report says so.

Two roll-ups are printed under the table: `= matvec` (every weight-matrix
product) and `= grouped` (the shared-activation dispatches, a subset of
`attn_qkv`).

### Outside the decode loop

`model_load`, `prompt_render`, `prefill`, `sampler`, `tokenizer_stream`.
These are not part of the decode wall clock and are reported separately -
a sampler that costs 1.7 ms per token is not 1.7 ms *of* a 1264 ms token,
it is 1.7 ms beside it.

### Environment

Every report and every `--profile-json` carries: OS and architecture, CPU
model and detected feature set, physical and logical core counts, the
kernel thread count and numba threading layer, the BLAS identity **and its
live thread count**, and the Python/NumPy/Numba/llvmlite versions.

The BLAS thread count is not decoration. A BLAS fanning out to every
logical CPU while the kernels hold the physical cores is the two-pool
contention the fused attention kernel exists to avoid, and it is invisible
without that number.

---

## 3. Is this machine even comparable? Cold, warm, and clock stability

```powershell
alpaccaroo bench --model qwen3bmed
alpaccaroo bench --model qwen3bmed --shapes long-long --json b.json --csv b.csv
alpaccaroo bench --model qwen3bmed --stability
python tests/bench.py --model qwen3bmed --profile-json perf.json
```

**Cold and warm are different workloads and are never quoted as one
another.**

- `cold` - a freshly loaded model: GGUF open and unpack, the JIT compiling
  or loading its cache, the first touch of every weight page. This is what
  a one-shot `alpaccaroo run` pays.
- `warm` - the same model after a full discarded pass, KV cache reset. This
  is what `alpaccaroo serve` or an interactive session pays per turn.

One caveat worth knowing before quoting a cold number: a sweep loads the
model once per `--model` and per `--ctx`, so `--ctx 2048,4096` produces
**two** `cold` rows - but only the first is cold in the operating system's
eyes. The second reloads the same file through a warm page cache and will
report a load time several times lower (measured 1.23 s then 0.33 s for the
same 18 MiB file). Compare cold load times across processes, not across
rows of one sweep.

Model references resolve **locally first** - nicknames, installed canonical
names and file paths all resolve without touching the network - and a
reference that is not installed is an error, not a download, unless
`--allow-pull` is given.

Shapes: `short-short` (32x32), `short-long` (32x256), `long-short`
(1024x32), `long-long` (1024x256), `default` (512x128), or any
`PREFILLxDECODE`. Contexts: `--ctx 2048,4096,8192`.

Each row records tok/s prefill and decode, p50/p95/p99 token latency, time
to first token, model load seconds, peak RSS (on Windows as well as POSIX),
the selected execution paths, and the whole environment block above.

### `--stability`: does this machine hold its clocks?

```
clock stability: 2.19-3.96 GB/s over 70s (spread 1.81x) - UNSTEADY:
single runs on this machine are not comparable
```

A thin laptop cycling between its short- and long-duration power limits
will report a spread of 2x or more. **On such a machine, two separate
benchmark runs cannot be compared** - the earlier one may simply have been
cooler. Use `bench.paired_compare`, which alternates variants round-robin
so every one sees the same thermal window and the per-round *ratio* stays
meaningful even when the absolute numbers do not.

This is not a hypothetical caution. On the reference laptop below, the
serial-versus-parallel comparison for one matrix shape reported the
**opposite winner** depending on whether it was taken as two separate runs
or as a paired one. Re-measure this way, or not at all.

---

## 4. Knobs

Every optimization here is measurable and disableable.

| variable | effect |
|---|---|
| `ALPACCAROO_THREADS` | kernel pool size. **Wins over everything**, including autotuning. |
| `ALPACCAROO_AUTOTUNE=1` | apply a cached `alpaccaroo tune` result at load. Never benchmarks. |
| `ALPACCAROO_SERIAL_MATVEC_ELEMS` | weight count at or below which a matvec runs on one thread (default 131072) |
| `ALPACCAROO_SERIAL_QUANTIZE_COLS` | column count at or below which activation quantization runs on one thread. **Default 0 (off)** - see section 5, it wins per call and loses end to end |
| `ALPACCAROO_GROUP_KERNEL=0` | disable the grouped Q4_K+Q6_K kernel |
| `ALPACCAROO_KERNELS=0` | disable the JIT kernels entirely |
| `ALPACCAROO_INT_DOT=0` | keep the kernels, revert to f32-scale storage |
| `ALPACCAROO_FUSE=0` | disable load-time row fusion of same-dtype neighbours |
| `ALPACCAROO_DENSE_WEIGHT_MB` | expand this many MiB of quantized weights to dense float32 |
| `ALPACCAROO_HOT_WEIGHT_MB` | dense float32 cache budget, applied per matrix on demand |
| `ALPACCAROO_PURE=1` | standard-library backend only |
| `ALPACCAROO_PREFILL_CHUNK` | prefill batch size (default 256) |

### Thread controls

The kernels default to **physical cores**, not logical CPUs: decode is
memory-bandwidth-bound and SMT siblings contend for the same load ports.
`ALPACCAROO_THREADS` overrides that and beats every other mechanism.

Per dispatch, the pool is narrowed for matrices too small to repay a
fan-out (section 6). This changes *who* computes each row, never *what* is
computed - `prange` splits the row loop and each row's accumulation order
is fixed - so it is bit-identical by construction, which `tests/smoke.py`
pins for Q4_K, Q5_K and Q6_K.

### The dense-weight budget

`ALPACCAROO_DENSE_WEIGHT_MB` spends RAM on dense float32 weights, which
BLAS can GEMV at full speed. **With the native integer-dot kernels active
this is usually a pessimisation**: those kernels stream 0.578 (Q4_K) to
1.066 (Q6_K) bytes per weight against float32's 4, and decode is bandwidth
bound. The CLI's auto-budget knows this and leaves weights quantized
whenever the kernels are available. Spend the budget when the kernels are
*not* available - a machine with NumPy but no pinned Numba - where dense
BLAS is genuinely the fastest path on offer.

### Autotuning

```powershell
alpaccaroo tune -m qwen3bmed              # thread count, from the model's own shapes
alpaccaroo tune -m qwen3bmed --crossover  # the narrow-matrix threshold
alpaccaroo tune                            # a generic size ladder, no model needed
```

Opt-in, always. `alpaccaroo tune` is the only place a benchmark runs;
`ALPACCAROO_AUTOTUNE=1` only ever *reads* what it wrote, so a normal CLI
start never pays for one. Model shapes are read from the **GGUF header**,
so tuning an 8B model costs milliseconds rather than a five-gigabyte load.

The cache is keyed on everything that can change the answer: CPU model and
feature set, physical and logical core counts, OS, the
Python/NumPy/Numba/llvmlite versions, the alpaccaroo version, a hash of
`kernels.py` itself, and the model's shape class. Any difference is a miss,
so an upgrade re-measures instead of serving a stale winner.

**Tune on a machine that holds its clocks.** Check `--stability` first: a
threshold measured while a laptop is cycling its power limits describes
that moment, not that machine.

### Resident model workflow

Decode speed is not the whole of perceived speed. A one-shot
`alpaccaroo run` pays process startup, imports, model load, JIT
cache-load and prompt setup before the first token; on the reference
laptop that is 30-49 s of load against a ~1 s token, so **more than half
the wall clock of a short answer is load**. Keeping the model resident
removes all of it from every turn after the first:

```powershell
alpaccaroo serve qwen3bmed --port 8080   # OpenAI- and Ollama-compatible API
alpaccaroo run qwen3bmed                 # interactive: one load, many turns
alpaccaroo run qwen3bmed --connect       # use a server that is already up
alpaccaroo run qwen3bmed --connect http://host:9000 "why is the sky blue?"
```

`--connect` is the fast reconnect: it probes `/health`, streams through the
server's OpenAI-compatible endpoint, and **loads the model locally instead
if nothing answers** - so it is opt-in, never mandatory, and the one-shot
CLI is unchanged for anyone who does not ask for it. With no URL it uses
`$ALPACCAROO_HOST:$ALPACCAROO_PORT`, the same variables `serve` reads.

Because the server streams text rather than token counts, `--connect`
reports the number of stream chunks and the elapsed seconds -
`[41 chunks in 12.3s]` - rather than implying a tok/s it cannot observe.
A chunk is one streamed delta, which is usually but not always one token.

The `cold` and `warm` bench phases measure exactly these two workflows.

---

## 5. Measured results

### Reference machine A - thin-and-light laptop, power-limited

```
os              Windows 11 (10.0.26200), AMD64
cpu             Intel Tiger Lake, 4 physical / 8 logical cores
cpu features    sse4.2 avx f16c fma avx2 avx512f avx512bw avx512vl avx512dq avx512vnni
blas            scipy-openblas 0.3.31.dev, 8 threads (threaded)
kernels         alpaccaroo-kernels active, numba==0.65.1, 4 threads
versions        python 3.13.14, numpy 2.4.3, numba 0.65.1, llvmlite 0.47.0
model           qwen2.5-3b-medieval Q4_K_M - qwen2, 36 layers, embd 2048,
                heads 16/2, ff 11008, vocab 151936, ~3086M params,
                1.8 GiB quantized weights, ctx 4096
```

**Decode profile, warm, 56 tokens, 128-token prompt:**

| bucket | per token | share |
|---|---|---|
| ffn_gate_up | 495.0 ms | 39.1% |
| ffn_down | 301.8 ms | 23.9% |
| attn_qkv | 207.4 ms | 16.4% |
| attn_out | 130.5 ms | 10.3% |
| output_proj | 56.2 ms | 4.4% |
| attention | 50.2 ms | 4.0% |
| ffn_act | 8.4 ms | 0.7% |
| norm | 8.1 ms | 0.6% |
| rope | 6.2 ms | 0.5% |
| embedding | 0.2 ms | 0.01% |
| orchestration (residue) | 0.6 ms | 0.05% |
| **= matvec** | **1191 ms** | **94.2%** |

Decode 0.79 tok/s, prefill 2.15 tok/s, model load 29.6 s, peak RSS 4088 MiB.
Sampler 1.67 ms/token (0.13%), outside the decode clock.

**Python overhead is not the problem.** Orchestration is 0.05% of a token.
94.2% is weight-matrix products, and the engine is bandwidth-bound.

### The finding that matters: burst versus sustained

The same kernels, measured in isolation on the same machine minutes apart:

| shape | isolated, machine idle | sustained, 70 s continuous |
|---|---|---|
| Q4_K 22016x2048 | 2.85 ms (9.2 GB/s) | 6.6-11.9 ms (2.2-4.0 GB/s) |
| Q6_K 151936x2048 | 16.8 ms (19.8 GB/s) | - |

Predicted per-token matvec cost from the *idle* kernel rates: **199.6 ms**.
Measured inside real decode: **1191 ms**. Six times worse.

Two hypotheses were tested and rejected before the right one was found:

- *OpenBLAS thread interference* - `rmsnorm` calls BLAS `sdot` 73 times per
  token, and OpenBLAS held 8 threads against the kernels' 4. Tested by
  interleaving a BLAS dot between matvecs and by re-running decode under
  `OPENBLAS_NUM_THREADS=1`: no consistent effect (0.82x, 2.16x, 0.87x
  across three shapes).
- *Cold working set* - real decode streams 1.8 GiB across 253 arrays per
  token, where a micro-benchmark hammers one. Tested by sweeping 36
  distinct matrices versus hammering one of the same total size: **1.01x**.
  Not the cause.

The cause is the machine. Held under continuous load for 70 s, throughput
oscillates between 2.19 and 3.96 GB/s with no trend - the package is
cycling between its power limits - and multi-threaded work degrades far
harder than single-threaded work does, because four active cores must each
drop further to fit one package power budget. Late in a long session the
same shapes measured 4-9x slower than at its start.

**Decode on this laptop runs at roughly its sustained memory bandwidth.**
There is no large algorithmic win hiding here; the plan predicted exactly
this ("the laptop CPU appeared to sustain low clocks under load") and said
not to design around it. The engine has not been.

The remaining lever on hardware like this is bytes per weight, not
instructions. Q4_K already streams at 0.578 B/weight; Q6_K stores its
6-bit codes unpacked at 1.066 B/weight where 0.82 is achievable, and that
is worth roughly 10% of the bytes a token of this model touches. It is
recorded in `kernels.py` as future work.

### Package results, paired and round-robin

All ratios below are the median per-round ratio from
`bench.paired_compare`, which alternates variants so a wandering clock
cancels.

**Package C - fused greedy sampling.** 203/203 differential comparisons
against the float64 reference path selected the identical token, including
ties whose lowest index sits outside any top-k partition, wholly penalized
vectors, NaN, infinities and degenerate penalties. On a 151936-token
vocabulary: **168.5 us -> 35.4 us, 4.76x**. That is 0.1% of a token on this
machine; it would be ~3% of a 50 ms token on hardware that is not
bandwidth-starved. An earlier attempt using `argpartition` to find the
runner-up measured **2x slower** than the path it replaced and was
discarded - a quickselect over 151936 entries costs more than the float64
copy it saves.

**Package D - grouped Q4_K+Q6_K kernel.** Bit-identical (relative error
exactly 0) on every shape tested.

| group | separate | grouped | ratio |
|---|---|---|---|
| qwen2.5-3B attention: Q4_K 2304x2048 + Q6_K 256x2048 | 3074 us | 2521 us | **0.837** |
| llama-3-8B attention: Q4_K 6144x4096 + Q6_K 1024x4096 | 7580 us | 7031 us | 0.969 |
| narrow GQA: Q4_K 1024x2048 + Q6_K 128x2048 | 2197 us | 2265 us | 1.003 |

16% off the qwen attention group, which is 16.4% of decode - about 2.6% of
a token. It does nothing for the narrow shape, which is why it is a flag.

**Package E - narrow-matrix dispatch.** Bit-identical for Q4_K, Q5_K and
Q6_K. Paired, on an idle machine:

| shape | elements | t=1 | t=2 | t=4 | best |
|---|---|---|---|---|---|
| Q4_K 64x2048 | 131072 | 49.0 us | 49.5 us | 63.9 us | **1** (1.30x) |
| Q4_K 128x2048 | 262144 | 76.3 us | 69.8 us | 67.7 us | 4 |
| Q4_K 256x2048 | 524288 | 157.6 us | 114.0 us | 102.7 us | 4 (1.53x) |
| Q6_K 256x2048 | 524288 | 220.6 us | 327.0 us | 420.2 us | **1** (1.90x) |
| Q4_K 512x2048 | 1048576 | 307.0 us | 202.4 us | 270.4 us | 2 |
| Q4_K 2048x2048 | 4194304 | 2443.5 us | 1849.3 us | 1755.6 us | 4 |

The crossover is real but shape- *and* dtype-dependent, so the built-in
default (131072) is the smallest size at which every probed dtype agreed.
It can only help shapes that were losing and cannot regress a larger one.
`alpaccaroo tune --crossover` measures the real threshold for a given
machine.

Per-dispatch floor at 4 threads, measured with a matvec small enough that
the arithmetic is negligible: **64.4 us**, against 17.8 us on one thread
and 182.3 us on eight. `numba.set_num_threads` itself costs 3.3 us, which
is why the pool is sized set-and-leave rather than set-and-restore.

**Package F - thread autotuning.** Per-token matvec cost for this model's
shape class: 1 thread 724.3 ms, 2 threads 467.1 ms, **4 threads 199.6 ms**,
8 threads 268.8 ms. The tuner selected 4 - the physical-core default - at
1.00x. The existing default was already right on this machine, which is
the result worth having from a tuner: confirmation, not a change.

### The end-to-end check, and the one result that changed a default

Every measurement above is per call. A decode token is 181 calls, and the
two are not the same claim. So the whole loop was A/B'd in **one process**,
from one loaded model and one KV state, 25 ABBA rounds, with the token ids
compared as well as the times.

The first attempt turned on everything at once - grouped kernel, narrow
matvec dispatch, and narrow activation quantization at 8192 columns - and
it lost **22 of 25 rounds**, median 1.20x. Under a null hypothesis of no
effect, 3 wins out of 25 is not a wandering clock; it is a regression.

Rerunning the identical A/B with only the activation-quantization
threshold returned to 0 gave **14 of 25 rounds** and a median of 0.976 -
noise, and the regression gone. The effect tracked that one setting.

| configuration | rounds won by "on" | median ratio | ratio of medians |
|---|---|---|---|
| grouped kernel + narrow matvec + narrow quantize (8192) | 3 / 25 | 1.197 | 1.093 |
| grouped kernel + narrow matvec, quantize threshold off | 14 / 25 | 0.976 | 0.853 |

`ALPACCAROO_SERIAL_QUANTIZE_COLS` therefore ships at **0, disabled**,
despite a clean 6-10% per-call win in isolation. The knob remains.

**The mechanism is not established, and this document will not pretend
otherwise.** The obvious suspect was the thread-pool resize the threshold
forces about 216 times per token - a 2048-wide model quantizes below the
threshold and then matvecs above it, three times per layer. A direct probe
of exactly that (identical kernel work, with and without a resize between
each pair) measured **no cost at all**: ratio 1.06, 11 of 25 rounds. So the
empirical result stands and its cause does not. The next engineer should
treat "resizing the pool is expensive" as *unsupported*, not as received
wisdom.

**The lesson, which is worth more than the 10%: a microbenchmark win is not
an end-to-end win.** Validate whole-loop changes whole-loop, with a sign
test - with a 2x spread the median is not enough, and 3 wins out of 25 is a
result where 1.20x alone would have been a shrug.

### Track 4: what these Python loops actually compile to

```powershell
alpaccaroo tune --asm                      # matvec_q4k_int by default
alpaccaroo tune --asm matvec_q6k_int --json q6k.s
```

The plan asks what instructions the kernels become on a given CPU rather
than assuming. Numba knows; `--asm` surfaces it with no build step, so an
engineer on AMD Zen, on a machine without AVX-512, or on Linux rather than
Windows can check whether the integer contraction the kernels depend on
survived. On the reference machine, `matvec_q4k_int`:

| instruction | count | meaning |
|---|---|---|
| `vpdpwssd` | 31 | AVX-512 VNNI int16 dot-product accumulate - the best case |
| `vpmullw` | 32 | the 6-bit sub-scale premultiplied into the codes |
| `vpmaddwd` | 1 | the non-VNNI fallback |
| `vpmuldq` | **0** | no int64 promotion - the `np.int32(acc + ...)` idiom is holding |
| `ymm` / `zmm` | 66 / **0** | 256-bit registers, **not** 512-bit |

Two things worth carrying forward. The int32 re-cast idiom that
`kernels.py` depends on is verifiably working - zero 64-bit lane
multiplies. And LLVM is emitting the AVX-512 VNNI instruction on **256-bit
ymm operands, never zmm**: a deliberate vector-width cap, and on a
downclocking Tiger Lake part not obviously the wrong choice. It was not
pursued here because this machine is bandwidth-bound, not
instruction-bound - but on a part that sustains its AVX-512 clocks it is
the first thing to re-measure.

One trap, handled: numba refuses to disassemble code it loaded from its
on-disk cache and returns an empty listing, which reads as "no SIMD at
all". `--asm` points it at a fresh cache directory so the numbers are real,
and says so explicitly if they still come back all-zero.

### What has not been measured

One machine, one OS, one CPU vendor, three model sizes' worth of *shapes*
but only one real model end to end. The plan's benchmark matrix asks for
Linux, AMD, mini-PC and desktop classes, and a CUDA device. `--json` and
`--csv` carry everything needed to merge results from those machines into
one table; nothing above should be read as characterising them.

---

## 7. What is left

The tools are built and the reference machine is characterised. What
remains is mostly *coverage* and one real optimisation. In rough order of
expected value:

**1. Q6_K 6-bit packing - the only large lever left.** Q6_K codes are
stored unpacked at 1.066 bytes per weight where the format itself needs
0.82. On a Q4_K_M qwen2.5-3B, Q6_K carries 31% of the weights, so packing
is worth roughly **10% of the bytes a token touches** - and decode is
bandwidth-bound, so bytes per weight is the lever that actually moves.
`quants.py` already flags it ("the 6-bit code packing stays future work").
It needs a new pack/unpack plus a kernel that reads the packed form, and it
should be held to the same bit-identity bar as Packages D and E.

**2. Run the benchmark matrix.** The harness exists; the data does not.
Cheapest wins first, because two more real models are *already installed*
on the reference machine and were never benchmarked:

```powershell
alpaccaroo bench --model bartllama3b --shapes short-short,long-long --json llama3b.json
alpaccaroo bench --model hermes8b   --shapes short-short,long-long --json hermes8b.json
```

Then the parts that need other hardware: Linux, an AMD Zen part, a desktop
or server CPU that holds its clocks, and a CUDA device. Also unmeasured on
any machine: Q5_K_M and F16/F32 dense storage, and contexts 2048/8192 on a
real model.

**3. Validate Package E on a model that actually triggers it.** The default
threshold is 131072 weights. qwen2.5-3B's *smallest* decode matvec is
524288, so the narrow-dispatch path never engages on it - every measurement
of it here is per-call, not end-to-end. A model with narrow GQA (n_kv 1,
head_dim 64, embd 2048 gives exactly 131072) would exercise it. Until then,
treat the feature as measured-in-isolation only.

**4. Extend the grouped kernel to Q4_K+Q5_K.** Package D covers the
Q4_K+Q6_K pairing a Q4_K_M file produces. A **Q4_K_S** file stores attn_v
as Q5_K, so it falls back to per-matrix dispatch and gets nothing. The
kernel is a mechanical copy of the existing one with the Q5_K branch
substituted, and `_pair_indices` already refuses unknown pairings safely.

**5. Two open questions, both needing a machine that holds its clocks.**
Do not attempt either on a laptop that `--stability` reports as unsteady:

  - *Why did the activation-quantization threshold regress?* It lost 22 of
    25 end-to-end rounds; the obvious cause (thread-pool resize) was tested
    directly and found to cost nothing. The knob ships off. Either explain
    it or leave it off.
  - *Should the kernels use 512-bit registers?* `tune --asm` shows LLVM
    emitting `vpdpwssd` on 256-bit ymm and never zmm. On a downclocking
    Tiger Lake that is plausibly correct; on a part that sustains AVX-512
    clocks it may be leaving throughput on the table.

**6. Track 1 leftovers, deliberately not done.** Fusing top-k into the
output projection, and top-p over a reduced candidate set. Measured ceiling
on qwen2.5-3B: the sampler is 1.67 ms of a 1264 ms token and the logits
materialisation is ~0.2 ms, so the whole track is worth **~0.15%** here.
Revisit only if a profile on faster hardware shows the sampler above a few
percent - the fused greedy path already took the cheap 4.76x.

**7. The GPU tier is untouched and was not exercised.** `cuda.py` has zero
changes on this branch, and the GPU branch of `matvec_group` returns before
any new code runs, so Track 7's "CPU work must not block GPU gains"
guardrail holds by construction rather than by testing. Nobody has run this
branch on a CUDA device; do that before trusting it there.

### Two claims that were wrong, and are now fixed

Recorded because the same drift is easy to reintroduce:

- The docs said `--connect` reports "chunks per second". It reports chunks
  and elapsed seconds. Fixed above.
- The docs defined `cold` as "the model's first use in this process", which
  stops being true the moment a sweep has more than one `--ctx` or
  `--model`: each combination reloads, but only the first is cold to the
  page cache (1.23 s then 0.33 s for the same file). Fixed above.

`README.md`'s "what did not help" note about fusing attn_q+attn_k launches
refers to *load-time row fusion* on a different machine, not to Package D's
grouped kernel - different mechanism, and not a contradiction.

---

## 6. For the next engineer

- **Measure before you optimise, and measure paired.** The profiler will
  tell you where the time is in one run. If `--stability` reports a spread
  above ~1.3, no single-shot A/B on that machine means anything.
- **Then measure end to end, with a sign test.** Per-call wins do not add
  up to a per-token win by default, and with a 2x spread the median lies.
  Count rounds won: near half is noise, 3 out of 25 is a regression. This
  is how the activation-quantization threshold was caught after it had
  already passed its own benchmark.
- **Prefer work reduction to cleverness.** The two changes that paid here
  reduced dispatches and bytes touched. The two that did not pay both added
  something to save something: an `argpartition` pass to avoid a float64
  copy, and a thread-pool resize to avoid a fan-out.
- **Bit-identity is cheap to keep and expensive to lose.** Packages D and E
  are bit-identical *by construction*, because they change which thread
  computes a row rather than how a row is computed. Any future kernel
  should aim for the same property, and the smoke suite should pin it.
- **Read `prompts/03-RESULTS.md`** for the earlier experiment log - the
  int32 re-cast idiom that unlocked AVX-512 VNNI is recorded there, and
  every integer kernel depends on it.
