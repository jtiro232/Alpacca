# Results log: portable performance architecture (prompts/04-portable-performance.md)

One line per avenue: what was tried, what was expected, what was measured,
kept or discarded. Same format as `03-RESULTS.md`. Numbers are end-to-end
qwen2.5-3B Q4_K_M decode on **machine B** unless marked "micro".

**Read this before re-running any of it**, and read `03-RESULTS.md` too -
several avenues here are re-openings of things round 3 measured shut on
different hardware, and the reason they reopen is the hardware, not the
idea.

## The two machines are not comparable, and that is the headline

| | machine A (03-RESULTS) | machine B (this round) |
|---|---|---|
| CPU | Ryzen 5 7640HS, 6C/12T, DDR5 | Intel Tiger Lake, 4C/8T, LPDDR4x |
| sustained streaming bandwidth | **59-60 GB/s**, flat 4-12 threads | **2.2-4.0 GB/s**, oscillating |
| burst (idle, short window) | - | 9-20 GB/s |
| clock stability | flat, wall is real | spread 1.4-1.8x, no trend |
| 3B Q4_K_M decode | not measured | 0.7-1.0 tok/s |
| 8B Q4_K_M decode | 9.1-9.9 tok/s | not measured |

Machine A is **ALU-bound at a 60 GB/s wall**. Machine B is **bandwidth-bound
at ~3 GB/s**, roughly 20x less bandwidth for broadly similar per-core ALU.
Any avenue whose verdict was "the ALU cost exactly cancels the bandwidth
saving" was decided by a ratio that does not hold here. That is not a
contradiction of round 3; it is the reason the plan asked for a portable
architecture instead of one machine's tuning.

## FINAL SUMMARY (2026-08-03)

Six packages plus Tracks 4/5/6 delivered. **No end-to-end speedup is
demonstrable on machine B**, and the log says so rather than quoting the
per-call wins as if they were per-token wins.

qwen2.5-3B Q4_K_M, warm, 56 tokens, ctx 4096, machine B:

| bucket | per token | share |
|---|---:|---:|
| ffn_gate_up | 495.0 ms | 39.1% |
| ffn_down | 301.8 ms | 23.9% |
| attn_qkv | 207.4 ms | 16.4% |
| attn_out | 130.5 ms | 10.3% |
| output_proj | 56.2 ms | 4.4% |
| attention | 50.2 ms | 4.0% |
| ffn_act + norm + rope + embedding | 22.8 ms | 1.8% |
| orchestration (residue) | 0.6 ms | **0.05%** |
| **= matvec** | **1191 ms** | **94.2%** |

Python overhead is not the problem and never was. 94.2% is weight-matrix
products against a ~3 GB/s wall.

## Avenues

| avenue | expected | measured | verdict |
|---|---|---|---|
| decode profiler + path labels (A) | find where time goes | 94.2% matvec, 0.05% orchestration | KEPT, answered the question |
| benchmark harness, cold/warm (B) | repeatable numbers | works; nickname resolution bug found and fixed | KEPT |
| fused greedy sampler (C) | "large-vocab win" | 168.5 -> 35.4 us micro (4.76x); **0.13% of a token** | KEPT, honestly small |
| sampler via argpartition (C, first try) | avoid float64 copy | **2x SLOWER** - quickselect over 151936 costs more than the copy | DISCARDED |
| grouped Q4_K+Q6_K kernel (D) | fewer dispatches | 0.837 on the qwen attn group micro; 14/25 rounds end-to-end | KEPT, flagged |
| narrow-matrix dispatch (E) | serial wins when small | 1.30-1.90x micro at <=131072 w; **inert on every model tested** | KEPT, under-validated |
| narrow activation quantize (E) | 6-10% per call | per call yes; **end-to-end lost 22/25 rounds** | REVERTED, ships off |
| thread-pool resize cost (E follow-up) | explains the above | **no cost measured** (11/25, ratio 1.06) | hypothesis DISPROVED |
| thread autotuner (F) | beat physical-core default | 1t 724ms, 2t 467ms, **4t 200ms**, 8t 269ms | KEPT; default already right |
| OpenBLAS thread contention | suspected root cause | 0.82x/2.16x/0.87x across shapes, no consistent effect | REJECTED |
| cold working set / TLB | suspected root cause | sweep vs hammer **1.01x** at equal bytes | REJECTED |
| burst vs sustained clocks | - | **199.6 ms predicted vs 1191 ms actual** | THE key result |
| codegen census (Track 4) | is VNNI emitted? | 31 vpdpwssd, 0 vpmuldq, **66 ymm / 0 zmm** | KEPT as a tool |
| CLI -> server reconnect (Track 6) | skip 30-49s load | works; load is >50% of a short answer's wall clock | KEPT |

### The key result: burst is not sustained

The same kernels, same machine, minutes apart:

| shape | isolated, idle | sustained, 70 s continuous |
|---|---|---|
| Q4_K 22016x2048 | 2.85 ms (9.2 GB/s) | 6.6-11.9 ms (2.2-4.0 GB/s) |
| Q6_K 151936x2048 | 16.8 ms (19.8 GB/s) | - |

Predicted per-token matvec from idle rates: **199.6 ms**. Measured inside
real decode: **1191 ms**. Two hypotheses were tested and rejected before
this one (OpenBLAS contention; cold working set). The package cycles
between its power limits, and multi-threaded work degrades harder than
single-threaded because four active cores must each drop further to fit one
power budget. Late in a long session the same shapes measured 4-9x slower
than at its start.

**Consequence for method:** on machine B a single-shot A/B is worthless.
Everything above was measured with `bench.paired_compare` (round-robin,
ABBA ordering, sign test). Taken as two separate runs, the
serial-vs-parallel comparison for Q4_K 256x2048 reported the **opposite
winner**.

### NEGATIVE 1 - narrow activation quantization: per-call win, per-token loss

`ALPACCAROO_SERIAL_QUANTIZE_COLS=8192` was set on a clean per-call
benchmark (serial wins 0.84-0.94 at every width up to 11008; loses at
16384). End-to-end, 25 ABBA rounds, one process, one KV state:

| configuration | rounds won | median ratio | ratio of medians |
|---|---:|---:|---:|
| grouped kernel + narrow matvec + narrow quantize (8192) | **3 / 25** | 1.197 | 1.093 |
| grouped kernel + narrow matvec, quantize off | 14 / 25 | 0.976 | 0.853 |

3 wins in 25 is not a wandering clock. The regression tracked the setting;
it now ships at 0. **The mechanism was NOT established.** The obvious
suspect - the ~216 thread-pool resizes per token it forces - was probed
directly (identical kernel work with and without a resize between each
pair) and measured **no cost at all**. Do not write a confident
explanation into the source until someone has one.

The generalisable lesson, and the most useful line in this file: **a
microbenchmark win is not an end-to-end win.** Validate whole-loop changes
whole-loop, with a sign test; with a 2x spread the median alone would have
shrugged this off as noise.

### NEGATIVE 2 - argpartition in the greedy sampler

First implementation found the runner-up behind the penalized tokens with
`np.argpartition`, to avoid the general path's float64 copy. Correct
(203/203 differential) but **2x slower**: a quickselect over 151936 entries
costs more than the copy it saves. Replaced with a single `np.argmax` pass
that does double duty (finiteness gate + winner), falling back to the
general path in the rare case where the global max is itself penalized.
4.76x, same 203/203.

### Package E is the least-validated thing on the branch

Default threshold 131072 weights. qwen2.5-3B's **smallest** decode matvec is
524288, so the path never engages on the only model measured end to end.
Every number for it is per-call. It is bit-identical by construction
(`prange` splits the row loop; each row's accumulation order is unchanged),
so the risk is bounded - but it is unvalidated in situ and this log says so.

## Round 5: what is left

### 1. Q6_K 6-bit packing - REOPENED, but read this first

`03-RESULTS.md` NEGATIVE 1 measured this shut on machine A: packed 52.1
Gw/s vs unpacked 52.2, because "the 6-bit unpack ALU exactly cancels the
bandwidth saving at this machine's ratio". That verdict was correct **at 60
GB/s**.

Machine B sustains 2.2-4.0 GB/s with comparable per-core ALU - the ratio is
inverted by roughly 20x. A trade that is a wash when bandwidth is cheap
should win when bandwidth is the binding constraint, and 03-RESULTS itself
notes the avenue reopens if the ALU/bandwidth balance changes. Q6_K carries
31% of a Q4_K_M 3B's weights at 1.066 B/w against 0.82 achievable, so the
prize is ~10% of bytes touched per token.

**Do not re-implement blind.** Re-run the machine-A microbenchmark on a
bandwidth-starved machine first; if the packed kernel wins there, it is a
storage mode selected by measured ALU:bandwidth ratio, not a global switch.

### 2. Run the benchmark matrix - the harness exists, the data does not

Cheapest first: **two real models are already installed on machine B and
were never benchmarked.**

```powershell
alpaccaroo bench --model bartllama3b --shapes short-short,long-long --json llama3b.json
alpaccaroo bench --model hermes8b   --shapes short-short,long-long --json hermes8b.json
```

Then the parts needing other hardware: Linux, AMD Zen, a desktop/server CPU
that holds its clocks, and a CUDA device. Unmeasured anywhere: Q5_K_M and
F16/F32 dense storage, contexts 2048/8192 on a real model, and three of the
four named prompt shapes on a real model.

Always run `--stability` first and record the spread beside the numbers.

### 3. Validate Package E on a model that triggers it

Needs narrow GQA: n_kv 1, head_dim 64, embd 2048 gives exactly 131072.
`bartllama3b` and `hermes8b` may straddle it - check with
`alpaccaroo tune -m <model>`, which prints the model's distinct matvec
shapes from the GGUF header without loading it.

### 4. Extend the grouped kernel to Q4_K+Q5_K

Package D covers the Q4_K+Q6_K pairing a **Q4_K_M** file produces. A
**Q4_K_S** file stores attn_v as Q5_K and gets nothing. The kernel is a
mechanical copy with the Q5_K branch substituted; `_pair_indices` already
refuses unknown pairings safely, so the fallback is correct today.

### 5. Two open questions, both needing stable clocks

- Why did the activation-quantization threshold regress? (NEGATIVE 1 above.)
- Should the kernels use zmm? `alpaccaroo tune --asm` shows LLVM emitting
  `vpdpwssd` on 256-bit ymm and never zmm. Plausibly correct on a
  downclocking Tiger Lake; possibly leaving throughput on a part that
  sustains AVX-512 clocks. Note 03-RESULTS' standing observation that both
  Q6_K packing and speculative decoding reopen if LLVM learns to emit
  `vpdpbusd` (4 int8 MACs/lane vs vpdpwssd's 2) - `tune --asm` now reports
  that instruction directly, so the check is one command.

### 6. Deliberately not done, with the number that says why

Track 1's fused top-k and top-p over a reduced candidate set. The sampler
is 1.67 ms of a 1264 ms token and the logits materialisation ~0.2 ms, so
the whole track is worth **~0.15%** on machine B. The cheap 4.76x was
already taken by the fused greedy path. Revisit only if a profile on faster
hardware puts the sampler above a few percent.

### 7. The GPU tier was not exercised

`cuda.py` has **zero changes** on this branch, and the GPU branch of
`matvec_group` returns before any new code runs, so the plan's "CPU work
must not block GPU gains" guardrail holds by construction rather than by
testing. Nobody has run this branch on a CUDA device.

## Tooling added this round (use it, do not rebuild it)

| tool | what it answers |
|---|---|
| `alpaccaroo run <m> --profile` | where a token's time goes, and which kernel ran each matrix |
| `alpaccaroo bench --model <m>` | cold vs warm prefill/decode, p50/p95, TTFT, peak RSS |
| `alpaccaroo bench --stability` | does this machine hold its clocks (spread >1.3 = single runs are worthless) |
| `bench.paired_compare` | ABBA round-robin A/B with a sign test - the only trustworthy A/B here |
| `alpaccaroo tune -m <m>` | best thread count for a model's own shapes, from the GGUF header |
| `alpaccaroo tune --crossover` | the narrow-matrix serial/parallel threshold |
| `alpaccaroo tune --asm` | which SIMD instructions the kernels became on this CPU |
| `alpaccaroo run <m> --connect` | reuse a resident server instead of paying the load |

`docs/PERFORMANCE.md` documents all of it, every knob, and the measured
results with their hardware.
