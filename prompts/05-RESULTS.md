# Results log: portable performance on a third machine (prompts/05-second-machine.md)

One line per avenue: what was tried, what was expected, what was measured,
kept or discarded. Same format as `03-RESULTS.md` and `04-RESULTS.md`.
Written as work happens.

**Read `03-RESULTS.md` and `04-RESULTS.md` first.** Several avenues here are
re-openings of things those rounds measured shut, and the reason they reopen
is the hardware, not the idea.

---

## Step 0 - this machine, before anything was changed

Call it **machine C**. It is the first of the three that holds its clocks,
which is the single fact that unlocks most of this round.

```
branch          Alpaccaroo2, clean (## Alpaccaroo2)
os              Ubuntu (live session), Linux 7.0.0-14-generic, x86_64
cpu             AMD Ryzen 7 7730U (znver3, Zen 3), 8 physical / 16 logical
cpu features    sse4.2, avx, f16c, fma, avx2      <- NO AVX-512, NO VNNI
memory          30.7 GiB, NO SWAP
blas            scipy-openblas 0.3.31.188.0, 16 threads (threaded)
kernels         alpaccaroo-kernels active, numba==0.65.1 (pin), 8 threads
versions        python 3.14.4, numpy 2.4.6, numba 0.65.1, llvmlite 0.47.0
gpu             none detected - alpaccaroo-gpu inactive
governor        amd-pstate powersave / balance_performance, boost enabled
smoke suite     all 544 checks passed (56.8 s), PYTHONUTF8=1, green before
                any measurement
```

### The environment caveat that qualifies every load number below

This is a **live-ISO session**: `/` is an overlayfs whose upper layer is
tmpfs, so the filesystem holding the model store is **RAM-backed**. Two
consequences, and both are stated wherever they matter:

- **Every `cold` model-load number here is a lower bound.** A cold load on
  this machine never touches a disk; it is a memory copy. Do not compare
  these load seconds against machine A's or machine B's, which were real
  storage. Decode and prefill throughput are unaffected.
- Disk capacity is RAM capacity. 5.7 GiB free at the start of the round is
  what bounded the model matrix (see the capacity boundary in Package G).

### Clock stability - the headline for method

```
alpaccaroo bench --model llama3.2:1b --stability --stability-seconds 70
clock stability: 25.36-26.82 GB/s over 70s (spread 1.06x)
                 - steady, single runs are comparable
```

| | machine A (03) | machine B (04) | **machine C (this round)** |
|---|---|---|---|
| CPU | Ryzen 5 7640HS, Zen 4, 6C/12T | Intel Tiger Lake, 4C/8T | **Ryzen 7 7730U, Zen 3, 8C/16T** |
| memory | DDR5 | LPDDR4x | DDR4 |
| sustained bandwidth | 59-60 GB/s | 2.2-4.0 GB/s | **25.4-26.8 GB/s** |
| clock stability | flat | spread 1.4-1.8x | **spread 1.06x** |
| AVX-512 / VNNI | yes | yes | **no** |

Machine C sits between the other two on bandwidth and is the **only one of
the three that holds its clocks under sustained load**. Per the plan's rule,
spread < 1.3 means short exploratory timing is useful here - but every
decision below still goes through `bench.paired_compare` with a sign test,
because that is what round 4 proved was necessary and cheap.

### Codegen census - `alpaccaroo tune --asm`

The interesting result is a negative one, and it is about portability rather
than about this machine.

| kernel | vpdpwssd | vpdpbusd | vpmaddwd | vpmullw | vpmuldq | ymm | zmm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `matvec_q4k_int` | **0** | 0 | 32 | 32 | **0** | 66 | 0 |
| `matvec_q5k_int` | **0** | 0 | 14 | 0 | **0** | 234 | 0 |
| `matvec_q6k_int` | **0** | 0 | 32 | 32 | **0** | 38 | 0 |

Zen 3 has no AVX-512 and no VNNI in any form, so `vpdpwssd` - the
instruction `03-RESULTS.md` §6.7 identified as the thing that unlocked this
whole kernel design - **is simply unavailable**. What survives is the part
that actually matters portably: **`vpmuldq` is 0 in all three kernels**, so
the `np.int32(acc + i32*i32)` re-cast idiom is still defeating Numba's
int64 promotion, and LLVM falls back cleanly to 256-bit `vpmaddwd`.

The lesson worth carrying: the recast idiom is **not** a VNNI trick. It is
what keeps the contraction in 32-bit lanes, and its value on a non-VNNI part
is the difference between `vpmaddwd` and 8-lane `vpmuldq`. The kernels
degrade gracefully onto a CPU generation their key optimisation predates.

**This settles Package H2 by hardware.** There is no AVX-512 path on this
CPU, so there is no zmm question to ask. Per the plan ("If this CPU has no
AVX-512/VNNI path, record that and do not pursue this package here"), H2 is
closed here and remains open for a part that sustains AVX-512 clocks.

**This settles Package L by hardware too.** `alpaccaroo doctor` reports
`gpu: none detected`; there is no CUDA device in this machine. The GPU tier
remains unexercised - `cuda.py` still has zero changes on this branch, and
nobody has yet run round 4's work on a CUDA device.

### Thread autotuning - the finding that contradicts a standing default

`alpaccaroo tune -m <model>`, per-token matvec cost by thread count. The
default is **physical cores (8)**, on the documented reasoning that decode is
bandwidth-bound and SMT siblings contend for the same load ports.

| model | quant | 1 thr | 4 thr | 8 thr | 16 thr | selected |
|---|---|---:|---:|---:|---:|---|
| qwen2.5-3b-medieval | Q4_K_M | 256.94 ms | 87.16 ms | 57.25 ms | **47.01 ms** | **16** (1.22x) |
| Qwen2.5-0.5B-Instruct | Q4_K_M* | 42.69 ms | 15.69 ms | 11.73 ms | **11.54 ms** | **16** (1.02x) |
| llama3.2:1b | Q8_0 | 113.79 ms | 42.68 ms | **29.14 ms** | 53.84 ms | **8** (1.00x) |

On this part SMT is **worth 1.22x** on the K-quant 3B and **costs 1.85x** on
the Q8_0 1B. The physical-core reasoning is not wrong in general - it is
right for one of these three models and wrong for another, on the same CPU.

This is a per-call tuner prediction, not a per-token measurement, and this
project's most expensive lesson is that those are different claims. It is
carried forward as **Package N** and validated end-to-end below before any
recommendation is made. It does **not** change a global default from one
machine.

\* the Qwen2.5-0.5B "Q4_K_M" file is mostly not K-quantized; see Package G.

---
