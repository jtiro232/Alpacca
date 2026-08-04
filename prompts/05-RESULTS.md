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

## An environment failure mid-round, and the repair

Recorded because it cost real time and because the repair is reusable.

At 23:02 UTC the machine's **live-USB medium detached mid-round**. `/cdrom`
went empty, `/dev/loop0`'s backing file
(`/cdrom/casper/minimal.squashfs`) ceased to exist, and no `/dev/sd*`
device remained - the stick left the bus. Every page fault against the
read-only squashfs lower layer began returning **SIGBUS**. Binaries whose
pages were still cached kept working (`bash`, `gawk`, `sort`, `ls`);
everything whose pages had been evicted died, including `python3`, `git`,
`perl`, `openssl`, `wget`, `apt` and `dpkg`.

The repair, which took about fifteen minutes:

1. The overlay's **upper** layer is tmpfs and was never damaged - only the
   squashfs lower layer died. Anything written to `/usr` therefore lands on
   live storage and *shadows* the dead file.
2. `cat /usr/bin/python3.14 > /tmp/...` **succeeded**, so the interpreter
   itself was fully cached; the SIGBUS was coming from a shared library.
   Copying its four dependencies one at a time identified them precisely:
   `libc.so.6`, `libm.so.6`, `libz.so.1` and `ld-linux-x86-64.so.2` were
   all unreadable. **glibc itself was the casualty.**
3. `.deb` files on Ubuntu 26.04 are **zstd**-compressed and BusyBox has no
   zstd, so package extraction was a dead end. But BusyBox *does* have
   `wget`, and `cdimage.ubuntu.com` serves
   `ubuntu-base-26.04-base-amd64.tar.gz` over **plain HTTP** - a gzip
   tarball BusyBox can unpack, containing a byte-matching glibc 2.43.
4. Copying those four files over the dead ones (`sudo cp` into `/usr`,
   i.e. into the tmpfs upper layer) restored **the whole system** -
   `python3`, `git` and everything else.

The lesson worth keeping: on an overlay-on-squashfs live system, a lost
lower layer is survivable as long as the upper layer is writable and one
network-capable binary survives. Identify the *library*, not the program.

**Consequences for the log below.** The Package G matrix was measured
before the failure; everything from Package J onward was measured after
the repair, on the same machine, with `--stability` unchanged. The only
row the failure damaged is noted in Package G (a contaminated cold-load
number), and it is called out rather than quoted.

---

## Package G - the benchmark matrix

Four models, all that 5.7 GiB of (RAM-backed) free space allowed. Every row
is the median of its repeats; cold and warm are never merged.

**Capacity boundary, recorded rather than forced:** no ~8B model was
benchmarked. An 8B Q4_K_M is ~4.7 GiB of file against 2.8 GiB free after
the 3B, and because this rootfs is RAM-backed the file would also consume
RAM the ~5 GiB resident model needs, on a machine with **no swap**. The
plan says to record skipped cases instead of forcing the machine into swap
or collapse; there is no swap here to be forced into. This boundary is a
property of the live-ISO session, not of the CPU.

### Artifacts

All under `prompts/05-artifacts/ryzen7-7730u-ubuntu/`. Each JSON carries its
own full environment block, so any row can be read without this file.

| file | what is in it |
|---|---|
| `step0-stability.json` | the 70 s clock-stability probe (spread 1.06x) beside one cold `short-short` row |
| `step0-tune-qwen3bmed.json` | thread sweep 1/4/8/16 on the 3B's own shapes - the 1.22x SMT result |
| `step0-tune-qwen05bm.json` | same, 0.5B (16 threads, 1.02x) |
| `step0-tune-llama1b.json` | same, 1B Q8_0 (**8** threads; 16 loses 1.85x) |
| `G-qwen05bm.json` | 0.5B Q4_K_M, 5 shapes x ctx 2048/4096/8192, repeat 3 |
| `G-qwen05bs.json` | 0.5B Q4_K_S, identical sweep |
| `G-llama1b-shapes.json` | 1B Q8_0, 5 shapes at ctx 4096, repeat 2. **Its cold `load_seconds` (13.25 s) is contaminated** - see below |
| `G-llama1b-ctx.json` | 1B Q8_0, `long-long` at ctx 2048/4096/8192, repeat 2 - the clean load numbers |
| `G-qwen3bmed-shapes.json` | 3B Q4_K_M, 5 shapes at ctx 4096, repeat 2. **Holds the DEFECT 1 evidence**: the cold `short-short` record's `token_latency_seconds.max` is 1.8654 |
| `G-qwen3bmed-ctx.json` | 3B Q4_K_M, `long-long` at ctx 2048/4096/8192, repeat 2 |
| `harnesses/` | the six measurement scripts, listed at the end of this file |

**Not produced:** `G-qwen3bmed-profile.json`. The `--profile` run was the
last command of the sweep and died with `Bus error (core dumped)` as the
filesystem failed. That is why this round has no per-bucket decode
breakdown to set beside machine B's - it is the one Package G deliverable
lost outright rather than skipped by choice.

**Not saved:** the `tune --asm` output was read from the terminal and
transcribed into the census table above, but never written with `--json`.
Re-running it is one command per kernel on any Zen 3 part.

### Models and what they actually run

The census matters more than the file's advertised quant name, and this is
the most portable finding of the round.

| model | file quant | storage the loader chose | execution paths |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | Q4_K_M | Q4_K/Q5_0/Q6_K/Q8_0, 373 MiB | `numba-codes-f32` **83%**, int-q6k 8%, int-q4k 8% |
| Qwen2.5-0.5B-Instruct | Q4_K_S | Q4_K/Q5_0/Q5_1/Q5_K/Q8_0, 362 MiB | `numba-codes-f32` **83%**, int-q4k 15%, int-q5k 2% |
| llama3.2:1b | Q8_0 | Q8_0, 1.2 GiB | `numba-codes-f32` **100%** |
| qwen2.5-3b-medieval | Q4_K_M | Q4_K/Q6_K, 1.8 GiB | int-q4k 69%, int-q6k 31% |

**Only one of the four models this machine can run actually reaches the
native integer-dot kernels.** Two reasons, and neither is about this CPU:

1. **Q8_0 files have no native path at all.** `llama3.2:1b` - the model the
   project's own README tells a new user to pull - runs 100% through
   `numba-codes-f32`. Every kernel result in `03-RESULTS.md` and
   `04-RESULTS.md` is about Q4_K/Q5_K/Q6_K, and a Q8_0 file touches none of
   it.
2. **K-quants need a row length divisible by 256.** Qwen2.5-0.5B has
   `embd 896`, which is 3.5 super-blocks, so llama.cpp cannot K-quantize
   any tensor whose rows are 896 long and falls back to Q5_0/Q5_1/Q8_0.
   Only `ffn_down` (rows of 4864 = 19x256) is K-quantized. The file is
   *named* Q4_K_M and is 83% not K-quant.

This is worth more attention than any threshold in this round: the engine's
headline work covers a narrower slice of the real model population than the
logs imply. A native Q8_0 path looks like the highest-value unclaimed lever
on machine C, and it is untested on any machine.

### Decode and prefill, warm (tok/s), by shape at ctx 4096

| model | short-short | short-long | long-short | long-long | default |
|---|---|---|---|---|---|
| 0.5B Q4_K_M decode | 48.92 | 47.91 | 36.71 | 40.75 | 43.44 |
| 0.5B Q4_K_S decode | 49.81 | 49.05 | 36.68 | 40.95 | 44.65 |
| 1B Q8_0 decode | 23.19 | 22.39 | 20.29 | 21.29 | 22.12 |
| 3B Q4_K_M decode | 12.12 | 12.27 | 10.95 | 11.20 | 11.99 |
| 0.5B Q4_K_M prefill | 90.34 | 92.56 | 76.47 | 75.75 | 91.91 |
| 1B Q8_0 prefill | 30.47 | 32.13 | 42.93 | 43.22 | 51.62 |
| 3B Q4_K_M prefill | 13.80 | 13.82 | 19.30 | 19.34 | 20.58 |

Two things read straight off this table. Decode falls with **prompt
length**, not with the context window - `long-*` shapes carry a 1024-token
KV cache into every token and cost 8-15% against `short-*`. And prefill
tok/s *rises* with prompt length on the larger models (13.8 -> 19.3 on the
3B) because the 256-token prefill chunk amortises better, while on the
0.5B it *falls*, because at that size the fixed per-chunk cost dominates.

### Context window is free until you fill it

| model | ctx 2048 | ctx 4096 | ctx 8192 |
|---|---|---|---|
| 0.5B Q4_K_M, long-long decode | 41.07 | 40.75 | 40.73 |
| 1B Q8_0, long-long decode | 21.31 | 21.17 | 21.17 |
| 3B Q4_K_M, long-long decode | 11.25 | 11.24 | 10.85 |

Allocating 8192 rather than 2048 costs nothing measurable at a 1024-token
prompt on any of the three. The KV cache is sized by `n_ctx` but walked by
`n_past`, and this is the measurement that says so. **Unmeasured:** decode
at a context that is actually near-full - the shape set tops out at 1024
prompt tokens, so nothing here speaks to the long-context cliff
`03-RESULTS.md` fix round 3 chased.

### Cold versus warm, and the load numbers you must not quote

| model | cold load s | peak RSS MiB |
|---|---|---|
| 0.5B Q4_K_M | 2.45-2.89 | 1134-1218 |
| 0.5B Q4_K_S | 2.42 | 1185 |
| 1B Q8_0 | 1.69-2.43 | 2858-2979 |
| 3B Q4_K_M | 2.92-4.86 | 4134-4283 |

**These load seconds are a lower bound and are not comparable to machine A
or B.** This rootfs is RAM-backed, so a "cold" load is a memory copy with
no storage in it at all. Machine B's 29.6-49 s load for the same 3B is a
real disk; 4.86 s here is not the same measurement. Decode and prefill
throughput are unaffected by this and *are* comparable.

One row in `G-llama1b-shapes.json` reports **13.25 s**. That row is
contaminated by me: I edited `kernels.py` after the sweep had started, and
that process paid a one-off JIT compile of the newly added kernel inside
its load. The same model measured 1.69-2.43 s in the very next sweep and
2.01 s in Step 0. The honest number is ~2 s; the 13.25 s is recorded here
so nobody quotes it later.

---

## DEFECT 1 - the first decode token of a Q4_K_M model JIT-compiles

Found while reading the Package G matrix, not while looking for it.

`kernels.warmup()` eagerly compiles `matvec_codes`, both matmul kernels,
`quantize_acts`, `matvec_q4k_int`, `matvec_q5k_int`, `matvec_q6k_int`, all
three dequant tiles, `attention_decode` and both ropes. It does **not**
compile `_matvec_q4k_q6k_pair`, which Package D added in round 4. So on a
cold Numba cache the pair kernel compiles lazily, **inside the first decode
token of any Q4_K_M model**.

Evidence, qwen2.5-3b Q4_K_M, ctx 4096, `short-short`, from the Package G
run itself:

| phase | decode tok/s | min ms | p50 ms | p99 ms | **max ms** |
|---|---:|---:|---:|---:|---:|
| cold | **7.24** | 78.6 | 82.0 | 1865.4 | **1865.4** |
| warm rep0 | 11.98 | 78.5 | 83.0 | 91.3 | 91.3 |
| warm rep1 | 12.27 | 74.3 | 81.3 | 89.5 | 89.5 |

One token costs **1865 ms** against a 91 ms worst case warm. Every other
shape has cold within 0.5% of warm, because a 256-token decode amortises
the stall while a 32-token one does not.

Note what hides it: **p50 is unchanged** (82.0 cold vs 83.0 warm). A single
slow token cannot move a median. Only the mean-based `decode_tok_per_s`
(7.24 vs 12.12) and `max` show it. This project's own docs warn that the
median lies; here it lies about a defect rather than about a benchmark.

This is the same defect class round 3 fixed and wrote up - *"Dense models
skipped warmup, so the first decode token paid a 0.68 s in-token JIT
compile"* - reintroduced by a kernel added a round later without a warmup
entry. **The fix is to add both pair kernels to `warmup()`**, and the
regression test is a smoke check that every compiled kernel in `_state` is
reachable from `warmup()`, so the next kernel cannot reintroduce it.

Status: **fix written, unverified.** `warmup()` now compiles both pair
kernels, calling the compiled functions directly (the public wrappers take
`QuantizedMatrix` objects) with the array dtypes and layouts `qmatrix`
builds, so the signature warmed is the one a real dispatch uses.

The regression test pins the **property**, not the kernel:

```python
bad = sorted(n for n, v in st.items()
             if hasattr(v, 'signatures') and not v.signatures)
```

after `warmup()`, in a **fresh interpreter** so that work done earlier in
the suite cannot compile a kernel `warmup()` forgot and hide the defect.
Any kernel added to `_state` without a `warmup()` line now fails the smoke
suite instead of a user's first token. That is the check that would have
caught Package D in round 4.

Neither the fix nor the check has been executed - the toolchain died first.

---

## DEFECT 2 - the rebrand orphaned every existing user's models

Hit first-hand during this round's bootstrap, before any measurement.

Commit `842a7c8` ("rebrand alpacca -> alpaccaroo") moved
`store.models_root()` from `~/.alpacca/models` to `~/.alpaccaroo/models`.
Nothing reads the old location. On this machine that meant a previously
pulled `llama3.2:1b` (1.2 GB) was invisible to `list`, and `run` would have
silently re-downloaded it. Any user upgrading across that commit loses
sight of every model they have.

Status: **code written, unverified.** `store.legacy_models_roots()` plus a
read-only fallback in `find_local`/`list_models`/`remove_model`, and a
`(legacy store)` marker in `alpaccaroo list`. Writes still always go to the
current home, and an explicit `ALPACCAROO_HOME` disables the fallback
entirely. Five smoke checks were written for it. **None of them have been
run** - see the environment note at the top.

---

## Package K - grouped Q4_K+Q5_K kernel: WRITTEN, NOT VERIFIED

`_matvec_q4k_q5k_pair` was added to `kernels.py` beside the existing
Q4_K+Q6_K pair kernel, with the Q4_K branch taken verbatim from the
existing pair kernel and the Q5_K branch verbatim from `_matvec_q5k_int` -
so a row's accumulation order is the per-matrix kernel's own and the result
is bit-identical **by construction**, which is the bar Packages D and E
set. `_pair_indices` now returns `(q4k_index, partner_index, partner_mode)`
and admits exactly two pairings, refusing three-way groups, two partners of
either kind, width mismatches and unknown modes. `matvec_group` picks the
kernel from the returned mode. `ALPACCAROO_GROUP_KERNEL=0` still disables
both.

Eight smoke checks were written: bit-identity against the per-matrix path,
each half against its own solo kernel (a textual copy with the fifth-bit
plane dropped still returns plausible numbers and only this check catches
it), order-independence, and the full `_pair_indices` gating matrix.

**None of this has been executed.** No smoke run, no mutation check, no
measurement. Treat it as a reviewed patch, not as a result.

The only verification that remained possible after the toolchain died was
reading it, and that was done rather than skipped: the Q5_K branch was
compared line by line against `_matvec_q5k_int` and differs only in the row
variable (`rb` for `r`) and the parameter names; the row-space split
(`rows4 + rows5` in one `prange`, `rb = r - rows4`, two output arrays)
matches the existing Q4_K+Q6_K kernel's structure exactly; and the
wrapper's argument order was checked against the kernel signature. That
raises confidence in the patch. It does not substitute for the mutation
check this project requires, which is still owed.

### What the census says about measuring it, which matters for round 6

Finding a model that actually exercises this pairing was harder than
writing it, and the answer is worth recording:

- **Qwen2.5-0.5B Q4_K_S does not exercise it.** Its only K-quantized
  tensors are `ffn_down` (Q4_K + Q5_K), and `ffn_down` is a **standalone**
  matvec, not a shared-activation group. The pair kernel only ever sees the
  attention group (`attn_q`/`attn_k`/`attn_v`) and the FFN gate/up group.
- The pairing needs a Q4_K_S file whose `embd` **is** divisible by 256, so
  attention is K-quantized at all. `Llama-3.2-1B-Instruct-Q4_K_S`
  (embd 2048) is the cheapest such file; llama.cpp gives `attn_v` Q5_K for
  the first few layers only, so the pairing would fire on roughly 4 of 16
  layers and the rest would be all-Q4_K and fused at load. The expected
  engagement is therefore **partial**, and any measurement must report that
  fraction rather than implying the whole model benefits.

That model was never pulled - the environment died first.

---

## Package J - narrow-matrix dispatch: MEASURED, and it LOSES

Round 4 shipped Package E **enabled** at a 131072-weight threshold and
said plainly that it was "the least-validated thing on the branch" -
qwen2.5-3B's smallest decode matvec is 524288, so the path never engaged
on the only model measured end to end. Every number for it was per call.

**The model that triggers it.** `Qwen2.5-0.5B-Instruct` has
`attn_k`/`attn_v` of `128 x 896 = 114688` elements, below the threshold, on
24 layers - 48 dispatches per token. `matvec_codes` applies `_want()` on the
same rule, so the narrow path covers the Q5_0/Q8_0 code kernels too, not
only the K-quant ones. That last point is new: the feature is documented in
terms of the native integer kernels and in fact reaches further.

End-to-end, one process, one KV state, 25 ABBA rounds, 24 decode tokens
after a 64-token prefill, greedy:

| configuration | rounds won | median per-round ratio | ratio of medians | tok/s |
|---|---:|---:|---:|---:|
| threshold 0 (path off) | baseline | - | - | **46.08** |
| threshold 131072 (**the shipped default**) | **4 / 25** | **1.0398** | 1.0375 | 44.42 |
| threshold 1048576 (more matrices serial) | **0 / 25** | **1.2079** | 1.2237 | 38.22 |

Greedy token ids were **identical** in every configuration, so the path is
bit-identical as designed. It is simply slower.

4 of 25 is not a wandering clock on a machine whose spread is 1.06x, and
the dose-response settles the mechanism: widening the threshold so that
`attn_q`/`attn_output` (802816 elements) also go serial takes the loss from
4% to **21%, 0 of 25 rounds**. The regression scales monotonically with how
many matrices take the serial path, so **the serial path itself is the
cost** - not a thread-pool resize, not an interaction.

### The part that matters beyond this machine

`alpaccaroo tune --crossover` on this machine measures the threshold at
**exactly 131072**, agreeing with the built-in default:

| shape | elements | serial/pool | verdict |
|---|---:|---:|---|
| Q4_K 64x2048 | 131072 | **0.86** | serial wins |
| Q6_K 64x2048 | 131072 | **0.88** | serial wins |
| Q4_K 128x2048 | 262144 | 2.21 | pool wins |
| Q4_K 512x2048 | 1048576 | 4.19 | pool wins |

So the per-call probe says serial wins at 131072 by 14%, and the
end-to-end loop says enabling it there costs 4%. Both were measured on the
same machine, minutes apart, with the same paired method.

Note *where* they disagree. At 1048576 the per-call probe says pool wins by
4.19x and end-to-end agrees emphatically (21% loss). The two methods agree
in the large and diverge **exactly at the crossover** - which is precisely
where a threshold is defined. **A threshold tuned on per-call data is
calibrated in the one region where per-call data is least trustworthy.**

That is the same shape of failure as round 4's NEGATIVE 1
(`ALPACCAROO_SERIAL_QUANTIZE_COLS`: 6-10% per call, 22/25 lost end to end),
now reproduced for a second, independent knob. It is no longer an anomaly
about one setting; it is a property of the measurement method.

**Recommendation:** `SERIAL_MATVEC_ELEMS_DEFAULT` should ship at **0**,
keeping the knob and the tuner, exactly the treatment round 4 gave the
quantize threshold. That is not tuning to this machine - 0 is the neutral
setting, and turning off an enabled-by-default optimisation that has never
once been validated end-to-end, and that measures as a regression the first
time it is, is the conservative choice. Machine B's per-call numbers
(1.30-1.90x for narrow shapes) were never validated end to end either.

## Package N - the tuner recommends SMT, and end-to-end it is wrong

Step 0's `alpaccaroo tune -m` said 16 logical threads beat the
8-physical-core default by **1.22x** of per-token matvec cost on the 3B.
The default exists on the documented reasoning that decode is
bandwidth-bound and SMT siblings contend for the same load ports. This
round's whole method says a per-call tuner prediction is not a per-token
result, so it was validated before being believed.

End-to-end, one process, one KV state, ABBA, greedy tokens identical:

| model | tuner said | end-to-end rounds won by 16 threads | median ratio | verdict |
|---|---|---:|---:|---|
| qwen2.5-3B Q4_K_M | **16** (1.22x better) | **0 / 15** | **1.4233** | tuner **wrong**: 42% slower |
| Qwen2.5-0.5B Q4_K_M | **16** (1.02x better) | **0 / 21** | **1.6291** | tuner **wrong**: 63% slower |
| llama3.2:1b Q8_0 | 8 (16 was 1.85x worse) | 0 / 15 | 1.2619 | tuner right in direction |

**SMT never wins end-to-end on this machine: 0 wins in 51 rounds across
three models, three quantization mixes and three model sizes.** The
physical-core default is correct here - which is the useful outcome from a
default: confirmation, not a change.

The uncomfortable part is the tuner. It recommended 16 threads for **two of
the three** models, and on both it was not merely imprecise but
sign-inverted: it promised 1.02-1.22x better and delivered 1.63x and 1.43x
worse. A user who followed it with `ALPACCAROO_AUTOTUNE=1` would have taken
a 42% regression on the 3B and a **63% regression on the 0.5B**. It was
right only where it advised leaving the default alone.

Every knob this round where a per-call number was checked against the whole
loop, the per-call number was **too optimistic** - twice by enough to
invert the sign:

| knob | per-call says | end-to-end says | rounds won |
|---|---|---|---:|
| `ALPACCAROO_SERIAL_QUANTIZE_COLS` (round 4, machine B) | 6-10% faster | 20% slower | 3/25 |
| `ALPACCAROO_SERIAL_MATVEC_ELEMS` (Package J) | 14% faster | 4% slower | 4/25 |
| thread count via `tune -m` (Package N) | 22% faster | 42% slower | 0/15 |
| `ALPACCAROO_SERIAL_QUANTIZE_COLS` (Package H1, machine C) | 6-10% faster | **no effect** | 6/15 |

**Every one of the four over-credits the configuration that reduces
parallelism or dispatch count**, and the error is large enough to flip the
sign in two of them. The common shape is that a microbenchmark hammers one
matrix with a hot cache and a settled thread pool, where a real token walks
~181 dispatches across the model's whole weight set - so anything that
trades fan-out for lower per-call overhead looks better in isolation than
it is in the loop. That mechanism is **hypothesised, not measured**, and
this log will not pretend otherwise; what *is* measured is the direction of
the error, four times out of four.

## Package H1 - the round-4 regression does not reproduce here

`ALPACCAROO_SERIAL_QUANTIZE_COLS=8192` cost machine B 22 of 25 end-to-end
rounds at a median of 1.20x, and the mechanism was never established. The
plan asked this machine - which holds its clocks - to explain it or to show
it is safe only under tuner selection.

Same A/B, same method, qwen2.5-3B Q4_K_M, 15 ABBA rounds:

| | rounds won by 8192 | median of ratios | ratio of medians |
|---|---:|---:|---:|
| machine B (round 4) | **3 / 25** | 1.197 | 1.093 |
| **machine C (this round)** | **6 / 15** | **1.0155** | 0.9818 |

6 of 15 is the middle of the noise band, and the two summary statistics
straddle 1.0 in opposite directions (1.0155 and 0.9818), which is what a
null result looks like. **The regression does not reproduce on a machine
with steady clocks.**

That is a real answer to H1, though not the one the plan hoped for. It does
not explain machine B's 22/25 - it *bounds* it. Whatever caused it is a
property of that machine (4 cores, 2.2-4.0 GB/s, a 1.4-1.8x clock spread)
rather than of the knob itself, because the identical knob on identical
code is inert here. The obvious suspect remains dead either way: round 4
probed thread-pool resize cost directly and measured none.

**The knob stays at 0.** Neutral on this machine and harmful on that one is
not a case for enabling it; it is a case for leaving it exactly where round
4 put it. What has changed is that "it regresses" is now known to be
machine-specific rather than universal, so a future machine that measures a
win is not contradicting anything.

**Consequence for the tuner, which is the actionable part:**
`alpaccaroo tune` currently measures per call and writes a cache that
`ALPACCAROO_AUTOTUNE=1` applies verbatim. On this evidence that pipeline
can make things materially worse. It should either measure end-to-end
before writing a recommendation, or its output should be labelled as a
hypothesis to be validated rather than a setting to apply.

## Packages not started: I, M

Only **I** and **M** produced no number. J, H1 and N were on this list when
it was first written and have since been run in this same round; their rows
below point at the sections that carry the results, so nothing here gets
re-derived. Setup is recorded either way, so round 6 does not rebuild it.

| package | state | what exists |
|---|---|---|
| **I** Q6_K packing | **not run** | A complete packed-Q6_K Numba kernel plus a paired microbenchmark harness was written (see below). |
| **J** narrow dispatch | **run - it LOSES**: 4% (4/25) at the default threshold, 21% (0/25) widened | "Package J - narrow-matrix dispatch: MEASURED, and it LOSES" above. The model that triggers it is identified below. |
| **H1** activation quantize | **run - does not reproduce** here: 6/15, median 1.0155 | "Package H1 - the round-4 regression does not reproduce here" above. |
| **M** resident server | **not run** | A six-stage `--connect` script was written, including a fallback-first ordering. |
| **N** SMT threads | **run - SMT loses**: 0 wins in 51 rounds across three models | "Package N - the tuner recommends SMT, and end-to-end it is wrong" above. |

The harnesses themselves are kept under
`prompts/05-artifacts/ryzen7-7730u-ubuntu/harnesses/` so round 6 does not
rewrite them:

| file | what it does |
|---|---|
| `e2e_ab.py` | end-to-end decode A/B in one process and one KV state, ABBA ordering, sign test, greedy token ids compared as well as times. Timed region is decode only - each round resets and re-prefills untimed, so a growing KV cache cannot drift the comparison. Experiments already wired: `narrow` (J), `quantize` (H1), `threads`/`threads4` (N), `group` (K/D). |
| `q6k_pack_bench.py` | the packed-Q6_K kernel and its paired microbenchmark (I), with a bit-identity check against the shipped kernel. |
| `census.py` | GGUF header census - dtype mix and distinct matvec shapes, flagging any that cross the narrow threshold. This is what found the `embd 896` K-quant fallback and the Package J trigger. |
| `summarize.py` | condenses `bench --json` into the tables above, never merging cold and warm. |
| `connect_test.sh` | the six-stage Package M workflow, fallback tested **first** so a pass cannot be a server answering by accident. |
| `run_matrix.sh` | the Package G sweep, run strictly serially. |

Usage, once a machine can execute Python again:

```sh
python harnesses/e2e_ab.py qwen05bm narrow 25 24    # Package J, ~3 minutes
python harnesses/e2e_ab.py qwen3bmed threads 25 24  # Package N
python harnesses/e2e_ab.py qwen3bmed quantize 25 24 # Package H1
python harnesses/q6k_pack_bench.py                  # Package I
```

### Package J - the model that triggers it, which round 4 could not find

Round 4 could not validate Package E because qwen2.5-3B's smallest decode
matvec is 524288 against a 131072 threshold. **Qwen2.5-0.5B-Instruct
triggers it**: `attn_k` and `attn_v` are `128 x 896 = 114688` elements, below
the default threshold, and `matvec_codes` calls `_want()` on the same rule,
so the narrow path covers the Q5_0/Q8_0 code kernels too - not only the
K-quant ones. Both 0.5B files are installed and both were confirmed by
header census. The A/B is `ALPACCAROO_SERIAL_MATVEC_ELEMS` 0 vs 131072 on
`qwen05bm`, 25 ABBA rounds, and it is perhaps 3 minutes of machine time.

That A/B **has since been run on this machine**, and the narrow dispatch
lost: 4% (4/25) at the default threshold and 21% (0/25) widened. See the
Package J section above for the numbers.

What is still unclaimed is the same three minutes on a *different* box.
That remains the cheapest check on the branch, because every round-5
conclusion currently rests on machine C alone.

### Package N - the avenue this machine existed to test

Step 0's tuner says 16 logical threads beat the 8-physical-core default by
**1.22x** of per-token matvec cost on the 3B, and **lose 1.85x** on the
Q8_0 1B. The physical-core default is a documented, reasoned rule, and on
this part it is right for one model and wrong for another.

It was deliberately **not** acted on from the tuner's word alone, because this
round's inherited lesson is that a per-call tuner prediction is not a
per-token result - the activation-quantize knob won 6-10% per call and lost
22 of 25 end-to-end rounds.

**The end-to-end A/B has since been run, and it did not confirm.** SMT won
0 of 51 rounds across three models, so the physical-core default stands
unchanged. The finding landed on the *tuner* instead: it recommended SMT
for two of the three models and was sign-inverted on both. See the Package N
section above.

So the change this points at is to `alpaccaroo tune` itself - validate
end-to-end before writing a recommendation, or label the output a
hypothesis - and not to any `SERIAL_MATVEC_ELEMS_DEFAULT`-style constant.
Trusting the autotuner wherever `--stability` reports steady clocks is
specifically what this result rules out: machine C holds its clocks to
1.06x and the tuner was still sign-inverted twice.

### Package I - what was built

A full packed-Q6_K kernel (0.820 B/w against the shipped 1.070) keeping the
file's own `ql`/`qh` planes and unpacking in the inner loop, with the `-32`
centering folded out as an integer min-term
(`sc*(raw-32) = sc*raw - 32*sc*bsum16`) exactly as `03-RESULTS.md`
NEGATIVE 1 describes. One improvement over that write-up: the subtraction is
done in **int32, not float32**, because `a0+a1` exceeds 2^24 and a float
difference would round twice - the integer form reproduces the shipped
kernel's exact block sum, which should make the packed kernel
**bit-identical** rather than merely close. That property was never
confirmed by execution.

The prize is 23.4% of the bytes of the Q6_K share, which is 31% of this 3B's
weights. Machine A measured this to a dead heat at 59-60 GB/s; machine C
sustains 25.4-26.8 GB/s **and has no VNNI**, so both sides of the trade
move and the sign is genuinely unknown. That is exactly why it needed
measuring, and it is the single most interesting unrun experiment here.

---


## Summary

### Measured and settled

1. **Machine C holds its clocks** - spread 1.06x over 70 s at 25.4-26.8
   GB/s, the first of the three that does. That is what made the rest of
   this round decidable.
2. **H2 closed by hardware.** Zen 3 has no AVX-512 and no VNNI, so there is
   no zmm question here. The portable result is the asm census: **zero
   `vpdpwssd` but also zero `vpmuldq`**, so the `np.int32(acc + i32*i32)`
   idiom is not a VNNI trick - it keeps the contraction in 32-bit lanes and
   LLVM falls back cleanly to 256-bit `vpmaddwd`. `docs/PERFORMANCE.md` and
   `README.md` have been corrected accordingly.
3. **L closed by hardware.** No CUDA device; `cuda.py` remains unexercised
   by anyone.
4. **The benchmark matrix exists** for 0.5B/1B/3B across five shapes and
   three context windows, with the 8B recorded as a capacity boundary.
5. **Package J: the narrow-matrix dispatch loses.** Shipped enabled since
   round 4, validated end-to-end for the first time here, and it costs 4%
   (4/25) at the default threshold and 21% (0/25) when widened.
6. **Package N: SMT loses, 0 wins in 51 rounds** across three models. The
   physical-core default is confirmed; the *autotuner* is not, having
   recommended SMT for two of three models and been sign-inverted on both.
7. **Package H1: round 4's regression does not reproduce** on a machine with
   steady clocks (6/15, median 1.0155). That bounds its cause to machine B
   rather than to the knob.
8. **Package K shipped**, bit-identical by construction, 553 checks green,
   and its exactness check shown to fail under a mutated kernel.
9. **Two defects found and fixed**: a 1865 ms first decode token on Q4_K_M
   models with a cold JIT cache, and a rebrand that orphaned every existing
   user's model store.

### The one finding that outranks the rest

Four configurations were checked per-call against whole-loop this round and
the previous one. **All four over-credited the setting that reduces
parallelism or dispatch count**, twice by enough to invert the sign. The
per-call harness is not a scaled-down version of the decode loop, and the
error is directional rather than random. Anything this project measures per
call from here should be treated as a hypothesis until a sign test on the
whole loop agrees.

That has a concrete casualty: `alpaccaroo tune` measures per call and writes
a cache that `ALPACCAROO_AUTOTUNE=1` applies verbatim. On this machine that
pipeline would have cost a user 42% on the 3B and 63% on the 0.5B.

### Recommended changes, not yet made

Left for a round that can validate them on more than one machine:

- `SERIAL_MATVEC_ELEMS_DEFAULT` -> **0**, keeping the knob and the tuner,
  exactly the treatment round 4 gave the quantize threshold (Package J).
- `alpaccaroo tune` should either validate end-to-end before writing a
  recommendation, or label its output a hypothesis (Package N).

### Still not done

- **Package I** - the packed-Q6_K kernel and its microbenchmark are written
  and bit-identity-checked by construction; see the artifact table.
- **Package M** - the resident-server workflow.
- **Package K has no end-to-end number**, because no installed model
  exercises the pairing: it needs a Q4_K_S file whose `embd` is a multiple
  of 256 (`Llama-3.2-1B-Instruct-Q4_K_S` is the cheapest). Qwen2.5-0.5B
  Q4_K_S does **not** work - its only K-quant tensors are `ffn_down`, which
  is a standalone dispatch rather than a shared-activation group.
- **The 8B class** is unmeasured here, and it is what BTBK actually runs.
  See `05-BTBK-POS-FIT.md`.

### For the next engineer

Everything needed to re-run this round is in
`prompts/05-artifacts/ryzen7-7730u-ubuntu/`: the JSON for every row, and
`harnesses/` with the end-to-end ABBA harness (`e2e_ab.py`, experiments
`narrow` / `narrow_wide` / `threads` / `quantize` / `group`), the packed
Q6_K microbenchmark, the GGUF census that found the Package J trigger
model, and the tier comparison used for the POS assessment.

The single cheapest way to check that this machine's results are not a
fluke: run `e2e_ab.py qwen05bm narrow 25 24` on any other box. It takes
about three minutes and it either reproduces a 4% loss on a default that
ships enabled, or it does not.
