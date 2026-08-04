# Alpaccaroo round 6: run the 8B on machine C, and serve BTBK from it

This prompt is for a fresh instance on **machine C** (AMD Ryzen 7 7730U,
Zen 3, 8 physical / 16 logical cores, 30 GiB RAM, no VNNI, no CUDA device,
sustained streaming reads 25.4-26.8 GB/s). The mission is narrow and
falsifiable:

> Get `hermes3:8b-llama3.1-q4_k_m` - or a shape-identical stand-in - running
> on this machine at the fastest tier it can reach, measure it honestly, and
> decide whether it can serve BTBK-Looper-2 inside BTBK's **75 s per letter**
> budget. If it cannot, find the gap and close it.

Round 5 left this as an explicit gap: *"The 8B class is unmeasured here, and
it is what BTBK actually runs."* This round closes it.

---

## Read before touching anything

- `prompts/05-RESULTS.md` - the round-5 record. In particular "The one
  finding that outranks the rest": **four configurations were checked
  per-call against whole-loop and all four over-credited the setting that
  reduces parallelism or dispatch count**, twice by enough to invert the
  sign. Nothing you measure per call is a result until a whole-loop sign
  test agrees.
- `prompts/05-BTBK-POS-FIT.md` - the POS assessment. **Read the scoping note
  below before you act on it.**
- `docs/PERFORMANCE.md` - the tooling reference.
- `prompts/05-artifacts/ryzen7-7730u-ubuntu/harnesses/` - do not rewrite
  these. `e2e_ab.py` in particular is the whole-loop ABBA harness with a
  sign test, and it already compares greedy token ids as well as times.

### A scoping correction that matters

`05-BTBK-POS-FIT.md` concludes "no" to running Alpaccaroo at peak. **Every
one of its three blockers is a property of POS, not of machine C**, and a
previous instance conflated the two. On machine C specifically:

| blocker in the fit doc | applies to machine C? |
|---|---|
| POS ships no numpy (pure tier only) | **No.** numpy 2.5.1 and `numba==0.65.1` both install here; numba 0.65.1 publishes a **cp314** wheel, so the pinned pair works on this box's system Python 3.14. |
| 4 GiB VM cannot hold 5.78 GiB of model | **No.** 30 GiB total, ~27 GiB available. 4.6x headroom. |
| 4 vCPUs, opaque topology | **No.** 8 real physical cores, correctly reported. |

So the 8B is not blocked here. It has never been tried here. Those are very
different things, and the difference is the reason for this round.

---

## The number that frames the whole round

Decode is memory-bandwidth-bound. 8B Q4_K_M is **5.78 GiB resident**
(`03-RESULTS.md`), which is 6.21 GB and **0.773 bytes per weight**.

| machine | sustained BW | ceiling = BW / 6.21 GB | measured | efficiency |
|---|---:|---:|---:|---:|
| A (Ryzen 5 7640HS) | 59-60 GB/s | 9.51 tok/s | **5.18** tok/s @ 6 threads | **54%** |
| C (this box) | 25.4-26.8 GB/s | **4.09-4.32 tok/s** | *unknown* | *unknown* |

Two things follow, and they set the round's two questions.

**Q1 - where does machine C land against its own wall?** If it reaches ~54%
like machine A, expect **2.2-2.4 tok/s**. That is the prediction to falsify.

**Q2 - what is machine A's missing 46%?** Machine A left nearly half its
bandwidth ceiling unclaimed and nobody has ever explained why. If that gap
is structural rather than machine-specific, it is the largest single lever
in the project and it is worth more than any kernel tweak. Machine C is a
2.3x-lower-bandwidth part, so the two regimes bracket the question.

Against BTBK's 75 s per letter:

| decode rate | tokens inside 75 s |
|---:|---:|
| 2.2 tok/s (predicted) | 165 |
| 2.4 tok/s | 180 |
| 4.1 tok/s (at the wall) | 307 |
| 4.3 tok/s | 322 |

---

## Step 1 - resolve the one input that decides everything (do this FIRST)

**How many tokens is a BTBK letter?** Nothing else in this round matters
until this is known, and it costs no machine time.

The budget is `letter_system.py:146` -> `{"ollama": 24.0, "alpaccaroo":
75.0}` seconds, and `test_llm_prompt_prefix_order.py:289` asserts
"alpaccaroo's budget leaves no room above its measured worst case". So 75 s
is a *measured tight* bound, not a guess - which means someone has already
run this workload somewhere. Find:

1. **Tokens generated per letter** - max tokens, stop conditions, and the
   observed distribution if BTBK logs it.
2. **Prompt tokens per letter** - system prompt + context. This sets prefill
   cost, and prefill is *not* free at 8B.
3. **Whether the system prompt is stable across letters.** If it is,
   Alpaccaroo's multi-slot prefix cache already amortises it and the
   effective prompt is only the per-letter delta. This is built and shipped;
   verify it engages rather than assuming it.
4. **Where the 75 s was measured** - which machine, which tier. If it was
   measured at the kernels tier on a fast box, machine C will not match it
   and the budget may need renegotiating rather than the engine optimising.

Write the answer into this file before proceeding. If a letter is ~150
tokens, the predicted 2.2-2.4 tok/s **already fits** and this round is a
confirmation. If it is 400+, no amount of tuning on this hardware closes it
and the honest recommendation is a smaller model or a longer budget - decide
that on evidence, not after a week of kernel work.

## Step 2 - build the environment, and prove the tier is real

PEP 668 marks this system Python externally managed, so use a venv. Do not
use `--break-system-packages`.

```sh
python3 -m venv ~/.venvs/alpaccaroo && . ~/.venvs/alpaccaroo/bin/activate
python -m pip install numpy "numba==0.65.1"
```

The pin is load-bearing: a different Numba deactivates the kernels and
`alpaccaroo doctor` says so. Then:

```sh
alpaccaroo doctor
```

**Gate:** it must report the kernels tier active, not `pure-python` and not
`numpy (no kernels)`. If it does not, stop and fix that - every number below
is meaningless otherwise. Record `doctor` output verbatim as the round's
first artifact.

Note the launcher at `~/.local/bin/alpaccaroo` execs the system `python3`
against the checkout. Inside the venv, invoke `python -m alpaccaroo` (or
repoint the launcher) so you actually get the venv's numpy.

Re-run the clock-stability probe. Machine C held 1.06x over 70 s in round 5;
confirm it still does before trusting any A/B.

## Step 3 - get the model

Prefer Hugging Face over `registry.ollama.ai` - both are plain HTTPS file
fetches, but HF keeps this round free of the Ollama registry entirely.

```sh
alpaccaroo pull hf:<org>/<repo>:Q4_K_M
```

The target is BTBK's own `hermes3:8b-llama3.1-q4_k_m`. A Hermes-3-Llama-3.1-8B
GGUF is the exact match; a Meta-Llama-3.1-8B-Instruct Q4_K_M is an acceptable
**performance** stand-in (same architecture, parameter count and quant mix)
but not an output stand-in. **Verify the repo and quant file actually exist
before planning around a name** - do not trust a remembered repo path.
~4.9 GB download; 425 GB free, so disk is not a constraint.

Then, before benchmarking:

```sh
alpaccaroo show <model>          # confirm Q4_K_M, arch llama, param count
python harnesses/census.py <path-to.gguf>
```

The census is what found round 5's `embd 896` K-quant fallback. Llama-3.1-8B
has `embd 4096`, a multiple of 256, so the grouped kernels *should* apply -
**confirm it rather than assume it.** Any tensor that falls back to dense
float32 streams 4 B/w instead of 0.578, a ~7x bandwidth penalty on that
matrix, and on a bandwidth-bound machine that is the whole ballgame.

## Step 4 - measure, with round 5's discipline

```sh
alpaccaroo bench --model <model> --json prompts/06-artifacts/<machine>/8b-base.json
```

Rules, all inherited from round 5 and all non-negotiable:

- **Never merge cold and warm.** The first decode token of a Q4_K_M model
  used to pay a 1865 ms in-token JIT compile; that is fixed and regression-
  tested, but cold/warm separation is what made it visible.
- Report prefill and decode separately. At 8B with a long letter prompt,
  prefill may dominate the 75 s budget - the fused matmul kernels give
  2.3-2.7x TTFT on 16-32 token turns, but a 500-token system prompt is a
  different regime and has not been measured.
- Record peak RSS. Predicted ~5.78 GiB; a large deviation means the tier is
  not what `doctor` claimed.
- Run the matrix strictly serially (`run_matrix.sh`).

**Then answer Q1 explicitly**, in writing: measured decode tok/s, the
4.09-4.32 ceiling, and the efficiency percentage. That single ratio decides
the rest of the round:

- **Near 100%** - you are at the memory wall. Only *bytes per weight* helps.
  Go to Step 5a and ignore everything else.
- **Near 54%** (machine A's number) - the inefficiency is reproducible
  across two very different bandwidth regimes, so it is structural and
  worth ~1.8x. This is the most valuable finding available this round.
  Go to Step 5b.
- **Well under 40%** - something is misconfigured. Do not optimise; diagnose.
  Go to Step 5c.

## Step 5 - the levers, in the order the measurement selects

### 5a. If you are at the wall: attack bytes per weight

**Package I - the packed Q6_K kernel.** Round 5 called this "the single most
interesting unrun experiment here" and it directly targets the only quantity
that matters in this branch. The kernel is written, bit-identity-checked by
construction, and never executed: `harnesses/q6k_pack_bench.py`. It takes
Q6_K from **1.070 to 0.820 B/w**, a 23.4% cut on the Q6_K share - and Q4_K_M
mixes Q6_K into exactly the largest tensors. Machine A measured this to a
dead heat at 59-60 GB/s; machine C at 25.4-26.8 GB/s **and no VNNI** moves
both sides of the trade, so the sign is genuinely unknown. Run it, then
validate end-to-end.

**The f16 host K/V cache.** The host KV cache is still float32 and its cost
grows with context. `ALPACCAROO_KV_F16` exists as a VRAM dial for the GPU
tier; extending it to the host cache is on the roadmap and not done. At 8B
with a long letter prompt this is real bandwidth. Quantify the KV bytes at
BTBK's actual context length before deciding whether it is worth the
accuracy question.

### 5b. If you are at ~54%: find the missing 46%

This is the interesting branch. Bandwidth is not the binding constraint, so
something else is. Candidates, cheapest first:

1. **Per-token non-weight work.** README's own estimate is ~8-9 ms per token
   for attention, rope, norms, sampling and activation quantization against
   llama.cpp's ~1-2 ms. At 2.3 tok/s (435 ms/token) that is only ~2%, so it
   is *probably not* the answer here - but the profiler now reports it
   directly and it is one run to check. **Note:** the decode profiler was
   only wired into the pure-Python forwards recently (`f321a74`); the NumPy
   forwards have always had it, so `--profile` is trustworthy at this tier.
2. **Thread scaling.** Round 5 says physical cores (8) beat SMT (16), 0 wins
   in 51 rounds, across three models - but **none of them was an 8B**, and
   an 8B has a very different working set. Confirm with
   `e2e_ab.py <model> threads 25 24`. Do not re-derive it by hand and do not
   trust `alpaccaroo tune` (see the prohibitions below).
3. **Dense fallbacks**, per the census in Step 3. A single large tensor on
   the dense path would explain a large efficiency gap on its own.
4. **Prefetch / access pattern.** If achieved bandwidth during decode is far
   below the 25.4 GB/s the stability probe measures, the kernel's read
   pattern is leaving throughput on the table. Measure achieved GB/s during
   decode directly (bytes streamed / decode seconds) and compare to the
   probe. This has never been done and it is the cleanest way to tell
   "bandwidth-bound" from "we think we are bandwidth-bound".

### 5c. If you are well under 40%: diagnose, do not tune

Work the list in order and stop at the first hit: `doctor` says kernels
inactive; the census shows dense fallbacks; cold numbers were mistaken for
warm; thermal throttling (re-run the stability probe *during* a decode, not
before it); swap (8 GB swap file exists - if the model is touching it, RSS
and timings both go strange).

### Prohibitions - these are settled, do not re-litigate

- **Do not set `ALPACCAROO_AUTOTUNE=1`.** Round 5 Package N: the tuner
  measures per call, recommended SMT for two of three models, was
  sign-inverted on both, and would have cost 42% on the 3B and 63% on the
  0.5B. It is unsafe until it validates end-to-end.
- **Do not enable `ALPACCAROO_SERIAL_MATVEC_ELEMS`.** Package J: lost 4/25
  at the default threshold and 21/25 widened.
- **Do not reach for `ALPACCAROO_DENSE_WEIGHT_MB` first.** There is ~20 GiB
  of headroom on this box and it is tempting, but densifying converts
  0.578 B/w into 4 B/w - *more* bytes read per token, which is backwards on
  a bandwidth-bound machine. If you test it, predict a loss and be ready to
  record a negative.
- **Do not chase concurrent generation.** README calls it the largest
  throughput lever in the project and that is true - but it does nothing for
  single-stream latency, and BTBK generates one letter at a time. Wrong
  lever for this round.

## Step 6 - the pure-Python track (separate, and bounded)

POS ships no numpy, so this is a real and separate goal - but be clear about
its ceiling. Measured on machine C, Python 3.14, no numpy:

- **41.58 bytes per weight**, 100% `pure-python-dense`. A Q4_0 file at
  0.655 B/w on disk expands **63x** at load.
- **~27M weight-multiply-adds/s** end-to-end (2.17 tok/s on 12.5M weights),
  cross-validated by microbenchmark at 29.1M/s.

An 8B at this tier needs **334 GB** and ~297 s **per token**. It is not a
tuning problem and no stdlib trick closes two orders of magnitude. The pure
tier's honest target is the **0.5-1B class**, not BTBK's 8B.

Three levers, all stdlib, with measured numbers:

| change | effect |
|---|---|
| `math.sumprod` for the inner loop | 29.1 -> **91.0 M mul-add/s (3.1x)** |
| `array('f')` storage | ~41.6 -> ~4 B/w (**9x memory**) but sumprod drops to 39.3 M/s |
| wire up the dead quantized pure path | ~0.58 B/w; see below |

Notes a future instance will need:

- `math.sumprod` is 3.12+, project floor is 3.10, so it needs a fallback.
- It is **not** bit-identical to `sum(w*x for ...)`: 830/2000 mixed-magnitude
  rows differ. It is *more* accurate (Neumaier-style accumulation), and the
  cross-tier gate at `smoke.py:2656` is a **1e-3 tolerance**, not exact
  equality - so it should pass. Verify, do not assume.
- Memory and speed pull against each other here: `array('f')` boxes a float
  per access, cutting sumprod's win from 3.1x to 1.35x. The roadmap does not
  say this.
- **`QuantMatrix` is only constructed when `T.HAS_NUMPY`** (`model.py:535`).
  So `qmatrix.py`'s `_matvec_pure` and its `pure-python-quant` path label are
  unreachable in *both* configurations today. The quantized pure path is
  gated off, not missing - wiring it up is the ~70x memory fix.

## Step 7 - what "done" looks like

1. `prompts/06-RESULTS.md`, written to round 5's standard: every claim with
   its artifact, cold and warm never merged, negatives reported as results.
2. An **8B row** in machine C's benchmark matrix - the gap round 5 named.
3. **A yes/no on the 75 s budget, with the arithmetic shown**: prompt tokens,
   prefill seconds, tokens per letter, decode seconds, total, margin.
4. **Q1 answered**: machine C's efficiency against its 4.09-4.32 tok/s wall.
5. **Q2 addressed**: whether machine A's missing 46% reproduces here.
6. `prompts/05-BTBK-POS-FIT.md:81` still contains an **unfilled
   `<!--TIERS-->` placeholder** - the "measured cost of that package set"
   table that document cites as measured has never been populated. Fill it,
   or delete the claim.
7. If the answer is no: the specific gap in seconds, the ranked levers, and
   what each was measured to be worth. "It does not fit" is a valid result
   only if it comes with the number it missed by.

## The standing rule

Round 5's inherited lesson, restated because it is the one that keeps being
learned the hard way: **the per-call harness is not a scaled-down decode
loop, and its error is directional.** Every candidate change goes through
`e2e_ab.py` with ABBA ordering, 25 rounds, a sign test, and greedy token ids
compared alongside times. Four for four, per-call over-credited the setting
that reduces parallelism or dispatch count. Assume you are the fifth.
