# Alpaccaroo round 5: continue portable performance work on another PC

This prompt is for a fresh Codex instance on a different PC. The mission is
to improve Alpaccaroo2 into a broadly reliable, faster local inference
engine. This is **not** a compare-and-contrast project between computers.
Hardware differences are useful only because they expose which
optimisations generalise, which must remain opt-in/tuned, and which should
be discarded.

Round 4 (`prompts/04-portable-performance.md`) built a measurement-driven
performance architecture and characterised **one** machine: a
bandwidth-starved, power-cycling Intel laptop sustaining 2.2-4.0 GB/s.
Round 3 (`prompts/03-close-the-ollama-gap.md`) characterised a different
one: an ALU-bound Ryzen sustaining 59-60 GB/s. Treat those results as
constraints and prior evidence, not as the thing you are trying to
recreate. Use this PC to validate the current defaults, find portable
improvements, and implement only changes that survive correctness and
end-to-end measurement.

The tools already exist. Do not rebuild them. Most of this round is
running them, recording what they say, and acting only where the data
supports it.

## Read these first, in this order

1. `prompts/04-RESULTS.md` - the previous round's log, including the
   continuation notes this plan expands, and the two-machine comparison.
2. `prompts/03-RESULTS.md` - the earlier log. Several avenues here are
   re-openings of things it measured shut; know why before re-opening one.
3. `docs/PERFORMANCE.md` - the reference for every tool, label, knob and
   profiler field used below.

If any of those three contradicts this plan, they were written against
measurements and this plan was not. Trust them unless the current code or
new measurements prove they have gone stale; if that happens, state the
discrepancy in `prompts/05-RESULTS.md` before acting on it.

## Bootstrap

Start from a clean checkout or a new local branch. Do not overwrite an
existing dirty worktree.

```sh
git clone https://github.com/jtiro232/Alpacca.git
cd Alpacca
git checkout Alpaccaroo2
python -m pip install ".[kernels]"     # numpy + the PINNED numba==0.65.1
```

The branch is `Alpaccaroo2` - two `c`s, matching the project name. If it is
not present, list remote branches (`git ls-remote --heads origin`) and use
the one carrying the round-4 performance work; do not silently fall back to
`main`, and do not use `Alpaccaroo` (no `2`), which is its stale parent.

Run the smoke suite before measuring:

```sh
python tests/smoke.py                  # must print "all NNN checks passed"
```

On Windows, if the suite reaches tokenizer checks and fails with a
`UnicodeEncodeError` while printing labels, rerun it with temporary UTF-8
mode instead of changing code:

```powershell
$env:PYTHONUTF8 = "1"
python tests\smoke.py
```

On Linux/macOS, this is equivalent:

```sh
PYTHONUTF8=1 python tests/smoke.py
```

The pin is not advisory: a different Numba deactivates the kernels and
`alpaccaroo doctor` will say so. If the smoke suite fails on your platform
**that is the first finding of the round** - record it and fix it before
measuring anything, because every number below assumes a green suite.

You will need at least one real model. Any GGUF works; the round-4 numbers
used `hf:LaRocca/qwen2.5-3b-medieval-npc-gguf` (Q4_K_M, 3B).

```sh
alpaccaroo pull hf:LaRocca/qwen2.5-3b-medieval-npc-gguf
```

## Step 0 - characterise this machine BEFORE changing anything

Nothing in this plan is meaningful without this. Run it, save it, and put
it at the top of your results log:

```sh
git status --short --branch                         # exact branch and cleanliness
alpaccaroo doctor                                   # cpu, features, cores, BLAS threads
alpaccaroo bench --model <model> --stability        # does it hold its clocks?
alpaccaroo tune --asm                               # what the kernels compiled to
alpaccaroo tune -m <model>                          # best thread count for these shapes
```

The stability spread decides how cautious your method must be:

- **spread < 1.3** - this machine holds its clocks well enough that short
  exploratory timing is useful. Still use paired A/B plus a sign test before
  changing a default.
- **spread >= 1.3** - single runs are not decision-quality. Every A/B must
  go through `alpaccaroo.bench.paired_compare` (ABBA round-robin + sign
  test). Round 4 measured the *opposite winner* for one shape depending on
  whether the comparison was paired.

## Non-negotiable constraints (inherited, still binding)

- Keep the pure Python / NumPy / Numba / GPU tiering intact.
- Every optimisation must be measurable and disableable by an env var.
- Prefer runtime capability detection over machine-specific branches.
- Do not turn the work into a report comparing PCs. Benchmarks are
  evidence used to decide what Alpaccaroo2 should do by default, by
  autotune, or behind a flag.
- Do not tune the project around **your** machine either. Two machines'
  worth of over-fitting is what this round is correcting.
- Kernel changes must be **bit-identical by construction** where possible -
  Packages D and E achieve this by changing which thread computes a row,
  never how a row is computed. Where bit-identity is impossible, prove
  equivalence differentially and pin it in `tests/smoke.py`.
- `ALPACCAROO_THREADS` always wins over any tuning mechanism.
- Do not merge anything that is faster per call and unmeasured per token.
  Round 4 shipped exactly one such change and it lost 22 of 25 end-to-end
  rounds.
- **You are authorised to commit and push to `Alpaccaroo2`.** That
  authorisation is scoped: push to `Alpaccaroo2` or to a branch created
  from it for this work, and nothing else. Do not push to `main`,
  `Alpaccaroo`, or unrelated branches; do not force-push, rewrite
  published history, delete remote branches, or change the remote URL.
  Push reviewable increments instead of saving everything for one final
  lump.

## Work packages

Independent; take them in any order after Step 0. Each names what "done"
means, because "it felt faster" is what this whole architecture exists to
replace.

### Package G - the benchmark matrix (highest value, lowest risk)

The harness exists, but the cross-device data needed to choose safe
defaults is still incomplete. The goal is not to crown one machine faster
than another; the goal is to discover which defaults, tuning rules, and
fallbacks improve Alpaccaroo2 across realistic hardware.

Deliverables:
- `alpaccaroo bench --json` output for every model size this PC can run
  safely (~1B, ~3B, ~8B), using the named prompt shapes and context windows
  that fit in RAM/VRAM. Prefer 2048/4096/8192, but record skipped cases
  instead of forcing a machine into swap or thermal collapse.
- The same for at least one quantization other than Q4_K_M when practical -
  Q5_K_M, Q4_K_S (whose attn_v is Q5_K), or an F16/F32 dense build. If the
  machine cannot run those safely, log that as a capacity boundary.
- A `--stability` record beside the numbers.
- Results committed to `prompts/05-RESULTS.md`.

Acceptance criteria:
- Every row carries its environment block (the JSON does this for you).
- Cold and warm are never quoted as one another.
- Where your numbers contradict round 3 or 4, the contradiction is used to
  decide whether the behaviour should be a default, an autotuned choice, an
  env-flagged option, or discarded. Do not average contradictory machines
  into one vague conclusion.

### Package H - reduce the two open uncertainties

Both were blocked on round 4's machine being unable to measure reliably.
If your Step 0 spread is < 1.3, you can produce decision-quality evidence
for this class of machine. One PC does **not** settle a global default by
itself; it can justify an autotuned rule, an env-flagged path, or a follow-up
matrix requirement.

**H1. Why did the narrow activation-quantization threshold regress?**
`ALPACCAROO_SERIAL_QUANTIZE_COLS` wins 6-10% per call and lost 22/25
end-to-end rounds. The obvious cause - thread-pool resizes - was probed
directly and measured to cost nothing. Reproduce the A/B here
(`prompts/04-RESULTS.md` NEGATIVE 1 has the method). Either explain it, or
show that it is safe only when the tuner selects it. Do not enable it
globally from one machine's win.

**H2. Should the kernels use 512-bit registers?** `alpaccaroo tune --asm`
on round 4's Tiger Lake shows `vpdpwssd` on 256-bit `ymm` and never `zmm`.
On a downclocking part that is plausibly correct. On a part that sustains
AVX-512 clocks it may be leaving throughput unused. If this CPU has no
AVX-512/VNNI path, record that and do not pursue this package here.

Acceptance criteria:
- H1: a mechanism supported by a direct measurement, or a documented
  "safe only when tuning selects it" result with the data. Do not make it a
  global default from one PC.
- H2: an asm census from your CPU, plus - only if it shows a plausible and
  controllable gap - a measured attempt at raising the vector width, kept
  only if it wins end-to-end.

### Package I - Q6_K 6-bit packing, re-measured before implemented

**Read `prompts/03-RESULTS.md` NEGATIVE 1 first.** It measured packed Q6_K
at 52.1 Gw/s against unpacked 52.2 - the 6-bit unpack ALU exactly
cancelling the bandwidth saving - and that verdict is correct at that
machine's 59-60 GB/s. Round 4's machine sustains 2.2-4.0 GB/s with
comparable per-core ALU, which inverts the ratio the verdict rested on.

The hypothesis is worth testing where bandwidth binds: Q6_K carries ~31%
of a Q4_K_M 3B's weights at 1.066 B/weight against 0.82 achievable, so the
upper bound is roughly 10% of the bytes a token touches. That is not a
speedup claim until the packed path wins end-to-end.

Deliverables, **in this order**:
1. A microbenchmark of the packed vs unpacked Q6_K kernel on **your**
   machine, reported alongside your sustained GB/s.
2. Only if packed wins: the storage mode, selected by measured
   ALU:bandwidth ratio rather than a global switch, behind an env var.

Acceptance criteria:
- Step 1 is logged whether or not step 2 happens. A reproduced negative on
  another machine is a valuable result and should stop this branch from
  spending more time here unless a future toolchain changes the codegen.
- If implemented: bit-identical or differentially proven, flag-disableable,
  and a measured end-to-end win with a sign test.

### Package J - validate the narrow-matrix dispatch in situ

Package E ships enabled at a 131072-weight threshold and is **inert on
every model measured end to end** - qwen2.5-3B's smallest decode matvec is
524288. Every number for it is per-call.

Deliverables:
- `alpaccaroo tune -m <model>` on each model you have (it prints the
  distinct matvec shapes from the GGUF header without loading weights) to
  find one whose shapes cross the threshold. Narrow GQA does it: n_kv 1,
  head_dim 64, embd 2048 is exactly 131072.
- An end-to-end paired A/B on that model with the threshold on and off.
- `alpaccaroo tune --crossover` on your machine, and the cached threshold.

Acceptance criteria:
- Either an end-to-end result that justifies the default, or a changed
  default with the data, or an honest "still inert on everything available
  here".

### Package K - extend the grouped kernel to Q4_K+Q5_K

Package D fuses the Q4_K+Q6_K pairing a **Q4_K_M** file produces. A
**Q4_K_S** file stores attn_v as Q5_K and currently falls back to
per-matrix dispatch, getting nothing.

Deliverables:
- A `_matvec_q4k_q5k_pair` kernel alongside the existing one. It is expected
  to be close to the Q4_K+Q6_K pair kernel, but do not assume a textual copy
  is correct; use the existing Q5_K exactness checks as the oracle.
- Dispatch via `tensor._pair_indices`, which already refuses unknown
  pairings safely - so the current fallback is correct and the change is
  additive.
- Smoke coverage matching the Q4_K+Q6_K checks.

Acceptance criteria:
- Bit-identical to the per-matrix path (the existing pair kernel is).
- `ALPACCAROO_GROUP_KERNEL=0` still disables it.
- A measured result on a Q4_K_S model; keep it only if it does not regress.

### Package L - the GPU tier, if you have a CUDA device

`alpaccaroo/cuda.py` has **zero changes** on this branch and nobody has run
round 4's work on a GPU. The guardrail ("CPU work must not block GPU
gains") holds by construction - the GPU branch of `matvec_group` returns
before any new code - but by construction is not by testing.

Deliverables:
- `python tests/smoke.py` with the GPU tier enabled.
- `alpaccaroo bench --model <m> --json` on the GPU tier, and a
  `--profile` run showing `gpu-cuda` in the path report.

Acceptance criteria:
- No regression against the CPU tier's correctness checks.
- Any GPU-path defect found is fixed or documented, not worked around by
  disabling CPU features.

### Package M - practical user workflow, not just kernel speed

The user-visible problem is not only tokens/second. On machines where model
load is 30-50 seconds, short prompts feel slow even when decode is
unchanged. Round 4 added `alpaccaroo run <model> --connect` so one-shot CLI
calls can stream through an already-running `alpaccaroo serve`.

Deliverables:
- Start `alpaccaroo serve` with one real installed model.
- Run `alpaccaroo run <model> --connect "short prompt"` against it.
- Record whether it avoids reload, streams correctly, and fails back to
  local loading when no server answers.
- If the workflow is confusing, fix the smallest code or documentation
  surface that makes the intended resident workflow obvious, then test the
  fallback path.

Acceptance criteria:
- The one-shot CLI remains unchanged unless `--connect` is requested.
- Any reported speedup separates load avoidance from decode throughput.
- The result is written as a user-facing workflow finding in
  `prompts/05-RESULTS.md`.

## Correctness requirements

Unchanged from round 4, and all currently green:

- `python tests/smoke.py` passes; the current suite prints at least 544
  checks on this branch, and the exact count may increase.
- Greedy output is character-identical with the profiler on and off.
- Quantized/dense parity, JSON-only generation, chat templates, model store
  compatibility, and Windows **and** Linux importability all hold.
- Every new kernel gets an exactness check in `tests/smoke.py` that you have
  **shown to fail** under a deliberately mutated kernel. This project's
  standing rule; do not skip the mutation check.

## What not to do

- Do not re-derive a measured negative. `03-RESULTS.md` and
  `04-RESULTS.md` list them with numbers; check both before starting.
- Do not turn the deliverable into "which computer is better." Use hardware
  variation to make Alpaccaroo2 better.
- Do not use a single-shot A/B to change defaults. If Step 0 showed spread
  < 1.3, single shots are only for exploration; decisions still need paired
  end-to-end measurement.
- Do not quote a per-call speedup as a per-token speedup.
- Do not tune defaults to your machine. Defaults must be safe on unknown
  hardware; measured per-machine values belong in the `alpaccaroo tune`
  cache, which is already keyed on CPU, cores, OS and toolchain versions.
- Do not remove fallback paths to chase a benchmark.
- Do not hard-code device names, core counts, or OS branches outside
  `_platform.py`.
- Do not push to GitHub branches outside the scoped `Alpaccaroo2` work,
  create releases, change repository settings, or edit unrelated remote
  state unless the user explicitly asks for that separate action.

## Recommended order of work

1. Bootstrap, green smoke suite, Step 0 characterisation.
2. Package G - the matrix. It is cheap and it informs everything else.
3. Package J - it needs only a model whose shapes cross the threshold.
4. Package H - if and only if Step 0 says your clocks are stable.
5. Package I - microbenchmark first, implement only on a win.
6. Package K - additive and bounded.
7. Package L - if you have the hardware.
8. Package M - validate the practical resident-server workflow.

## Deliverable

`prompts/05-RESULTS.md`, in the same format as 03 and 04: one line per
avenue - what was tried, what was expected, what was measured, kept or
discarded - written as work happens rather than at the end, opening with
this machine's Step 0 characterisation so every number below it can be
read in context.

If benchmark JSON/CSV/asm artifacts are small enough to review, keep them
under a machine-named subfolder such as `prompts/05-artifacts/<machine>/`
and link them from `prompts/05-RESULTS.md`. If an artifact is too large,
record the command, the summary table, and where the local file lives.

Commit and push to `Alpaccaroo2` or to a branch created from it as work
becomes reviewable. This prompt is the explicit approval for that scoped
push authority.

### If the push fails, read this before spending time on it

Round 4 lost time to credential ambiguity. Separate read access from write
access before debugging anything deeper:

```sh
git ls-remote --heads origin      # succeeds => the credential can READ
git push origin Alpaccaroo2       # 403 => it cannot WRITE this branch
```

- If read works but push is denied, assume the machine is authenticated as
  the wrong GitHub account or lacks write permission. Ask the user to fix
  the GitHub identity or permission; do not work around it by pushing to an
  unrelated branch or account.
- If Git cannot prompt for credentials because the harness has no
  controlling terminal, ask the user to run the push in a real terminal or
  preconfigure a credential with write access to this repository.
- Do not read secrets out of credential stores, paste tokens into logs, mint
  OAuth tokens through unrelated apps, or change the remote URL to embed a
  token. If no safe credential path is available, leave local commits and
  report the exact branch and commit SHA.
