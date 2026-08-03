# Alpaccaroo portable performance plan

This plan is for future agents improving Alpaccaroo performance without tailoring the project to one machine. It should work across ThinkPads, Intel/AMD mini PCs, gaming desktops, Windows, Linux, CPU-only systems, and optional GPU systems.

The goal is not to tune around one observed laptop result. The goal is to build a measurement-driven performance architecture that can make large gains where hardware allows, while preserving correctness and useful fallbacks everywhere.

## Current baseline and observed issue

On one Windows ThinkPad-class machine, Alpaccaroo ran a Qwen2.5 3B Q4_K_M model around 2-3 tok/s even after installing the pinned Numba CPU kernel stack. Investigation showed:

- `numba==0.65.1` was installed and detected.
- Alpaccaroo reported `alpaccaroo-kernels active`.
- The Qwen model used native `q4k_int` / `q6k_int` quantized matrix modes.
- Dense weight mode was slower.
- Disabling the integer-dot path was slower.
- Changing thread count did not produce a reliable gain.
- The laptop CPU appeared to sustain low clocks under load.

This is a useful data point, not a design target. Do not hard-code around it.

## Non-negotiable design constraints

- Keep the pure Python / NumPy / Numba / GPU tiering model intact.
- Keep correctness reproducible across backends.
- Do not hard-code device names, ThinkPad assumptions, core counts, or OS-specific heuristics as universal policy.
- Prefer runtime capability detection over machine-specific branches.
- Every optimization must be measurable and disableable.
- Keep environment-variable escape hatches for users and agents.
- Do not make CPU optimizations block future GPU gains.
- Do not make Windows fast at the expense of Linux, or Linux fast at the expense of Windows.
- Avoid clever changes that improve one benchmark while regressing real chat usage.

## Required measurement work before major optimization

Performance work should start by making Alpaccaroo explain where time goes.

Add a structured decode profiler that can report, per run:

- model load time
- prompt render/tokenize time
- prefill time
- per-token decode time
- quantized matvec time
- grouped matvec time
- attention time
- output projection time
- sampler/top-k/top-p/repeat-penalty time
- tokenizer streaming time
- Python orchestration overhead
- selected backend path for each major operation
- thread count and BLAS/Numba thread settings
- CPU/GPU capability summary when available

The current `backend numpy` label is too coarse. It should distinguish at least:

- pure Python
- NumPy dense
- NumPy quantized fallback
- Numba fused f32-scale matvec
- Numba native integer-dot Q4_K/Q5_K/Q6_K
- CUDA/GPU path

The profiler should be available from the CLI, for example:

```powershell
alpaccaroo run qwen3bmed --profile
```

and from benchmarks, for example:

```powershell
python tests/bench.py --model qwen3bmed --profile-json perf.json
```

## Benchmark matrix

Create a benchmark suite that separates cold start, warm start, prefill, and steady decode.

Minimum benchmark dimensions:

- OS: Windows, Linux
- CPU class:
  - laptop 4-core/8-thread Intel
  - mini PC Intel/AMD
  - desktop/gaming CPU
  - server/workstation CPU when available
- GPU:
  - CPU only
  - NVIDIA CUDA where available
- model sizes:
  - ~1B
  - ~3B
  - ~8B
- quantization:
  - Q4_K_M / Q4_K
  - Q5_K_M / Q5_K if available
  - Q6_K if available
  - F16/F32 dense where practical
- prompt shapes:
  - short prompt, short answer
  - short prompt, long answer
  - long prompt, short answer
  - long prompt, long answer
- context:
  - 2048
  - 4096
  - 8192
  - larger contexts only where memory allows

Each run should record:

- tok/s prefill
- tok/s decode
- p50/p95 token latency
- model load seconds
- peak RSS where supported
- selected execution path
- CPU thread count
- BLAS thread count if knowable
- package versions: Python, NumPy, Numba, llvmlite
- OS and architecture

Do not compare cold one-shot CLI runs to warm resident-server runs as if they are the same thing.

## Optimization tracks

### Track 1: output projection and sampling fusion

Large-vocabulary models such as Qwen have expensive output projection and logits scanning. Qwen2.5 3B has a vocabulary around 151k tokens. Computing a full logits vector and then scanning it separately may waste time.

Investigate fused paths for common sampling cases:

- greedy argmax directly during output projection
- top-k directly during output projection
- top-p after a smaller candidate set where mathematically valid
- repeat penalty applied only to affected tokens before final candidate selection

Guardrails:

- Preserve exact behavior for existing sampler modes unless an approximation is explicitly opted in.
- Keep a fallback full-logits path.
- Make the optimized path conditional on sampler settings.
- Add tests comparing selected token IDs against the baseline for deterministic cases.

Potential benefit:

- General across devices.
- Especially relevant for Qwen-family and other large-vocabulary models.

### Track 2: grouped native matvec kernels

Alpaccaroo already shares activation quantization across multiple native quantized matrices in `matvec_group`. However, it still calls separate matrix kernels.

Investigate kernels that process multiple matrices sharing the same input vector in one dispatch.

Candidate groups:

- attention q/k/v or fused qk/v
- FFN gate/up or fused gate-up/down patterns
- output projection with sampler fusion

Guardrails:

- Do not remove current individual-matrix kernels.
- Keep grouped kernels shape-gated and benchmark-gated.
- Avoid layout choices that only help one architecture.
- Keep CPU and GPU dispatch independent.

Potential benefit:

- General across CPU systems.
- Helps reduce Python dispatch and thread-pool overhead.
- More important on laptops and small CPUs, but not unique to them.

### Track 3: shape-specific kernels for narrow GQA matrices

Some models have narrow grouped-query attention matrices. For example, Qwen2.5 3B has KV dimensions much smaller than the main FFN matrices.

Current broad parallel kernels may be too heavy for narrow matrices.

Investigate:

- low-overhead single-thread kernels for small row counts
- two-thread or chunked kernels for narrow GQA projections
- shape-based dispatch thresholds measured per backend
- avoiding expensive thread-pool wakeups for small work items

Guardrails:

- Thresholds must be measured and overrideable.
- Do not assume Qwen-specific dimensions.
- Avoid fragile branch explosions; use a small dispatch table keyed by dtype, rows, cols, and hardware capabilities.

Potential benefit:

- General model-shape optimization.
- More visible on laptop CPUs, but useful anywhere small matrices are frequent.

### Track 4: CPU feature-aware integer-dot improvements

The native Q4_K/Q5_K/Q6_K integer-dot path is active, but it may not generate optimal instructions on every CPU.

Investigate:

- generated LLVM/assembly for Numba kernels
- AVX2 behavior
- AVX-VNNI / VNNI availability where present
- AMD Zen behavior
- Windows vs Linux codegen differences
- memory alignment and cache-line behavior
- scale/min conversion overhead
- activation quantization overhead

Guardrails:

- Feature detection must happen at runtime or install/runtime capability boundaries.
- Never assume AVX-VNNI.
- Keep portable kernels as fallback.
- Do not introduce native compiled source unless that is an explicit project decision.

Potential benefit:

- Large on newer Intel/AMD CPUs if codegen can exploit available instructions.
- Not ThinkPad-specific.

### Track 5: thread scheduling and autotuning

The current physical-core default is reasonable, but no single thread count is best for all devices or matrix shapes.

Implement a small, bounded autotuner:

- run tiny warm benchmark on first use or explicit command
- test a small set of thread counts: `1`, physical cores / 2, physical cores, logical cores
- cache result per machine + Python + NumPy + Numba + model-shape class
- keep `ALPACCAROO_THREADS` as an override
- keep startup cost bounded

Also investigate BLAS/Numba thread-pool coordination:

- when dense NumPy paths are active, BLAS threads matter
- when Numba quantized paths are active, BLAS thread pools may hurt if they wake unnecessarily
- avoid setting global BLAS thread limits blindly without measurement

Guardrails:

- Autotune must be opt-in first.
- Cached tuning must be invalidated when dependencies or hardware change.
- Do not run long benchmarks during normal CLI startup.

Potential benefit:

- Broadly useful across laptops, mini PCs, and desktops.
- Prevents project defaults from being trapped by one machine class.

### Track 6: resident model workflow

This does not improve steady decode tok/s, but it improves real user experience.

One-shot CLI use repeatedly pays:

- process startup
- imports
- model load
- possible JIT warmup/cache load
- prompt setup

Improve and document resident workflows:

- `alpaccaroo serve`
- interactive `alpaccaroo run`
- optional local daemon
- fast reconnect from CLI to local daemon

Guardrails:

- Do not make daemon use mandatory.
- Keep simple one-shot CLI working.
- Keep model unload controls.

Potential benefit:

- Strong perceived-speed improvement on all systems.
- Especially useful for large models and slower disks.

### Track 7: GPU path isolation and future GPU gains

CPU optimizations must not block GPU work.

Maintain a clean dispatch model:

- CPU dense
- CPU quantized fallback
- CPU Numba native
- GPU CUDA

Do not entangle CPU-specific matrix layout changes with GPU assumptions. If a layout is changed for CPU, provide adapters or parallel storage only when memory allows and measurement justifies it.

GPU-specific future work should be allowed to pursue:

- resident GPU weights
- fused decode chain
- KV cache GPU residency
- batched prefill/decode improvements
- mixed precision where correct

Guardrails:

- CPU-only systems must remain first-class.
- GPU systems must not pay extra CPU layout cost unless needed.
- GPU dispatch should remain explicit and inspectable.

## Agent work packages

Future agents can take these as independent tasks.

### Package A: profiler and path reporting

Deliverables:

- CLI `--profile` option
- benchmark JSON output
- detailed backend/path labels
- tests that profile mode does not change generated output

Acceptance criteria:

- A run identifies where decode time is spent.
- A user can tell whether native quantized kernels are actually used.
- The old coarse `backend numpy` ambiguity is resolved.

### Package B: benchmark harness

Deliverables:

- repeatable benchmark command
- warm/cold separation
- model reference support including nicknames
- JSON/CSV output
- Windows and Linux compatibility

Acceptance criteria:

- Benchmarks do not accidentally pull remote models when a nickname exists locally.
- Results include enough environment data to compare devices.
- Benchmarks can be run in CI with tiny fixtures and locally with real models.

### Package C: output projection + sampler fusion prototype

Deliverables:

- optimized greedy path
- optional top-k path if practical
- correctness tests against baseline
- benchmark report on 1B/3B/8B models

Acceptance criteria:

- No behavioral regression for deterministic greedy tests.
- Clear fallback to existing full-logits path.
- Demonstrated improvement on at least one large-vocab model without regression on small-vocab models.

### Package D: grouped native matvec prototype

Deliverables:

- grouped Q4_K/Q6_K native kernel prototype
- dispatch integration behind a feature flag
- correctness tests against current per-matrix path
- benchmark report across different model shapes

Acceptance criteria:

- Bit-identical or tolerance-equivalent output where expected.
- Feature flag can disable it.
- No regression on models where grouping is not beneficial.

### Package E: narrow-matrix dispatch thresholds

Deliverables:

- small-matrix benchmark
- threshold table or measured heuristic
- environment override
- tests for dispatch correctness

Acceptance criteria:

- Narrow GQA matrices do not pay excessive thread overhead.
- Thresholds are data-backed, not device-name-backed.
- The default remains safe on unknown hardware.

### Package F: autotuning experiment

Deliverables:

- opt-in `alpaccaroo tune` or `ALPACCAROO_AUTOTUNE=1`
- cached tuning result
- invalidation key
- clear output explaining selected settings

Acceptance criteria:

- No long implicit startup delay by default.
- User override always wins.
- Tuning improves or matches default on tested systems.

## Correctness and regression requirements

Every performance change must preserve:

- existing smoke tests
- quantized vs dense parity where currently expected
- deterministic greedy output tests
- JSON-only generation guarantees
- chat-template behavior
- model store compatibility
- Windows and Linux importability

Add performance-specific tests where practical:

- optimized path selects the same greedy token as baseline
- grouped matvec output matches individual matvec output
- dispatch flags select expected paths
- fallback works when Numba is absent
- fallback works when NumPy is absent

## Documentation requirements

Update docs when performance behavior changes:

- explain backend labels
- explain when kernels are active
- explain dense budget tradeoffs
- explain thread controls
- explain resident-server workflow
- document benchmark commands
- document profiler output fields

Avoid over-promising performance. Report measured numbers with hardware, OS, model, quantization, context, and dependency versions.

## What not to do

Do not:

- hard-code ThinkPad-specific behavior
- assume 4 physical cores
- assume Windows or Linux is faster
- assume dense weights are faster
- assume quantized weights are always faster
- assume one thread count fits all systems
- hide backend decisions behind vague labels
- remove fallback paths to chase one benchmark
- make one-shot CLI the only optimized workflow
- merge CPU layout changes that block GPU-resident layouts later

## Recommended order of work

1. Add profiler and clear path reporting.
2. Fix benchmark model/nickname resolution and JSON output.
3. Benchmark current code across representative systems.
4. Prototype output projection + greedy sampler fusion.
5. Prototype grouped native matvec.
6. Add narrow-matrix dispatch thresholds.
7. Add opt-in autotuning.
8. Revisit CPU feature-specific kernels with measured evidence.
9. Expand resident model workflow if user experience remains dominated by load/startup.

The project should stay broad: portable first, measurable always, specialized only behind capability detection and safe fallbacks.
