# Alpacca optional native kernel: fused quantized GEMV, opt-in, user-compiled

Follow-up to `prompts/01-performance-overhaul.md`. **Verify before starting** that the
overhaul has landed: `alpacca/qmatrix.py` (QuantMatrix tiled dequant-matvec),
`tests/bench.py`, batched prefill, KV prefix reuse, and the `ALPACCA_F32=1` escape
hatch. If any of those are missing, stop and report instead of improvising.

## Why this exists

The NumPy tiled path still materializes f32 tiles and pays NumPy dispatch per tile, so
decode runs ~1.5–2x slower than llama.cpp on the same machine. The remaining gap lives
in exactly one operation: the quantized matrix–vector product. llama.cpp wins by fusing
dequantization into the dot product inside SIMD registers, never writing f32 weights to
memory. This prompt adds that one kernel — and **deliberately lifts prompt 01's
"no native code" restriction, and only that restriction**.

Because this crosses the project's "100% Python" line, it must be: **opt-in, off by
default, compiled by the user from a single auditable C file in the repo, never
shipped as a binary, never fetched from a package index.** Without it, Alpacca must
behave exactly as it does today.

## Hard constraints

1. **No new Python dependencies** (no cffi, no Numba). ctypes + stdlib only.
2. **No committed binaries, no install-time or import-time compilation.** One portable
   C11 source file, `alpacca/quantgemv.c`, target ≤ ~300 lines, libc only. The user
   compiles it explicitly via `scripts/build_kernel.sh` (cc/clang) or
   `scripts/build_kernel.ps1` (MSVC), matching the style of the existing install
   scripts. Output goes to `$ALPACCA_HOME/lib/` (default `~/.alpacca/lib/`), not the
   repo.
3. **Auditability over peak speed**: portable scalar C that the compiler
   auto-vectorizes (`-O3 -march=native` / `/O2`). NO hand-written intrinsics, NO
   inline assembly, NO OpenMP. The C must mirror the arithmetic of the pure-Python
   decoders in `alpacca/quants.py` exactly.
4. **Default behavior unchanged**: with no library built (or `ALPACCA_NATIVE=0`),
   every code path, test, and output is identical to before this change. Loader
   failures must never crash — fall back to the NumPy path with one stderr note.
5. House style, MIT headers, cross-platform CI green, honest README claims.

## Specification

### C kernel (`alpacca/quantgemv.c`)

One function per format, uniform signature, operating on a half-open row range so
Python can parallelize:

```c
int  quantgemv_abi(void);  /* returns QUANTGEMV_ABI, bump on any signature change */
void gemv_q8_0(const uint8_t *blocks, const float *x, float *y,
               int32_t cols, int32_t r0, int32_t r1);
void gemv_q4_0(...);  void gemv_q4_k(...);  void gemv_q6_k(...);
```

- Formats: **Q8_0, Q4_0** (tiny/test models, stories15M) and **Q4_K, Q6_K** (covers
  real Q4_K_M models). Nothing else.
- `blocks` is the raw GGUF block bytes for the whole matrix (row stride =
  `cols / block_elems * block_bytes`); each row's result is the sum over its blocks of
  `scale * dot(int_quants, x_slice)` — decode scales/quants in registers, accumulate
  in a float (or double) register, write only `y[r]`. Never materialize a dequantized
  row.
- Port the decode arithmetic from the pure decoders in `quants.py` (`_deq_q8_0`,
  `_deq_q4_0`, `_deq_q4_k`, `_deq_q6_k` and `_scale_min_k4`); keep variable names
  recognizably parallel so the two can be audited side by side. f16 scales: decode
  with a small portable half→float helper (no `_Float16` dependence).

### Python loader (`alpacca/native.py`)

- ctypes; search `$ALPACCA_HOME/lib/` for `libquantgemv.so` / `libquantgemv.dylib` /
  `quantgemv.dll`. Honor `ALPACCA_NATIVE=0` (and `ALPACCA_PURE=1` implies disabled).
- Check `quantgemv_abi()` against the expected constant; on mismatch, warn once on
  stderr and ignore the library.
- Expose `available() -> bool` and `supported_formats() -> set[str]`. Import must
  never raise.
- `alpacca doctor` reports kernel status (found/ABI/formats or "not built").

### Integration (`alpacca/qmatrix.py`)

- `QuantMatrix.matvec` dispatch order: native kernel (if available and format
  supported) → existing NumPy tiled path. The raw block bytes QuantMatrix already
  holds are passed to C directly (ensure C-contiguous uint8 buffer).
- **Threading**: ctypes releases the GIL during the call, so split `[0, rows)` into
  contiguous chunks across a small persistent `concurrent.futures.ThreadPoolExecutor`
  (`min(os.cpu_count() or 1, 8)` workers, env override `ALPACCA_THREADS`); call the
  kernel once per chunk with disjoint `[r0, r1)`. Single call when rows are small
  (e.g. < 4 * workers * 64).
- `matmul_t` (batched prefill) **stays on NumPy GEMM** — do not port it to C; BLAS is
  already compute-efficient there and prefill is not the bottleneck.

## Tests (extend `tests/smoke.py`; skip cleanly and visibly when no kernel is built)

- Per-format parity: random valid quantized matrices (reuse prompt 01's block
  generation helpers); native `matvec` vs NumPy tiled `matvec`, max-abs diff < 1e-3;
  threaded result identical to single-call result.
- End-to-end: tiny Q8_0 and Q4_0 models — greedy argmax token sequences identical
  with the kernel enabled vs `ALPACCA_NATIVE=0` over a short generation.
- ABI guard: a deliberately wrong-ABI stub is rejected with a warning, not a crash.
- CI: add a job per OS in `.github/workflows/build.yml` that runs the build script,
  then the kernel parity tests. `tests/real_model_test.py` must pass with the kernel
  enabled.

## Performance gates (via `tests/bench.py`, same machine, same commit)

- Decode tok/s with kernel ≥ **1.5x** the NumPy tiled path on a quantized real model
  (1B-class if available, else stories15M-q4_0), at unchanged peak RSS.
- Report absolute numbers and thread-count scaling (1, 4, 8) in the commit message.
- If llama.cpp/Ollama happens to be installed, include a side-by-side as
  information only — never as a gate.

## Documentation

README gains an "Optional native kernel" section: what it is (~300 lines of C you
compile yourself), why it is opt-in (preserves the no-binaries, no-package-index
supply-chain story), build commands for all three OSes, `ALPACCA_NATIVE` /
`ALPACCA_THREADS`, and measured before/after numbers. Update the roadmap ("small
native kernels exposed through Python" → done). Keep the honest framing: this is
llama.cpp's *technique*, credited accordingly in THIRD-PARTY-NOTICES if any layout
documentation was consulted.

## Git protocol

- Work on the branch this session designates; create it if needed.
- Commits: `perf: native quantized GEMV kernel (opt-in)` style, measured numbers in
  the body. Run `python3 tests/smoke.py` (with and without the kernel built) before
  each commit. Push with `git push -u origin <branch>`. No PR unless asked.

## Mandatory closure protocol

Before ending the session, read `C:\Users\Jon\Desktop\MANDATORY-CLOSURE-PROTOCOL.md`
and execute every step it contains. If the session is not running on the machine
where that path exists (e.g. a remote Linux container), state explicitly that the
protocol file was not accessible and complete the Git protocol above in full instead.

## Explicitly out of scope

GPU; SIMD intrinsics or assembly; OpenMP or pthreads inside the C file; porting
prefill/GEMM to C; additional quant formats; Numba/cffi/Cython; changing defaults;
server or sampler changes; shipping prebuilt binaries or auto-building on import.

## Definition of done

- [ ] `quantgemv.c` + build scripts + loader landed; default behavior bit-identical
      without the kernel
- [ ] Parity and end-to-end tests green with and without the kernel; CI builds and
      tests it on Linux/macOS/Windows
- [ ] Decode ≥ 1.5x vs NumPy tiled path, measured and recorded
- [ ] README + roadmap updated; work committed and pushed
- [ ] `C:\Users\Jon\Desktop\MANDATORY-CLOSURE-PROTOCOL.md` read and executed (or its
      inaccessibility explicitly reported)
