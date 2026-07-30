# Results log: closing the decode gap (prompts/03-close-the-ollama-gap.md)

One line per avenue: what was tried, what was expected, what was measured,
kept or discarded. Written as work happens, not at the end. Numbers are
end-to-end 8B Q4_K_M decode unless marked "micro".

## Baseline reproduction (2026-07-30) — ALL REPRODUCED

- [x] smoke.py system python: 402 checks pass
- [x] smoke.py ALPACCA_PURE=1: 305 checks pass
- [x] smoke.py in numba venv (0.65.1 + numpy 1.26.4): 403 checks pass
- [x] Alpacca 8B decode: 4.41-4.49 tok/s @ 12 threads, 5.18 tok/s @ 6 threads
      (16 tokens, warm JIT, ALPACCA_DENSE_WEIGHT_MB=0, load 14.3s)
- [x] Ollama nemotron:latest (= Llama-3.1-Nemotron-Nano-8B Q4_K_M, 8.03B,
      llama arch): decode 11.95 tok/s, prefill 31.5 tok/s, size_vram=0
      (CPU-only confirmed via /api/ps)
- CPU flags: avx512f/bw/vl/dq + avx512_vnni + avx512_bf16 present (Zen 4).
  VNNI exists in hardware; the §6.7 question is purely LLVM codegen.

## Avenues

| avenue | expected | measured | verdict |
|---|---|---|---|
| NUMBA_NUM_THREADS=6 (§6.1) | +9% | 4.45→5.18 tok/s (+16%) | KEEP |

### §6.7 VNNI — SOLVED at the codegen level (2026-07-30)

llvmlite 0.47 = LLVM 20.1.8, host znver4, avx512vnni present. Scalar int
arithmetic in Numba promotes to int64 (documented Numba semantics), so every
prior integer-dot experiment compiled to 8-lane i64 vpmuldq/vpaddq — THAT is
why "no VNNI was ever emitted", not an LLVM limitation. Accumulating through
an int32 array cell (`acc = np.zeros(1, np.int32); acc[0] += ...`) keeps the
IR in i32 and LLVM 20 forms vpmaddwd → folds into vpdpwssd (VNNI, 32 i16
MACs/zmm-instr). Confirmed emitted + bit-exact for:
  - u8 x s8 flat dot           -> vpdpwssd
  - interleaved-nibble dot     -> vpdpwssd (unpack fused in the same loop!)
  - split-nibble (llama-style) -> vpdpwssd + vpmaddwd
Caveat: a constant 32-trip inner loop gets const-unrolled and the pattern
dies; runtime trip counts (as the real kernels have) are required.
Consequence: §7's "4-bit packing is a wash" was measured against a crippled
dot. Packed-4-bit + int8 activations + VNNI is back on the table as the main
route. Scripts: ~/.claude/jobs/e07cca0c/tmp/vnni_probe{,2}.py

Recast idiom (`acc = np.int32(acc + i32*i32)`) keeps i32 in pure SSA, survives
parallel=True + prange + runtime sub_len (probe3). Flat VNNI dot measures
220.8 Gw/s ALU (vs 104.9 f32 ceiling from §7); per-sub-block boundary work is
the cost driver.

### §6.1 thread/layer sweep (2026-07-30), 8B decode end-to-end

threads: 4 -> 4.32 | 6 -> 5.07-5.20 | 8 -> 4.58 | 12 -> 4.41-4.49 tok/s.
Layers: default==omp (confirmed via numba.threading_layer()); omp ~= default
(5.20 vs 5.07 same-noise), workqueue consistently worse (4.19@6), tbb not
loadable from pip wheel in venv. VERDICT: 6 threads (physical cores), omp.
Worth defaulting to physical-core count in kernels.py with an env knob.

### Packed-Q4_K microbenchmarks, 14336x4096, 6 threads (2026-07-30)

Format: 128B split-nibble codes + sc/mn u8 + d/dmin f16 per 256-block
= 0.578 B/w (native 0.5625). Activations int8 per-256 block (Q8-style, ours),
bsums per 32. Integer inner algebra identical to what llama.cpp's format
implies; implemented from the Q4_K spec. Error decomposition on synthetic
gaussian data: method (activation-quant) 2.5e-2 max/rms; kernel vs exact
integer simulation 6.4e-7 (exact).

| variant (all vpdpwssd) | ms | Gw/s | vs cur 1.28ms |
|---|---:|---:|---:|
| cur int8+f32 (reference)          | 1.28 | 45.7 | 1.00x |
| f16s: int8 codes + f16-LUT scales | 1.15 | 50.9 | 1.11x |
| pk2_inline (8 reduces/blk)        | 0.73 | 80.2 | 1.75x |
| pk3_premul (4 reduces/blk)        | 0.94 | 62.8 | regression |
| pk3_jouter (1 reduce/blk, sc premul via vpmullw) | **0.53** | **110.5** | **2.42x** |

pk3_jouter: j-loop outer over 32 bytes, 4 chunk streams unrolled inline, sc
premultiplied into i16 codes in-register (exact: sc*q <= 945 < 2^15), one
vector reduce per 256 weights. 63.9 GB/s effective = at the memory wall for
0.578 B/w. THIS is the integration candidate.

