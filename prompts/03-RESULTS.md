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

### Integration results (2026-07-30, commits e728089 + Q6_K follow-up)

End-to-end 8B Q4_K_M decode, warm JIT, ALPACCA_DENSE_WEIGHT_MB=0:
  5.18 tok/s (best pre-existing) -> 8.42 (int-dot v1) -> 9.85 tok/s after
  the Q6_K j-outer kernel (101.6 ms/token). Ollama: 11.95.
Load time 14.3s -> 7.8s (native unpack is lighter than int8+f32 expand).
Numerics: greedy decode 48 tokens, int-dot vs exact-f32 path: 48/48
identical tokens (twice, incl. after the Q6_K kernel change), coherent
text, min top-1 margin 0.112. Kernel-vs-integer-simulation exact (<=1e-5
gate in smoke.py, mutation-tested). Suites: 402 numpy / 305 pure / 422
kernels.
Decode profile after: Q4_K matvecs 67.3ms (at the wall), Q6_K 45.7->~30ms,
quantize_acts 2.2ms, rope 1.45, attention 1.17, rmsnorm 1.04, embed+rest
0.6. The old "20% non-matvec" is now ~6.5ms total.
Q6_K kernel: per-sub-block reduces 33.8 Gw/s -> j-outer all-16-scales
premultiplied, two acc chains 52.2 Gw/s = 94% of the 1.066 B/w wall.
(pair-variant regression 17.2 Gw/s - kept in log per 5.2.)

### 6.1 ceiling probe: the ~59 GB/s wall is REAL (2026-07-30)

Multi-thread disjoint-chunk streaming reads, 1 GiB working set:
4 thr 59.5 | 6 thr 60.1 | 8 thr 59.5 | 12 thr 58.7 GB/s. Flat across
thread counts => hard DRAM limit, not per-thread/prefetcher. Both engines
sit on the same wall.

### 6.2 THP: NEGATIVE - unmaterializable from userspace here, and ~zero prize

- No root; system-wide 'always' untestable (sudo needs a password).
- stdlib mmap.madvise(MADV_HUGEPAGE) on MAP_PRIVATE|MAP_ANONYMOUS: VMA gets
  the hg flag, but fault-in allocates 0 huge pages (AnonHugePages: 0) and
  MADV_COLLAPSE returns EINVAL, despite enabled=[madvise],
  defrag=[madvise], hugepages-2048kB=[inherit] and ~2400 free order-9
  blocks. This kernel (7.0.0-28-generic) will not hand THP to userspace.
- Sizing evidence that makes it moot: streaming on plain 4K pages already
  hits 60.0 GB/s = the DRAM wall, so TLB misses are not limiting
  sequential streams. Expected prize ~0 for decode. CLOSED.
- Python gotcha for the record: mmap.mmap(-1, n) defaults to MAP_SHARED =
  shmem, governed by shmem_enabled=[never] here - madvise silently no-ops.
  MAP_PRIVATE|MAP_ANONYMOUS is required for anon THP.

### 6.5 f16 scales: SUBSUMED by the native int format (better than planned)

Measured standalone (+11%: 1.15ms vs 1.28ms micro) but the shipped Q4_K
format stores the file's own f16 supers + 6-bit int sub-scales at 0.578
B/w, strictly better than the planned 1.125. No separate work remains.

### 6.4 f16 KV cache: DEFERRED with rationale

At the benchmark's context (<=512) attention costs 1.17 ms/token; halving
KV traffic buys <1ms now. It matters at long context (ctx 4096: ~1 GB/token
of KV reads), but numba 0.65 cannot take f16 arrays, so the route is a
u16-bits + LUT attention kernel - real work, low benchmark impact. Future.

