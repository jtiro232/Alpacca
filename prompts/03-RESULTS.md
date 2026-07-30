# Results log: closing the decode gap (prompts/03-close-the-ollama-gap.md)

One line per avenue: what was tried, what was expected, what was measured,
kept or discarded. Written as work happens, not at the end. Numbers are
end-to-end 8B Q4_K_M decode unless marked "micro".

## FINAL SUMMARY (2026-07-30)

8B Q4_K_M, single stream, CPU only, warm JIT, model loaded, same machine,
same day:

|  | Ollama (llama.cpp) | Alpacca before | Alpacca after |
|---|---:|---:|---:|
| Decode | 11.91-11.95 tok/s | 5.07-5.20 tok/s | **9.1-9.9 tok/s** |
| Prefill 911 tok | 31.5 tok/s | 5.9 (same-day) | 7.2-8.0 tok/s |
| Weight RAM | ~4.6 GiB | 9.35 GiB | **5.03 GiB** (RSS 5.78) |
| Load time | - | 14.3 s | 7.8-9.4 s |

Alpacca decode: 101.6/101.8/101.8 ms/token in quiet windows (9.83-9.85
tok/s), 104.8-109.4 ms in windows with background load; final pairing in
one session: Alpacca 9.14/9.40/9.54 vs Ollama 11.91. Decode gap closed
from 2.3x to 1.21-1.30x. Every avenue below; commits on
perf/fused-quantized-matmul. Both suites green throughout, counts
402->403 numpy / 305 pure / 403->423 kernels; CI green.

| avenue | expected | measured | verdict |
|---|---|---|---|
| 6 threads not 12 (6.1) | +9% | +16% (5.18 vs 4.45) | KEPT, auto physical cores |
| threading layer (6.1) | ? | default==omp best; workqueue -19%; tbb n/a | default KEPT |
| ceiling probe (6.1) | maybe >59 GB/s | 58.7-60.1 flat at 4-12 thr | wall is REAL |
| THP (6.2) | few % | unmaterializable from userspace; streaming already at wall | CLOSED, ~0 |
| non-matvec 20% (6.3) | 1.25x cap | it was launch-bound f32 kernels; now 6.5 ms/token total | mooted by int path |
| QK+gate/up fusion (6.3) | 1-2 ms | 0.0 ms here, bit-identical logits | KEPT (structure) |
| head shortcut (6.3) | - | sampler needs full logits, no exact shortcut | REJECTED |
| f16 KV (6.4) | <1 ms at short ctx | analysis only; needs u16+LUT attention kernel | DEFERRED |
| f16 scales (6.5) | +11% | +11% micro, then subsumed by native format | SUBSUMED |
| VNNI (6.7) | decides everything | vpdpwssd EMITTED once int64 promotion defeated | THE key result |
| packed Q4_K + int8 acts (7 revisited) | "a wash" per prior pass | 2.42x micro, wall-bound | KEPT (0.578 B/w) |
| Q6_K int kernel | memory-bound | 33.8 -> 52.2 Gw/s (94% of wall) after j-outer | KEPT (1.066 B/w) |
| Q6_K 6-bit packing (0.82 B/w) | 1.3x for Q6_K share | ALU headroom insufficient (52 < 72 Gw/s needed) | NOT PURSUED, future |
| prefill dequant tiles | fix regression | numpy unpack 2x slower than JIT kernel; pool ping-pong; page faults | FIXED, net +1.2-1.35x |

### Verdict on the central question

Can pure Python reach llama.cpp's decode throughput on this hardware?
**Within ~20%: yes, measured. Exact parity: no, and the remaining gap is
now bytes-per-weight and per-token fixed cost, not kernel quality.**

The binding constraints, quantified:
1. DRAM wall 59-60 GB/s (measured four ways, flat across thread counts).
2. Alpacca streams 0.578 B/w for Q4_K vs llama.cpp's 0.5625 (+2.7%), and
   1.066 B/w for Q6_K vs 0.82 (+30% on 19% of weights) - worth ~6 ms of
   the 17 ms/token gap. Closing it needs the 6-bit code packing whose
   unpack currently costs more ALU than the bandwidth it saves (52 Gw/s
   measured vs 72 needed); a smarter packed layout might close it and is
   the highest-value remaining lead.
3. ~8-9 ms/token of non-weight work (attention, rope, norms, sampling,
   activation quantization, Python dispatch) that llama.cpp's C runtime
   does in ~1-2 ms. Fusing these into JIT kernels is possible in
   principle; each is small and the sum matters at this speed.
Evidence that would change the answer: a Q6_K (and Q4_K sc/mn) packed
layout whose in-kernel unpack stays under ~15 ops/32 weights, or a Numba
release whose f16 support removes the LUT detour.

What made the difference: Numba promotes scalar integer arithmetic to
int64 (documented semantics), which had compiled every prior integer-dot
experiment to 8-lane vpmuldq - the prior pass's "VNNI never emitted" and
"4-bit packing is a wash" conclusions were both artifacts of that. One
idiom (`acc = np.int32(acc + i32*i32)`) unlocked vpdpwssd at 220 Gw/s
flat, and everything else followed from spending the recovered ALU on
narrower storage.

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

### 6.3 fusion + prefill tiled path (2026-07-30)

- attn_q+attn_k and ffn_gate+ffn_up fused by raw-byte concat (row-major
  blocks make this valid): launches 226 -> 162, logits bit-identical
  (new smoke check, guards fusion engaged). Decode unchanged on this
  machine (101.8 vs 101.6 ms/token) - omp fork/join was already cheap.
  KEPT for structure; ALPACCA_FUSE=0 reverts.
- Output-head shortcut (6.3c): REJECTED - sampler needs full logits
  (global top-k, repeat penalty on arbitrary ids); no exact-equivalent
  shortcut exists.
- Prefill scare, resolved: native storage first made the batch>64 tiled
  path unpack via NumPy (911-tok prefill measured 6.9 tok/s vs the
  brief's 10). Fixes: JIT dequant-tile kernels (2.1x the NumPy unpack),
  32 MB tiles (4 MB tiles ping-pong the numba and BLAS pools: 1077 vs
  271 ms for the same matmul), reused tile buffer (fresh 32 MB allocs
  fault their pages every call). Final same-day pairing: NEW 7.2-8.0
  tok/s vs OLD-PATH 5.9 tok/s - the brief's "10 tok/s" does not
  reproduce today (background load); same-day, prefill IMPROVED
  1.2-1.35x. GEMM itself is at the 240 GFLOP/s BLAS floor (17.1s of a
  29s chunk); int8 GEMM a la llama.cpp remains the only route past it.
  Prefill runs are 2-minute windows and jitter +-15% here; one
  contended run showed 3.0 tok/s prefill / 118 ms decode, both noise.

### 6.4 f16 KV cache: DEFERRED with rationale

At the benchmark's context (<=512) attention costs 1.17 ms/token; halving
KV traffic buys <1ms now. It matters at long context (ctx 4096: ~1 GB/token
of KV reads), but numba 0.65 cannot take f16 arrays, so the route is a
u16-bits + LUT attention kernel - real work, low benchmark impact. Future.

