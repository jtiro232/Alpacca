# Alpacca - our own fused quantized-matvec kernels, written in Python.
# MIT License. See LICENSE.
"""Alpacca's native-speed kernels: Python source, optionally JIT-compiled.

Every kernel here is Alpacca's own algorithm, authored and maintained in
this file as ordinary Python. When the OPTIONAL, PINNED Numba JIT is
installed (``pip install alpacca[kernels]``), these functions are compiled
at runtime to native SIMD machine code and the quantized decode path runs
at memory-bandwidth speed while weights stay quantized in RAM. Without
Numba - or with ``ALPACCA_KERNELS=0`` - nothing changes: the NumPy and
pure-Python paths remain the reference implementations and the fallback.

Pin policy: Numba is locked to ``NUMBA_PIN`` below, a combination
validated against this code and the supported NumPy range. The pin is
never updated implicitly; a different installed Numba version deactivates
the kernels (set ``ALPACCA_KERNELS=force`` to override at your own risk).
"""

from __future__ import annotations

import os

NUMBA_PIN = "0.65.1"

# Batch at which the wide (batch-contiguous) matmul kernel overtakes the
# narrow one. Measured on a 14336x4096 Q4_K matrix, milliseconds:
#     batch      1     4     8    16    32    64   128
#     narrow   1.5   4.3  11.3  23.7  39.0  87.1  248.9
#     wide    15.6  23.0  16.9  19.4  23.5  41.6   76.4
# The wide kernel allocates one row accumulator, which dominates when there
# are few columns to amortize it over; the narrow one strides the batch,
# which dominates once there are many. They cross between 8 and 16.
NARROW_BATCH = 8

_state: dict | None = None  # lazy: {"matvec": compiled fn} or {} if inactive


def _init() -> dict:
    global _state
    if _state is not None:
        return _state
    _state = {}
    mode = os.environ.get("ALPACCA_KERNELS", "").strip().lower()
    if mode in ("0", "off", "no") or os.environ.get("ALPACCA_PURE"):
        return _state
    try:
        import numpy as np
        import numba
        from numba import njit, prange
    except Exception:
        return _state
    if numba.__version__ != NUMBA_PIN and mode != "force":
        import sys
        print(f"alpacca: numba {numba.__version__} != pinned {NUMBA_PIN}; "
              f"kernels disabled (ALPACCA_KERNELS=force to override)",
              file=sys.stderr)
        return _state

    @njit(parallel=True, fastmath=True, cache=True)
    def _matvec_codes(q3, d, m, xs, xsums, affine):
        # out[r] = sum_s d[r,s] * (q3[r,s,:] . xs[s,:])  (+ m[r,s]*xsums[s])
        rows, nsub, sub_len = q3.shape
        out = np.empty(rows, np.float32)
        for r in prange(rows):
            acc = np.float32(0.0)
            for s in range(nsub):
                dot = np.float32(0.0)
                qb = q3[r, s]
                xb = xs[s]
                for j in range(sub_len):
                    dot += np.float32(qb[j]) * xb[j]
                acc += d[r, s] * dot
            if affine:
                macc = np.float32(0.0)
                for s in range(nsub):
                    macc += m[r, s] * xsums[s]
                acc += macc
            out[r] = acc
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _matmul_codes(q3, d, m, Xs, Xsums, affine):
        # out[b,r] = sum_s d[r,s]*(q3[r,s,:] . Xs[b,s,:]) (+ m[r,s]*Xsums[b,s])
        #
        # The row loop is outermost so each row's codes are read from RAM
        # once and reused across the whole batch. The alternative - the
        # NumPy path's dequantize-to-float32-then-GEMM - pays a cost
        # proportional to the *weights*, not the batch, which is why a
        # one-token batch used to cost as much as a 256-token one.
        rows, nsub, sub_len = q3.shape
        batch = Xs.shape[0]
        out = np.zeros((batch, rows), np.float32)
        for r in prange(rows):
            for s in range(nsub):
                qb = q3[r, s]
                drs = d[r, s]
                for b in range(batch):
                    xb = Xs[b, s]
                    dot = np.float32(0.0)
                    for j in range(sub_len):
                        dot += np.float32(qb[j]) * xb[j]
                    out[b, r] += drs * dot
            if affine:
                for s in range(nsub):
                    mrs = m[r, s]
                    for b in range(batch):
                        out[b, r] += mrs * Xsums[b, s]
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _matmul_codes_wide(q3, d, m, Xt, XsumsT, affine):
        # Same result as _matmul_codes, laid out for wider batches: Xt is
        # (nsub, sub_len, batch) so the innermost loop sweeps the batch
        # contiguously and vectorizes, where the narrow kernel strides it.
        # The scale is folded into the code value, which keeps the row
        # accumulator the only scratch and matches the tiled path's
        # scale-then-sum order.
        rows, nsub, sub_len = q3.shape
        batch = Xt.shape[2]
        out = np.empty((rows, batch), np.float32)
        for r in prange(rows):
            acc = np.zeros(batch, np.float32)
            for s in range(nsub):
                qb = q3[r, s]
                drs = d[r, s]
                for j in range(sub_len):
                    qv = drs * np.float32(qb[j])
                    xj = Xt[s, j]
                    for b in range(batch):
                        acc[b] += qv * xj[b]
                if affine:
                    mrs = m[r, s]
                    xsb = XsumsT[s]
                    for b in range(batch):
                        acc[b] += mrs * xsb[b]
            for b in range(batch):
                out[r, b] = acc[b]
        return out

    _state = {"np": np, "matvec": _matvec_codes, "matmul": _matmul_codes,
              "matmul_wide": _matmul_codes_wide,
              "numba_version": numba.__version__}
    return _state


def available() -> bool:
    """True when the pinned JIT is importable and kernels are enabled."""
    return bool(_init())


def status() -> str:
    st = _init()
    if st:
        return (f"alpacca-kernels active (numba=={st['numba_version']}, "
                f"pin {NUMBA_PIN}, our Python source)")
    return "alpacca-kernels inactive (pure/NumPy paths in use)"


def matvec_codes(q3, d_eff, m_eff, x):
    """Fused quantized matvec over int8 codes + per-sub-block scales.

    q3: int8 (rows, n_sub, sub_len) C-contiguous, element order.
    d_eff/m_eff: float32 (rows, n_sub); m_eff may be None.
    x: float32 (cols,). Returns float32 (rows,).
    """
    st = _init()
    np = st["np"]
    rows, nsub, sub_len = q3.shape
    xs = np.ascontiguousarray(x, dtype=np.float32).reshape(nsub, sub_len)
    if m_eff is None:
        xsums = xs[:1, :1].reshape(1)  # unused dummy
        return st["matvec"](q3, d_eff, d_eff, xs, xsums, False)
    xsums = xs.sum(axis=1)
    return st["matvec"](q3, d_eff, m_eff, xs, xsums, True)


def matmul_codes(q3, d_eff, m_eff, X):
    """Fused quantized matmul: X (batch, cols) @ self.T -> (batch, rows).

    Same layout contract as :func:`matvec_codes`, with X float32
    (batch, cols). Weights are streamed once for the whole batch.
    """
    st = _init()
    np = st["np"]
    rows, nsub, sub_len = q3.shape
    batch = X.shape[0]
    Xs = np.ascontiguousarray(X, dtype=np.float32).reshape(batch, nsub, sub_len)
    affine = m_eff is not None
    m_arg = m_eff if affine else d_eff  # unused when not affine
    if batch <= NARROW_BATCH:
        # few columns: keep the batch in the outer loop, so no per-row
        # scratch buffer is allocated and one-token batches stay ~1.5 ms
        xsums = Xs.sum(axis=2) if affine else Xs[:, :1, 0]
        return st["matmul"](q3, d_eff, m_arg, Xs, xsums, affine)
    Xt = np.ascontiguousarray(Xs.transpose(1, 2, 0))
    XsumsT = (np.ascontiguousarray(Xs.sum(axis=2).T) if affine
              else Xt[:, :1, 0])
    wide = st["matmul_wide"](q3, d_eff, m_arg, Xt, XsumsT, affine)
    # the wide kernel accumulates into (rows, batch) so its innermost loop
    # stays contiguous; every other path returns a C-contiguous (batch, rows),
    # so pay the transpose rather than hand back a view that behaves
    # differently. Measured at 1.4% of the matmul it follows.
    return np.ascontiguousarray(wide.T)


def warmup() -> None:
    """Trigger JIT compilation once (cached on disk afterwards)."""
    st = _init()
    if not st:
        return
    np = st["np"]
    q = np.zeros((2, 1, 32), dtype=np.int8)
    d = np.zeros((2, 1), dtype=np.float32)
    matvec_codes(q, d, None, np.zeros(32, dtype=np.float32))
    matvec_codes(q, d, d, np.zeros(32, dtype=np.float32))
    for b in (2, NARROW_BATCH + 1):          # both matmul kernels
        X = np.zeros((b, 32), dtype=np.float32)
        matmul_codes(q, d, None, X)
        matmul_codes(q, d, d, X)
