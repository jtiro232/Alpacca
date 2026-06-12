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

    _state = {"np": np, "matvec": _matvec_codes}
    return _state


def available() -> bool:
    """True when the pinned JIT is importable and kernels are enabled."""
    return bool(_init())


def status() -> str:
    if available():
        return f"alpacca-kernels active (numba=={NUMBA_PIN}, our Python source)"
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
