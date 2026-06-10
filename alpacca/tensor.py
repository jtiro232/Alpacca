# Alpacca - minimal tensor backend: NumPy when available, pure Python
# otherwise (set ALPACCA_PURE=1 to force the pure path).
# MIT License. See LICENSE.
"""The handful of dense operations the transformer needs.

Vectors are NumPy 1-D float32 arrays or Python lists of floats; matrices
are NumPy 2-D arrays or lists of row-lists. The pure path is exact but
slow - it exists so the engine runs with zero dependencies, and so the two
implementations can verify each other in tests.
"""

from __future__ import annotations

import math
import os

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

if os.environ.get("ALPACCA_PURE"):
    _np = None

HAS_NUMPY = _np is not None


def backend_name() -> str:
    return "numpy" if HAS_NUMPY else "pure-python"


# ---- construction --------------------------------------------------------

def vector(values) -> "object":
    if HAS_NUMPY:
        return _np.asarray(values, dtype=_np.float32)
    return list(values)


def matrix(flat, rows: int, cols: int):
    """Build a matrix from flat data in row-major order (row = output)."""
    if HAS_NUMPY:
        return _np.asarray(flat, dtype=_np.float32).reshape(rows, cols)
    if isinstance(flat, list):
        return [flat[r * cols:(r + 1) * cols] for r in range(rows)]
    flat = list(flat)
    return [flat[r * cols:(r + 1) * cols] for r in range(rows)]


def zeros(n: int):
    if HAS_NUMPY:
        return _np.zeros(n, dtype=_np.float32)
    return [0.0] * n


def to_list(v) -> list:
    if HAS_NUMPY and isinstance(v, _np.ndarray):
        return v.tolist()
    return list(v)


# ---- core ops -------------------------------------------------------------

def matvec(W, x):
    """W (rows x cols) times x (cols) -> rows."""
    if HAS_NUMPY:
        return W @ x
    return [sum(w * xv for w, xv in zip(row, x)) for row in W]


def matrix_row(W, r: int):
    if HAS_NUMPY:
        return W[r].copy()
    return list(W[r])


def dot(a, b) -> float:
    if HAS_NUMPY:
        return float(a @ b)
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    if HAS_NUMPY:
        return a + b
    return [x + y for x, y in zip(a, b)]


def add_(a, b):
    """In-place a += b (returns a)."""
    if HAS_NUMPY:
        a += b
        return a
    for i, y in enumerate(b):
        a[i] += y
    return a


def mul(a, b):
    if HAS_NUMPY:
        return a * b
    return [x * y for x, y in zip(a, b)]


def scale(a, s: float):
    if HAS_NUMPY:
        return a * s
    return [x * s for x in a]


def silu(x):
    if HAS_NUMPY:
        return x / (1.0 + _np.exp(-x))
    out = [0.0] * len(x)
    for i, v in enumerate(x):
        if v >= 0:
            out[i] = v / (1.0 + math.exp(-v))
        else:  # avoid overflow in exp for very negative values
            e = math.exp(v)
            out[i] = v * e / (1.0 + e)
    return out


def rmsnorm(x, weight, eps: float):
    if HAS_NUMPY:
        inv = 1.0 / _np.sqrt(_np.mean(x.astype(_np.float32) ** 2) + eps)
        return x * inv * weight
    ss = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(ss + eps)
    return [v * inv * w for v, w in zip(x, weight)]


def softmax(x):
    if HAS_NUMPY:
        m = _np.max(x)
        e = _np.exp(x - m)
        return e / _np.sum(e)
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def argmax(x) -> int:
    if HAS_NUMPY:
        return int(_np.argmax(x))
    best, besti = x[0], 0
    for i, v in enumerate(x):
        if v > best:
            best, besti = v, i
    return besti
