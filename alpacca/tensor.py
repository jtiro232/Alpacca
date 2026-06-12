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
import struct
import weakref

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

if os.environ.get("ALPACCA_PURE"):
    _np = None

HAS_NUMPY = _np is not None

_HOT_WEIGHT_ENV = "ALPACCA_HOT_WEIGHT_MB"
_UNPACKED_WEIGHT_ENV = "ALPACCA_UNPACKED_WEIGHT_MB"
_UNPACKED_CACHE_DEFAULT_MB = 2048.0
_HOT_CACHE_LIMIT_BYTES = None
_HOT_CACHE_USED_BYTES = 0
_HOT_CACHE_MATRICES = 0
_HOT_CACHE_OWNERS = weakref.WeakSet()
_UNPACKED_CACHE_LIMIT_BYTES = None
_UNPACKED_CACHE_USED_BYTES = 0
_UNPACKED_CACHE_MATRICES = 0
_UNPACKED_CACHE_OWNERS = weakref.WeakSet()

QK = 32
QK_K = 256
_QUANTIZED_BLOCK_ELEMENTS = {
    "Q8_0": QK,
    "Q4_0": QK,
    "Q4_K": QK_K,
    "Q5_K": QK_K,
    "Q6_K": QK_K,
}
_QUANTIZED_BLOCK_BYTES = {
    "Q8_0": 34,
    "Q4_0": 18,
    "Q4_K": 144,
    "Q5_K": 176,
    "Q6_K": 210,
}
QUANTIZED_MATVEC_DTYPES = frozenset(_QUANTIZED_BLOCK_BYTES)


def backend_name() -> str:
    return "numpy" if HAS_NUMPY else "pure-python"


class QuantizedMatrix:
    """Row-major quantized matrix with owned bytes.

    GGUF tensors are mmap-backed during loading; this type intentionally owns a
    bytes copy so matrices remain valid after the GGUF file is closed.
    """

    __slots__ = ("data", "dtype", "rows", "cols", "block_elements", "block_bytes",
                 "blocks_per_row", "_np_cache", "_dense_cache",
                 "_dense_cache_bytes", "_q6_cache", "_q6_cache_bytes",
                 "__weakref__")

    def __init__(self, data, dtype: str, rows: int, cols: int):
        if dtype not in _QUANTIZED_BLOCK_BYTES:
            raise ValueError(f"quantized matvec does not support {dtype}")
        block_elements = _QUANTIZED_BLOCK_ELEMENTS[dtype]
        if cols % block_elements:
            raise ValueError(
                f"{dtype} matrix columns must be a multiple of {block_elements}")
        self.data = bytes(data)
        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.block_elements = block_elements
        self.block_bytes = _QUANTIZED_BLOCK_BYTES[dtype]
        self.blocks_per_row = cols // block_elements
        expected = rows * self.blocks_per_row * self.block_bytes
        if len(self.data) != expected:
            raise ValueError(
                f"{dtype} matrix has {len(self.data)} bytes, expected {expected}")
        self._np_cache = None
        self._dense_cache = None
        self._dense_cache_bytes = 0
        self._q6_cache = None
        self._q6_cache_bytes = 0

    def __del__(self):  # pragma: no cover - depends on interpreter shutdown
        try:
            if self._dense_cache is not None:
                _release_hot_cache_bytes(self._dense_cache_bytes)
            if self._q6_cache is not None:
                _release_unpacked_cache_bytes(self._q6_cache_bytes)
        except Exception:
            pass

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def __matmul__(self, x):
        return matvec(self, x)

    def matvec(self, x):
        return matvec(self, x)

    def matmul_t(self, X):
        return matmul_t(X, self)

    def row(self, i: int):
        return matrix_row(self, i)

    def rows_at(self, rows):
        return matrix_rows(self, rows)

    def _np_views(self):
        if not HAS_NUMPY:
            return None
        if self._np_cache is None:
            b = _np.frombuffer(self.data, dtype=_np.uint8).reshape(
                self.rows, self.blocks_per_row, self.block_bytes)
            if self.dtype == "Q8_0":
                d = b[:, :, 0:2].copy().view(_np.float16).astype(
                    _np.float32).reshape(self.rows, self.blocks_per_row)
                q = b[:, :, 2:34].view(_np.int8)
                self._np_cache = d, q
            elif self.dtype == "Q4_0":
                d = b[:, :, 0:2].copy().view(_np.float16).astype(
                    _np.float32).reshape(self.rows, self.blocks_per_row)
                q = b[:, :, 2:18]
                self._np_cache = d, q
            elif self.dtype == "Q4_K":
                d = b[:, :, 0:2].copy().view(_np.float16).astype(
                    _np.float32).reshape(self.rows, self.blocks_per_row)
                dmin = b[:, :, 2:4].copy().view(_np.float16).astype(
                    _np.float32).reshape(self.rows, self.blocks_per_row)
                scales = b[:, :, 4:16]
                sc = _np.empty((self.rows, self.blocks_per_row, 8),
                               dtype=_np.float32)
                mn = _np.empty((self.rows, self.blocks_per_row, 8),
                               dtype=_np.float32)
                for j in range(8):
                    if j < 4:
                        sc[:, :, j] = (scales[:, :, j] & 63).astype(_np.float32)
                        mn[:, :, j] = (scales[:, :, j + 4] & 63).astype(_np.float32)
                    else:
                        sc[:, :, j] = (
                            (scales[:, :, j + 4] & 0x0F) |
                            ((scales[:, :, j - 4] >> 6) << 4)
                        ).astype(_np.float32)
                        mn[:, :, j] = (
                            (scales[:, :, j + 4] >> 4) |
                            ((scales[:, :, j] >> 6) << 4)
                        ).astype(_np.float32)
                q = b[:, :, 16:144]
                self._np_cache = d, dmin, sc, mn, q
            elif self.dtype == "Q5_K":
                d = b[:, :, 0:2].copy().view(_np.float16).astype(
                    _np.float32).reshape(self.rows, self.blocks_per_row)
                dmin = b[:, :, 2:4].copy().view(_np.float16).astype(
                    _np.float32).reshape(self.rows, self.blocks_per_row)
                scales = b[:, :, 4:16]
                sc = _np.empty((self.rows, self.blocks_per_row, 8),
                               dtype=_np.float32)
                mn = _np.empty((self.rows, self.blocks_per_row, 8),
                               dtype=_np.float32)
                for j in range(8):
                    if j < 4:
                        sc[:, :, j] = (scales[:, :, j] & 63).astype(_np.float32)
                        mn[:, :, j] = (scales[:, :, j + 4] & 63).astype(_np.float32)
                    else:
                        sc[:, :, j] = (
                            (scales[:, :, j + 4] & 0x0F) |
                            ((scales[:, :, j - 4] >> 6) << 4)
                        ).astype(_np.float32)
                        mn[:, :, j] = (
                            (scales[:, :, j + 4] >> 4) |
                            ((scales[:, :, j] >> 6) << 4)
                        ).astype(_np.float32)
                qh = b[:, :, 16:48]
                ql = b[:, :, 48:176]
                self._np_cache = d, dmin, sc, mn, qh, ql
            else:  # Q6_K
                ql = b[:, :, 0:128]
                qh = b[:, :, 128:192]
                sc = b[:, :, 192:208].view(_np.int8).astype(_np.float32)
                d = b[:, :, 208:210].copy().view(_np.float16).astype(
                    _np.float32).reshape(self.rows, self.blocks_per_row)
                self._np_cache = d, sc, ql, qh
        return self._np_cache

    def _dense_hot_cache(self):
        if not HAS_NUMPY:
            return None
        _sync_hot_cache_budget(_hot_cache_limit_bytes())
        if self._dense_cache is not None:
            return self._dense_cache
        nbytes = self.rows * self.cols * 4
        if not _reserve_hot_cache_bytes(nbytes):
            return None
        try:
            from .quants import dequantize
            dense = _np.asarray(
                dequantize(self.data, self.rows * self.cols, self.dtype),
                dtype=_np.float32,
            ).reshape(self.rows, self.cols)
        except Exception:
            _release_hot_cache_bytes(nbytes)
            raise
        self._dense_cache = dense
        self._dense_cache_bytes = nbytes
        _HOT_CACHE_OWNERS.add(self)
        return dense


    def _q6_unpacked_cache(self):
        if not HAS_NUMPY or self.dtype != "Q6_K":
            return None
        _sync_unpacked_cache_budget(_unpacked_cache_limit_bytes())
        if self._q6_cache is not None:
            return self._q6_cache
        nbytes = self.rows * self.cols
        if not _reserve_unpacked_cache_bytes(nbytes):
            return None
        try:
            _d, _sc, ql, qh = self._np_views()
            q = _np.empty((self.rows, self.blocks_per_row, 16, 16),
                          dtype=_np.int8)
            for half in range(2):
                qlb = ql[:, :, half * 64:(half + 1) * 64]
                qhb = qh[:, :, half * 32:(half + 1) * 32]
                q1 = ((qlb[:, :, :32] & 0x0F) |
                      (((qhb >> 0) & 3) << 4)).astype(_np.int16) - 32
                q2 = ((qlb[:, :, 32:] & 0x0F) |
                      (((qhb >> 2) & 3) << 4)).astype(_np.int16) - 32
                q3 = ((qlb[:, :, :32] >> 4) |
                      (((qhb >> 4) & 3) << 4)).astype(_np.int16) - 32
                q4 = ((qlb[:, :, 32:] >> 4) |
                      (((qhb >> 6) & 3) << 4)).astype(_np.int16) - 32
                base = half * 8
                q[:, :, base + 0, :] = q1[:, :, :16]
                q[:, :, base + 1, :] = q1[:, :, 16:]
                q[:, :, base + 2, :] = q2[:, :, :16]
                q[:, :, base + 3, :] = q2[:, :, 16:]
                q[:, :, base + 4, :] = q3[:, :, :16]
                q[:, :, base + 5, :] = q3[:, :, 16:]
                q[:, :, base + 6, :] = q4[:, :, :16]
                q[:, :, base + 7, :] = q4[:, :, 16:]
            self._q6_cache = q
            self._q6_cache_bytes = q.nbytes
            _UNPACKED_CACHE_OWNERS.add(self)
            return q
        except Exception:
            _release_unpacked_cache_bytes(nbytes)
            raise


def _cache_limit_bytes(env_name: str, default_mb: float = 0.0) -> int:
    if not HAS_NUMPY:
        return 0
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        mb = default_mb
    else:
        try:
            mb = float(raw.strip())
        except ValueError:
            return 0
    if not math.isfinite(mb) or mb <= 0.0:
        return 0
    return int(mb * 1024 * 1024)


def _hot_cache_limit_bytes() -> int:
    return _cache_limit_bytes(_HOT_WEIGHT_ENV)


def _unpacked_cache_limit_bytes() -> int:
    return _cache_limit_bytes(_UNPACKED_WEIGHT_ENV, _UNPACKED_CACHE_DEFAULT_MB)


def _clear_hot_caches() -> None:
    global _HOT_CACHE_USED_BYTES, _HOT_CACHE_MATRICES
    for owner in list(_HOT_CACHE_OWNERS):
        owner._dense_cache = None
        owner._dense_cache_bytes = 0
    try:
        _HOT_CACHE_OWNERS.clear()
    except Exception:
        pass
    _HOT_CACHE_USED_BYTES = 0
    _HOT_CACHE_MATRICES = 0


def _clear_unpacked_caches() -> None:
    global _UNPACKED_CACHE_USED_BYTES, _UNPACKED_CACHE_MATRICES
    for owner in list(_UNPACKED_CACHE_OWNERS):
        owner._q6_cache = None
        owner._q6_cache_bytes = 0
    try:
        _UNPACKED_CACHE_OWNERS.clear()
    except Exception:
        pass
    _UNPACKED_CACHE_USED_BYTES = 0
    _UNPACKED_CACHE_MATRICES = 0


def _sync_hot_cache_budget(limit: int) -> None:
    global _HOT_CACHE_LIMIT_BYTES
    if _HOT_CACHE_LIMIT_BYTES != limit:
        _clear_hot_caches()
        _HOT_CACHE_LIMIT_BYTES = limit


def _sync_unpacked_cache_budget(limit: int) -> None:
    global _UNPACKED_CACHE_LIMIT_BYTES
    if _UNPACKED_CACHE_LIMIT_BYTES != limit:
        _clear_unpacked_caches()
        _UNPACKED_CACHE_LIMIT_BYTES = limit


def _reserve_hot_cache_bytes(nbytes: int) -> bool:
    global _HOT_CACHE_USED_BYTES, _HOT_CACHE_MATRICES
    limit = _hot_cache_limit_bytes()
    _sync_hot_cache_budget(limit)
    if limit <= 0 or nbytes > limit - _HOT_CACHE_USED_BYTES:
        return False
    _HOT_CACHE_USED_BYTES += nbytes
    _HOT_CACHE_MATRICES += 1
    return True


def _release_hot_cache_bytes(nbytes: int) -> None:
    global _HOT_CACHE_USED_BYTES, _HOT_CACHE_MATRICES
    _HOT_CACHE_USED_BYTES = max(0, _HOT_CACHE_USED_BYTES - nbytes)
    _HOT_CACHE_MATRICES = max(0, _HOT_CACHE_MATRICES - 1)


def _reserve_unpacked_cache_bytes(nbytes: int) -> bool:
    global _UNPACKED_CACHE_USED_BYTES, _UNPACKED_CACHE_MATRICES
    limit = _unpacked_cache_limit_bytes()
    _sync_unpacked_cache_budget(limit)
    if limit <= 0 or nbytes > limit - _UNPACKED_CACHE_USED_BYTES:
        return False
    _UNPACKED_CACHE_USED_BYTES += nbytes
    _UNPACKED_CACHE_MATRICES += 1
    return True


def _release_unpacked_cache_bytes(nbytes: int) -> None:
    global _UNPACKED_CACHE_USED_BYTES, _UNPACKED_CACHE_MATRICES
    _UNPACKED_CACHE_USED_BYTES = max(0, _UNPACKED_CACHE_USED_BYTES - nbytes)
    _UNPACKED_CACHE_MATRICES = max(0, _UNPACKED_CACHE_MATRICES - 1)


def _reset_hot_cache_state() -> None:
    global _HOT_CACHE_LIMIT_BYTES, _UNPACKED_CACHE_LIMIT_BYTES
    _clear_hot_caches()
    _clear_unpacked_caches()
    _HOT_CACHE_LIMIT_BYTES = None
    _UNPACKED_CACHE_LIMIT_BYTES = None


def hot_cache_stats() -> dict[str, int]:
    limit = _hot_cache_limit_bytes()
    _sync_hot_cache_budget(limit)
    unpacked_limit = _unpacked_cache_limit_bytes()
    _sync_unpacked_cache_budget(unpacked_limit)
    return {
        "limit_bytes": limit,
        "used_bytes": _HOT_CACHE_USED_BYTES,
        "matrices": _HOT_CACHE_MATRICES,
        "unpacked_limit_bytes": unpacked_limit,
        "unpacked_used_bytes": _UNPACKED_CACHE_USED_BYTES,
        "unpacked_matrices": _UNPACKED_CACHE_MATRICES,
    }


def quantized_matrix(data, dtype: str, rows: int, cols: int) -> QuantizedMatrix:
    return QuantizedMatrix(data, dtype, rows, cols)


def can_quantized_matvec(dtype: str, cols: int) -> bool:
    block_elements = _QUANTIZED_BLOCK_ELEMENTS.get(dtype)
    return block_elements is not None and cols % block_elements == 0


def is_quantized_matrix(W) -> bool:
    return isinstance(W, QuantizedMatrix)


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

def _f16(data: bytes, off: int) -> float:
    return struct.unpack_from("<e", data, off)[0]


def _scale_min_k4(j: int, scales) -> tuple[int, int]:
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
    mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, mn


def _q8_0_matvec_pure(W: QuantizedMatrix, x) -> list[float]:
    out = [0.0] * W.rows
    for r in range(W.rows):
        acc = 0.0
        row_off = r * W.blocks_per_row * W.block_bytes
        for b in range(W.blocks_per_row):
            off = row_off + b * W.block_bytes
            d = _f16(W.data, off)
            base = b * QK
            for j, q in enumerate(struct.unpack_from("<32b", W.data, off + 2)):
                acc += d * q * x[base + j]
        out[r] = acc
    return out


def _q4_0_matvec_pure(W: QuantizedMatrix, x) -> list[float]:
    out = [0.0] * W.rows
    for r in range(W.rows):
        acc = 0.0
        row_off = r * W.blocks_per_row * W.block_bytes
        for b in range(W.blocks_per_row):
            off = row_off + b * W.block_bytes
            d = _f16(W.data, off)
            base = b * QK
            for j in range(16):
                q = W.data[off + 2 + j]
                acc += d * ((q & 0x0F) - 8) * x[base + j]
                acc += d * ((q >> 4) - 8) * x[base + j + 16]
        out[r] = acc
    return out


def _q4_k_matvec_pure(W: QuantizedMatrix, x) -> list[float]:
    out = [0.0] * W.rows
    for r in range(W.rows):
        acc = 0.0
        row_off = r * W.blocks_per_row * W.block_bytes
        for b in range(W.blocks_per_row):
            off = row_off + b * W.block_bytes
            d = _f16(W.data, off)
            dmin = _f16(W.data, off + 2)
            scales = W.data[off + 4:off + 16]
            qoff = off + 16
            base = b * QK_K
            is_ = 0
            for half in range(4):
                sc, mn = _scale_min_k4(is_, scales)
                d1, m1 = d * sc, dmin * mn
                sc, mn = _scale_min_k4(is_ + 1, scales)
                d2, m2 = d * sc, dmin * mn
                packed = qoff + half * 32
                xb = base + half * 64
                for j in range(32):
                    q = W.data[packed + j]
                    acc += (d1 * (q & 0x0F) - m1) * x[xb + j]
                    acc += (d2 * (q >> 4) - m2) * x[xb + j + 32]
                is_ += 2
        out[r] = acc
    return out


def _q5_k_matvec_pure(W: QuantizedMatrix, x) -> list[float]:
    out = [0.0] * W.rows
    for r in range(W.rows):
        acc = 0.0
        row_off = r * W.blocks_per_row * W.block_bytes
        for b in range(W.blocks_per_row):
            off = row_off + b * W.block_bytes
            d = _f16(W.data, off)
            dmin = _f16(W.data, off + 2)
            scales = W.data[off + 4:off + 16]
            qhoff = off + 16
            qloff = off + 48
            base = b * QK_K
            is_ = 0
            u1, u2 = 1, 2
            for half in range(4):
                sc, mn = _scale_min_k4(is_, scales)
                d1, m1 = d * sc, dmin * mn
                sc, mn = _scale_min_k4(is_ + 1, scales)
                d2, m2 = d * sc, dmin * mn
                packed = qloff + half * 32
                xb = base + half * 64
                for j in range(32):
                    q = W.data[packed + j]
                    qh = W.data[qhoff + j]
                    q1 = (q & 0x0F) + (16 if qh & u1 else 0)
                    q2 = (q >> 4) + (16 if qh & u2 else 0)
                    acc += (d1 * q1 - m1) * x[xb + j]
                    acc += (d2 * q2 - m2) * x[xb + j + 32]
                is_ += 2
                u1 <<= 2
                u2 <<= 2
        out[r] = acc
    return out


def _q6_k_matvec_pure(W: QuantizedMatrix, x) -> list[float]:
    out = [0.0] * W.rows
    for r in range(W.rows):
        acc = 0.0
        row_off = r * W.blocks_per_row * W.block_bytes
        for b in range(W.blocks_per_row):
            off = row_off + b * W.block_bytes
            sc = struct.unpack_from("<16b", W.data, off + 192)
            d = _f16(W.data, off + 208)
            base = b * QK_K
            for half in range(2):
                qloff = off + half * 64
                qhoff = off + 128 + half * 32
                soff = half * 8
                xb = base + half * 128
                for j in range(32):
                    is_ = j // 16
                    ql1 = W.data[qloff + j]
                    ql2 = W.data[qloff + j + 32]
                    qh = W.data[qhoff + j]
                    q1 = ((ql1 & 0x0F) | (((qh >> 0) & 3) << 4)) - 32
                    q2 = ((ql2 & 0x0F) | (((qh >> 2) & 3) << 4)) - 32
                    q3 = ((ql1 >> 4) | (((qh >> 4) & 3) << 4)) - 32
                    q4 = ((ql2 >> 4) | (((qh >> 6) & 3) << 4)) - 32
                    acc += d * sc[soff + is_] * q1 * x[xb + j]
                    acc += d * sc[soff + is_ + 2] * q2 * x[xb + j + 32]
                    acc += d * sc[soff + is_ + 4] * q3 * x[xb + j + 64]
                    acc += d * sc[soff + is_ + 6] * q4 * x[xb + j + 96]
        out[r] = acc
    return out


def _quantized_matvec_np(W: QuantizedMatrix, x):
    dense = W._dense_hot_cache()
    if dense is not None:
        return dense @ _np.asarray(x, dtype=_np.float32)
    if W.dtype == "Q6_K" and _unpacked_cache_limit_bytes() > 0:
        return _quantized_matvec_blocks_np(W, x)
    return _quantized_matvec_tiled_np(W, x)


def _quantized_matvec_blocks_np(W: QuantizedMatrix, x):
    xv = _np.asarray(x, dtype=_np.float32)
    out = _np.zeros(W.rows, dtype=_np.float32)
    if W.dtype == "Q4_K":
        d, dmin, sc, mn, q = W._np_views()
        for b in range(W.blocks_per_row):
            xb = xv[b * QK_K:(b + 1) * QK_K]
            for half in range(4):
                j1, j2 = half * 2, half * 2 + 1
                chunk = q[:, b, half * 32:(half + 1) * 32]
                lo = (chunk & 0x0F).astype(_np.float32)
                hi = (chunk >> 4).astype(_np.float32)
                x1 = xb[j1 * 32:(j1 + 1) * 32]
                x2 = xb[j2 * 32:(j2 + 1) * 32]
                out += d[:, b] * sc[:, b, j1] * (lo @ x1)
                out -= dmin[:, b] * mn[:, b, j1] * _np.sum(x1)
                out += d[:, b] * sc[:, b, j2] * (hi @ x2)
                out -= dmin[:, b] * mn[:, b, j2] * _np.sum(x2)
        return out
    if W.dtype == "Q5_K":
        d, dmin, sc, mn, qh, ql = W._np_views()
        for b in range(W.blocks_per_row):
            xb = xv[b * QK_K:(b + 1) * QK_K]
            qhb = qh[:, b, :]
            for half in range(4):
                j1, j2 = half * 2, half * 2 + 1
                chunk = ql[:, b, half * 32:(half + 1) * 32]
                hb1 = ((qhb >> j1) & 1).astype(_np.float32) * 16.0
                hb2 = ((qhb >> j2) & 1).astype(_np.float32) * 16.0
                lo = (chunk & 0x0F).astype(_np.float32) + hb1
                hi = (chunk >> 4).astype(_np.float32) + hb2
                x1 = xb[j1 * 32:(j1 + 1) * 32]
                x2 = xb[j2 * 32:(j2 + 1) * 32]
                out += d[:, b] * sc[:, b, j1] * (lo @ x1)
                out -= dmin[:, b] * mn[:, b, j1] * _np.sum(x1)
                out += d[:, b] * sc[:, b, j2] * (hi @ x2)
                out -= dmin[:, b] * mn[:, b, j2] * _np.sum(x2)
        return out
    if W.dtype == "Q6_K":
        qcache = W._q6_unpacked_cache()
        if qcache is not None:
            d, sc, _ql, _qh = W._np_views()
            xb = xv.reshape(W.blocks_per_row, 16, 16)
            dots = _np.einsum("rbsj,bsj->rbs", qcache, xb, optimize=False)
            return _np.sum(
                d[:, :, None] * sc * dots, axis=(1, 2), dtype=_np.float32)
        d, sc, ql, qh = W._np_views()
        for b in range(W.blocks_per_row):
            xb = xv[b * QK_K:(b + 1) * QK_K]
            for half in range(2):
                qlb = ql[:, b, half * 64:(half + 1) * 64]
                qhb = qh[:, b, half * 32:(half + 1) * 32]
                scb = sc[:, b, half * 8:(half + 1) * 8]
                q1 = ((qlb[:, :32] & 0x0F) |
                      (((qhb >> 0) & 3) << 4)).astype(_np.int16) - 32
                q2 = ((qlb[:, 32:] & 0x0F) |
                      (((qhb >> 2) & 3) << 4)).astype(_np.int16) - 32
                q3 = ((qlb[:, :32] >> 4) |
                      (((qhb >> 4) & 3) << 4)).astype(_np.int16) - 32
                q4 = ((qlb[:, 32:] >> 4) |
                      (((qhb >> 6) & 3) << 4)).astype(_np.int16) - 32
                base = half * 128
                for sub, q in enumerate((q1, q2, q3, q4)):
                    scales = _np.repeat(
                        scb[:, [sub * 2, sub * 2 + 1]], 16, axis=1)
                    xsub = xb[base + sub * 32:base + (sub + 1) * 32]
                    out += d[:, b] * ((q.astype(_np.float32) * scales) @ xsub)
        return out
    d, q = W._np_views()
    for b in range(W.blocks_per_row):
        xb = xv[b * QK:(b + 1) * QK]
        if W.dtype == "Q8_0":
            dot = q[:, b, :].astype(_np.float32) @ xb
        else:
            qb = q[:, b, :]
            lo = ((qb & 0x0F).astype(_np.int8) - 8).astype(_np.float32)
            hi = ((qb >> 4).astype(_np.int8) - 8).astype(_np.float32)
            dot = lo @ xb[:16] + hi @ xb[16:]
        out += d[:, b] * dot
    return out


def _quantized_matvec_tiled_np(W: QuantizedMatrix, x):
    xv = _np.asarray(x, dtype=_np.float32)
    out = _np.empty(W.rows, dtype=_np.float32)
    tile_rows = max(32, (2 << 20) // max(W.cols * 4, 1))
    for r0 in range(0, W.rows, tile_rows):
        r1 = min(W.rows, r0 + tile_rows)
        tile = _quantized_dequant_rows_np(W, r0, r1)
        out[r0:r1] = tile @ xv
    return out


def _quantized_matvec(W: QuantizedMatrix, x):
    if HAS_NUMPY:
        return _quantized_matvec_np(W, x)
    if W.dtype == "Q8_0":
        return _q8_0_matvec_pure(W, x)
    if W.dtype == "Q4_0":
        return _q4_0_matvec_pure(W, x)
    if W.dtype == "Q4_K":
        return _q4_k_matvec_pure(W, x)
    if W.dtype == "Q5_K":
        return _q5_k_matvec_pure(W, x)
    if W.dtype == "Q6_K":
        return _q6_k_matvec_pure(W, x)
    raise ValueError(f"quantized matvec does not support {W.dtype}")


def _quantized_row_pure(W: QuantizedMatrix, r: int) -> list[float]:
    if not 0 <= r < W.rows:
        raise IndexError(r)
    out = [0.0] * W.cols
    row_off = r * W.blocks_per_row * W.block_bytes
    for b in range(W.blocks_per_row):
        off = row_off + b * W.block_bytes
        d = _f16(W.data, off)
        base = b * W.block_elements
        if W.dtype == "Q8_0":
            for j, q in enumerate(struct.unpack_from("<32b", W.data, off + 2)):
                out[base + j] = d * q
        elif W.dtype == "Q4_0":
            for j in range(16):
                q = W.data[off + 2 + j]
                out[base + j] = d * ((q & 0x0F) - 8)
                out[base + j + 16] = d * ((q >> 4) - 8)
        elif W.dtype == "Q4_K":
            dmin = _f16(W.data, off + 2)
            scales = W.data[off + 4:off + 16]
            qoff = off + 16
            is_ = 0
            for half in range(4):
                sc, mn = _scale_min_k4(is_, scales)
                d1, m1 = d * sc, dmin * mn
                sc, mn = _scale_min_k4(is_ + 1, scales)
                d2, m2 = d * sc, dmin * mn
                packed = qoff + half * 32
                xb = base + half * 64
                for j in range(32):
                    q = W.data[packed + j]
                    out[xb + j] = d1 * (q & 0x0F) - m1
                    out[xb + j + 32] = d2 * (q >> 4) - m2
                is_ += 2
        elif W.dtype == "Q5_K":
            dmin = _f16(W.data, off + 2)
            scales = W.data[off + 4:off + 16]
            qhoff = off + 16
            qloff = off + 48
            is_ = 0
            u1, u2 = 1, 2
            for half in range(4):
                sc, mn = _scale_min_k4(is_, scales)
                d1, m1 = d * sc, dmin * mn
                sc, mn = _scale_min_k4(is_ + 1, scales)
                d2, m2 = d * sc, dmin * mn
                packed = qloff + half * 32
                xb = base + half * 64
                for j in range(32):
                    q = W.data[packed + j]
                    qh = W.data[qhoff + j]
                    q1 = (q & 0x0F) + (16 if qh & u1 else 0)
                    q2 = (q >> 4) + (16 if qh & u2 else 0)
                    out[xb + j] = d1 * q1 - m1
                    out[xb + j + 32] = d2 * q2 - m2
                is_ += 2
                u1 <<= 2
                u2 <<= 2
        else:  # Q6_K
            sc = struct.unpack_from("<16b", W.data, off + 192)
            d = _f16(W.data, off + 208)
            for half in range(2):
                qloff = off + half * 64
                qhoff = off + 128 + half * 32
                soff = half * 8
                xb = base + half * 128
                for j in range(32):
                    is_ = j // 16
                    ql1 = W.data[qloff + j]
                    ql2 = W.data[qloff + j + 32]
                    qh = W.data[qhoff + j]
                    q1 = ((ql1 & 0x0F) | (((qh >> 0) & 3) << 4)) - 32
                    q2 = ((ql2 & 0x0F) | (((qh >> 2) & 3) << 4)) - 32
                    q3 = ((ql1 >> 4) | (((qh >> 4) & 3) << 4)) - 32
                    q4 = ((ql2 >> 4) | (((qh >> 6) & 3) << 4)) - 32
                    out[xb + j] = d * sc[soff + is_] * q1
                    out[xb + j + 32] = d * sc[soff + is_ + 2] * q2
                    out[xb + j + 64] = d * sc[soff + is_ + 4] * q3
                    out[xb + j + 96] = d * sc[soff + is_ + 6] * q4
    return out


def _quantized_row_np(W: QuantizedMatrix, r: int):
    if not 0 <= r < W.rows:
        raise IndexError(r)
    if W.dtype == "Q4_K":
        d, dmin, sc, mn, q = W._np_views()
        out = _np.empty(W.cols, dtype=_np.float32)
        for b in range(W.blocks_per_row):
            for half in range(4):
                j1, j2 = half * 2, half * 2 + 1
                chunk = q[r, b, half * 32:(half + 1) * 32]
                lo = (chunk & 0x0F).astype(_np.float32)
                hi = (chunk >> 4).astype(_np.float32)
                base = b * QK_K + half * 64
                out[base:base + 32] = (
                    d[r, b] * sc[r, b, j1] * lo -
                    dmin[r, b] * mn[r, b, j1])
                out[base + 32:base + 64] = (
                    d[r, b] * sc[r, b, j2] * hi -
                    dmin[r, b] * mn[r, b, j2])
        return out
    if W.dtype == "Q5_K":
        d, dmin, sc, mn, qh, ql = W._np_views()
        out = _np.empty(W.cols, dtype=_np.float32)
        for b in range(W.blocks_per_row):
            qhb = qh[r, b, :]
            for half in range(4):
                j1, j2 = half * 2, half * 2 + 1
                chunk = ql[r, b, half * 32:(half + 1) * 32]
                hb1 = ((qhb >> j1) & 1).astype(_np.float32) * 16.0
                hb2 = ((qhb >> j2) & 1).astype(_np.float32) * 16.0
                lo = (chunk & 0x0F).astype(_np.float32) + hb1
                hi = (chunk >> 4).astype(_np.float32) + hb2
                base = b * QK_K + half * 64
                out[base:base + 32] = (
                    d[r, b] * sc[r, b, j1] * lo -
                    dmin[r, b] * mn[r, b, j1])
                out[base + 32:base + 64] = (
                    d[r, b] * sc[r, b, j2] * hi -
                    dmin[r, b] * mn[r, b, j2])
        return out
    if W.dtype == "Q6_K":
        d, sc, ql, qh = W._np_views()
        out = _np.empty(W.cols, dtype=_np.float32)
        for b in range(W.blocks_per_row):
            for half in range(2):
                qlb = ql[r, b, half * 64:(half + 1) * 64]
                qhb = qh[r, b, half * 32:(half + 1) * 32]
                scb = sc[r, b, half * 8:(half + 1) * 8]
                q1 = ((qlb[:32] & 0x0F) |
                      (((qhb >> 0) & 3) << 4)).astype(_np.int16) - 32
                q2 = ((qlb[32:] & 0x0F) |
                      (((qhb >> 2) & 3) << 4)).astype(_np.int16) - 32
                q3 = ((qlb[:32] >> 4) |
                      (((qhb >> 4) & 3) << 4)).astype(_np.int16) - 32
                q4 = ((qlb[32:] >> 4) |
                      (((qhb >> 6) & 3) << 4)).astype(_np.int16) - 32
                base = b * QK_K + half * 128
                for sub, q in enumerate((q1, q2, q3, q4)):
                    scales = _np.repeat(scb[[sub * 2, sub * 2 + 1]], 16)
                    out[base + sub * 32:base + (sub + 1) * 32] = (
                        d[r, b] * scales * q.astype(_np.float32))
        return out
    d, q = W._np_views()
    if W.dtype == "Q8_0":
        return (d[r, :, None] * q[r].astype(_np.float32)).reshape(-1)
    qb = q[r]
    lo = ((qb & 0x0F).astype(_np.int8) - 8).astype(_np.float32)
    hi = ((qb >> 4).astype(_np.int8) - 8).astype(_np.float32)
    vals = _np.concatenate([lo, hi], axis=1)
    return (d[r, :, None] * vals).reshape(-1)


def _quantized_dequant_rows_np(W: QuantizedMatrix, r0: int, r1: int):
    from .quants import dequantize
    row_bytes = W.blocks_per_row * W.block_bytes
    data = W.data[r0 * row_bytes:r1 * row_bytes]
    vals = dequantize(data, (r1 - r0) * W.cols, W.dtype)
    return _np.asarray(vals, dtype=_np.float32).reshape(r1 - r0, W.cols)


def _quantized_matmul_t_np(W: QuantizedMatrix, X):
    X = _np.asarray(X, dtype=_np.float32)
    if X.ndim != 2 or X.shape[1] != W.cols:
        raise ValueError(f"expected ({X.shape[0]}, {W.cols}) input, got {X.shape}")
    dense = W._dense_hot_cache()
    if dense is not None:
        return X @ dense.T
    out = _np.empty((X.shape[0], W.rows), dtype=_np.float32)
    tile_rows = max(32, (2 << 20) // max(W.cols * 4, 1))
    for r0 in range(0, W.rows, tile_rows):
        r1 = min(W.rows, r0 + tile_rows)
        tile = _quantized_dequant_rows_np(W, r0, r1)
        out[:, r0:r1] = X @ tile.T
    return out


def matvec(W, x):
    """W (rows x cols) times x (cols) -> rows."""
    if isinstance(W, QuantizedMatrix):
        if HAS_NUMPY:
            dense = W._dense_hot_cache()
            if dense is not None:
                return dense @ _np.asarray(x, dtype=_np.float32)
        return _quantized_matvec(W, x)
    if HAS_NUMPY:
        return W @ x
    return [sum(w * xv for w, xv in zip(row, x)) for row in W]


def matmul_t(X, W):
    """X (batch x cols) times W.T (cols x rows) -> batch x rows."""
    if not HAS_NUMPY:
        return [matvec(W, row) for row in X]
    if isinstance(W, QuantizedMatrix):
        return _quantized_matmul_t_np(W, X)
    return _np.asarray(X, dtype=_np.float32) @ W.T


def matrix_row(W, r: int):
    if isinstance(W, QuantizedMatrix):
        if HAS_NUMPY:
            _sync_hot_cache_budget(_hot_cache_limit_bytes())
            _sync_unpacked_cache_budget(_unpacked_cache_limit_bytes())
            if W._dense_cache is not None:
                return W._dense_cache[r].copy()
            return _quantized_row_np(W, r)
        return _quantized_row_pure(W, r)
    if HAS_NUMPY:
        return W[r].copy()
    return list(W[r])


def matrix_rows(W, rows):
    if isinstance(W, QuantizedMatrix):
        if HAS_NUMPY:
            _sync_hot_cache_budget(_hot_cache_limit_bytes())
            _sync_unpacked_cache_budget(_unpacked_cache_limit_bytes())
            idx = _np.asarray(rows, dtype=_np.int64)
            if W._dense_cache is not None:
                return W._dense_cache[idx].copy()
            return _np.stack([_quantized_row_np(W, int(r)) for r in idx], axis=0)
        return [_quantized_row_pure(W, int(r)) for r in rows]
    if HAS_NUMPY:
        return W[_np.asarray(rows, dtype=_np.int64)].copy()
    return [list(W[int(r)]) for r in rows]


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
        arr = x if getattr(x, "dtype", None) == _np.float32 else x.astype(_np.float32)
        inv = 1.0 / _np.sqrt(_np.mean(arr * arr, axis=-1, keepdims=True) + eps)
        return arr * inv * weight
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
