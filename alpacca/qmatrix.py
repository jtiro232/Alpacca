# Alpacca - quantized matrix storage and matvec/matmul kernels.
# MIT License. See LICENSE.
"""Row-major quantized GGUF matrix weights.

NumPy backend
    At construction the raw GGUF blocks are unpacked once (via
    :func:`alpacca.quants.np_unpack`) into int8 quant codes plus per-sub-block
    effective float32 scales/offsets, all owned copies - the source mmap can
    close immediately and no raw block bytes are retained. This keeps RAM at
    roughly 1.1-1.3 bytes per weight (vs 4 for float32) while decode reads
    the codes directly; nothing is re-unpacked per token.

    matvec uses the fastest exact-scale kernels measured for this layout
    (see README "Honest performance expectations"): a batched-matmul
    block-dot for small matrices and an einsum block-dot for large ones.
    Both compute out = sum_s d_eff[:, s] * (codes[:, s] @ x_s) (+ offsets),
    which is bit-equivalent algebra to dequantize-then-GEMV.

    The optional hot-cache budget globals (``ALPACCA_HOT_WEIGHT_MB``) are not
    synchronized for concurrent matvec callers: alpacca's own server
    serializes generation behind a lock, but embedders doing concurrent
    inference with the budget set should serialize calls or leave it unset.

Pure backend
    Owns a copy of the raw block bytes and decodes rows on the fly with the
    pure decoders in :mod:`alpacca.quants`. Slow but dependency-free; the
    model loader never wraps matrices in pure mode, this path exists so both
    backends can verify each other in tests.
"""

from __future__ import annotations

import math
import os
import weakref

from .quants import QUANT_GEOMETRY, dequantize

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

if os.environ.get("ALPACCA_PURE"):
    _np = None

HAS_NUMPY = _np is not None

QUANTIZED_MATVEC_DTYPES = frozenset(QUANT_GEOMETRY)

# below this many elements the batched-matmul kernel beats the einsum kernel
# (crossover measured around 1M elements on a 4-core AVX2 box; re-measure
# before changing)
_SMALL_MATVEC_ELEMS = 1 << 20

_HOT_WEIGHT_ENV = "ALPACCA_HOT_WEIGHT_MB"
_HOT_CACHE_LIMIT_BYTES = None
_HOT_CACHE_USED_BYTES = 0
_HOT_CACHE_MATRICES = 0
_HOT_CACHE_OWNERS = weakref.WeakSet()


class QuantMatrix:
    """Quantized (rows x cols) matrix; quant blocks run along columns.

    A contiguous row range therefore maps to a contiguous range of the
    underlying blocks, which is what row slicing and the pure backend rely
    on.
    """

    __slots__ = ("dtype", "rows", "cols", "block_elements", "block_bytes",
                 "blocks_per_row", "sub_len", "n_sub", "data",
                 "_q", "_q3", "_small", "_d", "_m",
                 "_dense_cache", "_dense_cache_bytes", "__weakref__")

    def __init__(self, data, dtype: str, rows: int, cols: int):
        geom = QUANT_GEOMETRY.get(dtype)
        if geom is None:
            raise ValueError(f"quantized matvec does not support {dtype}")
        block_elements, block_bytes, sub_len, _affine = geom
        if cols % block_elements:
            raise ValueError(
                f"{dtype} matrix columns must be a multiple of {block_elements}")
        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.block_elements = block_elements
        self.block_bytes = block_bytes
        self.blocks_per_row = cols // block_elements
        self.sub_len = sub_len
        self.n_sub = cols // sub_len
        expected = rows * self.blocks_per_row * block_bytes
        if len(data) != expected:
            raise ValueError(
                f"{dtype} matrix has {len(data)} bytes, expected {expected}")
        self._dense_cache = None
        self._dense_cache_bytes = 0
        if HAS_NUMPY:
            from .quants import np_unpack
            q, d_eff, m_eff = np_unpack(data, rows * cols, dtype)
            self.data = None  # unpacked copies own everything; mmap may close
            q3 = q.reshape(rows, self.n_sub, sub_len)
            self._small = rows * cols < _SMALL_MATVEC_ELEMS
            if self._small:
                # store small matrices in the batched-matmul kernel layout so
                # the per-token astype reads contiguously; _q3 stays the
                # logical element-order (rows, n_sub, sub_len) view of it
                self._q = _np.ascontiguousarray(q3.transpose(1, 0, 2))
                self._q3 = self._q.transpose(1, 0, 2)
            else:
                self._q = q3
                self._q3 = q3
            self._d = d_eff.reshape(rows, self.n_sub)
            self._m = None if m_eff is None else m_eff.reshape(rows, self.n_sub)
        else:
            self.data = bytes(data)
            self._small = False
            self._q = self._q3 = self._d = self._m = None

    def __del__(self):  # pragma: no cover - depends on interpreter shutdown
        try:
            if self._dense_cache is not None:
                _release_hot_cache_bytes(self._dense_cache_bytes)
        except Exception:
            pass

    # ---- introspection ---------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def storage_nbytes(self) -> int:
        """Bytes held by the quantized representation (excl. hot cache)."""
        if HAS_NUMPY:
            n = self._q.nbytes + self._d.nbytes
            if self._m is not None:
                n += self._m.nbytes
            return n
        return len(self.data)

    # ---- matvec (decode path) ---------------------------------------------

    def __matmul__(self, x):
        return self.matvec(x)

    def matvec(self, x):
        """self (rows x cols) @ x (cols) -> (rows)."""
        if not HAS_NUMPY:
            return self._matvec_pure(x)
        if self._dense_cache is not None or _HOT_WEIGHT_ENV in os.environ:
            dense = self._dense_hot_cache()
            if dense is not None:
                return dense @ _np.asarray(x, dtype=_np.float32)
        xs = _np.asarray(x, dtype=_np.float32).reshape(self.n_sub, self.sub_len)
        if self._small:
            # batched (n_sub) BLAS matmuls of (rows, sub_len) @ (sub_len, 1)
            blockdot = _np.matmul(self._q.astype(_np.float32),
                                  xs[:, :, None])[:, :, 0]      # (n_sub, rows)
            out = _np.einsum("sr,rs->r", blockdot, self._d)
        else:
            blockdot = _np.einsum("rsl,sl->rs", self._q3, xs)   # (rows, n_sub)
            out = _np.einsum("rs,rs->r", blockdot, self._d)
        if self._m is not None:
            out += self._m @ xs.sum(axis=1)
        return out

    def _matvec_pure(self, x):
        out = [0.0] * self.rows
        row_bytes = self.blocks_per_row * self.block_bytes
        for r in range(self.rows):
            wrow = dequantize(self.data[r * row_bytes:(r + 1) * row_bytes],
                              self.cols, self.dtype)
            out[r] = sum(w * xv for w, xv in zip(wrow, x))
        return out

    # ---- batched matmul (prefill path) -------------------------------------

    def matmul_t(self, X):
        """X (batch x cols) @ self.T -> (batch x rows)."""
        if not HAS_NUMPY:
            return [self.matvec(row) for row in X]
        X = _np.asarray(X, dtype=_np.float32)
        if X.ndim != 2 or X.shape[1] != self.cols:
            raise ValueError(
                f"expected ({X.shape[0]}, {self.cols}) input, got {X.shape}")
        if self._dense_cache is not None or _HOT_WEIGHT_ENV in os.environ:
            dense = self._dense_hot_cache()
            if dense is not None:
                return X @ dense.T
        out = _np.empty((X.shape[0], self.rows), dtype=_np.float32)
        x_sub_sums = None
        if self._m is not None:
            x_sub_sums = X.reshape(X.shape[0], self.n_sub, self.sub_len).sum(axis=2)
        # dequantize ~4 MB row tiles; GGUF rows are contiguous so each tile
        # is a plain slice of the code/scale arrays
        tile_rows = max(16, (4 << 20) // max(self.cols * 4, 1))
        q3 = self._q3
        for r0 in range(0, self.rows, tile_rows):
            r1 = min(self.rows, r0 + tile_rows)
            tile = _np.ascontiguousarray(q3[r0:r1], dtype=_np.float32)
            tile *= self._d[r0:r1, :, None]
            out[:, r0:r1] = X @ tile.reshape(r1 - r0, self.cols).T
            if self._m is not None:
                out[:, r0:r1] += x_sub_sums @ self._m[r0:r1].T
        return out

    # ---- row access (embedding lookups) ------------------------------------

    def row(self, i: int):
        if not 0 <= i < self.rows:
            raise IndexError(i)
        if not HAS_NUMPY:
            row_bytes = self.blocks_per_row * self.block_bytes
            return dequantize(self.data[i * row_bytes:(i + 1) * row_bytes],
                              self.cols, self.dtype)
        if self._dense_cache is not None or _HOT_WEIGHT_ENV in os.environ:
            _sync_hot_cache_budget(_hot_cache_limit_bytes())
            if self._dense_cache is not None:
                return self._dense_cache[i].copy()
        v = self._q3[i].astype(_np.float32)
        v *= self._d[i, :, None]
        if self._m is not None:
            v += self._m[i, :, None]
        return v.reshape(-1)

    def rows_at(self, rows):
        if not HAS_NUMPY:
            return [self.row(int(r)) for r in rows]
        idx = _np.asarray(rows, dtype=_np.int64)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self.rows):
            raise IndexError("row index out of range")
        if self._dense_cache is not None or _HOT_WEIGHT_ENV in os.environ:
            _sync_hot_cache_budget(_hot_cache_limit_bytes())
            if self._dense_cache is not None:
                return self._dense_cache[idx].copy()
        v = self._q3[idx].astype(_np.float32)
        v *= self._d[idx][:, :, None]
        if self._m is not None:
            v += self._m[idx][:, :, None]
        return v.reshape(idx.size, self.cols)

    # ---- optional dense f32 cache (ALPACCA_HOT_WEIGHT_MB) -------------------

    def _dense_from_storage(self):
        v = self._q3.astype(_np.float32)
        v *= self._d[:, :, None]
        if self._m is not None:
            v += self._m[:, :, None]
        return v.reshape(self.rows, self.cols)

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
            dense = self._dense_from_storage()
        except Exception:
            _release_hot_cache_bytes(nbytes)
            raise
        self._dense_cache = dense
        self._dense_cache_bytes = nbytes
        _HOT_CACHE_OWNERS.add(self)
        return dense


# ---- optional hot-cache budget (module state) ------------------------------

def _hot_cache_limit_bytes() -> int:
    if not HAS_NUMPY:
        return 0
    raw = os.environ.get(_HOT_WEIGHT_ENV)
    if raw is None or not raw.strip():
        return 0
    try:
        mb = float(raw.strip())
    except ValueError:
        return 0
    if not math.isfinite(mb) or mb <= 0.0:
        return 0
    return int(mb * 1024 * 1024)


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


def _sync_hot_cache_budget(limit: int) -> None:
    global _HOT_CACHE_LIMIT_BYTES
    if _HOT_CACHE_LIMIT_BYTES != limit:
        _clear_hot_caches()
        _HOT_CACHE_LIMIT_BYTES = limit


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


def _reset_hot_cache_state() -> None:
    global _HOT_CACHE_LIMIT_BYTES
    _clear_hot_caches()
    _HOT_CACHE_LIMIT_BYTES = None


def hot_cache_stats() -> dict[str, int]:
    limit = _hot_cache_limit_bytes()
    _sync_hot_cache_budget(limit)
    return {
        "limit_bytes": limit,
        "used_bytes": _HOT_CACHE_USED_BYTES,
        "matrices": _HOT_CACHE_MATRICES,
    }


def can_quantized_matvec(dtype: str, cols: int) -> bool:
    geom = QUANT_GEOMETRY.get(dtype)
    return geom is not None and cols % geom[0] == 0


__all__ = ["QuantMatrix", "QUANTIZED_MATVEC_DTYPES", "can_quantized_matvec",
           "hot_cache_stats"]
