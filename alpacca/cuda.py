# Alpacca - our own GPU tier: quantized matvec/matmul CUDA kernels,
# written in Python. MIT License. See LICENSE.
"""Alpacca's GPU tier: Python-source CUDA kernels, optionally JIT-compiled.

Weight matrices stay quantized in VRAM in the same universal codes layout
the NumPy backend unpacks (int8 codes + per-sub-block effective float32
scales/offsets, :func:`alpacca.quants.np_unpack`). Every kernel here is
Alpacca's own algorithm, authored in this file as ordinary Python and
compiled at runtime by the OPTIONAL, PINNED numba-cuda JIT
(``pip install alpacca[gpu]``). Without a CUDA device or the pinned JIT -
or with ``ALPACCA_GPU=0`` - nothing changes: the CPU tiers (kernels/NumPy/
pure) remain the reference implementations and the fallback, and any error
in this tier degrades to them rather than failing the engine.

v1 placement: only weight matrices move to VRAM, and only whole matrices -
per-matrix dispatch makes mixed GPU/CPU placement exact, so running out of
VRAM mid-load just leaves the rest on the CPU tiers. Activations, the KV
cache and attention stay host-side; each matvec uploads one activation
vector and downloads one result (measured tax below). The token embedding
never uploads: it is a row-gather workload, the one shape these kernels
are wrong for, and it must stay gatherable on the host.

Pin policy: numba-cuda is locked to ``NUMBA_CUDA_PIN`` below, mirroring
the kernels tier. The pin is never updated implicitly; a different
installed version deactivates the tier (``ALPACCA_GPU=force`` overrides at
your own risk).

Known trap, recorded because the failure is silent and misattributed: with
``nvidia-nvjitlink`` missing, cuda.core's DLL search can find a torch
wheel's bundled nvJitLink_120_0.dll instead, and every kernel then dies
with nvJitLinkError ERROR_OUTDATED_LIBRARY(14). The availability probe
below therefore compiles and RUNS a real kernel, catches Exception
broadly, and reports the cause through status() instead of activating a
tier that cannot launch.

Buffer reuse contract: the per-size pinned/device activation buffers are
not synchronized for concurrent matvec callers - alpacca's own server
serializes generation behind a lock, same caveat as the hot-weight cache
in :mod:`alpacca.qmatrix`. Every matvec ends with a synchronous D2H on the
legacy default stream, which drains the queued async H2D and kernel, so
rewriting the staging buffer on the next call is safe.
"""

from __future__ import annotations

import math
import os
import sys

from .quants import QUANT_GEOMETRY, np_unpack

NUMBA_CUDA_PIN = "0.30.4"

# VRAM headroom never spent on weights: the WDDM display path, the CUDA
# context and the JIT each claim device memory after load, and exhausting
# the last MiB turns into unrelated launch failures much later.
GPU_RESERVE_MB = 512

# Threads per block for the row-per-block matvec; also the shared-memory
# reduction width, so it must stay a power of two.
_TPB = 256

# Wide-matmul tile shape: 32 batch lanes (one warp, coalesced activation
# reads) x 8 rows per block.
_MM_BATCH_TILE = 32
_MM_ROW_TILE = 8

_state: dict | None = None  # lazy: {"ok": True, ...} or {} / {"error"} off


# Batched matmul picks between two kernels by MATRIX size, not batch: the
# looped matvec is Python-launch-bound at prefill batches (256 launches
# make it a flat ~27 ms whatever the shape), while the 2-D wide kernel is
# one launch but ALU-bound on the byte unpack (~830 GFLOP/s). Milliseconds
# at batch 256 (RTX 4090):
#     elements   1.0M  4.2M  16.8M  33.6M  58.7M
#     looped     27.3  23.9   26.9   28.7   27.5
#     wide        2.9   5.8   16.2   28.6   36.9
# They cross at ~34M elements - VRAM streaming beats redundant decode only
# once the codes dwarf the launch overhead. Below the big-matrix regime
# the wide kernel also wins every batch >= 2 (fewer launches, same reads).
# Default is just under the measured tie; a knob so different hardware can
# re-measure rather than re-guess.
def _wide_matmul_max_elems() -> int:
    raw = os.environ.get("ALPACCA_GPU_WIDE_MATMUL_ELEMS", "")
    try:
        return max(0, int(raw))
    except ValueError:
        return 32 << 20


def _vram_cap_bytes() -> int:
    """ALPACCA_GPU_VRAM_MB caps total uploaded weight bytes - the test
    hook for the mixed-placement fallback; unset/0 lets free VRAM minus
    GPU_RESERVE_MB decide."""
    raw = os.environ.get("ALPACCA_GPU_VRAM_MB", "")
    if not raw.strip():
        return 0
    try:
        mb = float(raw)
    except ValueError:
        return 0
    if not math.isfinite(mb) or mb <= 0.0:
        return 0
    try:
        return int(mb * 1024 * 1024)
    except (OverflowError, ValueError):
        return 0


def _init() -> dict:
    global _state
    if _state is not None:
        return _state
    _state = {}
    mode = os.environ.get("ALPACCA_GPU", "").strip().lower()
    if mode in ("0", "off", "no") or os.environ.get("ALPACCA_PURE"):
        return _state
    try:
        import numpy as np
        import numba_cuda
        from numba import cuda, float32, int32
        from cuda.bindings import runtime as rt
    except Exception:
        return _state
    try:
        # a 3-row fixture matrix legitimately launches a 3-block grid; the
        # per-launch "Grid size N will likely result in GPU
        # under-utilization" advice is noise at that scale, and only
        # there. numba-cuda VENDORS its warning class (numba.cuda.core.
        # errors, not numba.core.errors - verified distinct classes), so
        # the filter must name the vendored one to have any effect.
        import warnings
        from numba.cuda.core.errors import NumbaPerformanceWarning
        warnings.filterwarnings("ignore", message=".*Grid size.*",
                                category=NumbaPerformanceWarning)
    except Exception:
        pass
    if numba_cuda.__version__ != NUMBA_CUDA_PIN and mode != "force":
        print(f"alpacca: numba-cuda {numba_cuda.__version__} != pinned "
              f"{NUMBA_CUDA_PIN}; gpu tier disabled (ALPACCA_GPU=force to "
              f"override)", file=sys.stderr)
        return _state

    try:
        if not cuda.is_available():
            return _state
        dev = cuda.get_current_device()
        name = (dev.name.decode() if isinstance(dev.name, bytes)
                else str(dev.name))
        err, free_b, total_b = rt.cudaMemGetInfo()
        if int(err):
            raise RuntimeError(f"cudaMemGetInfo: {err}")

        # The probe compiles AND runs before the tier may activate: import
        # succeeds even when the JIT-link stack is broken (see the
        # nvJitLink trap in the module docstring), so only a kernel that
        # produced correct output proves the tier can work.
        @cuda.jit(cache=True)
        def _probe(a, b):
            i = cuda.grid(1)
            if i < a.size:
                b[i] = a[i] * float32(2.0) + b[i]

        pa = np.arange(32, dtype=np.float32)
        pb = np.ones(32, dtype=np.float32)
        d_a, d_b = cuda.to_device(pa), cuda.to_device(pb)
        _probe[1, 32](d_a, d_b)
        got = d_b.copy_to_host()
        if float(abs(got[31] - 63.0)) != 0.0:
            raise RuntimeError("probe kernel returned wrong values")
    except Exception as e:
        # broad on purpose: report the real cause (missing driver, stale
        # nvJitLink, exhausted device...) through status() instead of
        # crashing whoever asked whether a GPU exists
        _state = {"error": f"{type(e).__name__}: {e}"}
        return _state

    @cuda.jit(fastmath=True, cache=True)
    def _matvec_codes(qw, d, m, x, out, shift, affine):
        # out[r] = sum_c (d[r, c>>shift] * code + m[...]) * x[c]: the same
        # per-element algebra as the effective scales np_unpack feeds every
        # CPU path. One block per row; qw is the int8 codes viewed as
        # int32, so each thread pulls four codes per load and a warp moves
        # 128 contiguous bytes - 734 GB/s measured vs 500 for byte loads.
        sh = cuda.shared.array(_TPB, float32)
        r = cuda.blockIdx.x
        t = cuda.threadIdx.x
        cw = qw.shape[1]
        acc = float32(0.0)
        c = t
        while c < cw:
            v = qw[r, c]
            c0 = c << 2
            s = c0 >> shift  # 4-byte groups never straddle a sub-block
            b0 = float32(((v & int32(0xFF)) ^ int32(0x80)) - int32(0x80))
            b1 = float32((((v >> 8) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            b2 = float32((((v >> 16) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            b3 = float32((((v >> 24) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            ds = d[r, s]
            if affine:
                ms = m[r, s]
                acc += (ds * b0 + ms) * x[c0]
                acc += (ds * b1 + ms) * x[c0 + 1]
                acc += (ds * b2 + ms) * x[c0 + 2]
                acc += (ds * b3 + ms) * x[c0 + 3]
            else:
                acc += ds * (b0 * x[c0] + b1 * x[c0 + 1]
                             + b2 * x[c0 + 2] + b3 * x[c0 + 3])
            c += _TPB
        sh[t] = acc
        cuda.syncthreads()
        i = _TPB // 2
        while i > 0:
            if t < i:
                sh[t] += sh[t + i]
            cuda.syncthreads()
            i >>= 1
        if t == 0:
            out[r] = sh[0]

    @cuda.jit(fastmath=True, cache=True)
    def _matmul_codes_wide(qw, d, m, xt, out, shift, affine):
        # xt (cols, batch) so the 32 batch lanes of a warp read
        # consecutive floats; out (rows, batch) keeps the store coalesced
        # and the host transposes after download (same trade the CPU wide
        # kernel documents). Used below the _wide_matmul_max_elems
        # crossover, where one launch beats one launch per batch row.
        b = cuda.blockIdx.y * _MM_BATCH_TILE + cuda.threadIdx.x
        r = cuda.blockIdx.x * _MM_ROW_TILE + cuda.threadIdx.y
        if r >= out.shape[0] or b >= out.shape[1]:
            return
        cw = qw.shape[1]
        acc = float32(0.0)
        for c in range(cw):
            v = qw[r, c]
            c0 = c << 2
            s = c0 >> shift
            b0 = float32(((v & int32(0xFF)) ^ int32(0x80)) - int32(0x80))
            b1 = float32((((v >> 8) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            b2 = float32((((v >> 16) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            b3 = float32((((v >> 24) & int32(0xFF)) ^ int32(0x80))
                         - int32(0x80))
            ds = d[r, s]
            if affine:
                ms = m[r, s]
                acc += (ds * b0 + ms) * xt[c0, b]
                acc += (ds * b1 + ms) * xt[c0 + 1, b]
                acc += (ds * b2 + ms) * xt[c0 + 2, b]
                acc += (ds * b3 + ms) * xt[c0 + 3, b]
            else:
                acc += ds * (b0 * xt[c0, b] + b1 * xt[c0 + 1, b]
                             + b2 * xt[c0 + 2, b] + b3 * xt[c0 + 3, b])
        out[r, b] = acc

    _state = {
        "ok": True, "np": np, "cuda": cuda, "rt": rt,
        "H2D": rt.cudaMemcpyKind.cudaMemcpyHostToDevice,
        "D2H": rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        "version": numba_cuda.__version__,
        "name": name, "free_mb": free_b >> 20, "total_mb": total_b >> 20,
        "matvec": _matvec_codes, "matmul_wide": _matmul_codes_wide,
        # per-size activation buffers, reused across calls (a cudaMalloc
        # per matvec costs more than the transfer it serves)
        "dx": {}, "hx": {}, "hout": {}, "dout": {},
        "matrices": 0, "uploaded_bytes": 0, "skipped": 0,
    }
    return _state


def available() -> bool:
    """True when the pinned JIT probe compiled, launched and verified."""
    return bool(_init().get("ok"))


def status() -> str:
    st = _init()
    if st.get("ok"):
        return (f"alpacca-gpu active (numba-cuda=={st['version']}, "
                f"pin {NUMBA_CUDA_PIN}, our Python source)")
    if st.get("error"):
        return f"alpacca-gpu inactive ({st['error']})"
    return "alpacca-gpu inactive (CPU tiers in use)"


def doctor_line() -> str:
    """One line for `alpacca doctor`: device + VRAM, or why there is none."""
    st = _init()
    if st.get("ok"):
        return (f"{st['name']} ({st['free_mb']} MiB free of "
                f"{st['total_mb']} MiB) - {status()}")
    return f"none detected - {status()}"


def vram_stats() -> dict[str, int]:
    st = _init()
    return {"matrices": st.get("matrices", 0),
            "uploaded_bytes": st.get("uploaded_bytes", 0),
            "skipped": st.get("skipped", 0)}


# ---- transfers -------------------------------------------------------------
# Raw runtime-API memcpys against pinned staging, not the numba copy
# helpers: per 16 KiB activation upload the numba path measured 47 us of
# Python argument handling, the raw call 31; the result download 17 vs 10.
# At ~160 matvec calls per decoded token that difference is milliseconds.

def _memcpy_h2d_async(st, dev, pinned, nbytes: int) -> None:
    err, = st["rt"].cudaMemcpyAsync(
        dev.__cuda_array_interface__["data"][0], pinned.ctypes.data,
        nbytes, st["H2D"], 0)
    if int(err):
        raise RuntimeError(f"cudaMemcpyAsync H2D: {err}")


def _memcpy_d2h(st, pinned, dev, nbytes: int) -> None:
    # synchronous: also the point where the legacy default stream drains,
    # which is what makes reusing the staging buffers safe (see docstring)
    err, = st["rt"].cudaMemcpy(
        pinned.ctypes.data, dev.__cuda_array_interface__["data"][0],
        nbytes, st["D2H"])
    if int(err):
        raise RuntimeError(f"cudaMemcpy D2H: {err}")


def _upload_x(st, x):
    """Stage one float32 activation vector into the per-size device buffer."""
    np_ = st["np"]
    cols = x.shape[0]
    hx = st["hx"].get(cols)
    if hx is None:
        hx = st["cuda"].pinned_array(cols, np_.float32)
        st["hx"][cols] = hx
    hx[:] = x
    dx = st["dx"].get(cols)
    if dx is None:
        dx = st["cuda"].device_array(cols, np_.float32)
        st["dx"][cols] = dx
    _memcpy_h2d_async(st, dx, hx, cols * 4)
    return dx


def upload_vector(x):
    """Upload one activation vector so several GpuMatrix.matvec calls can
    share a single H2D (tensor.matvec_group's wqk+wv chain). Returns the
    device buffer, or None when the tier is inactive or the upload fails -
    callers then degrade to per-matvec uploads."""
    st = _init()
    if not st.get("ok"):
        return None
    try:
        x = st["np"].ascontiguousarray(x, dtype=st["np"].float32)
        return _upload_x(st, x)
    except Exception:
        return None


def _download_out(st, dout, rows: int):
    np_ = st["np"]
    hout = st["hout"].get(rows)
    if hout is None:
        hout = st["cuda"].pinned_array(rows, np_.float32)
        st["hout"][rows] = hout
    _memcpy_d2h(st, hout, dout, rows * 4)
    return np_.array(hout)  # fresh array: callers keep results across calls


class _VramBudget(Exception):
    """Upload refused by the free-VRAM reserve or ALPACCA_GPU_VRAM_MB."""


class GpuMatrix:
    """Quantized (rows x cols) matrix resident in VRAM, codes layout.

    Built from the raw GGUF block bytes at load time (the host QuantMatrix
    frees its unpack source, so wrapping one after construction is not
    possible) and holding ONLY device arrays - a matrix that fails to
    upload never constructs, and the loader falls back to the exact
    QuantMatrix it would have built anyway. A matrix that fails at runtime
    downloads its arrays once and computes on the host from then on.
    """

    # duck-typed marker checked by tensor.py: an isinstance test there
    # would drag the CUDA stack into every CPU-only import
    is_gpu_matrix = True

    __slots__ = ("dtype", "rows", "cols", "sub_len", "n_sub", "vram_nbytes",
                 "_shift", "_dq", "_dd", "_dm", "_dead",
                 "_hq3", "_hd", "_hm")

    def __init__(self, data, dtype: str, rows: int, cols: int):
        st = _init()
        if not st.get("ok"):
            raise RuntimeError("gpu tier inactive")
        geom = QUANT_GEOMETRY.get(dtype)
        if geom is None:
            raise ValueError(f"gpu matvec does not support {dtype}")
        block_elements, _bb, sub_len, _aff = geom
        if cols % block_elements:
            raise ValueError(
                f"{dtype} matrix columns must be a multiple of "
                f"{block_elements}")
        if sub_len & (sub_len - 1) or cols % 4:
            # the kernels index scales by c >> log2(sub_len) and load
            # codes four at a time; every GGUF geometry satisfies both
            raise ValueError(f"unsupported geometry {dtype}/{cols}")
        np_ = st["np"]
        q, d_eff, m_eff = np_unpack(data, rows * cols, dtype)
        qw = np_.ascontiguousarray(q.reshape(rows, cols)).view(np_.int32)
        d = np_.ascontiguousarray(d_eff.reshape(rows, -1))
        m = (None if m_eff is None
             else np_.ascontiguousarray(m_eff.reshape(rows, -1)))
        need = qw.nbytes + d.nbytes + (0 if m is None else m.nbytes)
        cap = _vram_cap_bytes()
        if cap and st["uploaded_bytes"] + need > cap:
            raise _VramBudget(f"cap {cap >> 20} MiB")
        err, free_b, _total = st["rt"].cudaMemGetInfo()
        if int(err):
            raise RuntimeError(f"cudaMemGetInfo: {err}")
        if need + (GPU_RESERVE_MB << 20) > free_b:
            raise _VramBudget(f"{free_b >> 20} MiB free")
        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.sub_len = sub_len
        self.n_sub = cols // sub_len
        self._shift = sub_len.bit_length() - 1
        cuda = st["cuda"]
        self._dq = cuda.to_device(qw)
        self._dd = cuda.to_device(d)
        self._dm = None if m is None else cuda.to_device(m)
        self.vram_nbytes = need
        self._dead = False
        self._hq3 = self._hd = self._hm = None
        st["matrices"] += 1
        st["uploaded_bytes"] += need
        if st["matrices"] == 1:
            from . import tensor as _t
            _t._GPU_ACTIVE = True  # backend_name() flips to "gpu (cuda)"

    # ---- introspection ---------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def storage_nbytes(self) -> int:
        """Bytes resident on the device for this matrix."""
        return self.vram_nbytes

    def _release_vram(self) -> None:
        """Return this matrix's bytes to the budget. Idempotent; called on
        degrade (device arrays dropped) and on garbage collection, so a
        freed model no longer counts against ALPACCA_GPU_VRAM_MB and a
        later budget exhaustion warns again."""
        st = _state
        if st and self.vram_nbytes:
            st["uploaded_bytes"] = max(
                0, st["uploaded_bytes"] - self.vram_nbytes)
            st["matrices"] = max(0, st["matrices"] - 1)
            st["skip_warned"] = False
            if st["matrices"] == 0:
                from . import tensor as _t
                _t._GPU_ACTIVE = False  # nothing device-resident any more
        self.vram_nbytes = 0

    def __del__(self):
        try:
            self._release_vram()
        except Exception:
            pass  # interpreter teardown: module globals may already be gone

    # ---- degraded host path ----------------------------------------------

    def _host_arrays(self):
        """Download codes+scales once; afterwards this matrix computes on
        the CPU with the same effective-scale algebra as QuantMatrix."""
        st = _init()
        np_ = st["np"]
        if self._hq3 is None:
            q = self._dq.copy_to_host().view(np_.int8)
            self._hq3 = q.reshape(self.rows, self.n_sub, self.sub_len)
            self._hd = self._dd.copy_to_host()
            self._hm = None if self._dm is None else self._dm.copy_to_host()
        return self._hq3, self._hd, self._hm

    def _degrade(self, exc) -> None:
        if not self._dead:
            # download while the device still answers. If even the download
            # fails the weights are unrecoverable (there is deliberately no
            # host copy) and the original error propagates - the one case
            # this tier cannot degrade from is losing the device itself.
            self._host_arrays()
            self._dead = True
            # a degraded matrix computes on the host from here on: drop the
            # device arrays (numba frees them on collection) and give the
            # bytes back to the budget instead of holding VRAM that will
            # never be read again
            self._dq = self._dd = self._dm = None
            self._release_vram()
            print(f"alpacca-gpu: matrix {self.rows}x{self.cols} {self.dtype} "
                  f"degraded to CPU ({type(exc).__name__}: {exc})",
                  file=sys.stderr)

    def _matvec_host(self, x):
        st = _init()
        np_ = st["np"]
        q3, d, m = self._host_arrays()
        xs = np_.asarray(x, dtype=np_.float32).reshape(self.n_sub,
                                                       self.sub_len)
        blockdot = np_.einsum("rsl,sl->rs", q3, xs)
        out = np_.einsum("rs,rs->r", blockdot, d)
        if m is not None:
            out += m @ xs.sum(axis=1)
        return out

    def _matmul_host(self, X):
        st = _init()
        np_ = st["np"]
        q3, d, m = self._host_arrays()
        out = np_.empty((X.shape[0], self.rows), dtype=np_.float32)
        x_sub_sums = (X.reshape(X.shape[0], self.n_sub,
                                self.sub_len).sum(axis=2)
                      if m is not None else None)
        tile_rows = max(16, (4 << 20) // max(self.cols * 4, 1))
        for r0 in range(0, self.rows, tile_rows):
            r1 = min(self.rows, r0 + tile_rows)
            tile = np_.ascontiguousarray(q3[r0:r1], dtype=np_.float32)
            tile *= d[r0:r1, :, None]
            out[:, r0:r1] = X @ tile.reshape(r1 - r0, self.cols).T
            if m is not None:
                out[:, r0:r1] += x_sub_sums @ m[r0:r1].T
        return out

    # ---- matvec (decode path) ---------------------------------------------

    def __matmul__(self, x):
        return self.matvec(x)

    def matvec(self, x, dx=None):
        """self (rows x cols) @ x (cols) -> (rows) float32.

        `dx` optionally carries upload_vector(x) so matrices sharing an
        input vector upload it once (tensor.matvec_group).
        """
        st = _init()
        np_ = st["np"]
        x = np_.ascontiguousarray(x, dtype=np_.float32)
        if x.shape[0] != self.cols:
            raise ValueError(
                f"expected ({self.cols},) input, got {x.shape}")
        if self._dead:
            return self._matvec_host(x)
        try:
            if dx is None:
                dx = _upload_x(st, x)
            dout = st["dout"].get(self.rows)
            if dout is None:
                dout = st["cuda"].device_array(self.rows, np_.float32)
                st["dout"][self.rows] = dout
            st["matvec"][self.rows, _TPB](
                self._dq, self._dd,
                self._dd if self._dm is None else self._dm,
                dx, dout, self._shift, self._dm is not None)
            return _download_out(st, dout, self.rows)
        except Exception as e:
            self._degrade(e)
            return self._matvec_host(x)

    # ---- batched matmul (prefill path) -------------------------------------

    def matmul_t(self, X):
        """X (batch x cols) @ self.T -> (batch x rows) float32."""
        st = _init()
        np_ = st["np"]
        X = np_.ascontiguousarray(X, dtype=np_.float32)
        if X.ndim != 2 or X.shape[1] != self.cols:
            raise ValueError(
                f"expected ({X.shape[0]}, {self.cols}) input, got {X.shape}")
        if self._dead:
            return self._matmul_host(X)
        batch = X.shape[0]
        try:
            cuda = st["cuda"]
            if batch > 1 and self.rows * self.cols <= _wide_matmul_max_elems():
                xt = cuda.to_device(np_.ascontiguousarray(X.T))
                dout = cuda.device_array((self.rows, batch), np_.float32)
                gx = (self.rows + _MM_ROW_TILE - 1) // _MM_ROW_TILE
                gy = (batch + _MM_BATCH_TILE - 1) // _MM_BATCH_TILE
                st["matmul_wide"][(gx, gy), (_MM_BATCH_TILE, _MM_ROW_TILE)](
                    self._dq, self._dd,
                    self._dd if self._dm is None else self._dm,
                    xt, dout, self._shift, self._dm is not None)
                return np_.ascontiguousarray(dout.copy_to_host().T)
            # looped matvec: codes re-stream from VRAM per batch row, and
            # still win every prefill-chunk batch (table above); results
            # land directly in (batch, rows) with one download
            dX = cuda.to_device(X)
            dO = cuda.device_array((batch, self.rows), np_.float32)
            kern = st["matvec"]
            dm = self._dd if self._dm is None else self._dm
            affine = self._dm is not None
            for i in range(batch):
                kern[self.rows, _TPB](self._dq, self._dd, dm, dX[i], dO[i],
                                      self._shift, affine)
            return dO.copy_to_host()
        except Exception as e:
            self._degrade(e)
            return self._matmul_host(X)

    # ---- row access (rare: only reached if a GpuMatrix is row-gathered) ----

    def row(self, i: int):
        if not 0 <= i < self.rows:
            raise IndexError(i)
        st = _init()
        np_ = st["np"]
        q3, d, m = ((self._hq3, self._hd, self._hm) if self._dead
                    else (None, None, None))
        if q3 is None:
            try:
                q = self._dq[i].copy_to_host().view(np_.int8)
                drow = self._dd[i].copy_to_host()
                mrow = None if self._dm is None \
                    else self._dm[i].copy_to_host()
            except Exception as e:
                self._degrade(e)
                q3, d, m = self._host_arrays()
        if q3 is not None:
            q = q3[i].reshape(-1)
            drow = d[i]
            mrow = None if m is None else m[i]
        v = q.reshape(self.n_sub, self.sub_len).astype(np_.float32)
        v *= drow[:, None]
        if mrow is not None:
            v += mrow[:, None]
        return v.reshape(-1)

    def rows_at(self, rows):
        st = _init()
        np_ = st["np"]
        idx = np_.asarray(rows, dtype=np_.int64)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self.rows):
            raise IndexError("row index out of range")
        if not idx.size:
            return np_.empty((0, self.cols), dtype=np_.float32)
        return np_.stack([self.row(int(i)) for i in idx])


def gpu_matrix(data, dtype: str, rows: int, cols: int):
    """Build a VRAM-resident matrix from raw GGUF block bytes.

    Returns None - and the caller keeps its exact CPU QuantMatrix - when
    the tier is inactive, the budget is spent, or anything at all fails;
    this function never raises.
    """
    st = _init()
    if not st.get("ok"):
        return None
    try:
        return GpuMatrix(data, dtype, rows, cols)
    except _VramBudget as e:
        st["skipped"] += 1
        if not st.get("skip_warned"):
            st["skip_warned"] = True
            print(f"alpacca-gpu: VRAM budget reached ({e}) after "
                  f"{st['matrices']} matrices "
                  f"({st['uploaded_bytes'] >> 20} MiB); remaining matrices "
                  f"stay on the CPU tiers", file=sys.stderr)
        return None
    except Exception as e:
        st["skipped"] += 1
        if not st.get("fail_warned"):
            st["fail_warned"] = True
            print(f"alpacca-gpu: upload failed ({type(e).__name__}: {e}); "
                  f"matrix stays on the CPU tiers", file=sys.stderr)
        return None


def warmup() -> None:
    """Trigger JIT compilation once (cached on disk afterwards)."""
    st = _init()
    if not st.get("ok"):
        return
    try:
        np_ = st["np"]
        cuda = st["cuda"]
        qw = cuda.to_device(np_.zeros((2, 8), np_.int32))
        d = cuda.to_device(np_.zeros((2, 1), np_.float32))
        x = cuda.to_device(np_.zeros(32, np_.float32))
        out = cuda.device_array(2, np_.float32)
        st["matvec"][2, _TPB](qw, d, d, x, out, 5, False)
        st["matvec"][2, _TPB](qw, d, d, x, out, 5, True)
        xt = cuda.to_device(np_.zeros((32, 2), np_.float32))
        o2 = cuda.device_array((2, 2), np_.float32)
        st["matmul_wide"][(1, 1), (_MM_BATCH_TILE, _MM_ROW_TILE)](
            qw, d, d, xt, o2, 5, True)
        cuda.synchronize()
    except Exception as e:
        # warn, but do NOT clear _state: matrices already resident hold
        # references into it, and their per-call degrade path (download,
        # then compute on the host) needs the state alive to run at all
        print(f"alpacca: gpu warmup failed ({type(e).__name__}: {e}); "
              f"kernel calls will degrade per-matrix to the CPU tiers",
              file=sys.stderr)


__all__ = ["GpuMatrix", "NUMBA_CUDA_PIN", "available", "status",
           "doctor_line", "gpu_matrix", "upload_vector", "vram_stats",
           "warmup"]
