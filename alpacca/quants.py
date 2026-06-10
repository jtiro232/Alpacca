# Alpacca — GGUF quantization formats, implemented from the spec in pure
# Python (with optional NumPy fast paths). MIT License. See LICENSE.
"""Dequantize GGUF tensor data to float32, and quantize for the writer.

Every decoder has a pure-Python implementation (standard library only).
When NumPy is importable, vectorized fast paths are used for the common
formats; the remaining ones fall back to the pure code transparently.
"""

from __future__ import annotations

import struct

try:  # optional accelerator only — everything works without it
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

import os

if os.environ.get("ALPACCA_PURE"):
    _np = None

QK = 32      # block size of the classic quants
QK_K = 256   # block size of the K-quants


def _f16(b: bytes | memoryview, off: int) -> float:
    return struct.unpack_from("<e", b, off)[0]


# ---- pure-Python decoders ------------------------------------------------

def _deq_f32(data, n):
    return list(struct.unpack(f"<{n}f", bytes(data)))


def _deq_f16(data, n):
    return list(struct.unpack(f"<{n}e", bytes(data)))


def _deq_bf16(data, n):
    out = [0.0] * n
    raw = struct.unpack(f"<{n}H", bytes(data))
    for i, h in enumerate(raw):
        out[i] = struct.unpack("<f", struct.pack("<I", h << 16))[0]
    return out


def _deq_q8_0(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 34
        d = _f16(data, off)
        qs = struct.unpack_from("<32b", data, off + 2)
        base = i * QK
        for j in range(QK):
            out[base + j] = d * qs[j]
    return out


def _deq_q4_0(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 18
        d = _f16(data, off)
        base = i * QK
        for j in range(16):
            q = data[off + 2 + j]
            out[base + j] = d * ((q & 0x0F) - 8)
            out[base + j + 16] = d * ((q >> 4) - 8)
    return out


def _deq_q4_1(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 20
        d = _f16(data, off)
        m = _f16(data, off + 2)
        base = i * QK
        for j in range(16):
            q = data[off + 4 + j]
            out[base + j] = d * (q & 0x0F) + m
            out[base + j + 16] = d * (q >> 4) + m
    return out


def _deq_q5_0(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 22
        d = _f16(data, off)
        qh, = struct.unpack_from("<I", data, off + 2)
        base = i * QK
        for j in range(16):
            q = data[off + 6 + j]
            xh0 = ((qh >> j) << 4) & 0x10
            xh1 = (qh >> (j + 12)) & 0x10
            out[base + j] = d * (((q & 0x0F) | xh0) - 16)
            out[base + j + 16] = d * (((q >> 4) | xh1) - 16)
    return out


def _deq_q5_1(data, n):
    out = [0.0] * n
    nb = n // QK
    for i in range(nb):
        off = i * 24
        d = _f16(data, off)
        m = _f16(data, off + 2)
        qh, = struct.unpack_from("<I", data, off + 4)
        base = i * QK
        for j in range(16):
            q = data[off + 8 + j]
            xh0 = ((qh >> j) << 4) & 0x10
            xh1 = (qh >> (j + 12)) & 0x10
            out[base + j] = d * ((q & 0x0F) | xh0) + m
            out[base + j + 16] = d * ((q >> 4) | xh1) + m
    return out


def _scale_min_k4(j, scales):
    """6-bit packed scale/min pairs used by Q4_K / Q5_K."""
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
    mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, mn


def _deq_q4_k(data, n):
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 144
        d = _f16(data, off)
        dmin = _f16(data, off + 2)
        scales = data[off + 4:off + 16]
        qs = data[off + 16:off + 144]
        y = i * QK_K
        q = 0
        is_ = 0
        for _ in range(0, QK_K, 64):
            sc, mn = _scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * mn
            sc, mn = _scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * mn
            for l in range(32):
                out[y] = d1 * (qs[q + l] & 0xF) - m1
                y += 1
            for l in range(32):
                out[y] = d2 * (qs[q + l] >> 4) - m2
                y += 1
            q += 32
            is_ += 2
    return out


def _deq_q5_k(data, n):
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 176
        d = _f16(data, off)
        dmin = _f16(data, off + 2)
        scales = data[off + 4:off + 16]
        qh = data[off + 16:off + 48]
        ql = data[off + 48:off + 176]
        y = i * QK_K
        q = 0
        is_ = 0
        u1, u2 = 1, 2
        for _ in range(0, QK_K, 64):
            sc, mn = _scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * mn
            sc, mn = _scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * mn
            for l in range(32):
                out[y] = d1 * ((ql[q + l] & 0xF) + (16 if qh[l] & u1 else 0)) - m1
                y += 1
            for l in range(32):
                out[y] = d2 * ((ql[q + l] >> 4) + (16 if qh[l] & u2 else 0)) - m2
                y += 1
            q += 32
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return out


def _deq_q6_k(data, n):
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 210
        ql = data[off:off + 128]
        qh = data[off + 128:off + 192]
        sc = struct.unpack_from("<16b", data, off + 192)
        d = _f16(data, off + 208)
        y = i * QK_K
        qloff = 0
        qhoff = 0
        soff = 0
        for _ in range(0, QK_K, 128):
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[qloff + l] & 0xF) | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[qloff + l] >> 4) | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[qloff + l + 32] >> 4) | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32
                out[y + l] = d * sc[soff + is_] * q1
                out[y + l + 32] = d * sc[soff + is_ + 2] * q2
                out[y + l + 64] = d * sc[soff + is_ + 4] * q3
                out[y + l + 96] = d * sc[soff + is_ + 6] * q4
            y += 128
            qloff += 64
            qhoff += 32
            soff += 8
    return out


def _deq_q2_k(data, n):
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 84
        scales = data[off:off + 16]
        qs = data[off + 16:off + 80]
        d = _f16(data, off + 80)
        dmin = _f16(data, off + 82)
        y = i * QK_K
        is_ = 0
        qoff = 0
        for _ in range(0, QK_K, 128):
            shift = 0
            for _j in range(4):
                sc = scales[is_]
                is_ += 1
                dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
                for l in range(16):
                    out[y] = dl * ((qs[qoff + l] >> shift) & 3) - ml
                    y += 1
                sc = scales[is_]
                is_ += 1
                dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
                for l in range(16):
                    out[y] = dl * ((qs[qoff + 16 + l] >> shift) & 3) - ml
                    y += 1
                shift += 2
            qoff += 32
    return out


def _deq_q3_k(data, n):
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    out = [0.0] * n
    nb = n // QK_K
    for i in range(nb):
        off = i * 110
        hmask = data[off:off + 32]
        qs = data[off + 32:off + 96]
        aux = list(struct.unpack_from("<3I", data, off + 96))
        d_all = _f16(data, off + 108)
        tmp = aux[2]
        a0 = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        a1 = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        a2 = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        a3 = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        packed = struct.pack("<4I", a0, a1, a2, a3)
        scales = struct.unpack("<16b", packed)
        y = i * QK_K
        is_ = 0
        m = 1
        qoff = 0
        for _ in range(0, QK_K, 128):
            shift = 0
            for _j in range(4):
                dl = d_all * (scales[is_] - 32)
                is_ += 1
                for l in range(16):
                    q = (qs[qoff + l] >> shift) & 3
                    h = 0 if (hmask[l] & m) else 4
                    out[y] = dl * (q - h)
                    y += 1
                dl = d_all * (scales[is_] - 32)
                is_ += 1
                for l in range(16):
                    q = (qs[qoff + 16 + l] >> shift) & 3
                    h = 0 if (hmask[16 + l] & m) else 4
                    out[y] = dl * (q - h)
                    y += 1
                shift += 2
                m <<= 1
            qoff += 32
    return out


_PURE_DECODERS = {
    "F32": _deq_f32, "F16": _deq_f16, "BF16": _deq_bf16,
    "Q8_0": _deq_q8_0, "Q4_0": _deq_q4_0, "Q4_1": _deq_q4_1,
    "Q5_0": _deq_q5_0, "Q5_1": _deq_q5_1,
    "Q4_K": _deq_q4_k, "Q5_K": _deq_q5_k, "Q6_K": _deq_q6_k,
    "Q2_K": _deq_q2_k, "Q3_K": _deq_q3_k,
}


# ---- NumPy fast paths ----------------------------------------------------

def _np_blocks(data, nb, block_bytes):
    return _np.frombuffer(bytes(data), dtype=_np.uint8).reshape(nb, block_bytes)


def _np_deq_q8_0(data, n):
    nb = n // QK
    b = _np_blocks(data, nb, 34)
    d = b[:, 0:2].copy().view(_np.float16).astype(_np.float32)
    qs = b[:, 2:34].view(_np.int8).astype(_np.float32)
    return (d * qs).reshape(-1)


def _np_deq_q4_0(data, n):
    nb = n // QK
    b = _np_blocks(data, nb, 18)
    d = b[:, 0:2].copy().view(_np.float16).astype(_np.float32)
    qs = b[:, 2:18]
    lo = (qs & 0x0F).astype(_np.int8) - 8
    hi = (qs >> 4).astype(_np.int8) - 8
    out = _np.concatenate([lo, hi], axis=1).astype(_np.float32)
    return (d * out).reshape(-1)


def _np_deq_q4_k(data, n):
    nb = n // QK_K
    b = _np_blocks(data, nb, 144)
    d = b[:, 0:2].copy().view(_np.float16).astype(_np.float32)      # (nb,1)
    dmin = b[:, 2:4].copy().view(_np.float16).astype(_np.float32)
    scales = b[:, 4:16]
    qs = b[:, 16:144]
    sc = _np.empty((nb, 8), dtype=_np.float32)
    mn = _np.empty((nb, 8), dtype=_np.float32)
    for j in range(8):  # unpack 6-bit scale/min pairs
        if j < 4:
            sc[:, j] = (scales[:, j] & 63).astype(_np.float32)
            mn[:, j] = (scales[:, j + 4] & 63).astype(_np.float32)
        else:
            sc[:, j] = ((scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)).astype(_np.float32)
            mn[:, j] = ((scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)).astype(_np.float32)
    out = _np.empty((nb, QK_K), dtype=_np.float32)
    for half in range(4):  # 4 chunks of 32 bytes -> 2 sub-blocks each
        chunk = qs[:, half * 32:(half + 1) * 32]
        lo = (chunk & 0x0F).astype(_np.float32)
        hi = (chunk >> 4).astype(_np.float32)
        j1, j2 = 2 * half, 2 * half + 1
        out[:, j1 * 32:(j1 + 1) * 32] = d * sc[:, j1:j1 + 1] * lo - dmin * mn[:, j1:j1 + 1]
        out[:, j2 * 32:(j2 + 1) * 32] = d * sc[:, j2:j2 + 1] * hi - dmin * mn[:, j2:j2 + 1]
    return out.reshape(-1)


def _np_deq_q5_k(data, n):
    nb = n // QK_K
    b = _np_blocks(data, nb, 176)
    d = b[:, 0:2].copy().view(_np.float16).astype(_np.float32)
    dmin = b[:, 2:4].copy().view(_np.float16).astype(_np.float32)
    scales = b[:, 4:16]
    qh = b[:, 16:48]
    qs = b[:, 48:176]
    sc = _np.empty((nb, 8), dtype=_np.float32)
    mn = _np.empty((nb, 8), dtype=_np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = (scales[:, j] & 63).astype(_np.float32)
            mn[:, j] = (scales[:, j + 4] & 63).astype(_np.float32)
        else:
            sc[:, j] = ((scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)).astype(_np.float32)
            mn[:, j] = ((scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)).astype(_np.float32)
    out = _np.empty((nb, QK_K), dtype=_np.float32)
    for half in range(4):
        chunk = qs[:, half * 32:(half + 1) * 32]
        j1, j2 = 2 * half, 2 * half + 1
        hb1 = ((qh >> j1) & 1).astype(_np.float32) * 16.0
        hb2 = ((qh >> j2) & 1).astype(_np.float32) * 16.0
        lo = (chunk & 0x0F).astype(_np.float32) + hb1
        hi = (chunk >> 4).astype(_np.float32) + hb2
        out[:, j1 * 32:(j1 + 1) * 32] = d * sc[:, j1:j1 + 1] * lo - dmin * mn[:, j1:j1 + 1]
        out[:, j2 * 32:(j2 + 1) * 32] = d * sc[:, j2:j2 + 1] * hi - dmin * mn[:, j2:j2 + 1]
    return out.reshape(-1)


def _np_deq_q6_k(data, n):
    nb = n // QK_K
    b = _np_blocks(data, nb, 210)
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    sc = b[:, 192:208].view(_np.int8).astype(_np.float32)
    d = b[:, 208:210].copy().view(_np.float16).astype(_np.float32)
    out = _np.empty((nb, QK_K), dtype=_np.float32)
    for half in range(2):  # two 128-element halves
        qlh = ql[:, half * 64:(half + 1) * 64]
        qhh = qh[:, half * 32:(half + 1) * 32]
        sch = sc[:, half * 8:(half + 1) * 8]
        q1 = ((qlh[:, :32] & 0xF) | (((qhh >> 0) & 3) << 4)).astype(_np.int16) - 32
        q2 = ((qlh[:, 32:] & 0xF) | (((qhh >> 2) & 3) << 4)).astype(_np.int16) - 32
        q3 = ((qlh[:, :32] >> 4) | (((qhh >> 4) & 3) << 4)).astype(_np.int16) - 32
        q4 = ((qlh[:, 32:] >> 4) | (((qhh >> 6) & 3) << 4)).astype(_np.int16) - 32
        base = half * 128
        for sub, q in enumerate((q1, q2, q3, q4)):
            # each 32-element sub-block uses two scales, one per 16 elements
            s = _np.repeat(sch[:, [sub * 2, sub * 2 + 1]], 16, axis=1)
            out[:, base + sub * 32: base + (sub + 1) * 32] = d * s * q.astype(_np.float32)
    return out.reshape(-1)


_NP_DECODERS = {
    "Q8_0": _np_deq_q8_0,
    "Q4_0": _np_deq_q4_0,
    "Q4_K": _np_deq_q4_k,
    "Q5_K": _np_deq_q5_k,
    "Q6_K": _np_deq_q6_k,
}


def dequantize(data, n: int, dtype: str):
    """Decode `n` elements of GGUF tensor `data` to float32.

    Returns a NumPy float32 array when NumPy is available, else list[float].
    """
    if dtype not in _PURE_DECODERS:
        raise ValueError(
            f"tensor type {dtype} is not supported by the alpacca engine "
            f"(supported: {', '.join(sorted(_PURE_DECODERS))})")
    if _np is not None:
        if dtype == "F32":
            return _np.frombuffer(bytes(data), dtype=_np.float32).copy()
        if dtype == "F16":
            return _np.frombuffer(bytes(data), dtype=_np.float16).astype(_np.float32)
        if dtype == "BF16":
            raw = _np.frombuffer(bytes(data), dtype=_np.uint16).astype(_np.uint32) << 16
            return raw.view(_np.float32).copy()
        fast = _NP_DECODERS.get(dtype)
        if fast is not None:
            return fast(data, n)
        return _np.asarray(_PURE_DECODERS[dtype](data, n), dtype=_np.float32)
    return _PURE_DECODERS[dtype](data, n)


# ---- quantizers (used by the GGUF writer / tests) ------------------------

def quantize_q8_0(values: list[float]) -> bytes:
    if len(values) % QK:
        raise ValueError("Q8_0 needs a multiple of 32 values")
    out = bytearray()
    for i in range(0, len(values), QK):
        block = values[i:i + QK]
        amax = max(abs(v) for v in block)
        d = amax / 127.0 if amax else 0.0
        inv = 1.0 / d if d else 0.0
        qs = [max(-128, min(127, round(v * inv))) for v in block]
        out += struct.pack("<e32b", d, *qs)
    return bytes(out)


def quantize_q4_0(values: list[float]) -> bytes:
    if len(values) % QK:
        raise ValueError("Q4_0 needs a multiple of 32 values")
    out = bytearray()
    for i in range(0, len(values), QK):
        block = values[i:i + QK]
        vmax = max(block, key=abs)
        d = vmax / -8.0
        inv = 1.0 / d if d else 0.0
        q = [max(0, min(15, int(v * inv + 8.5))) for v in block]
        packed = bytes((q[j] & 0x0F) | (q[j + 16] << 4) for j in range(16))
        out += struct.pack("<e", d) + packed
    return bytes(out)
