# Alpacca - GGUF quantization formats, implemented from the spec in pure
# Python (with optional NumPy fast paths). MIT License. See LICENSE.
"""Dequantize GGUF tensor data to float32, and quantize for the writer.

Every decoder has a pure-Python implementation (standard library only).
When NumPy is importable, vectorized fast paths are used for the common
formats; the remaining ones fall back to the pure code transparently.
"""

from __future__ import annotations

import struct

try:  # optional accelerator only - everything works without it
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


# ---- quantized block geometry ---------------------------------------------
# dtype -> (block_elements, block_bytes, sub_block_len, affine)
# `sub_block_len` is the run of consecutive elements sharing one effective
# scale (and offset, when `affine`). Available without NumPy so the pure
# backend can slice rows out of raw block bytes.
QUANT_GEOMETRY = {
    "Q8_0": (QK, 34, 32, False),
    "Q4_0": (QK, 18, 32, False),
    "Q4_1": (QK, 20, 32, True),
    "Q5_0": (QK, 22, 32, False),
    "Q5_1": (QK, 24, 32, True),
    "Q4_K": (QK_K, 144, 32, True),
    "Q5_K": (QK_K, 176, 32, True),
    "Q6_K": (QK_K, 210, 16, False),
}


# ---- NumPy fast paths ----------------------------------------------------
#
# Each unpacker decodes raw blocks into the shared compact representation
# used by both `dequantize` and `alpacca.qmatrix.QuantMatrix`:
#   codes int8 (nb, block_elements)  - quant codes in element order
#   d_eff float32 (nb, n_sub)        - effective scale per sub-block
#   m_eff float32 (nb, n_sub) | None - effective offset per sub-block
# so that value = d_eff * code (+ m_eff).

def _np_blocks(data, nb, block_bytes):
    return _np.frombuffer(data, dtype=_np.uint8).reshape(nb, block_bytes)


def _np_f16_col(b, off):
    return b[:, off:off + 2].copy().view(_np.float16).astype(_np.float32)


def _np_unpack_q8_0(b):
    d = _np_f16_col(b, 0)
    q = b[:, 2:34].view(_np.int8).copy()
    return q, d, None


def _np_unpack_q4_0(b):
    d = _np_f16_col(b, 0)
    qs = b[:, 2:18]
    q = _np.empty((b.shape[0], QK), dtype=_np.int8)
    q[:, :16] = (qs & 0x0F).view(_np.int8)
    q[:, 16:] = (qs >> 4).view(_np.int8)
    q -= 8
    return q, d, None


def _np_unpack_q4_1(b):
    d = _np_f16_col(b, 0)
    m = _np_f16_col(b, 2)
    qs = b[:, 4:20]
    q = _np.empty((b.shape[0], QK), dtype=_np.int8)
    q[:, :16] = (qs & 0x0F).view(_np.int8)
    q[:, 16:] = (qs >> 4).view(_np.int8)
    return q, d, m


def _np_high_bits(qh_u32):
    """Per-element 5th bit (already shifted to 0x10) from a u32 mask column."""
    shifts = _np.arange(32, dtype=_np.uint32)
    return (((qh_u32 >> shifts) & 1) << 4).astype(_np.uint8)


def _np_unpack_q5_0(b):
    d = _np_f16_col(b, 0)
    qh = b[:, 2:6].copy().view(_np.uint32)
    hi5 = _np_high_bits(qh)
    qs = b[:, 6:22]
    q = _np.empty((b.shape[0], QK), dtype=_np.int8)
    q[:, :16] = ((qs & 0x0F) | hi5[:, :16]).view(_np.int8)
    q[:, 16:] = ((qs >> 4) | hi5[:, 16:]).view(_np.int8)
    q -= 16
    return q, d, None


def _np_unpack_q5_1(b):
    d = _np_f16_col(b, 0)
    m = _np_f16_col(b, 2)
    qh = b[:, 4:8].copy().view(_np.uint32)
    hi5 = _np_high_bits(qh)
    qs = b[:, 8:24]
    q = _np.empty((b.shape[0], QK), dtype=_np.int8)
    q[:, :16] = ((qs & 0x0F) | hi5[:, :16]).view(_np.int8)
    q[:, 16:] = ((qs >> 4) | hi5[:, 16:]).view(_np.int8)
    return q, d, m


def _np_unpack_k_scales(scales):
    """6-bit packed scale/min pairs of Q4_K/Q5_K -> float32 (nb, 8) each."""
    nb = scales.shape[0]
    sc = _np.empty((nb, 8), dtype=_np.float32)
    mn = _np.empty((nb, 8), dtype=_np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = (scales[:, j] & 63).astype(_np.float32)
            mn[:, j] = (scales[:, j + 4] & 63).astype(_np.float32)
        else:
            sc[:, j] = ((scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)).astype(_np.float32)
            mn[:, j] = ((scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)).astype(_np.float32)
    return sc, mn


def _np_unpack_q4_k(b):
    d = _np_f16_col(b, 0)
    dmin = _np_f16_col(b, 2)
    sc, mn = _np_unpack_k_scales(b[:, 4:16])
    qs = b[:, 16:144]
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    for half in range(4):  # 4 chunks of 32 bytes -> 2 sub-blocks each
        chunk = qs[:, half * 32:(half + 1) * 32]
        q[:, half * 64:half * 64 + 32] = (chunk & 0x0F).view(_np.int8)
        q[:, half * 64 + 32:half * 64 + 64] = (chunk >> 4).view(_np.int8)
    return q, d * sc, -(dmin * mn)


def _np_unpack_q5_k(b):
    d = _np_f16_col(b, 0)
    dmin = _np_f16_col(b, 2)
    sc, mn = _np_unpack_k_scales(b[:, 4:16])
    qh = b[:, 16:48]
    qs = b[:, 48:176]
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    for half in range(4):
        chunk = qs[:, half * 32:(half + 1) * 32]
        j1, j2 = 2 * half, 2 * half + 1
        hb1 = ((qh >> j1) & 1) << 4
        hb2 = ((qh >> j2) & 1) << 4
        q[:, j1 * 32:(j1 + 1) * 32] = ((chunk & 0x0F) | hb1).view(_np.int8)
        q[:, j2 * 32:(j2 + 1) * 32] = ((chunk >> 4) | hb2).view(_np.int8)
    return q, d * sc, -(dmin * mn)


def _np_unpack_q6_k(b):
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    sc = b[:, 192:208].view(_np.int8).astype(_np.float32)  # (nb, 16)
    d = _np_f16_col(b, 208)
    q = _np.empty((b.shape[0], QK_K), dtype=_np.int8)
    for half in range(2):  # two 128-element halves
        qlh = ql[:, half * 64:(half + 1) * 64]
        qhh = qh[:, half * 32:(half + 1) * 32]
        base = half * 128
        q[:, base + 0:base + 32] = (
            ((qlh[:, :32] & 0xF) | (((qhh >> 0) & 3) << 4)).view(_np.int8))
        q[:, base + 32:base + 64] = (
            ((qlh[:, 32:] & 0xF) | (((qhh >> 2) & 3) << 4)).view(_np.int8))
        q[:, base + 64:base + 96] = (
            ((qlh[:, :32] >> 4) | (((qhh >> 4) & 3) << 4)).view(_np.int8))
        q[:, base + 96:base + 128] = (
            ((qlh[:, 32:] >> 4) | (((qhh >> 6) & 3) << 4)).view(_np.int8))
    q -= 32
    return q, d * sc, None


_NP_UNPACKERS = {
    "Q8_0": _np_unpack_q8_0,
    "Q4_0": _np_unpack_q4_0,
    "Q4_1": _np_unpack_q4_1,
    "Q5_0": _np_unpack_q5_0,
    "Q5_1": _np_unpack_q5_1,
    "Q4_K": _np_unpack_q4_k,
    "Q5_K": _np_unpack_q5_k,
    "Q6_K": _np_unpack_q6_k,
}


def np_unpack(data, n: int, dtype: str):
    """Unpack `n` elements of raw GGUF blocks into (codes, d_eff, m_eff).

    codes is int8 (nb, block_elements) in element order; d_eff/m_eff are
    float32 (nb, n_sub) so that value = d_eff * code (+ m_eff) per sub-block
    of QUANT_GEOMETRY[dtype] sub_block_len elements. Requires NumPy.
    """
    if _np is None:
        raise RuntimeError("np_unpack requires NumPy")
    if dtype not in _NP_UNPACKERS:
        raise ValueError(f"np_unpack does not support {dtype}")
    block_n, block_b, _sub, _aff = QUANT_GEOMETRY[dtype]
    if n % block_n:
        raise ValueError(f"{dtype} needs a multiple of {block_n} elements")
    return _NP_UNPACKERS[dtype](_np_blocks(data, n // block_n, block_b))


def _np_assemble(q, d_eff, m_eff, sub_len):
    nb, block_n = q.shape
    out = q.astype(_np.float32).reshape(nb, block_n // sub_len, sub_len)
    out *= d_eff[:, :, None]
    if m_eff is not None:
        out += m_eff[:, :, None]
    return out.reshape(-1)


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
            return _np.frombuffer(data, dtype=_np.float32).copy()
        if dtype == "F16":
            return _np.frombuffer(data, dtype=_np.float16).astype(_np.float32)
        if dtype == "BF16":
            raw = _np.frombuffer(data, dtype=_np.uint16).astype(_np.uint32) << 16
            return raw.view(_np.float32).copy()
        if dtype in _NP_UNPACKERS:
            q, d_eff, m_eff = np_unpack(data, n, dtype)
            return _np_assemble(q, d_eff, m_eff, QUANT_GEOMETRY[dtype][2])
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
