"""Package I: packed (0.820 B/w) vs unpacked (1.070 B/w) Q6_K matvec.

03-RESULTS.md NEGATIVE 1 measured these to a dead heat on machine A
(52.1 vs 52.2 Gw/s) because the 6-bit unpack ALU exactly cancelled the
bandwidth saving at 59-60 GB/s. This re-runs that comparison on a machine
whose sustained bandwidth is different, which is the only thing that can
move the verdict.

The packed kernel keeps the file's own ql/qh planes and unpacks in the
inner loop, folding the -32 centering into an integer min-term
(sc*(raw-32) = sc*raw - 32*sc*bsum16) so the contraction stays in int16
lanes - the same algebra Q4_K's min term uses, and the thing that rescued
vectorization for machine A's attempt.
"""
import os
import sys
import time

import numpy as np
import numba
from numba import njit, prange

sys.path.insert(0, "/home/ubuntu/alpaccaroo")
from alpaccaroo import kernels as K            # noqa: E402
from alpaccaroo import bench as B              # noqa: E402

QK_K = 256


# ---- packed kernel -------------------------------------------------------
#
# Sub-block k of a 256-block covers elements [16k, 16k+16). Mapping k to the
# file's planes (derived from quants._np_q6_k_codes):
#   h = k // 8            which 128-element half
#   s = (k % 8) // 2      which of the half's four 32-element streams
#   off = 16 * (k % 2)    which half of that stream
#   ql column = 64h + 32*(s & 1) + off + j , low nibble if s < 2 else high
#   qh column = 32h + off + j             , shift 2s
@njit(parallel=True, fastmath=True, cache=True)
def _matvec_q6k_packed(ql, qh, sc, dh, lut, xq, ascale, bsums16):
    rows = ql.shape[0]
    nblk = ascale.shape[0]
    out = np.empty(rows, np.float32)
    for r in prange(rows):
        acc = np.float32(0.0)
        for b in range(nblk):
            l0 = b * 128
            h0 = b * 64
            e0 = b * 256
            k0 = b * 16
            t0 = np.int16(sc[r, k0 + 0])
            t1 = np.int16(sc[r, k0 + 1])
            t2 = np.int16(sc[r, k0 + 2])
            t3 = np.int16(sc[r, k0 + 3])
            t4 = np.int16(sc[r, k0 + 4])
            t5 = np.int16(sc[r, k0 + 5])
            t6 = np.int16(sc[r, k0 + 6])
            t7 = np.int16(sc[r, k0 + 7])
            t8 = np.int16(sc[r, k0 + 8])
            t9 = np.int16(sc[r, k0 + 9])
            t10 = np.int16(sc[r, k0 + 10])
            t11 = np.int16(sc[r, k0 + 11])
            t12 = np.int16(sc[r, k0 + 12])
            t13 = np.int16(sc[r, k0 + 13])
            t14 = np.int16(sc[r, k0 + 14])
            t15 = np.int16(sc[r, k0 + 15])
            a0 = np.int32(0)
            a1 = np.int32(0)
            for j in range(16):
                # half 0
                lo0 = ql[r, l0 + j]
                lo1 = ql[r, l0 + 16 + j]
                lo2 = ql[r, l0 + 32 + j]
                lo3 = ql[r, l0 + 48 + j]
                g0 = qh[r, h0 + j]
                g1 = qh[r, h0 + 16 + j]
                a0 = np.int32(
                    a0
                    + np.int32(np.int16(t0 * np.int16((lo0 & np.uint8(15)) | ((g0 & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + j])
                    + np.int32(np.int16(t1 * np.int16((lo1 & np.uint8(15)) | ((g1 & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 16 + j])
                    + np.int32(np.int16(t2 * np.int16((lo2 & np.uint8(15)) | (((g0 >> np.uint8(2)) & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 32 + j])
                    + np.int32(np.int16(t3 * np.int16((lo3 & np.uint8(15)) | (((g1 >> np.uint8(2)) & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 48 + j])
                    + np.int32(np.int16(t4 * np.int16((lo0 >> np.uint8(4)) | (((g0 >> np.uint8(4)) & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 64 + j])
                    + np.int32(np.int16(t5 * np.int16((lo1 >> np.uint8(4)) | (((g1 >> np.uint8(4)) & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 80 + j])
                    + np.int32(np.int16(t6 * np.int16((lo2 >> np.uint8(4)) | ((g0 >> np.uint8(6)) << np.uint8(4))))) * np.int32(xq[e0 + 96 + j])
                    + np.int32(np.int16(t7 * np.int16((lo3 >> np.uint8(4)) | ((g1 >> np.uint8(6)) << np.uint8(4))))) * np.int32(xq[e0 + 112 + j]))
                # half 1
                hi0 = ql[r, l0 + 64 + j]
                hi1 = ql[r, l0 + 80 + j]
                hi2 = ql[r, l0 + 96 + j]
                hi3 = ql[r, l0 + 112 + j]
                g2 = qh[r, h0 + 32 + j]
                g3 = qh[r, h0 + 48 + j]
                a1 = np.int32(
                    a1
                    + np.int32(np.int16(t8 * np.int16((hi0 & np.uint8(15)) | ((g2 & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 128 + j])
                    + np.int32(np.int16(t9 * np.int16((hi1 & np.uint8(15)) | ((g3 & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 144 + j])
                    + np.int32(np.int16(t10 * np.int16((hi2 & np.uint8(15)) | (((g2 >> np.uint8(2)) & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 160 + j])
                    + np.int32(np.int16(t11 * np.int16((hi3 & np.uint8(15)) | (((g3 >> np.uint8(2)) & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 176 + j])
                    + np.int32(np.int16(t12 * np.int16((hi0 >> np.uint8(4)) | (((g2 >> np.uint8(4)) & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 192 + j])
                    + np.int32(np.int16(t13 * np.int16((hi1 >> np.uint8(4)) | (((g3 >> np.uint8(4)) & np.uint8(3)) << np.uint8(4))))) * np.int32(xq[e0 + 208 + j])
                    + np.int32(np.int16(t14 * np.int16((hi2 >> np.uint8(4)) | ((g2 >> np.uint8(6)) << np.uint8(4))))) * np.int32(xq[e0 + 224 + j])
                    + np.int32(np.int16(t15 * np.int16((hi3 >> np.uint8(4)) | ((g3 >> np.uint8(6)) << np.uint8(4))))) * np.int32(xq[e0 + 240 + j]))
            # the -32 centering, folded out of the inner loop
            mi = np.int32(0)
            mi = np.int32(mi + t0 * bsums16[k0 + 0] + t1 * bsums16[k0 + 1]
                          + t2 * bsums16[k0 + 2] + t3 * bsums16[k0 + 3]
                          + t4 * bsums16[k0 + 4] + t5 * bsums16[k0 + 5]
                          + t6 * bsums16[k0 + 6] + t7 * bsums16[k0 + 7]
                          + t8 * bsums16[k0 + 8] + t9 * bsums16[k0 + 9]
                          + t10 * bsums16[k0 + 10] + t11 * bsums16[k0 + 11]
                          + t12 * bsums16[k0 + 12] + t13 * bsums16[k0 + 13]
                          + t14 * bsums16[k0 + 14] + t15 * bsums16[k0 + 15])
            # subtract in int32, not float32: a0+a1 exceeds 2**24 so a
            # float difference would round twice, where the integer form
            # reproduces the shipped kernel's exact block sum and converts
            # once - which makes this bit-identical rather than merely close
            acc += (ascale[b] * lut[dh[r, b]]
                    * np.float32(np.int32(a0 + a1 - np.int32(32) * mi)))
        out[r] = acc
    return out


# ---- fixtures ------------------------------------------------------------

def make_blocks(rows, cols, seed=1234):
    """Random Q6_K raw blocks, plus both storage forms derived from them."""
    rng = np.random.default_rng(seed)
    nblk = cols // QK_K
    raw = rng.integers(0, 256, size=(rows, nblk, 210), dtype=np.uint8)
    # keep the f16 super-scale in a sane range so the reference sum is finite
    d = (rng.random((rows, nblk), dtype=np.float32) * 0.02 + 0.001).astype(np.float16)
    raw[:, :, 208:210] = d.view(np.uint8).reshape(rows, nblk, 2)

    ql = np.ascontiguousarray(raw[:, :, 0:128].reshape(rows, -1))
    qh = np.ascontiguousarray(raw[:, :, 128:192].reshape(rows, -1))
    sc = np.ascontiguousarray(
        raw[:, :, 192:208].reshape(rows, -1).view(np.int8))
    dh = np.ascontiguousarray(d.view(np.uint16))

    # unpacked codes, exactly as quants._np_q6_k_codes builds them
    q = np.empty((rows, nblk, QK_K), dtype=np.int8)
    R = raw
    for half in range(2):
        qlh = R[:, :, half * 64:(half + 1) * 64]
        qhh = R[:, :, 128 + half * 32:128 + (half + 1) * 32]
        base = half * 128
        q[:, :, base + 0:base + 32] = ((qlh[:, :, :32] & 0xF) | (((qhh >> 0) & 3) << 4)).view(np.int8)
        q[:, :, base + 32:base + 64] = ((qlh[:, :, 32:] & 0xF) | (((qhh >> 2) & 3) << 4)).view(np.int8)
        q[:, :, base + 64:base + 96] = ((qlh[:, :, :32] >> 4) | (((qhh >> 4) & 3) << 4)).view(np.int8)
        q[:, :, base + 96:base + 128] = ((qlh[:, :, 32:] >> 4) | (((qhh >> 6) & 3) << 4)).view(np.int8)
    q -= 32
    codes = np.ascontiguousarray(q.reshape(rows, cols))
    return codes, ql, qh, sc, dh


def main():
    shapes = [(256, 2048, "attn_v      Q6_K"),
              (2048, 11008, "ffn_down    Q6_K"),
              (151936, 2048, "output head Q6_K")]
    if len(sys.argv) > 1 and sys.argv[1] == "--small":
        shapes = shapes[:2]

    K._init()
    st = K._state
    lut = st["f16_lut"]
    threads = K.threads()
    print(f"numba threads {numba.get_num_threads()} (pool {threads}), "
          f"rounds via bench.paired_compare\n")

    for rows, cols, label in shapes:
        codes, ql, qh, sc, dh = make_blocks(rows, cols)
        x = (np.random.default_rng(7).standard_normal(cols)
             .astype(np.float32))
        xq, ascale, _bs32 = K.quantize_acts(x)
        bsums16 = xq.reshape(-1, 16).sum(axis=1).astype(np.int32)

        packed_b = ql.nbytes + qh.nbytes + sc.nbytes + dh.nbytes
        unpack_b = codes.nbytes + sc.nbytes + dh.nbytes
        w = rows * cols

        # correctness first: the packed kernel must agree with the shipped one
        ref = K.matvec_q6k_int(codes.reshape(rows, cols // 16, 16), sc, dh,
                               x, pre=(xq, ascale, _bs32))
        got = _matvec_q6k_packed(ql, qh, sc, dh, lut, xq, ascale, bsums16)
        err = float(np.max(np.abs(ref - got)) /
                    max(1e-30, float(np.max(np.abs(ref)))))
        exact = bool(np.array_equal(ref, got))

        c3 = codes.reshape(rows, cols // 16, 16)
        res = B.paired_compare(
            [("unpacked", lambda: K.matvec_q6k_int(c3, sc, dh, x,
                                                   pre=(xq, ascale, _bs32))),
             ("packed", lambda: _matvec_q6k_packed(ql, qh, sc, dh, lut, xq,
                                                   ascale, bsums16))],
            rounds=15, warmup=3)

        u = res["variants"]["unpacked"]["median"]
        p = res["variants"]["packed"]["median"]
        rr = res["ratios"]["packed"]
        ratio = rr["median_vs_baseline"]
        wins = rr["rounds_won"]
        print(f"{label}  {rows}x{cols}  ({w/1e6:.1f} Mw)")
        print(f"  vs shipped kernel         : "
              f"{'BIT-IDENTICAL' if exact else f'rel err {err:.3e}'}")
        print(f"  unpacked {u*1e3:8.3f} ms  {w/u/1e9:7.2f} Gw/s  "
              f"{unpack_b/u/1e9:6.2f} GB/s  ({unpack_b/w:.4f} B/w)")
        print(f"  packed   {p*1e3:8.3f} ms  {w/p/1e9:7.2f} Gw/s  "
              f"{packed_b/p/1e9:6.2f} GB/s  ({packed_b/w:.4f} B/w)")
        print(f"  packed/unpacked median ratio {ratio:.4f}   "
              f"rounds won by packed: {wins}/15")
        print(f"  bytes saved {100*(1-packed_b/unpack_b):.1f}%  ->  "
              f"break-even needs ratio < 1.0\n")
        del codes, ql, qh, sc, dh, c3


if __name__ == "__main__":
    main()
