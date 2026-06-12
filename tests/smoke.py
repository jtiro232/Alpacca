#!/usr/bin/env python3
"""Alpacca offline smoke test - no network, no third-party packages.

Exercises the full cycle against a local mock of the Ollama registry and
the Hugging Face API, with tiny generated GGUFs: pull -> list -> show ->
run (real inference) -> serve (real HTTP API) -> rm, plus engine unit
checks (quant roundtrips, tokenizers, numpy/pure parity).

usage: python3 tests/smoke.py
"""
from __future__ import annotations

import json
import gc
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS
    if ok:
        print(f"ok   {label}")
        PASS += 1
    else:
        print(f"FAIL {label}" + (f"\n     | {detail}" if detail else ""))
        sys.exit(1)


def run_cli(*args, env=None, expect=0) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run([sys.executable, "-m", "alpacca", *args],
                       capture_output=True, text=True, env=e, cwd=str(REPO))
    if expect is not None and r.returncode != expect:
        print(f"FAIL alpacca {' '.join(args)} -> rc={r.returncode}")
        print("     | " + "\n     | ".join((r.stdout + r.stderr).splitlines()[-15:]))
        sys.exit(1)
    return r


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="alpacca-smoke-"))
    server = None
    try:
        # ---- engine unit checks -----------------------------------------
        print("== engine checks ==")
        from alpacca import quants
        vals = [(i % 97) / 7.0 - 6.5 for i in range(512)]
        for fmt, tol in (("Q8_0", 0.06), ("Q4_0", 0.6)):
            packed = (quants.quantize_q8_0(vals) if fmt == "Q8_0"
                      else quants.quantize_q4_0(vals))
            back = quants.dequantize(packed, len(vals), fmt)
            err = max(abs(a - b) for a, b in zip(vals, list(back)))
            check(f"{fmt} quantize/dequantize roundtrip (max err {err:.3f})", err < tol)

        from alpacca import tensor as T

        def q4_k_bytes(n: int) -> bytes:
            out = bytearray()
            for block in range(n // 256):
                d = 0.015625 + (block % 7) * 0.001953125
                dmin = 0.00390625 + (block % 5) * 0.0009765625
                scales = bytes(((block * 17 + i * 29) & 0xFF) for i in range(12))
                qs = bytes(((block * 31 + i * 7) & 0xFF) for i in range(128))
                out += struct.pack("<ee", d, dmin) + scales + qs
            return bytes(out)

        def q5_k_bytes(n: int) -> bytes:
            out = bytearray()
            for block in range(n // 256):
                d = 0.015625 + (block % 7) * 0.001953125
                dmin = 0.00390625 + (block % 5) * 0.0009765625
                scales = bytes(((block * 17 + i * 29) & 0xFF) for i in range(12))
                qh = bytes(((block * 23 + i * 13) & 0xFF) for i in range(32))
                ql = bytes(((block * 31 + i * 7) & 0xFF) for i in range(128))
                out += struct.pack("<ee", d, dmin) + scales + qh + ql
            return bytes(out)

        def q6_k_bytes(n: int) -> bytes:
            out = bytearray()
            for block in range(n // 256):
                ql = bytes(((block * 13 + i * 11) & 0xFF) for i in range(128))
                qh = bytes(((block * 19 + i * 5) & 0xFF) for i in range(64))
                sc = [((block * 7 + i * 9) % 63) - 31 for i in range(16)]
                d = 0.001953125 + (block % 5) * 0.000244140625
                out += ql + qh + struct.pack("<16b", *sc) + struct.pack("<e", d)
            return bytes(out)

        def q4_1_bytes(n: int) -> bytes:
            out = bytearray()
            for block in range(n // 32):
                d = 0.015625 + (block % 7) * 0.001953125
                m = -0.125 + (block % 5) * 0.0625
                qs = bytes(((block * 31 + i * 7) & 0xFF) for i in range(16))
                out += struct.pack("<ee", d, m) + qs
            return bytes(out)

        def q5_0_bytes(n: int) -> bytes:
            out = bytearray()
            for block in range(n // 32):
                d = 0.015625 + (block % 7) * 0.001953125
                qh = bytes(((block * 23 + i * 13) & 0xFF) for i in range(4))
                qs = bytes(((block * 31 + i * 7) & 0xFF) for i in range(16))
                out += struct.pack("<e", d) + qh + qs
            return bytes(out)

        def q5_1_bytes(n: int) -> bytes:
            out = bytearray()
            for block in range(n // 32):
                d = 0.015625 + (block % 7) * 0.001953125
                m = -0.125 + (block % 5) * 0.0625
                qh = bytes(((block * 23 + i * 13) & 0xFF) for i in range(4))
                qs = bytes(((block * 31 + i * 7) & 0xFF) for i in range(16))
                out += struct.pack("<ee", d, m) + qh + qs
            return bytes(out)

        def check_quantized_matvec(fmt: str, rows: int, cols: int) -> None:
            n = rows * cols
            weights = [((i * 37) % 251) / 17.0 - 7.0 for i in range(n)]
            x = [((i * 19) % 67) / 23.0 - 1.4 for i in range(cols)]
            if fmt == "Q8_0":
                packed = quants.quantize_q8_0(weights)
            elif fmt == "Q4_0":
                packed = quants.quantize_q4_0(weights)
            elif fmt == "Q4_1":
                packed = q4_1_bytes(n)
            elif fmt == "Q5_0":
                packed = q5_0_bytes(n)
            elif fmt == "Q5_1":
                packed = q5_1_bytes(n)
            elif fmt == "Q4_K":
                packed = q4_k_bytes(n)
            elif fmt == "Q5_K":
                packed = q5_k_bytes(n)
            else:
                packed = q6_k_bytes(n)
            qmat = T.quantized_matrix(packed, fmt, rows, cols)
            dense = T.matrix(quants.dequantize(packed, n, fmt), rows, cols)
            if T.HAS_NUMPY:
                # break the circular oracle: check the numpy decode against
                # the independent pure spec decoder on the same bytes
                pure_ref = quants._PURE_DECODERS[fmt](packed, n)
                np_deq = quants.dequantize(packed, n, fmt)
                err = max(abs(float(a) - float(b))
                          for a, b in zip(np_deq, pure_ref))
                check(f"{fmt} numpy dequantize matches pure spec decoder (diff {err:.2e})",
                      err < 1e-6)
            qout = T.to_list(T.matvec(qmat, T.vector(x)))
            dout = T.to_list(T.matvec(dense, T.vector(x)))
            err = max(abs(a - b) for a, b in zip(qout, dout))
            check(f"{T.backend_name()} {fmt} quantized matvec matches dense (diff {err:.2e})",
                  err < 1e-3)
            if T.HAS_NUMPY:
                import numpy as np
                X = np.asarray([x, [v * 0.5 - 0.1 for v in x]], dtype=np.float32)
                qbatch = T.matmul_t(X, qmat)
                stacked = np.stack([T.matvec(qmat, row) for row in X], axis=0)
                berr = float(np.max(np.abs(qbatch - stacked)))
                check(f"{fmt} quantized matmul_t matches stacked matvecs (diff {berr:.2e})",
                      berr < 1e-3)
            if T.HAS_NUMPY:
                old_hot_mb = os.environ.get("ALPACCA_HOT_WEIGHT_MB")
                try:
                    os.environ["ALPACCA_HOT_WEIGHT_MB"] = "8"
                    T._reset_hot_cache_state()
                    hot_mat = T.quantized_matrix(packed, fmt, rows, cols)
                    hot = T.to_list(T.matvec(hot_mat, T.vector(x)))
                    herr = max(abs(a - b) for a, b in zip(hot, dout))
                    stats = T.hot_cache_stats()
                    check(f"{fmt} hot-cache matvec matches dense (diff {herr:.2e})",
                          herr < 2e-3 and stats["matrices"] == 1,
                          str(stats))
                    if fmt == "Q8_0":
                        os.environ["ALPACCA_HOT_WEIGHT_MB"] = "0"
                        T.matvec(hot_mat, T.vector(x))
                        stats = T.hot_cache_stats()
                        check("hot-cache budget change clears live matrix",
                              hot_mat._dense_cache is None and
                              stats["matrices"] == 0 and stats["used_bytes"] == 0,
                              str(stats))
                        os.environ["ALPACCA_HOT_WEIGHT_MB"] = "8"
                        T._reset_hot_cache_state()
                        row_hot_mat = T.quantized_matrix(packed, fmt, rows, cols)
                        T.matvec(row_hot_mat, T.vector(x))
                        os.environ["ALPACCA_HOT_WEIGHT_MB"] = "0"
                        T.matrix_row(row_hot_mat, 0)
                        stats = T.hot_cache_stats()
                        check("hot-cache budget change clears live matrix row lookup",
                              row_hot_mat._dense_cache is None and
                              stats["matrices"] == 0 and stats["used_bytes"] == 0,
                              str(stats))
                        os.environ["ALPACCA_HOT_WEIGHT_MB"] = str(
                            (rows * cols * 4 - 1) / (1024 * 1024))
                        T._reset_hot_cache_state()
                        over_hot_mat = T.quantized_matrix(packed, fmt, rows, cols)
                        over_hot = T.to_list(T.matvec(over_hot_mat, T.vector(x)))
                        oherr = max(abs(a - b) for a, b in zip(over_hot, dout))
                        stats = T.hot_cache_stats()
                        check("hot-cache over-budget matrix is skipped",
                              oherr < 2e-3 and over_hot_mat._dense_cache is None and
                              stats["matrices"] == 0 and stats["used_bytes"] == 0,
                              str(stats))
                        os.environ["ALPACCA_HOT_WEIGHT_MB"] = "8"
                        T._reset_hot_cache_state()
                        gc_hot_mat = T.quantized_matrix(packed, fmt, rows, cols)
                        T.matvec(gc_hot_mat, T.vector(x))
                        del gc_hot_mat
                        gc.collect()
                        stats = T.hot_cache_stats()
                        check("hot-cache accounting releases on matrix GC",
                              stats["matrices"] == 0 and stats["used_bytes"] == 0,
                              str(stats))
                        # late env: budget set only after the matrix already
                        # served a matvec without any budget configured
                        os.environ.pop("ALPACCA_HOT_WEIGHT_MB", None)
                        T._reset_hot_cache_state()
                        late_mat = T.quantized_matrix(packed, fmt, rows, cols)
                        T.matvec(late_mat, T.vector(x))
                        check("hot-cache absent env builds no cache",
                              late_mat._dense_cache is None,
                              str(T.hot_cache_stats()))
                        os.environ["ALPACCA_HOT_WEIGHT_MB"] = "8"
                        late = T.to_list(T.matvec(late_mat, T.vector(x)))
                        lerr = max(abs(a - b) for a, b in zip(late, dout))
                        stats = T.hot_cache_stats()
                        check("hot-cache late env var is picked up "
                              f"(diff {lerr:.2e})",
                              late_mat._dense_cache is not None and
                              stats["matrices"] == 1 and lerr < 2e-3,
                              str(stats))
                        os.environ["ALPACCA_HOT_WEIGHT_MB"] = "0"
                        T.matvec(late_mat, T.vector(x))
                        stats = T.hot_cache_stats()
                        check("hot-cache late env budget zero clears cache",
                              late_mat._dense_cache is None and
                              stats["matrices"] == 0 and stats["used_bytes"] == 0,
                              str(stats))
                        # absurd budget: mb*1048576 overflows float; the
                        # guard parses it to 0, so no cache and no crash
                        os.environ["ALPACCA_HOT_WEIGHT_MB"] = "1e308"
                        T._reset_hot_cache_state()
                        inert_mat = T.quantized_matrix(packed, fmt, rows, cols)
                        inert_out = T.to_list(T.matvec(inert_mat, T.vector(x)))
                        ierr = max(abs(a - b)
                                   for a, b in zip(inert_out, dout))
                        stats = T.hot_cache_stats()
                        check("hot-cache absurd budget is inert "
                              f"(diff {ierr:.2e})",
                              ierr < 2e-3 and stats["matrices"] == 0 and
                              inert_mat._dense_cache is None,
                              str(stats))
                finally:
                    if old_hot_mb is None:
                        os.environ.pop("ALPACCA_HOT_WEIGHT_MB", None)
                    else:
                        os.environ["ALPACCA_HOT_WEIGHT_MB"] = old_hot_mb
                    T._reset_hot_cache_state()
            row = min(3, rows - 1)
            qrow = T.to_list(T.matrix_row(qmat, row))
            drow = T.to_list(T.matrix_row(dense, row))
            rerr = max(abs(a - b) for a, b in zip(qrow, drow))
            check(f"{T.backend_name()} {fmt} quantized row lookup matches dense (diff {rerr:.2e})",
                  rerr < 1e-5)
            idx = [0, rows - 1, min(1, rows - 1)]
            qrows = [T.to_list(r) for r in T.matrix_rows(qmat, idx)]
            drows = [T.to_list(T.matrix_row(dense, i)) for i in idx]
            gerr = max(abs(a - b)
                       for qr, dr in zip(qrows, drows)
                       for a, b in zip(qr, dr))
            check(f"{T.backend_name()} {fmt} quantized row gather matches dense (diff {gerr:.2e})",
                  gerr < 1e-5)
            try:
                T.matrix_rows(qmat, [rows])
                oob_raises = False
            except IndexError:
                oob_raises = True
            check(f"{fmt} row gather rejects out-of-range index", oob_raises)
            try:
                T.matrix_rows(qmat, [-1])
                neg_raises = False
            except IndexError:
                neg_raises = True
            check(f"{fmt} row gather rejects negative index", neg_raises)
            if T.HAS_NUMPY:
                nbytes = qmat.storage_nbytes()
                dense_bytes = rows * cols * 4
                check(f"{fmt} stays quantized in RAM "
                      f"({nbytes}B vs {dense_bytes}B dense)",
                      qmat.data is None and qmat._q.dtype.name == "int8" and
                      nbytes * 3 <= dense_bytes + 4096,
                      f"nbytes={nbytes}")

        check_quantized_matvec("Q8_0", 5, 64)
        check_quantized_matvec("Q4_0", 5, 64)
        check_quantized_matvec("Q4_1", 5, 64)
        check_quantized_matvec("Q5_0", 5, 64)
        check_quantized_matvec("Q5_1", 5, 64)
        check_quantized_matvec("Q4_K", 3, 512)
        check_quantized_matvec("Q5_K", 3, 512)
        check_quantized_matvec("Q6_K", 3, 512)
        if T.HAS_NUMPY:
            import numpy as np
            # large matrix exercises the einsum matvec kernel (small ones
            # take the batched-matmul kernel)
            big_rows, big_cols = 4096, 288
            big_n = big_rows * big_cols
            big_vals = [((i * 37) % 251) / 251.0 - 0.5 for i in range(big_n)]
            big_packed = quants.quantize_q8_0(big_vals)
            big_q = T.quantized_matrix(big_packed, "Q8_0", big_rows, big_cols)
            big_dense = T.matrix(quants.dequantize(big_packed, big_n, "Q8_0"),
                                 big_rows, big_cols)
            xb = T.vector([((i * 19) % 67) / 23.0 - 1.4 for i in range(big_cols)])
            big_err = float(np.max(np.abs(T.matvec(big_q, xb) -
                                          T.matvec(big_dense, xb))))
            check(f"Q8_0 large-matrix quantized matvec matches dense (diff {big_err:.2e})",
                  big_rows * big_cols >= 1 << 20 and big_err < 1e-3)
            # cover the >=1M-element einsum kernel branch for an affine
            # format (Q4_K, exercises m_eff) and the 16-wide-sub-block Q6_K
            for big_fmt, brows, bcols in (("Q4_K", 2112, 512),
                                          ("Q6_K", 2112, 512)):
                bn = brows * bcols
                bpacked = (q4_k_bytes(bn) if big_fmt == "Q4_K"
                           else q6_k_bytes(bn))
                bq = T.quantized_matrix(bpacked, big_fmt, brows, bcols)
                bdense = T.matrix(quants.dequantize(bpacked, bn, big_fmt),
                                  brows, bcols)
                xb2 = T.vector([((i * 19) % 67) / 23.0 - 1.4
                                for i in range(bcols)])
                berr2 = float(np.max(np.abs(T.matvec(bq, xb2) -
                                            T.matvec(bdense, xb2))))
                check(f"{big_fmt} large-matrix quantized matvec matches dense "
                      f"(diff {berr2:.2e})",
                      brows * bcols >= 1 << 20 and berr2 < 1e-3)
                X2 = np.stack([np.asarray(xb2),
                               np.asarray(xb2) * 0.5 - 0.1], axis=0)
                bbatch = T.matmul_t(X2, bq)
                bstacked = np.stack([T.matvec(bq, row) for row in X2], axis=0)
                bberr = float(np.max(np.abs(bbatch - bstacked)))
                check(f"{big_fmt} large-matrix matmul_t matches stacked "
                      f"matvecs (diff {bberr:.2e})",
                      bberr < 1e-3)
        dense = T.matrix([1.0, -2.0, 0.5, 3.0, 4.0, -1.0], 2, 3)
        dy = T.to_list(T.matvec(dense, T.vector([2.0, -1.0, 4.0])))
        check("dense matvec fallback still works",
              max(abs(a - b) for a, b in zip(dy, [6.0, -2.0])) < 1e-6)

        from alpacca import kernels as AK
        try:
            import numba as _numba
            has_pinned_numba = (T.HAS_NUMPY and
                                _numba.__version__ == AK.NUMBA_PIN)
        except Exception:
            has_pinned_numba = False
        if has_pinned_numba:
            check("alpacca kernels activate on the pinned numba",
                  AK.available(), AK.status())
            r = subprocess.run(
                [sys.executable, "-c",
                 "import os; os.environ['ALPACCA_KERNELS'] = '0'\n"
                 "import sys; sys.path.insert(0, '.')\n"
                 "from alpacca import kernels\n"
                 "assert not kernels.available()\n"
                 "print('off-ok')"],
                capture_output=True, text=True, cwd=str(REPO))
            check("ALPACCA_KERNELS=0 disables the kernels",
                  "off-ok" in r.stdout, r.stdout + r.stderr)
        else:
            check("alpacca kernels stay inactive without the pinned numba",
                  not AK.available(), AK.status())

        if T.HAS_NUMPY:
            import numpy as np
            from types import SimpleNamespace
            from alpacca.model import Model
            dummy = Model.__new__(Model)
            dummy.hp = SimpleNamespace(n_head=4, n_kv=2, head_dim=3)
            q = np.asarray([((i * 7) % 19) / 11.0 - 0.8 for i in range(12)],
                           dtype=np.float32).reshape(4, 3)
            K = np.asarray([((i * 5) % 23) / 13.0 - 0.7 for i in range(30)],
                           dtype=np.float32).reshape(5, 2, 3)
            V = np.asarray([((i * 3) % 17) / 9.0 - 0.6 for i in range(30)],
                           dtype=np.float32).reshape(5, 2, 3)
            group = 2
            inv_sqrt = 0.5773502691896258
            fast = dummy._attention_np(q, K, V, group, inv_sqrt)
            slow = np.empty((4, 3), dtype=np.float32)
            for hh in range(4):
                kvh = hh // group
                scores = K[:, kvh, :] @ q[hh] * inv_sqrt
                scores -= scores.max()
                w = np.exp(scores)
                w /= w.sum()
                slow[hh] = w @ V[:, kvh, :]
            aerr = float(np.max(np.abs(fast - slow)))
            check(f"numpy grouped attention matches per-head loop (diff {aerr:.2e})",
                  aerr < 1e-6)

        from alpacca.tokenizer import pretokenize
        toks = pretokenize("Hello there, world! It's 2026...\n  indented")
        check("BPE pretokenizer splits text", "".join(toks) == "Hello there, world! It's 2026...\n  indented",
              str(toks))

        from alpacca.cli import _auto_dense_budget_mb, _available_ram_mb
        check("auto dense budget formula spends what is left after reserves",
              _auto_dense_budget_mb(16000.0, 600.0) ==
              int(0.85 * (16000.0 - 1.2 * 600.0 - 2048.0)))
        check("auto dense budget formula floors at zero on tight RAM",
              _auto_dense_budget_mb(2500.0, 600.0) == 0)
        check("auto dense budget reserve scales with requested context",
              _auto_dense_budget_mb(16000.0, 600.0, 16384) ==
              int(0.85 * (16000.0 - 1.2 * 600.0 - 2048.0 * 4.0)))
        from alpacca.cli import _cgroup_limit_remaining_mb
        v = _cgroup_limit_remaining_mb()
        check("cgroup limit detection returns a sane value or None",
              v is None or v >= 0, str(v))
        detected_ram = _available_ram_mb()
        check("available-RAM detection returns a sane value or None",
              detected_ram is None or detected_ram > 0,
              str(detected_ram))

        from alpacca.pull import _hf_choose, _hf_collect_parts
        hf_files = [
            {"path": "toy-Q4_K_M-00001-of-00002.gguf", "size": 1, "sha256": ""},
            {"path": "toy-Q4_K_M-00002-of-00002.gguf", "size": 1, "sha256": ""},
            {"path": "toy-Q4_0.gguf", "size": 1, "sha256": ""},
        ]
        chosen = _hf_choose(hf_files, "")
        check("HF picker prefers single-file GGUF", chosen["path"] == "toy-Q4_0.gguf")
        split = _hf_choose(hf_files[:2], "")
        check("HF split GGUF parts are detected",
              len(_hf_collect_parts(hf_files[:2], split)) == 2)

        # ---- tiny models -------------------------------------------------
        print("== building tiny models (own GGUF writer) ==")
        srv = tmp / "srv"
        srv.mkdir()
        mk = REPO / "tests" / "make_tiny_model.py"
        for dtype, name in (("F32", "model.gguf"), ("F16", "tiny-f16.gguf"),
                            ("Q2_K", "tiny-q2k.gguf"),
                            ("Q8_0", "tiny-q8.gguf"),
                            ("Q4_0", "tiny-q4.gguf"), ("Q4_1", "tiny-q41.gguf"),
                            ("Q5_0", "tiny-q50.gguf"), ("Q5_1", "tiny-q51.gguf"),
                            ("Q4_K", "tiny-q4k.gguf"),
                            ("Q5_K", "tiny-q5k.gguf"), ("Q6_K", "tiny-q6k.gguf")):
            r = subprocess.run([sys.executable, str(mk), str(srv / name), dtype],
                               capture_output=True, text=True)
            check(f"write tiny {dtype} model", r.returncode == 0, r.stderr)

        from alpacca.model import Model, auto_budget_fit_mb
        fit = auto_budget_fit_mb(str(srv / "tiny-q4.gguf"))
        # tiny-q4 (untied): eligible = 6 ffn x 32768B + 8 attn x 16384B +
        # output 307*64*4B = 406272 B; fixed = embd residual + KV + 512 base
        check("auto budget fit sizing matches the header arithmetic",
              fit is not None and
              abs(fit[0] * 1024 * 1024 - 406272) < 1.0 and
              512.0 < fit[1] < 514.0,
              str(fit))
        check("auto budget fit sizing is None for unreadable models",
              auto_budget_fit_mb(str(srv / "does-not-exist.gguf")) is None)
        for fmt, name in (("Q8_0", "tiny-q8.gguf"), ("Q4_0", "tiny-q4.gguf"),
                          ("Q4_1", "tiny-q41.gguf"), ("Q5_0", "tiny-q50.gguf"),
                          ("Q5_1", "tiny-q51.gguf"),
                          ("Q4_K", "tiny-q4k.gguf"), ("Q5_K", "tiny-q5k.gguf"),
                          ("Q6_K", "tiny-q6k.gguf")):
            qm = Model.load(str(srv / name), progress=False)
            desc = qm.describe()
            if T.HAS_NUMPY:
                check(f"load tiny {fmt} keeps quantized weights",
                      f"weights quantized {fmt}" in desc, desc)
            else:
                check(f"load tiny {fmt} falls back to dense without NumPy",
                      "weights dense" in desc and f"dense fallback {fmt}" in desc,
                      desc)
            check(f"load tiny {fmt} reports backend",
                  f"backend {T.backend_name()}" in desc, desc)
            if fmt in ("Q4_1", "Q5_0", "Q5_1", "Q4_K", "Q5_K", "Q6_K"):
                ids = qm.tok.encode("hi") or [qm.tok.bos_id]
                logits = T.to_list(qm.prefill(ids[:1]))
                check(f"tiny {fmt} forward runs",
                      len(logits) == qm.hp.n_vocab and all(v == v for v in logits[:8]))

        f16_model = Model.load(str(srv / "tiny-f16.gguf"), progress=False)
        f16_desc = f16_model.describe()
        check("load tiny F16 stays dense float32 (no quantized wrap)",
              f16_model.weight_storage["dense"] == 16 and
              not f16_model.weight_storage["quantized"] and
              "weights dense" in f16_desc,
              f"{f16_model.weight_storage} {f16_desc}")
        f16_ids = f16_model.tok.encode("hi") or [f16_model.tok.bos_id]
        f16_logits = T.to_list(f16_model.prefill(f16_ids[:1]))
        check("tiny F16 forward runs",
              len(f16_logits) == f16_model.hp.n_vocab and
              all(v == v for v in f16_logits[:8]))

        if not T.HAS_NUMPY:
            old_budget = os.environ.get("ALPACCA_DENSE_WEIGHT_MB")
            try:
                os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "64"
                inert = Model.load(str(srv / "tiny-q4.gguf"), progress=False)
                check("dense budget is inert on the pure backend",
                      inert.weight_storage["densified"] == [] and
                      inert.weight_storage["dense"] == 16,
                      str(inert.weight_storage))
            finally:
                if old_budget is None:
                    os.environ.pop("ALPACCA_DENSE_WEIGHT_MB", None)
                else:
                    os.environ["ALPACCA_DENSE_WEIGHT_MB"] = old_budget

        if T.HAS_NUMPY:
            def greedy_trace(model, prompt: str, steps: int) -> tuple[list[int], list[list[float]]]:
                ids = model.tok.encode(prompt, add_bos=True)
                logits = model.prefill(ids)
                tokens: list[int] = []
                trace = [T.to_list(logits)]
                for _ in range(steps):
                    tid = T.argmax(logits)
                    tokens.append(tid)
                    if model.tok.is_eog(tid):
                        break
                    logits = model.forward(tid)
                    trace.append(T.to_list(logits))
                return tokens, trace

            for fmt, name in (("Q8_0", "tiny-q8.gguf"), ("Q4_0", "tiny-q4.gguf"),
                              ("Q4_1", "tiny-q41.gguf"), ("Q5_0", "tiny-q50.gguf"),
                              ("Q5_1", "tiny-q51.gguf")):
                old_f32 = os.environ.get("ALPACCA_F32")
                try:
                    os.environ.pop("ALPACCA_F32", None)
                    q_model = Model.load(str(srv / name), progress=False)
                    os.environ["ALPACCA_F32"] = "1"
                    d_model = Model.load(str(srv / name), progress=False)
                finally:
                    if old_f32 is None:
                        os.environ.pop("ALPACCA_F32", None)
                    else:
                        os.environ["ALPACCA_F32"] = old_f32
                qtoks, qlogits = greedy_trace(q_model, "hello world", 6)
                dtoks, dlogits = greedy_trace(d_model, "hello world", 6)
                logit_diff = max(abs(a - b)
                                 for qa, da in zip(qlogits, dlogits)
                                 for a, b in zip(qa, da))
                check(f"{fmt} quantized vs ALPACCA_F32 greedy generation/logits parity",
                      q_model.weight_storage["quantized"] == {fmt: 16} and
                      d_model.weight_storage["fallback"] == {fmt: 16} and
                      qtoks == dtoks and len(qlogits) == len(dlogits) and
                      logit_diff < 1e-2,
                      f"quant={qtoks} dense={dtoks} diff={logit_diff:.2e} "
                      f"qstore={q_model.weight_storage} dstore={d_model.weight_storage}")

            old_budget = os.environ.get("ALPACCA_DENSE_WEIGHT_MB")
            try:
                # tiny-q4: 6 ffn matrices of 32768B + 4 attn q/output of
                # 16384B = exactly 0.25 MiB; wk/wv/embd/output must stay
                # quantized
                os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "0.25"
                hybrid = Model.load(str(srv / "tiny-q4.gguf"), progress=False)
                expect_dense = {f"blk.{i}.{role}.weight"
                                for i in range(2)
                                for role in ("ffn_gate", "ffn_up", "ffn_down",
                                             "attn_q", "attn_output")}
                check("dense budget densifies FFN tier then attn q/output",
                      set(hybrid.weight_storage["densified"]) == expect_dense and
                      hybrid.weight_storage["quantized"] == {"Q4_0": 6} and
                      hybrid.weight_storage["densified_bytes"] == 262144 and
                      not hybrid.weight_storage["fallback"],
                      str(hybrid.weight_storage))
                check("describe reports the dense budget",
                      "dense budget 10 matrices" in hybrid.describe(),
                      hybrid.describe())
                os.environ.pop("ALPACCA_DENSE_WEIGHT_MB", None)
                qfull = Model.load(str(srv / "tiny-q4.gguf"), progress=False)
                htoks, hlogits = greedy_trace(hybrid, "hello world", 6)
                ftoks, flogits = greedy_trace(qfull, "hello world", 6)
                hdiff = max(abs(a - b)
                            for ha, fa in zip(hlogits, flogits)
                            for a, b in zip(ha, fa))
                check(f"dense-budget hybrid matches quantized generation (diff {hdiff:.2e})",
                      htoks == ftoks and len(hlogits) == len(flogits) and
                      hdiff < 1e-2,
                      f"hybrid={htoks} quant={ftoks}")
                os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "64"
                roomy = Model.load(str(srv / "tiny-q4.gguf"), progress=False)
                check("dense budget never densifies an untied token embedding",
                      len(roomy.weight_storage["densified"]) == 15 and
                      "token_embd.weight" not in roomy.weight_storage["densified"] and
                      roomy.weight_storage["quantized"] == {"Q4_0": 1},
                      str(roomy.weight_storage))
                os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "not-a-number"
                ignored = Model.load(str(srv / "tiny-q4.gguf"), progress=False)
                check("invalid dense budget is ignored",
                      ignored.weight_storage["quantized"] == {"Q4_0": 16} and
                      not ignored.weight_storage["densified"],
                      str(ignored.weight_storage))
                r = subprocess.run(
                    [sys.executable, str(REPO / "tests" / "make_bench_model.py"),
                     str(srv / "tied-bench.gguf"), "Q4_0", "--vocab", "320",
                     "--embd", "64", "--heads", "4", "--layers", "2",
                     "--ff", "128"],
                    capture_output=True, text=True)
                check("write tiny tied bench model", r.returncode == 0, r.stderr)
                os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "64"
                tied = Model.load(str(srv / "tied-bench.gguf"), progress=False)
                check("dense budget densifies a tied embedding as the output matrix",
                      "token_embd.weight" in tied.weight_storage["densified"] and
                      len(tied.weight_storage["densified"]) == 15 and
                      not tied.weight_storage["quantized"],
                      str(tied.weight_storage))
                # ALPACCA_F32 wins over the dense budget: everything is
                # already dense fallback, so no densify pass runs
                prior_f32 = os.environ.get("ALPACCA_F32")
                os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "64"
                os.environ["ALPACCA_F32"] = "1"
                try:
                    f32_wins = Model.load(str(srv / "tiny-q4.gguf"),
                                          progress=False)
                finally:
                    if prior_f32 is None:
                        os.environ.pop("ALPACCA_F32", None)
                    else:
                        os.environ["ALPACCA_F32"] = prior_f32
                check("ALPACCA_F32 wins over the dense budget",
                      f32_wins.weight_storage["densified"] == [] and
                      f32_wins.weight_storage["fallback"] == {"Q4_0": 16},
                      str(f32_wins.weight_storage))
                # absurd budget: mb*1048576 overflows float; the guard
                # parses it to 0 so the load stays fully quantized
                os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "1e308"
                absurd = Model.load(str(srv / "tiny-q4.gguf"), progress=False)
                check("absurd dense budget is inert",
                      absurd.weight_storage["densified"] == [] and
                      absurd.weight_storage["quantized"] == {"Q4_0": 16},
                      str(absurd.weight_storage))
                # first-fit spillover on a GQA untied shape: tier-2
                # attn_q/attn_output (16384B) do not fit, but a smaller
                # tier-3 attn_k matrix (8192B) still does
                r = subprocess.run(
                    [sys.executable,
                     str(REPO / "tests" / "make_bench_model.py"),
                     str(srv / "gqa-bench.gguf"), "Q4_0", "--vocab", "320",
                     "--embd", "64", "--heads", "4", "--kv", "2",
                     "--layers", "2", "--ff", "128", "--untied"],
                    capture_output=True, text=True)
                check("write tiny GQA untied bench model", r.returncode == 0,
                      r.stderr)
                # budget int(0.20*1048576)=209715B: six FFN mats use
                # 6*32768=196608B, then blk.0.attn_k lands at 204800B
                os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "0.20"
                spill = Model.load(str(srv / "gqa-bench.gguf"),
                                   progress=False)
                expect_spill = {f"blk.{i}.{role}.weight"
                                for i in range(2)
                                for role in ("ffn_gate", "ffn_up",
                                             "ffn_down")}
                expect_spill.add("blk.0.attn_k.weight")
                check("dense budget spills past oversized tier to smaller matrices",
                      set(spill.weight_storage["densified"]) == expect_spill and
                      spill.weight_storage["densified_bytes"] == 204800 and
                      len(spill.weight_storage["densified"]) == 7,
                      str(spill.weight_storage))
            finally:
                if old_budget is None:
                    os.environ.pop("ALPACCA_DENSE_WEIGHT_MB", None)
                else:
                    os.environ["ALPACCA_DENSE_WEIGHT_MB"] = old_budget

            old_f32 = os.environ.get("ALPACCA_F32")
            try:
                os.environ["ALPACCA_F32"] = "1"
                f32_forced = Model.load(str(srv / "tiny-q4.gguf"), progress=False)
                check("ALPACCA_F32 forces dense quantized matrix loading",
                      not f32_forced.weight_storage["quantized"] and
                      f32_forced.weight_storage["fallback"] == {"Q4_0": 16},
                      str(f32_forced.weight_storage))
            finally:
                if old_f32 is None:
                    os.environ.pop("ALPACCA_F32", None)
                else:
                    os.environ["ALPACCA_F32"] = old_f32

        q2 = Model.load(str(srv / "tiny-q2k.gguf"), progress=False)
        q2_desc = q2.describe()
        check("load tiny Q2_K falls back to dense matrices",
              q2.weight_storage["dense"] == 16 and
              q2.weight_storage["fallback"] == {"Q2_K": 16} and
              "Q2_K" not in q2.weight_storage["quantized"],
              str(q2.weight_storage))
        check("describe reports dense fallback Q2_K",
              "weights dense" in q2_desc and
              "dense fallback Q2_K" in q2_desc,
              q2_desc)

        (srv / "params.json").write_text(
            '{"temperature": 0.7, "num_ctx": 256, "top_k": 30}')
        (srv / "system.txt").write_text("You are a smoke test.")
        (srv / "license.txt").write_text("test license - MIT")

        # numpy/pure parity (when numpy is present)
        from alpacca import tensor
        if tensor.HAS_NUMPY:
            code = (
                "import json\n"
                "from alpacca.model import Model\n"
                "import alpacca.tensor as T\n"
                f"m = Model.load({str(srv / 'model.gguf')!r}, progress=False)\n"
                "l = m.prefill(m.tok.encode('hello world'))\n"
                "print(json.dumps(T.to_list(l)[:8]))\n")
            a = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, cwd=str(REPO))
            env = dict(os.environ, ALPACCA_PURE="1")
            b = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, cwd=str(REPO), env=env)
            la, lb = json.loads(a.stdout), json.loads(b.stdout)
            diff = max(abs(x - y) for x, y in zip(la, lb))
            check(f"numpy vs pure-python logits agree (diff {diff:.1e})", diff < 1e-3)

            import numpy as np
            ids = list(range(1, 45))
            seq = Model.load(str(srv / "model.gguf"), progress=False)
            bat = Model.load(str(srv / "model.gguf"), progress=False)
            slogits = None
            for tid in ids:
                slogits = seq.forward(tid)
            blogits = bat.forward_batch(ids)
            ldiff = float(np.max(np.abs(slogits - blogits)))
            kdiff = max(
                float(np.max(np.abs(seq.cache_k[li][:len(ids)] -
                                    bat.cache_k[li][:len(ids)])))
                for li in range(seq.hp.n_layer)
            )
            vdiff = max(
                float(np.max(np.abs(seq.cache_v[li][:len(ids)] -
                                    bat.cache_v[li][:len(ids)])))
                for li in range(seq.hp.n_layer)
            )
            check(f"forward_batch matches sequential logits (diff {ldiff:.2e})",
                  ldiff < 1e-4)
            check(f"forward_batch writes matching KV cache (K {kdiff:.2e}, V {vdiff:.2e})",
                  kdiff < 1e-5 and vdiff < 1e-5)

            pref = Model.load(str(srv / "model.gguf"), progress=False)
            base = ids[:12]
            longer = base + ids[12:24]
            diverged = base + ids[30:38]
            pref.prefill(base)
            check("prefill counter records initial prompt",
                  pref.last_prefill_forwarded == len(base),
                  str(pref.last_prefill_forwarded))
            pref.prefill(longer)
            check("prefill forwards only shared-prefix suffix",
                  pref.last_prefill_forwarded == len(longer) - len(base),
                  str(pref.last_prefill_forwarded))
            pref.prefill(longer)
            check("prefill regenerate re-forwards last token",
                  pref.last_prefill_forwarded == 1,
                  str(pref.last_prefill_forwarded))
            dlogits = pref.prefill(diverged)
            fresh = Model.load(str(srv / "model.gguf"), progress=False)
            flogits = fresh.prefill(diverged)
            pdiff = float(np.max(np.abs(dlogits - flogits)))
            check("prefill truncation keeps divergent conversation correct",
                  pdiff < 1e-4 and pref.cached_ids == diverged and
                  pref.n_past == len(diverged),
                  f"diff={pdiff:.2e} cached={pref.cached_ids} n_past={pref.n_past}")

        # ---- mock registry ------------------------------------------------
        print("== mock registry (offline) ==")
        port_file = tmp / "port"
        server = subprocess.Popen(
            [sys.executable, str(REPO / "tests" / "mock_registry.py"),
             str(srv), str(port_file)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(300):
            if port_file.exists() and port_file.read_text().strip():
                break
            if server.poll() is not None:
                check("mock registry starts", False, "server died")
            time.sleep(0.1)
        port = port_file.read_text().strip()
        check("mock registry starts", bool(port))

        env = {
            "ALPACCA_HOME": str(tmp / "home"),
            "ALPACCA_OLLAMA_REGISTRY": f"http://127.0.0.1:{port}",
            "ALPACCA_HF_ENDPOINT": f"http://127.0.0.1:{port}",
        }

        # ---- CLI: ollama path --------------------------------------------
        print("== ollama-registry path ==")
        run_cli("pull", "tiny", env=env)
        check("pull tiny", True)
        r = run_cli("pull", "tiny", env=env)
        check("pull is idempotent", "already installed" in r.stderr)
        r = run_cli("list", env=env)
        check("list shows tiny", any(line.startswith("tiny ") for line in r.stdout.splitlines()))
        r = run_cli("show", "tiny", env=env)
        check("show has params", '"temperature"' in r.stdout)
        check("show has system", "smoke test" in r.stdout)
        check("show has digest", '"digest": "sha256:' in r.stdout)
        lic = tmp / "home" / "models" / "ollama" / "library" / "tiny" / "latest" / "license.txt"
        check("license stored", lic.exists())

        # ---- inference ----------------------------------------------------
        print("== inference (the engine itself) ==")
        r = run_cli("run", "tiny", "hello there", "-n", "8", "--seed", "1", env=env)
        check("run one-shot generates", "tokens," in r.stderr)
        model_path = tmp / "home" / "models" / "ollama" / "library" / "tiny" / "latest" / "model.gguf"
        r = run_cli("run", str(model_path), "hi", "-n", "4", "--seed", "1", env=env)
        check("run by file path", "tokens," in r.stderr)
        r = run_cli("run", "tiny", "hi", "-n", "4", "--seed", "1",
                    env={**env, "ALPACCA_PURE": "1"})
        check("run with pure-python backend", "tokens," in r.stderr)
        r = run_cli("tokenize", "-m", "tiny", "-p", "hello", env=env)
        check("tokenize via model name", "\u2581hello" in r.stdout or "hello" in r.stdout)

        # ---- hugging-face path (incl. -GGUF fallback) ---------------------
        print("== hugging-face path ==")
        r = run_cli("pull", "hf:test/tiny", env=env)   # falls back to tiny-GGUF
        check("pull hf:test/tiny (fallback)", "trying test/tiny-GGUF" in r.stderr)
        run_cli("pull", "hf:test/tiny-GGUF:tiny-q4.gguf", env=env)
        check("pull hf exact file", True)
        r = run_cli("list", env=env)
        check("list shows hf models", "hf:test/tiny" in r.stdout)
        r = run_cli("run", "hf:test/tiny-GGUF:tiny-q4.gguf", "hi", "-n", "4",
                    "--seed", "1", env=env)
        check("run Q4_0 hf model", "tokens," in r.stderr)
        if T.HAS_NUMPY:
            if has_pinned_numba:
                check("run keeps weights quantized when kernels are active",
                      "alpacca-kernels active" in r.stderr and
                      "auto dense-weight budget:" not in r.stderr and
                      "weights quantized Q4_0 (16 matrices)" in r.stderr,
                      r.stderr[-500:])
            else:
                check("run defaults to an auto dense-weight budget",
                      "auto dense-weight budget:" in r.stderr and
                      "dense budget 15 matrices" in r.stderr,
                      r.stderr[-500:])
            r0 = run_cli("run", "hf:test/tiny-GGUF:tiny-q4.gguf", "hi", "-n", "4",
                         "--seed", "1",
                         env={**env, "ALPACCA_DENSE_WEIGHT_MB": "0"})
            check("ALPACCA_DENSE_WEIGHT_MB=0 keeps the CLI fully quantized",
                  "auto dense-weight budget:" not in r0.stderr and
                  "weights quantized Q4_0 (16 matrices)" in r0.stderr,
                  r0.stderr[-500:])
        else:
            check("pure backend skips the auto dense-weight budget",
                  "auto dense-weight budget:" not in r.stderr,
                  r.stderr[-500:])

        # ---- serve ---------------------------------------------------------
        print("== serve (OpenAI-compatible API) ==")
        sp = subprocess.Popen(
            [sys.executable, "-m", "alpacca", "serve", "tiny", "--port", "0"],
            env={**os.environ, **env}, cwd=str(REPO),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        import re
        sport = None
        deadline = time.time() + 60
        line = ""
        while time.time() < deadline:
            line = sp.stderr.readline()
            m = re.search(r"http://[^:]+:(\d+)", line)
            if m:
                sport = m.group(1)
                break
            if sp.poll() is not None:
                break
        check("serve starts", sport is not None, line)
        base = f"http://127.0.0.1:{sport}"
        try:
            with urllib.request.urlopen(base + "/health", timeout=10) as resp:
                check("serve /health", json.loads(resp.read())["status"] == "ok")
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": 6, "seed": 1}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
            check("serve /v1/chat/completions",
                  body["object"] == "chat.completion" and
                  "content" in body["choices"][0]["message"])
            req = urllib.request.Request(
                base + "/completion",
                data=json.dumps({"prompt": "hello", "n_predict": 4, "seed": 1}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
            check("serve /completion", "content" in body)
            req = urllib.request.Request(
                base + "/completion",
                data=json.dumps({"prompt": "", "n_predict": 4,
                                 "temperature": None, "stop": "\n"}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
            check("serve /completion empty prompt", "content" in body)
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}],
                                 "top_k": "not-an-int"}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=60)
                bad_param_is_400 = False
            except urllib.error.HTTPError as e:
                bad_param_is_400 = e.code == 400
            check("serve rejects invalid params", bad_param_is_400)
        finally:
            sp.terminate()
            sp.wait(timeout=10)

        import threading
        from alpacca import chat
        from alpacca.sample import SamplerParams
        from alpacca.serve import serve as serve_in_process
        api_model = Model.load(str(model_path), progress=False)
        ready = threading.Event()
        port_box: list[int] = []

        def ready_callback(port: int) -> None:
            port_box.append(port)
            ready.set()

        th = threading.Thread(
            target=serve_in_process,
            args=(api_model, "tiny"),
            kwargs={"host": "127.0.0.1", "port": 0,
                    "defaults": SamplerParams(temperature=0.0, seed=1),
                    "ready_callback": ready_callback},
            daemon=True,
        )
        th.start()
        check("in-process serve starts for prefix-cache check", ready.wait(10))
        ibase = f"http://127.0.0.1:{port_box[0]}"
        first_messages = [{"role": "user", "content": "hi"}]
        second_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
        ]
        req = urllib.request.Request(
            ibase + "/v1/chat/completions",
            data=json.dumps({"messages": first_messages, "max_tokens": 2,
                             "temperature": 0.0}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            json.loads(resp.read())
        fmt = chat.ChatFormat(api_model, chat.detect_format(api_model.metadata))
        second_prompt = fmt.render(second_messages)
        before_second = list(api_model.cached_ids)
        lcp = 0
        for a, b in zip(second_prompt, before_second):
            if a != b:
                break
            lcp += 1
        expected_forwarded = len(second_prompt) - lcp
        if expected_forwarded == 0:
            expected_forwarded = 1
        req = urllib.request.Request(
            ibase + "/v1/chat/completions",
            data=json.dumps({"messages": second_messages, "max_tokens": 2,
                             "temperature": 0.0}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            json.loads(resp.read())
        check("serve chat completions forwards exact shared-prefix suffix",
              api_model.last_prefill_forwarded == expected_forwarded,
              f"forwarded={api_model.last_prefill_forwarded} "
              f"expected={expected_forwarded} prompt={len(second_prompt)} lcp={lcp}")

        # ---- removal -------------------------------------------------------
        print("== removal ==")
        run_cli("rm", "tiny", env=env)
        check("rm tiny", True)
        run_cli("rm", "hf:test/tiny", "hf:test/tiny-GGUF:tiny-q4.gguf", env=env)
        check("rm hf models", True)
        r = run_cli("list", env=env)
        check("store empty after rm", "no models installed" in r.stdout)

        print(f"\nall {PASS} checks passed")
    finally:
        if server is not None:
            server.terminate()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
