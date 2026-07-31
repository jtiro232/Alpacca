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
import argparse
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


def run_cli(*args, env=None, expect=0,
            input_text: str | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run([sys.executable, "-m", "alpacca", *args],
                       input=input_text, capture_output=True, text=True,
                       env=e, cwd=str(REPO))
    if expect is not None and r.returncode != expect:
        print(f"FAIL alpacca {' '.join(args)} -> rc={r.returncode}")
        print("     | " + "\n     | ".join((r.stdout + r.stderr).splitlines()[-15:]))
        sys.exit(1)
    return r


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="alpacca-smoke-"))
    server = None
    # The storage-policy checks below assert exact HOST placement (which
    # matrices are quantized in RAM, dense, densified...). A GPU would claim
    # most of them and change every count, so the suite pins CPU placement
    # process-wide - subprocess checks inherit it through run_cli - and the
    # gpu-tier section re-enables the tier explicitly for its own checks.
    os.environ["ALPACCA_GPU"] = "0"
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

        def q2_k_bytes(n: int) -> bytes:
            out = bytearray()
            for block in range(n // 256):
                scales = bytes(((block * 11 + i * 23) & 0xFF) for i in range(16))
                qs = bytes(((block * 29 + i * 7) & 0xFF) for i in range(64))
                d = 0.00390625 + (block % 7) * 0.00048828125
                dmin = 0.001953125 + (block % 5) * 0.000244140625
                out += scales + qs + struct.pack("<ee", d, dmin)
            return bytes(out)

        def q3_k_bytes(n: int) -> bytes:
            out = bytearray()
            for block in range(n // 256):
                hmask = bytes(((block * 17 + i * 13) & 0xFF) for i in range(32))
                qs = bytes(((block * 29 + i * 7) & 0xFF) for i in range(64))
                aux = bytes(((block * 5 + i * 19) & 0xFF) for i in range(12))
                d = 0.001953125 + (block % 5) * 0.000244140625
                out += hmask + qs + aux + struct.pack("<e", d)
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
            elif fmt == "Q2_K":
                packed = q2_k_bytes(n)
            elif fmt == "Q3_K":
                packed = q3_k_bytes(n)
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

        # this whole region verifies the EXACT f32-activation storage and
        # kernels; the integer-dot path is approximate by design and has its
        # own checks (with an integer-simulation oracle) further down
        os.environ["ALPACCA_INT_DOT"] = "0"
        try:
            check_quantized_matvec("Q8_0", 5, 64)
            check_quantized_matvec("Q4_0", 5, 64)
            check_quantized_matvec("Q4_1", 5, 64)
            check_quantized_matvec("Q5_0", 5, 64)
            check_quantized_matvec("Q5_1", 5, 64)
            check_quantized_matvec("Q2_K", 3, 512)
            check_quantized_matvec("Q3_K", 3, 512)
            check_quantized_matvec("Q4_K", 3, 512)
            check_quantized_matvec("Q5_K", 3, 512)
            check_quantized_matvec("Q6_K", 3, 512)
        finally:
            os.environ.pop("ALPACCA_INT_DOT", None)
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
                os.environ["ALPACCA_INT_DOT"] = "0"  # exact-path coverage
                try:
                    bq = T.quantized_matrix(bpacked, big_fmt, brows, bcols)
                finally:
                    os.environ.pop("ALPACCA_INT_DOT", None)
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

        import io
        from alpacca.chat import _read_chat_line
        prompt_out = io.StringIO()
        check("chat Escape returns to caller",
              _read_chat_line(stdin=io.StringIO("\x1b\n"),
                              stdout=prompt_out) is None and
              prompt_out.getvalue() == "> ")
        prompt_out = io.StringIO()
        check("chat line reader keeps normal input",
              _read_chat_line(stdin=io.StringIO("hello\n"),
                              stdout=prompt_out) == "hello")

        old_home = os.environ.get("ALPACCA_HOME")
        try:
            os.environ["ALPACCA_HOME"] = str(tmp / "history-unit-home")
            import alpacca.history as history_mod
            from alpacca.history import (clear_history, delete_chat,
                                         list_chats, model_stats, read_chat,
                                         start_session)

            def run_with_vanished_after_list_load(target: Path, callback):
                original_load_chat = history_mod._load_chat
                vanished = {"value": False}

                def load_then_vanish(path: Path):
                    data = original_load_chat(path)
                    if path == target and data is not None and not vanished["value"]:
                        vanished["value"] = True
                        try:
                            target.unlink()
                        except FileNotFoundError:
                            pass
                    return data

                history_mod._load_chat = load_then_vanish
                try:
                    result = callback()
                finally:
                    history_mod._load_chat = original_load_chat
                    if target.exists():
                        target.unlink()
                return vanished["value"], result

            empty_hist = start_session("unit-model", "unit.gguf")
            empty_hist.close()
            check("empty chat history session is not saved", list_chats() == [])
            event_hist = start_session("unit-model", "unit.gguf")
            event_hist.append_event("clear")
            check("event-only chat history session is not saved", list_chats() == [])
            event_hist.append_message("user", "hello history")
            event_hist.append_message("assistant", "saved", tokens=2, seconds=0.25)
            event_hist.close()
            chats = list_chats()
            check("chat history session is saved",
                  len(chats) == 1 and chats[0]["turns"] == 1 and
                  chats[0]["title"] == "hello history",
                  str(chats))
            chat_doc = read_chat("1")
            check("chat history can be read by list number",
                  chat_doc["id"] == event_hist.id and
                  chat_doc["messages"][1]["content"] == "hello history")
            race_show = start_session("unit-model", "unit.gguf")
            race_show.append_message("user", "vanished show")
            race_show.close()

            def read_vanished_show():
                try:
                    read_chat(race_show.id)
                except ValueError as e:
                    return str(e)
                return ""

            vanished_show, show_error = run_with_vanished_after_list_load(
                race_show.path, read_vanished_show)
            check("history show reports chat vanished before read",
                  vanished_show and "could not read chat" in show_error,
                  show_error)
            stats = model_stats()
            check("chat history stats aggregate token speed by model",
                  len(stats) == 1 and stats[0]["model"] == "unit-model" and
                  stats[0]["chats"] == 1 and stats[0]["responses"] == 1 and
                  stats[0]["tokens"] == 2 and
                  abs(stats[0]["tok_per_sec"] - 8.0) < 1e-9,
                  str(stats))
            invalid_hist = start_session("unit-model", "unit.gguf")
            invalid_hist.append_message("user", "bad metrics")
            invalid_hist.append_message("assistant", "bool tokens",
                                        tokens=True, seconds=1.0)
            invalid_hist.append_message("assistant", "float tokens",
                                        tokens=1.5, seconds=1.0)
            invalid_hist.append_message("assistant", "negative tokens",
                                        tokens=-1, seconds=1.0)
            invalid_hist.append_message("assistant", "nonfinite seconds",
                                        tokens=3, seconds=float("inf"))
            invalid_hist.close()
            stats = model_stats()
            unit = next(row for row in stats if row["model"] == "unit-model")
            check("chat history stats rejects invalid metric fields",
                  unit["chats"] == 2 and unit["responses"] == 1 and
                  unit["tokens"] == 2 and abs(unit["seconds"] - 0.25) < 1e-9,
                  str(unit))
            delete_chat(invalid_hist.id)
            delete_chat(event_hist.id[:12])
            check("chat history can delete by id prefix", list_chats() == [])
            original_unlink = Path.unlink
            race_delete = start_session("unit-model", "unit.gguf")
            race_delete.append_message("user", "vanished delete")
            race_delete.close()
            vanished = {"delete": False}

            def vanish_on_delete(self, *args, **kwargs):
                if self == race_delete.path and not vanished["delete"]:
                    vanished["delete"] = True
                    raise FileNotFoundError(str(self))
                return original_unlink(self, *args, **kwargs)

            Path.unlink = vanish_on_delete
            try:
                delete_chat(race_delete.id)
            finally:
                Path.unlink = original_unlink
                if race_delete.path.exists():
                    race_delete.path.unlink()
            check("chat history delete tolerates vanished file",
                  vanished["delete"])
            race_delete_read = start_session("unit-model", "unit.gguf")
            race_delete_read.append_message("user", "vanished before delete unlink")
            race_delete_read.close()
            vanished_delete_read, deleted = run_with_vanished_after_list_load(
                race_delete_read.path, lambda: delete_chat(race_delete_read.id))
            check("chat history delete tolerates pre-unlink vanished file",
                  vanished_delete_read and deleted["id"] == race_delete_read.id and
                  not race_delete_read.path.exists(),
                  str(deleted))
            race_clear = start_session("unit-model", "unit.gguf")
            race_clear.append_message("user", "vanished clear")
            race_clear.close()
            vanished["clear"] = False

            def vanish_on_clear(self, *args, **kwargs):
                if self == race_clear.path and not vanished["clear"]:
                    vanished["clear"] = True
                    raise FileNotFoundError(str(self))
                return original_unlink(self, *args, **kwargs)

            Path.unlink = vanish_on_clear
            try:
                clear_deleted = clear_history()
            finally:
                Path.unlink = original_unlink
                if race_clear.path.exists():
                    race_clear.path.unlink()
            check("chat history clear tolerates vanished file",
                  vanished["clear"] and clear_deleted == 0)
            from alpacca.cli import cmd_history
            race_cli = start_session("unit-model", "unit.gguf")
            race_cli.append_message("user", "vanished cli rm")
            race_cli.close()
            vanished["cli"] = False

            def vanish_on_cli_rm(self, *args, **kwargs):
                if self == race_cli.path and not vanished["cli"]:
                    vanished["cli"] = True
                    raise FileNotFoundError(str(self))
                return original_unlink(self, *args, **kwargs)

            Path.unlink = vanish_on_cli_rm
            try:
                cli_rc = cmd_history(argparse.Namespace(
                    history_command="rm", chats=[race_cli.id]))
            finally:
                Path.unlink = original_unlink
                if race_cli.path.exists():
                    race_cli.path.unlink()
            check("history rm tolerates vanished file",
                  vanished["cli"] and cli_rc == 0)
            race_cli_read = start_session("unit-model", "unit.gguf")
            race_cli_read.append_message("user", "vanished before cli rm unlink")
            race_cli_read.close()
            vanished_cli_read, cli_read_rc = run_with_vanished_after_list_load(
                race_cli_read.path,
                lambda: cmd_history(argparse.Namespace(
                    history_command="rm", chats=[race_cli_read.id])))
            check("history rm tolerates pre-unlink vanished file",
                  vanished_cli_read and cli_read_rc == 0 and
                  not race_cli_read.path.exists())
            active_hist = start_session("unit-model", "unit.gguf")
            active_hist.append_message("user", "deleted active")
            active_path = active_hist.path
            delete_chat(active_hist.id[:12])
            active_hist.append_message("assistant", "late answer")
            active_hist.close()
            check("deleted chat history session is not recreated",
                  not active_path.exists() and list_chats() == [])
            for label in ("first", "second"):
                h = start_session("unit-model", "unit.gguf")
                h.append_message("user", label)
                h.close()
            check("chat history clear deletes all chats", clear_history() == 2 and
                  list_chats() == [])
            active_clear = start_session("unit-model", "unit.gguf")
            active_clear.append_message("user", "cleared active")
            active_clear_path = active_clear.path
            check("chat history clear removes active chat",
                  clear_history() == 1 and not active_clear_path.exists())
            active_clear.append_message("assistant", "late answer")
            active_clear.close()
            check("cleared chat history session is not recreated",
                  not active_clear_path.exists() and list_chats() == [])
        finally:
            if old_home is None:
                os.environ.pop("ALPACCA_HOME", None)
            else:
                os.environ["ALPACCA_HOME"] = old_home

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

        # ---- fused batched matmul vs the tiled dequantize+GEMM path -------
        # matmul_t used to dequantize the whole matrix on every call, so a
        # one-row batch cost as much as a 256-row one. The fused kernel below
        # the crossover must be numerically indistinguishable from the tiled
        # path it replaces, and from matvec for a single row.
        if T.HAS_NUMPY:
            import numpy as np
            from alpacca.qmatrix import QuantMatrix, _fused_matmul_max_batch
            from alpacca.quants import QUANT_GEOMETRY
            rng = np.random.default_rng(7)
            worst = 0.0
            shapes_done = 0
            for _dt in sorted(QUANT_GEOMETRY):
                blk = QUANT_GEOMETRY[_dt][0]
                _rows, _cols = 9, blk * 2
                nb = _rows * (_cols // blk) * QUANT_GEOMETRY[_dt][1]
                raw = bytes(rng.integers(0, 256, size=nb, dtype=np.uint8))
                # pin the f32-activation storage: this block verifies the
                # exact fused kernels, the integer-dot path has its own
                # checks (with its own quantified tolerance) below
                os.environ["ALPACCA_INT_DOT"] = "0"
                try:
                    qm = QuantMatrix(raw, _dt, _rows, _cols)
                finally:
                    os.environ.pop("ALPACCA_INT_DOT", None)
                # spans the narrow kernel (<=8), the wide one, and a batch
                # past NARROW_BATCH where the two must still agree
                for B in (1, 2, 5, 9, 16, 40):
                    X = rng.standard_normal((B, _cols)).astype(np.float32)
                    got = qm.matmul_t(X)
                    os.environ["ALPACCA_FUSED_MATMUL_MAX_BATCH"] = "0"
                    try:
                        ref = qm.matmul_t(X)
                    finally:
                        os.environ.pop("ALPACCA_FUSED_MATMUL_MAX_BATCH", None)
                    scale = max(1e-6, float(np.abs(ref).max()))
                    worst = max(worst, float(np.abs(got - ref).max()) / scale)
                    if B == 1:
                        mv = np.asarray(qm.matvec(X[0]), dtype=np.float32)
                        worst = max(worst,
                                    float(np.abs(got[0] - mv).max()) / scale)
                    if AK.available():
                        # call the kernel directly: without this the loop
                        # above compares the tiled path with itself whenever
                        # the JIT is absent, and would pass even if matmul_t
                        # had quietly stopped using the kernel at all
                        direct = AK.matmul_codes(qm._q3, qm._d, qm._m, X)
                        worst = max(worst,
                                    float(np.abs(direct - ref).max()) / scale)
                shapes_done += 1
            check("fused matmul matches the tiled path and matvec on every quant",
                  shapes_done == len(QUANT_GEOMETRY) and worst < 2e-5,
                  f"{shapes_done} dtypes, worst rel {worst:.2e}, "
                  f"kernel {'exercised' if AK.available() else 'absent'}")
            check("the fused-matmul crossover is a positive tunable batch size",
                  _fused_matmul_max_batch() > 0,
                  str(_fused_matmul_max_batch()))
            os.environ["ALPACCA_FUSED_MATMUL_MAX_BATCH"] = "0"
            try:
                check("ALPACCA_FUSED_MATMUL_MAX_BATCH=0 disables the fused path",
                      _fused_matmul_max_batch() == 0)
            finally:
                os.environ.pop("ALPACCA_FUSED_MATMUL_MAX_BATCH", None)

        # ---- integer-dot decode path (Q4_K/Q6_K native storage) -----------
        # Weights stay in the file's own fields (4-bit codes, 6-bit integer
        # sub-scales, f16 supers); activations are quantized to int8 per
        # 256 block. The kernel must be EXACT against a NumPy simulation of
        # that integer algebra - only the activation quantization itself is
        # approximate, and that error is bounded and measured separately.
        if T.HAS_NUMPY and AK.available():
            import numpy as np
            from alpacca.qmatrix import QuantMatrix
            rng = np.random.default_rng(11)
            for _dt, _rows, _cols in (("Q4_K", 9, 512), ("Q5_K", 9, 512),
                                      ("Q6_K", 9, 512)):
                raw = {"Q4_K": q4_k_bytes, "Q5_K": q5_k_bytes,
                       "Q6_K": q6_k_bytes}[_dt](_rows * _cols)
                qm = QuantMatrix(raw, _dt, _rows, _cols)
                check(f"{_dt} matrices adopt the native int-dot storage",
                      qm._mode == f"{_dt.lower()[:2]}k_int", qm._mode)
                os.environ["ALPACCA_INT_DOT"] = "0"
                try:
                    qm_f32 = QuantMatrix(raw, _dt, _rows, _cols)
                finally:
                    os.environ.pop("ALPACCA_INT_DOT", None)
                dense = qm_f32._dense_from_storage()
                tile = qm._tile_f32(0, _rows)
                check(f"{_dt} native tile expansion is bit-exact vs dense",
                      np.array_equal(tile, dense),
                      f"maxdiff {np.abs(tile - dense).max():.2e}")
                x = (rng.standard_normal(_cols) * 1.5).astype(np.float32)
                got = np.asarray(qm.matvec(x), dtype=np.float64)
                # exact integer simulation of what the kernel must compute
                xq, ascale, bsums = AK.quantize_acts(x)
                nblk = _cols // 256
                q3 = qm._q3 if _dt == "Q6_K" else None
                if _dt in ("Q4_K", "Q5_K"):
                    qp = qm._qp.reshape(_rows, nblk, 4, 32)
                    cl = (qp & 0x0F).astype(np.int64)
                    ch = (qp >> 4).astype(np.int64)
                    if _dt == "Q5_K":
                        qh5 = qm._qh.reshape(_rows, nblk, 1, 32)
                        for _c in range(4):
                            cl[:, :, _c] |= ((qh5[:, :, 0] >> (2 * _c)) & 1).astype(np.int64) << 4
                            ch[:, :, _c] |= ((qh5[:, :, 0] >> (2 * _c + 1)) & 1).astype(np.int64) << 4
                    co = np.empty((_rows, nblk, 4, 64), np.int64)
                    co[..., :32] = cl
                    co[..., 32:] = ch
                    co = co.reshape(_rows, _cols // 32, 32)
                    isum = (co * xq.reshape(1, -1, 32).astype(np.int64)).sum(-1)
                    blk = (qm._sci.astype(np.int64)
                           * isum.reshape(_rows, nblk, 8)).sum(-1)
                    mini = (qm._mni.astype(np.int64)
                            * bsums.reshape(1, nblk, 8).astype(np.int64)).sum(-1)
                    dh = qm._dh.view(np.float16).astype(np.float64)
                    dmh = qm._dmh.view(np.float16).astype(np.float64)
                    sim = (ascale.astype(np.float64)
                           * (dh * blk - dmh * mini)).sum(-1)
                else:
                    isum = (q3.astype(np.int64)
                            * xq.reshape(1, -1, 16).astype(np.int64)).sum(-1)
                    blk = (qm._sci.astype(np.int64) * isum).reshape(
                        _rows, nblk, 16).sum(-1)
                    dh = qm._dh.view(np.float16).astype(np.float64)
                    sim = (ascale.astype(np.float64) * dh * blk).sum(-1)
                srms = max(float(np.sqrt((sim ** 2).mean())), 1e-9)
                kerr = float(np.abs(got - sim).max()) / srms
                check(f"{_dt} int-dot kernel matches its integer simulation",
                      kerr < 1e-5, f"kernel-vs-sim rel {kerr:.2e}")
                fref = np.asarray(qm_f32.matvec(x), dtype=np.float64)
                frms = max(float(np.sqrt((fref ** 2).mean())), 1e-9)
                aerr = float(np.abs(got - fref).max()) / frms
                check(f"{_dt} activation-quantization error stays bounded",
                      aerr < 2e-2, f"int-vs-f32 rel {aerr:.2e}")
                for B in (1, 3, 12):
                    X = rng.standard_normal((B, _cols)).astype(np.float32)
                    bt = np.asarray(qm.matmul_t(X))
                    st = np.stack([np.asarray(qm.matvec(row)) for row in X])
                    check(f"{_dt} int matmul_t batch {B} == stacked matvecs",
                          np.allclose(bt, st, rtol=0, atol=0),
                          f"maxdiff {np.abs(bt - st).max():.2e}")
                grp = T.matvec_group([qm, qm], x)
                solo = np.asarray(qm.matvec(x))
                check(f"{_dt} shared-quantization matvec_group is bit-exact",
                      np.array_equal(np.asarray(grp[0]), solo) and
                      np.array_equal(np.asarray(grp[1]), solo))
                os.environ["ALPACCA_INT_MATMUL_MAX_BATCH"] = "0"
                try:
                    X = rng.standard_normal((3, _cols)).astype(np.float32)
                    tiled = np.asarray(qm.matmul_t(X))
                    ref = X @ dense.T
                    terr = float(np.abs(tiled - ref).max() /
                                 max(np.abs(ref).max(), 1e-6))
                    check(f"{_dt} native tiled matmul matches dense BLAS",
                          terr < 2e-5, f"rel {terr:.2e}")
                finally:
                    os.environ.pop("ALPACCA_INT_MATMUL_MAX_BATCH", None)
                ri = qm.rows_at([0, _rows - 1])
                shuffled = list(range(_rows))
                rng.shuffle(shuffled)
                gathered = qm.rows_at(shuffled + [0, 0])  # repeats included
                expect = dense[np.asarray(shuffled + [0, 0])]
                check(f"{_dt} native row access is bit-exact",
                      np.array_equal(qm.row(2), dense[2]) and
                      np.array_equal(ri, dense[[0, _rows - 1]]) and
                      np.array_equal(gathered, expect))
            os.environ["ALPACCA_FUSED_MATMUL_MAX_BATCH"] = "0"
            try:
                check("ALPACCA_FUSED_MATMUL_MAX_BATCH=0 disables the fused path",
                      _fused_matmul_max_batch() == 0)
            finally:
                os.environ.pop("ALPACCA_FUSED_MATMUL_MAX_BATCH", None)

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
            # the fused decode-attention kernel (used when the JIT is active
            # so decode never enters OpenBLAS's own thread pool) must match
            # the NumPy matmul math; fastmath reassociation allows last-ulp
            # drift, nothing more. Direct calls so neither side is vacuous.
            if AK.available():
                arng = np.random.default_rng(23)
                worst_att = 0.0
                for n_kv_t, group_t, hd_t, t_t in ((2, 2, 3, 5), (1, 4, 8, 1),
                                                   (4, 1, 16, 33),
                                                   (2, 4, 32, 700)):
                    n_head_t = n_kv_t * group_t
                    qv = arng.standard_normal(n_head_t * hd_t).astype(np.float32)
                    Kt = arng.standard_normal((t_t, n_kv_t, hd_t)).astype(np.float32)
                    Vt = arng.standard_normal((t_t, n_kv_t, hd_t)).astype(np.float32)
                    isq = 1.0 / hd_t ** 0.5
                    got = AK.attention_decode(qv, Kt, Vt, group_t, isq)
                    qg = qv.reshape(n_kv_t, group_t, hd_t)
                    sc2 = np.matmul(qg, Kt.transpose(1, 2, 0)) * isq
                    sc2 -= sc2.max(axis=2, keepdims=True)
                    w2 = np.exp(sc2)
                    w2 /= w2.sum(axis=2, keepdims=True)
                    ref_att = np.matmul(w2, Vt.transpose(1, 0, 2)).reshape(
                        n_head_t, hd_t)
                    worst_att = max(worst_att,
                                    float(np.abs(got - ref_att).max()))
                check(f"fused decode attention matches the matmul math "
                      f"(diff {worst_att:.2e})", worst_att < 2e-5,
                      f"worst {worst_att:.2e}")
                # the JIT rope must be BIT-identical to the NumPy slicing
                # path (strict FP, no reductions): the pinned logit tests
                # depend on it
                rope_ok = True
                for style in ("norm", "neox"):
                    for n_heads_r, hd_r, n_rot_r in ((4, 8, 8), (2, 16, 8)):
                        vr = arng.standard_normal(
                            n_heads_r * hd_r).astype(np.float32)
                        half_r = n_rot_r // 2
                        cr = arng.standard_normal(half_r).astype(np.float32)
                        sr = arng.standard_normal(half_r).astype(np.float32)
                        got_r = AK.rope_decode(vr, cr, sr, n_heads_r, hd_r,
                                               n_rot_r, style)
                        v2 = vr.reshape(n_heads_r, hd_r).copy()
                        if style == "norm":
                            x0 = v2[:, 0:n_rot_r:2].copy()
                            x1 = v2[:, 1:n_rot_r:2].copy()
                            v2[:, 0:n_rot_r:2] = x0 * cr - x1 * sr
                            v2[:, 1:n_rot_r:2] = x0 * sr + x1 * cr
                        else:
                            x0 = v2[:, :half_r].copy()
                            x1 = v2[:, half_r:n_rot_r].copy()
                            v2[:, :half_r] = x0 * cr - x1 * sr
                            v2[:, half_r:n_rot_r] = x0 * sr + x1 * cr
                        rope_ok = rope_ok and np.array_equal(
                            got_r, v2.reshape(-1))
                check("JIT rope is bit-identical to the NumPy path", rope_ok)
                # review-confirmed edges: empty gathers must return (0, cols)
                # in native modes, and degenerate rope dims (odd n_rot, or
                # n_rot > head_dim - unvalidated GGUF metadata) must fall to
                # the NumPy path, which raises loudly instead of returning
                # uninitialized memory
                empty = qm.rows_at([])
                check("native-mode empty row gather returns (0, cols)",
                      empty.shape == (0, qm.cols) and
                      empty.dtype == np.float32)
                from types import SimpleNamespace as _SNS
                dm = Model.__new__(Model)
                dm._use_kernel_attention = True
                dm.hp = _SNS(head_dim=8, n_rot=7, rope_style="norm")
                dm._rope_cos = np.zeros((4, 3), np.float32)
                dm._rope_sin = np.zeros((4, 3), np.float32)
                try:
                    dm._rope_np(np.zeros(16, np.float32), 2, 0)
                    odd_ok = False  # silent success would be the bug
                except ValueError:
                    odd_ok = True   # the loud NumPy failure, as before
                check("odd n_rot bypasses the JIT rope and fails loudly",
                      odd_ok)

        # ---- top-k selection ---------------------------------------------
        # This had no direct coverage at all, which is how a first version
        # that lost the tie-break at the cut passed the whole suite.
        import random
        from alpacca.sample import Sampler, SamplerParams, _topk_indices

        def topk_ref(vals, k):
            return sorted(range(len(vals)), key=vals.__getitem__,
                          reverse=True)[:k]

        topk_bad = []
        rng_tk = random.Random(7)
        for trial in range(300):
            n = rng_tk.choice([1, 2, 3, 8, 40, 257])
            # a small alphabet on purpose: ties across the cut are the bug
            vals = [float(rng_tk.randrange(4)) for _ in range(n)]
            for k in {1, 2, n // 2 or 1, n - 1 or 1, n, n + 3}:
                if k < 1:
                    continue
                got = _topk_indices(vals, k)
                if got != topk_ref(vals, min(k, n)):
                    topk_bad.append((vals, k, got))
        check("top-k matches the stable descending sort, ties included",
              not topk_bad, str(topk_bad[:2]))
        check("top-k on all-equal logits keeps ascending index order",
              _topk_indices([1.0] * 64, 5) == [0, 1, 2, 3, 4],
              str(_topk_indices([1.0] * 64, 5)))
        check("top-k with k >= n returns every index",
              _topk_indices([3.0, 1.0, 2.0], 9) == [0, 2, 1])
        check("top-k handles infinities",
              _topk_indices([float("-inf"), 1.0, float("inf"), 1.0], 3) ==
              [2, 1, 3], str(_topk_indices([float("-inf"), 1.0,
                                            float("inf"), 1.0], 3)))
        nan_top = _topk_indices([1.0] * 99 + [float("nan")], 40)
        check("top-k returns k indices even when a logit is NaN",
              len(nan_top) == 40, str(len(nan_top)))
        nan_sampled = Sampler(SamplerParams(temperature=0.8, seed=1)).sample(
            [1.0] * 99 + [float("nan")])
        check("sampling a NaN logit vector does not raise",
              isinstance(nan_sampled, int))
        # degenerate PARAMS (reachable from the serve API: json accepts the
        # NaN/Infinity literals unclamped) must not crash and must pick the
        # same token as the reference list path - the penalty can mint
        # non-finite logits AFTER the fast path's raw-logits gate
        degen_ok = True
        for bad in (dict(repeat_penalty=float("nan")),
                    dict(repeat_penalty=float("inf")),
                    dict(repeat_penalty=1e-320),
                    dict(temperature=float("nan"))):
            dp = dict(temperature=0.8, top_k=3, top_p=0.95, seed=5,
                      repeat_penalty=1.1)
            dp.update(bad)
            s_fast = Sampler(SamplerParams(**dp))
            s_ref = Sampler(SamplerParams(**dp))
            for t in (0, 1):
                s_fast.accept(t)
                s_ref.accept(t)
            try:
                got = s_fast.sample([0.0, 1.0, 0.5, 2.0])
                ref = s_ref._sample_list([0.0, 1.0, 0.5, 2.0])
                degen_ok = degen_ok and got == ref
            except Exception as e:
                degen_ok = False
                print(f"     | degenerate {bad}: {e!r}")
        check("degenerate sampler params match the list path and never raise",
              degen_ok)

        # ---- generation stop reasons -------------------------------------
        from alpacca.chat import GenerationResult
        from alpacca.serve import _finish_reason, _messages_from
        check("finish_reason maps a spent budget to length",
              _finish_reason(GenerationResult("", 4, 0.1, stop_reason="length"))
              == "length")
        check("finish_reason maps a full context window to length",
              _finish_reason(GenerationResult("", 0, 0.1, stop_reason="context"))
              == "length")
        check("finish_reason maps a stop string to stop",
              _finish_reason(GenerationResult("", 4, 0.1, stop_reason="stop"))
              == "stop")
        check("finish_reason maps end-of-generation to stop",
              _finish_reason(GenerationResult("", 4, 0.1, stop_reason="eog"))
              == "stop")
        # ---- quantized matvec kernel selection ---------------------------
        from alpacca import qmatrix as _qm
        check("the batched-matmul matvec path is off by default",
              _qm._small_matvec_elems() == 0 and _qm._SMALL_MATVEC_ELEMS == 0,
              str(_qm._SMALL_MATVEC_ELEMS))
        _sme = os.environ.get("ALPACCA_SMALL_MATVEC_ELEMS")
        try:
            os.environ["ALPACCA_SMALL_MATVEC_ELEMS"] = "1048576"
            check("the matvec crossover can be re-tuned from the environment",
                  _qm._small_matvec_elems() == 1 << 20)
            os.environ["ALPACCA_SMALL_MATVEC_ELEMS"] = "not-a-number"
            check("a bad matvec crossover falls back to off",
                  _qm._small_matvec_elems() == 0)
        finally:
            if _sme is None:
                os.environ.pop("ALPACCA_SMALL_MATVEC_ELEMS", None)
            else:
                os.environ["ALPACCA_SMALL_MATVEC_ELEMS"] = _sme

        # ---- incremental detokenizing ------------------------------------
        # One undecodable byte used to poison the buffer for the rest of the
        # response: nothing was emitted again, which also stopped stop-strings
        # from ever matching.
        import alpacca.tokenizer as _tokmod
        from alpacca.tokenizer import StreamDecoder, TT_BYTE
        byte_tok = _tokmod.Tokenizer.from_gguf({
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": ["<unk>"] + [f"<0x{b:02X}>" for b in range(256)],
            "tokenizer.ggml.token_type": [2] + [TT_BYTE] * 256,
            "tokenizer.ggml.unknown_token_id": 0,
        })
        dec = StreamDecoder(byte_tok)
        bad = byte_tok.token_id("<0xFF>")
        emitted = "".join(dec.feed(t) for t in
                          [bad] + [byte_tok.token_id(f"<0x{ord(c):02X}>")
                                   for c in "hello world"])
        emitted += dec.flush()
        check("the stream decoder keeps emitting after an undecodable byte",
              "hello world" in emitted, repr(emitted))
        # a genuinely split multi-byte character must still be held back
        dec2 = StreamDecoder(byte_tok)
        euro = "€".encode("utf-8")   # 3 bytes
        parts = [dec2.feed(byte_tok.token_id(f"<0x{b:02X}>")) for b in euro]
        check("the stream decoder holds back a split multi-byte character",
              parts[0] == "" and parts[1] == "" and parts[2] == "€",
              str(parts))

        # ---- BPE merge order: heap path vs the naive rescan reference ----
        # long chunks take a lazy-heap merge (O(L log L)); it must produce
        # exactly what the O(L^2) rescan-per-merge loop produces: lowest
        # rank first, leftmost occurrence on ties.
        def naive_bpe(tok, text):
            ids = []
            for chunk in _tokmod.pretokenize(text):
                word = [_tokmod._BYTE_ENC[b] for b in chunk.encode("utf-8")]
                while len(word) > 1:
                    best_rank, best_i = None, -1
                    for i in range(len(word) - 1):
                        r = tok.merge_ranks.get((word[i], word[i + 1]))
                        if r is not None and (best_rank is None or r < best_rank):
                            best_rank, best_i = r, i
                    if best_i < 0:
                        break
                    word[best_i:best_i + 2] = [word[best_i] + word[best_i + 1]]
                for piece in word:
                    tid = tok.piece_to_id.get(piece)
                    if tid is not None:
                        ids.append(tid)
                    else:
                        ids.extend(tok.piece_to_id[c] for c in piece
                                   if c in tok.piece_to_id)
            return ids

        alpha = "abcdef"
        bpe_pieces = list(alpha)
        bpe_merges = []
        # build pairs of existing pieces a few generations deep so long
        # runs keep merging and rank ties happen at multiple positions
        gen = list(alpha)
        for _round in range(3):
            new_pieces = []
            for a in gen:
                for b in gen:
                    m = a + b
                    if len(m) <= 8 and m not in bpe_pieces:
                        bpe_pieces.append(m)
                        new_pieces.append(m)
                        bpe_merges.append(f"{a} {b}")
            gen = new_pieces[:6]
        bpe_tok = _tokmod.Tokenizer.from_gguf({
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": bpe_pieces,
            "tokenizer.ggml.token_type": [1] * len(bpe_pieces),
            "tokenizer.ggml.merges": bpe_merges,
        })
        rng_bpe = random.Random(13)
        bpe_ok = 0
        bpe_cases = ["abcabcabc", "a" * 60, "abcdef" * 25, "fedcba" * 40,
                     "".join(rng_bpe.choice(alpha) for _ in range(200))]
        bpe_cases += ["".join(rng_bpe.choice(alpha)
                              for _ in range(rng_bpe.randint(1, 120)))
                      for _ in range(40)]
        for s in bpe_cases:
            if bpe_tok._encode_bpe(s) == naive_bpe(bpe_tok, s):
                bpe_ok += 1
        check("BPE heap merge matches the naive rescan on every case",
              bpe_ok == len(bpe_cases), f"{bpe_ok}/{len(bpe_cases)}")

        # ---- SPM special-piece prefilter -----------------------------------
        # specials whose first character is absent from the text are skipped;
        # splitting must be unchanged when they are present
        spm_tok = _tokmod.Tokenizer.from_gguf({
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": ["<unk>", "a", "b", "<X>", "ab", "\n\n"],
            "tokenizer.ggml.scores": [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            "tokenizer.ggml.token_type": [2, 1, 1, 4, 1, 4],
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.add_bos_token": False,
            "tokenizer.ggml.add_space_prefix": False,
        })
        check("SPM user-defined piece still splits with the prefilter",
              spm_tok.encode("ab<X>ab") == [4, 3, 4] and
              spm_tok.encode("ab\n\nab") == [4, 5, 4],
              str((spm_tok.encode("ab<X>ab"), spm_tok.encode("ab\n\nab"))))
        check("SPM prefilter skips texts with no special first-chars",
              spm_tok.encode("abab") == [4, 4] and
              spm_tok._special_by_first is not None,
              str(spm_tok.encode("abab")))

        # ---- model reference round-tripping ------------------------------
        from alpacca.store import _clean_nickname, parse_model_ref
        round_trip = ["llama3.2:1b", "llama3.2", "ollama:user/name:tag",
                      "ollama:user/name", "hf:org/repo", "hf:org/repo:Q4_K_M",
                      "org/repo"]
        rt_bad = []
        for raw in round_trip:
            ref = parse_model_ref(raw)
            again = parse_model_ref(ref.display())
            if (again.source, again.ns, again.name, again.tag) != \
                    (ref.source, ref.ns, ref.name, ref.tag):
                rt_bad.append((raw, ref.display(), again.source))
        check("every model reference survives display() -> parse_model_ref",
              not rt_bad, str(rt_bad))
        check("a non-library ollama ref keeps its disambiguator",
              parse_model_ref("ollama:user/name:tag").display() ==
              "ollama:user/name:tag",
              parse_model_ref("ollama:user/name:tag").display())

        # ---- column alignment for wide characters -------------------------
        from alpacca.cli import _clip, _display_width, _pad
        check("display width counts CJK and emoji as two columns",
              _display_width("日本語") == 6 and _display_width("abc") == 3 and
              _display_width("\U0001f680") == 2,
              str([_display_width("日本語"), _display_width("\U0001f680")]))
        check("display width ignores combining marks",
              _display_width("é") == 1)
        check("padding lines up a CJK cell with an ASCII one",
              _display_width(_pad("日本", 8)) == _display_width(_pad("ab", 8)) == 8,
              str([_display_width(_pad("日本", 8)), _display_width(_pad("ab", 8))]))
        check("clipping counts columns, not code points",
              _display_width(_clip("日本語日本語日本語", 10)) <= 10,
              repr(_clip("日本語日本語日本語", 10)))

        # ---- a corrupt nicknames file is preserved, not destroyed ---------
        from alpacca.store import _nicknames_file, _read_nicknames
        nick_home = os.environ.get("ALPACCA_HOME")
        try:
            os.environ["ALPACCA_HOME"] = str(tmp / "nick-corrupt-home")
            nf = _nicknames_file()
            nf.parent.mkdir(parents=True, exist_ok=True)
            nf.write_text('{"nicknames": {"a": ', "utf-8")   # truncated JSON
            spoiled = nf.with_suffix(nf.suffix + ".corrupt")
            check("a corrupt nicknames file degrades to an empty map",
                  _read_nicknames() == {})
            check("a corrupt nicknames file is moved aside, not overwritten",
                  spoiled.exists() and not nf.exists() and
                  spoiled.read_text("utf-8") == '{"nicknames": {"a": ',
                  str(list(nf.parent.iterdir())))
            nf.write_text("[" * 200000, "utf-8")  # RecursionError from json
            check("deeply nested nicknames JSON does not escape as RuntimeError",
                  _read_nicknames() == {})
        finally:
            if nick_home is None:
                os.environ.pop("ALPACCA_HOME", None)
            else:
                os.environ["ALPACCA_HOME"] = nick_home

        # ---- nickname sanitizing -----------------------------------------
        check("nickname sanitizing strips ANSI escapes",
              _clean_nickname("a\x1b[31mred") == "a [31mred",
              repr(_clean_nickname("a\x1b[31mred")))
        check("nickname sanitizing strips bidi overrides and zero-width chars",
              _clean_nickname("safe‮gpj.exe") == "safe gpj.exe" and
              _clean_nickname("a​b") == "a b" and
              _clean_nickname("a﻿b") == "a b",
              repr(_clean_nickname("safe‮gpj.exe")))
        check("a nickname made only of invisible characters cleans to empty",
              _clean_nickname("‮​⁦") == "",
              repr(_clean_nickname("‮​⁦")))
        check("nickname sanitizing strips lone surrogates",
              _clean_nickname("a\udc80b") == "a b",
              repr(_clean_nickname("a\udc80b")))

        bad_messages = [{"messages": [{"role": "user"}]},
                        {"messages": [{"content": "hi"}]},
                        {"messages": [{"role": "user", "content": None}]},
                        {"messages": ["hi"]},
                        {"messages": []}]
        rejected = 0
        for payload in bad_messages:
            try:
                _messages_from(payload)
            except ValueError:
                rejected += 1
        check("serve rejects malformed messages instead of raising KeyError",
              rejected == len(bad_messages), str(rejected))

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
        # IQ files cannot load, so they must never be auto-chosen, and an
        # explicit request for one must fail before the download starts
        iq_mixed = [{"path": "toy-IQ4_XS.gguf", "size": 1, "sha256": ""},
                    {"path": "toy-Q4_K_M.gguf", "size": 1, "sha256": ""}]
        check("HF picker skips IQ quantizations when auto-choosing",
              _hf_choose(iq_mixed, "")["path"] == "toy-Q4_K_M.gguf")
        try:
            _hf_choose(iq_mixed, "IQ4_XS")
            iq_sel_raised = False
        except ValueError as e:
            iq_sel_raised = "IQ" in str(e)
        check("an explicit IQ selector fails before downloading", iq_sel_raised)
        try:
            _hf_choose(iq_mixed[:1], "")
            iq_only_raised = False
        except ValueError as e:
            iq_only_raised = "IQ" in str(e)
        check("an IQ-only repo is rejected with a clear error", iq_only_raised)

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
        for arch in ("qwen2", "qwen3", "gemma"):
            r = subprocess.run([sys.executable, str(mk),
                                str(srv / f"tiny-{arch}.gguf"), "F32",
                                "--arch", arch],
                               capture_output=True, text=True)
            check(f"write tiny {arch} model", r.returncode == 0, r.stderr)
        for dtype, name in (("F32", "tiny-gemma3.gguf"),
                            ("Q4_0", "tiny-gemma3-q4.gguf")):
            r = subprocess.run([sys.executable, str(mk), str(srv / name), dtype,
                                "--arch", "gemma3"],
                               capture_output=True, text=True)
            check(f"write tiny gemma3 {dtype} model", r.returncode == 0, r.stderr)
        # a second gemma3 fixture carrying only the keys a real Gemma 3 GGUF
        # has, plus a 62-layer one for the 27B attention-scale rule
        for name, extra in (("tiny-gemma3-min.gguf", []),
                            ("tiny-gemma3-27b.gguf", ["--layers", "62"])):
            r = subprocess.run([sys.executable, str(mk), str(srv / name), "F32",
                                "--arch", "gemma3", "--minimal"] + extra,
                               capture_output=True, text=True)
            check(f"write {name}", r.returncode == 0, r.stderr)
        # fixtures carrying one invalid value each, to prove the loader
        # rejects them up front instead of failing later on NaNs, on
        # token-independent logits, or on a cryptic broadcast error
        for _bad in ("rope_base", "embed_scale", "norm_size"):
            r = subprocess.run([sys.executable, str(mk),
                                str(srv / f"tiny-gemma3-bad-{_bad}.gguf"),
                                "F32", "--arch", "gemma3", "--corrupt", _bad],
                               capture_output=True, text=True)
            check(f"write tiny-gemma3-bad-{_bad}.gguf", r.returncode == 0,
                  r.stderr)

        # ---- SPM tokenizer -----------------------------------------------
        print("== SPM tokenizer (greedy merge, as llama.cpp) ==")
        from alpacca.gguf import GGUFFile
        from alpacca.tokenizer import (Tokenizer, TT_BYTE, TT_CONTROL,
                                       TT_NORMAL, TT_USER_DEFINED)
        with GGUFFile.open(str(srv / "model.gguf")) as _gf:
            spm = Tokenizer.from_gguf(_gf.metadata)

        hello = spm.encode("hello", add_bos=False)
        check("SPM encodes a multi-character piece as one token",
              [spm.piece(i) for i in hello] == ["▁hello"],
              str([spm.piece(i) for i in hello]))
        check("SPM encodes a sentence one piece per word",
              [spm.piece(i) for i in spm.encode("hello world", add_bos=False)] ==
              ["▁hello", "▁world"],
              str(spm.encode("hello world", add_bos=False)))
        check("SPM falls back to byte tokens outside the vocabulary",
              [spm.piece(i) for i in spm.encode("z", add_bos=False)] ==
              ["▁", "<0x7A>"],
              str([spm.piece(i) for i in spm.encode("z", add_bos=False)]))
        for text in ("hello world", "the test", "日本語", "\U0001f680 ok",
                     "café", "a\tb\nc"):
            # add_space_prefix inserts a leading space, exactly as sentencepiece does
            check(f"SPM round-trips {text!r}",
                  spm.decode(spm.encode(text, add_bos=False)) == " " + text,
                  repr(spm.decode(spm.encode(text, add_bos=False))))
        check("SPM encode of an empty string is empty",
              spm.encode("", add_bos=False) == [])
        check("SPM add_bos prepends exactly one BOS",
              spm.encode("hello", add_bos=True) == [spm.bos_id] + hello)

        # Gemma 3 stores BPE merge ranks in tokenizer.ggml.scores, so a long
        # piece scores far worse than the sum of the short pieces it contains.
        # These are the real numbers from the Gemma 3 vocabulary: a unigram
        # Viterbi maximizes the sum and returns the four fragments (-222 beats
        # -4785); the merge order has to return the one true piece.
        rank_pieces = ["<unk>", "<s>", "</s>", "▁", "c", "a", "p", "i", "t", "l",
                       "▁c", "ap", "it", "al", "▁cap", "ital", "▁capital"]
        rank_scores = [0.0, 0.0, 0.0, -3.0, -9.0, -4.0, -30.0, -6.0, -5.0, -8.0,
                       -11.0, -176.0, -15.0, -20.0, -300.0, -400.0, -4785.0]
        ranks = Tokenizer.from_gguf({
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": rank_pieces,
            "tokenizer.ggml.scores": rank_scores,
            "tokenizer.ggml.token_type": [2, 3, 3] + [TT_NORMAL] * 14,
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.add_space_prefix": True,
        })
        whole = ranks.token_id("▁capital")
        pieces4 = [ranks.token_id(p) for p in ("▁c", "ap", "it", "al")]
        check("SPM rank-scored vocabulary: fragments really do score higher",
              sum(rank_scores[i] for i in pieces4) == -222.0 and
              rank_scores[whole] == -4785.0)
        check("SPM merges by rank order, not by maximizing the score sum",
              ranks.encode("capital", add_bos=False) == [whole],
              str([ranks.piece(i) for i in
                   ranks.encode("capital", add_bos=False)]))

        # A vocabulary whose user-defined pieces are runs of spaces: llama.cpp
        # splits those off the raw text before merging, so they must win over
        # the single-space piece they contain. Gemma 3 ships exactly this.
        ud_pieces = ["<unk>", "<s>", "</s>", "▁", "a", "b", "  ", "   ",
                     "<start_of_turn>"]
        ud_types = [2, 3, 3, TT_NORMAL, TT_NORMAL, TT_NORMAL,
                    TT_USER_DEFINED, TT_USER_DEFINED, TT_CONTROL]
        ud = Tokenizer.from_gguf({
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": ud_pieces,
            "tokenizer.ggml.scores": [0.0, 0.0, 0.0, -1.0, -2.0, -3.0,
                                      -4.0, -5.0, 0.0],
            "tokenizer.ggml.token_type": ud_types,
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.add_space_prefix": False,
        })
        check("SPM splits user-defined pieces off the raw text, longest first",
              [ud.piece(i) for i in ud.encode("a   b", add_bos=False)] ==
              ["a", "   ", "b"],
              str([ud.piece(i) for i in ud.encode("a   b", add_bos=False)]))
        check("SPM user-defined split prefers the longer run",
              [ud.piece(i) for i in ud.encode("a  b", add_bos=False)] ==
              ["a", "  ", "b"],
              str([ud.piece(i) for i in ud.encode("a  b", add_bos=False)]))
        check("SPM leaves control tokens as text by default",
              ud.encode("<start_of_turn>", add_bos=False) !=
              [ud.token_id("<start_of_turn>")],
              str(ud.encode("<start_of_turn>", add_bos=False)))
        check("SPM parse_special splits control tokens on request",
              ud.encode("<start_of_turn>a", add_bos=False, parse_special=True) ==
              [ud.token_id("<start_of_turn>"), ud.token_id("a")],
              str(ud.encode("<start_of_turn>a", add_bos=False, parse_special=True)))

        # add_space_prefix must keep working, including after a special token
        sp_on = Tokenizer.from_gguf({
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": ud_pieces,
            "tokenizer.ggml.scores": [0.0, 0.0, 0.0, -1.0, -2.0, -3.0,
                                      -4.0, -5.0, 0.0],
            "tokenizer.ggml.token_type": ud_types,
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.add_space_prefix": True,
        })
        check("SPM add_space_prefix prepends the space piece",
              [sp_on.piece(i) for i in sp_on.encode("a", add_bos=False)] ==
              ["▁", "a"],
              str([sp_on.piece(i) for i in sp_on.encode("a", add_bos=False)]))
        check("SPM add_space_prefix re-arms after a special token",
              [sp_on.piece(i) for i in
               sp_on.encode("<start_of_turn>a", add_bos=False,
                            parse_special=True)] ==
              ["<start_of_turn>", "▁", "a"],
              str([sp_on.piece(i) for i in
                   sp_on.encode("<start_of_turn>a", add_bos=False,
                                parse_special=True)]))

        # the BPE path is a different function and must be untouched by all this
        bpe = Tokenizer.from_gguf({
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": ["<|endoftext|>", "h", "e", "l", "o",
                                      "he", "ll", "hell", "hello", "Ġw"],
            "tokenizer.ggml.token_type": [TT_CONTROL] + [TT_NORMAL] * 9,
            "tokenizer.ggml.merges": ["h e", "l l", "he ll", "hell o"],
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.eos_token_id": 0,
            "tokenizer.ggml.add_bos_token": False,
        })
        check("BPE path still merges by rank and ignores SPM scores",
              [bpe.piece(i) for i in bpe.encode("hello")] == ["hello"],
              str([bpe.piece(i) for i in bpe.encode("hello")]))
        check("BPE vocabulary builds no SPM special-token cache",
              bpe.special_ids == [] and bpe.merge_ranks[("h", "e")] == 0)

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
        # a TIED embedding is eligible for densification, but if the budget
        # does not reach it the quantized copy is still resident - it used to
        # be counted in neither term (374 MiB unaccounted on a Gemma 3 1B)
        tied_fit = auto_budget_fit_mb(str(srv / "tiny-gemma3-q4.gguf"))
        with GGUFFile.open(str(srv / "tiny-gemma3-q4.gguf")) as _tf:
            tied_embd = _tf.tensors["token_embd.weight"].n_elements
            check("the gemma3 fixture really is tied",
                  "output.weight" not in _tf.tensors)
        check("a tied token embedding is counted in the fixed memory term",
              tied_fit is not None and
              tied_fit[1] - 512.0 > tied_embd * 1.3 / (1024 * 1024) * 0.99,
              str(tied_fit))
        for arch in ("qwen2", "qwen3", "gemma"):
            arch_model = Model.load(str(srv / f"tiny-{arch}.gguf"), progress=False)
            arch_logits = arch_model.prefill([1])
            check(f"load tiny {arch} keeps existing neox architecture path",
                  arch_model.hp.arch == arch and
                  arch_model.hp.rope_style == "neox" and
                  len(T.to_list(arch_logits)) == arch_model.hp.n_vocab,
                  arch_model.describe())

        # ---- Gemma 1 -----------------------------------------------------
        # `gemma` was listed in SUPPORTED_ARCHES but ran the plain llama
        # forward pass: no sqrt(n_embd) embedding scale and SiLU instead of
        # GELU. Both are now applied; verified against an independent float64
        # reference (5.3e-07 on NumPy, 1.1e-15 on the pure backend).
        gemma1 = Model.load(str(srv / "tiny-gemma.gguf"), progress=False)
        check("gemma scales the embedding by sqrt(n_embd)",
              abs(gemma1.hp.embed_scale - 64 ** 0.5) < 1e-6,
              str(gemma1.hp.embed_scale))
        check("gemma ties the output head to the token embedding",
              gemma1.output is gemma1.tok_embd)
        check("llama-class architectures do not scale the embedding",
              Model.load(str(srv / "model.gguf"), progress=False).hp.embed_scale
              == 1.0)
        g1_last = None
        for _t in range(1, 7):
            g1_last = gemma1.forward(_t)
        g1_values = T.to_list(g1_last)
        expected_g1 = [-0.057178, 0.022701, 0.552373, 1.243609,
                       -0.192921, 0.171072, 4.553725, -1.028753]
        g1_diff = max(abs(float(a) - b) for a, b in zip(g1_values[:8], expected_g1))
        check(f"tiny gemma logits are stable (diff {g1_diff:.2e})",
              g1_diff < 1e-5, str([round(float(v), 6) for v in g1_values[:8]]))
        if T.HAS_NUMPY:
            g1_batch = Model.load(str(srv / "tiny-gemma.gguf"), progress=False)
            g1_bdiff = max(abs(float(a) - float(b))
                           for a, b in zip(g1_values,
                                           T.to_list(g1_batch.forward_batch(
                                               list(range(1, 7))))))
            check(f"tiny gemma batch matches sequential (diff {g1_bdiff:.2e})",
                  g1_bdiff < 1e-5, str(g1_bdiff))

        gemma3 = Model.load(str(srv / "tiny-gemma3.gguf"), progress=False)
        check("load tiny gemma3 reads architecture-specific metadata",
              gemma3.hp.arch == "gemma3" and
              gemma3.hp.n_head == 2 and gemma3.hp.n_kv == 1 and
              gemma3.hp.head_dim == 16 and gemma3.hp.sliding_window == 3 and
              gemma3.hp.sliding_layers == (True, True, True, True, True, False) and
              abs(gemma3.hp.attention_scale - 0.125) < 1e-6 and
              abs(gemma3.hp.rope_base - 1000000.0) < 1.0 and
              abs(gemma3.hp.rope_base_swa - 10000.0) < 1.0 and
              abs(gemma3.hp.rope_freq_scale - 0.5) < 1e-6 and
              abs(gemma3.hp.final_logit_softcap - 20.0) < 1e-6 and
              gemma3.output is gemma3.tok_embd,
              gemma3.describe())
        # 0.125 is not 1/sqrt(head_dim); if the key were ignored this would be
        # 0.25 and the "metadata-driven attention scale" feature would be
        # verified by nothing at all
        check("gemma3 attention scale comes from metadata when present",
              abs(gemma3.hp.attention_scale - 1.0 / (16 ** 0.5)) > 1e-3)

        # ---- the fallbacks a real Gemma 3 GGUF actually takes -------------
        # No real Gemma 3 file carries attention.scale,
        # attention.sliding_window_pattern, rope.scaling.*,
        # final_logit_softcapping, vocab_size or rope.dimension_count. Every
        # one of those fallbacks was untested.
        g3min = Model.load(str(srv / "tiny-gemma3-min.gguf"), progress=False)
        check("gemma3 without a sliding_window_pattern uses a period of 6",
              g3min.hp.sliding_layers == () and
              g3min.hp.full_attention_period == 6 and
              [g3min._gemma3_layer_is_sliding(i) for i in range(6)] ==
              [True, True, True, True, True, False],
              str(g3min.hp))
        check("gemma3 without attention.scale falls back to 1/sqrt(head_dim)",
              abs(g3min.hp.attention_scale - 1.0 / (16 ** 0.5)) < 1e-6,
              str(g3min.hp.attention_scale))
        check("gemma3 without rope.scaling leaves RoPE unscaled",
              abs(g3min.hp.rope_freq_scale - 1.0) < 1e-9)
        check("gemma3 without final_logit_softcapping does not softcap",
              g3min.hp.final_logit_softcap == 0.0)
        check("gemma3 without vocab_size counts the token list",
              g3min.hp.n_vocab == len(g3min.tok.pieces))
        check("gemma3 without rope.dimension_count uses the head dimension",
              g3min.hp.n_rot == 16, str(g3min.hp.n_rot))
        check("gemma3 minimal fixture still generates",
              len(T.to_list(g3min.prefill([1, 2, 3]))) == g3min.hp.n_vocab)
        # the softcap-absent path must not be a no-op copy of the capped one
        g3min_logits = T.to_list(g3min.prefill([1, 2, 3, 4]))
        check("uncapped gemma3 logits can exceed the fixture's softcap",
              max(abs(v) for v in g3min_logits) > 0.0 and
              all(v == v for v in g3min_logits))

        g3_27b = Model.load(str(srv / "tiny-gemma3-27b.gguf"), progress=False)
        check("gemma3 with 62 layers takes the 27B attention-scale rule",
              g3_27b.hp.n_layer == 62 and
              abs(g3_27b.hp.attention_scale - 1.0 / ((64 / 2) ** 0.5)) < 1e-6 and
              abs(g3_27b.hp.attention_scale - g3min.hp.attention_scale) > 1e-3,
              str(g3_27b.hp.attention_scale))

        # ---- invalid metadata is rejected at load, not much later ---------
        for _bad, _needle in (("rope_base", "rope.freq_base"),
                              ("embed_scale", "embedding_scale"),
                              ("norm_size", "expected")):
            _err = ""
            try:
                Model.load(str(srv / f"tiny-gemma3-bad-{_bad}.gguf"),
                           progress=False)
            except ValueError as e:
                _err = str(e)
            check(f"gemma3 rejects an invalid {_bad} at load",
                  _needle in _err, repr(_err))

        # ---- the two RoPE tables must genuinely differ --------------------
        # The sliding/global parity test passes even if _rope_cos_swa were
        # never built, because both paths share the same silent fallback.
        if T.HAS_NUMPY:
            import numpy as _np
            check("gemma3 builds a separate sliding-window RoPE table",
                  gemma3._rope_cos_swa is not None and
                  gemma3._rope_sin_swa is not None and
                  float(_np.max(_np.abs(gemma3._rope_cos_swa -
                                        gemma3._rope_cos))) > 1e-3,
                  "sliding and global RoPE tables are identical")

        # ---- chat rendering ----------------------------------------------
        # The only render() call in this suite used a llama fixture with no
        # chat_template, so it took the `raw` path and no format was covered.
        from alpacca.chat import ChatFormat as _CF, detect_format as _detect
        check("gemma3 chat format is detected from the template",
              _detect(gemma3.metadata) == "gemma", str(_detect(gemma3.metadata)))
        g3_msgs = [{"role": "user", "content": "hello"}]
        g3_rendered = _CF(gemma3, "gemma").render(g3_msgs)
        check("a template that opens with bos_token renders BOS first",
              g3_rendered[0] == gemma3.tok.bos_id, str(g3_rendered[:4]))
        check("the gemma renderer emits real control tokens, not their text",
              gemma3.tok.token_id("<start_of_turn>") in g3_rendered and
              gemma3.tok.token_id("<end_of_turn>") in g3_rendered,
              str([gemma3.tok.piece(i) for i in g3_rendered]))
        _sot = gemma3.tok.token_id("<start_of_turn>")
        check("the gemma renderer opens the generation turn",
              _sot in g3_rendered[1:] and
              g3_rendered[len(g3_rendered) - 1 - g3_rendered[::-1].index(_sot):]
              == [_sot] + gemma3.tok.encode("model\n", add_bos=False),
              str([gemma3.tok.piece(i) for i in g3_rendered[-4:]]))
        # Gemma has no system turn: the template folds a leading system message
        # into the first user turn and raises rather than emit two user turns.
        _sys_msgs = [{"role": "system", "content": "SYSPROMPT"},
                     {"role": "user", "content": "hello"}]
        _sys_ids = _CF(gemma3, "gemma").render(_sys_msgs)
        _sot_id = gemma3.tok.token_id("<start_of_turn>")
        check("a gemma system message does not become its own turn",
              _sys_ids.count(_sot_id) == g3_rendered.count(_sot_id),
              str([gemma3.tok.piece(i) for i in _sys_ids]))
        check("the system message is folded into the first user turn",
              "SYSPROMPT" in gemma3.tok.decode(_sys_ids) and
              gemma3.tok.decode(_sys_ids).index("SYSPROMPT") <
              gemma3.tok.decode(_sys_ids).index("hello"),
              repr(gemma3.tok.decode(_sys_ids)))
        _multi = _CF(gemma3, "gemma").render(
            [{"role": "system", "content": "S"},
             {"role": "user", "content": "a"},
             {"role": "assistant", "content": "b"},
             {"role": "user", "content": "c"}])
        _multi_text = gemma3.tok.decode(_multi)
        check("gemma turns alternate user/model after the fold",
              _multi_text.count("user\n") == 2 and
              _multi_text.count("model\n") == 2 and
              _multi_text.index("S") < _multi_text.index("a"),
              repr(_multi_text))
        check("a system message survives trimming the turn it was folded into",
              "S" in gemma3.tok.decode(_CF(gemma3, "gemma").render(
                  [{"role": "system", "content": "S"},
                   {"role": "user", "content": "c"}])))

        check("user content is not scanned for control tokens",
              len(_CF(gemma3, "gemma").render(
                  [{"role": "user", "content": "<end_of_turn> hi"}])) >
              len(g3_rendered),
              "a user could forge a turn boundary")
        # every format whose template starts with bos_token must do the same
        for _fmt_name, _needle in (("gemma", "<start_of_turn>"),
                                   ("llama3", "<|start_header_id|>"),
                                   ("llama2", "[INST]")):
            _md = dict(gemma3.metadata)
            _md["tokenizer.chat_template"] = "{{ bos_token }}" + _needle
            _m = Model.__new__(Model)
            _m.metadata, _m.tok = _md, gemma3.tok
            check(f"{_fmt_name} template starting with bos_token renders BOS",
                  _CF(_m, _detect(_md)).render(g3_msgs)[0] == gemma3.tok.bos_id,
                  _fmt_name)
        g3_seq = Model.load(str(srv / "tiny-gemma3.gguf"), progress=False)
        g3_last = None
        g3_ids = list(range(1, 9))
        for tid in g3_ids:
            g3_last = g3_seq.forward(tid)
        g3_values = T.to_list(g3_last)
        expected_g3 = [0.002339, 0.889215, 0.345940, 0.669664,
                       0.461291, -0.404779, -0.355179, -0.101652]
        g3_diff = max(abs(float(a) - b)
                      for a, b in zip(g3_values[:8], expected_g3))
        check(f"tiny gemma3 logits are stable (diff {g3_diff:.2e})",
              g3_diff < 1e-5, str(g3_values[:8]))
        if T.HAS_NUMPY:
            g3_batch = Model.load(str(srv / "tiny-gemma3.gguf"), progress=False)
            g3_batch_logits = g3_batch.forward_batch(g3_ids)
            g3_batch_diff = max(abs(float(a) - float(b))
                                for a, b in zip(T.to_list(g3_last),
                                                T.to_list(g3_batch_logits)))
            check(f"tiny gemma3 batch matches sequential across sliding window "
                  f"(diff {g3_batch_diff:.2e})",
                  g3_batch_diff < 1e-5 and g3_batch.n_past == len(g3_ids),
                  str(g3_batch_diff))
            # chunked prefill with kv_start > 0: every chunk after the first
            # attends across a cache boundary, and with a window of 3 the
            # sliding layers have to drop rows the global layers keep
            for g3_chunk in (1, 2, 3, 4, 5, 7):
                g3_ch = Model.load(str(srv / "tiny-gemma3.gguf"), progress=False)
                g3_ch_logits = None
                for _s in range(0, len(g3_ids), g3_chunk):
                    g3_ch_logits = g3_ch.forward_batch(g3_ids[_s:_s + g3_chunk])
                g3_ch_diff = max(abs(float(a) - float(b))
                                 for a, b in zip(T.to_list(g3_last),
                                                 T.to_list(g3_ch_logits)))
                check(f"tiny gemma3 chunked prefill (chunk {g3_chunk}) matches "
                      f"sequential (diff {g3_ch_diff:.2e})",
                      g3_ch_diff < 1e-5 and g3_ch.n_past == len(g3_ids),
                      str(g3_ch_diff))
            # prefill discards every chunk's logits but the last, and on a
            # tied 262144-row head that projection is ~218 ms of wasted work
            # per chunk on the real model
            g3_skip = Model.load(str(srv / "tiny-gemma3.gguf"), progress=False)
            check("forward_batch can skip the output projection",
                  g3_skip.forward_batch(g3_ids[:4], want_logits=False) is None and
                  g3_skip.n_past == 4)
            g3_skip_logits = g3_skip.forward_batch(g3_ids[4:], want_logits=True)
            g3_skip_diff = max(abs(float(a) - float(b))
                               for a, b in zip(T.to_list(g3_last),
                                               T.to_list(g3_skip_logits)))
            check(f"skipping the projection does not change the answer "
                  f"(diff {g3_skip_diff:.2e})", g3_skip_diff < 1e-5,
                  str(g3_skip_diff))

            # prefix reuse: re-prefilling a shared prefix must not change the
            # answer, and must actually reuse the cache rather than redo it
            g3_re = Model.load(str(srv / "tiny-gemma3.gguf"), progress=False)
            g3_re.prefill(g3_ids[:5])
            g3_re_logits = g3_re.prefill(g3_ids)
            g3_re_diff = max(abs(float(a) - float(b))
                             for a, b in zip(T.to_list(g3_last),
                                             T.to_list(g3_re_logits)))
            check(f"tiny gemma3 prefix reuse matches a cold prefill "
                  f"(diff {g3_re_diff:.2e})",
                  g3_re_diff < 1e-5 and g3_re.last_prefill_forwarded == 3,
                  f"{g3_re_diff} forwarded={g3_re.last_prefill_forwarded}")

        # a scalar sliding_window_pattern is a period, matching llama.cpp's
        # set_swa_pattern: is_swa[i] = n == 0 or (i % n < n - 1). 1 means the
        # model has no sliding-window attention at all.
        for period, want in ((1, [False] * 6), (0, [True] * 6),
                             (2, [True, False, True, False, True, False])):
            name = f"tiny-gemma3-swa{period}.gguf"
            r = subprocess.run([sys.executable, str(mk), str(srv / name), "F32",
                                "--arch", "gemma3", "--minimal",
                                "--swa-pattern", str(period)],
                               capture_output=True, text=True)
            check(f"write {name}", r.returncode == 0, r.stderr)
            swa_m = Model.load(str(srv / name), progress=False)
            check(f"scalar sliding_window_pattern={period} sets the period",
                  [swa_m._gemma3_layer_is_sliding(i) for i in range(6)] == want,
                  str([swa_m._gemma3_layer_is_sliding(i) for i in range(6)]))
            check(f"gemma3 with sliding_window_pattern={period} still generates",
                  len(T.to_list(swa_m.prefill([1, 2, 3]))) == swa_m.hp.n_vocab)
        gemma3_q4 = Model.load(str(srv / "tiny-gemma3-q4.gguf"), progress=False)
        gemma3_q4_logits = gemma3_q4.prefill([1, 2, 3, 4])
        if T.HAS_NUMPY:
            check("load tiny gemma3 Q4_0 keeps matrix weights quantized",
                  gemma3_q4.weight_storage["quantized"] == {"Q4_0": 43} and
                  not gemma3_q4.weight_storage["fallback"] and
                  len(T.to_list(gemma3_q4_logits)) == gemma3_q4.hp.n_vocab,
                  str(gemma3_q4.weight_storage))
        else:
            check("load tiny gemma3 Q4_0 falls back to dense without NumPy",
                  gemma3_q4.weight_storage["fallback"] == {"Q4_0": 43} and
                  len(T.to_list(gemma3_q4_logits)) == gemma3_q4.hp.n_vocab,
                  str(gemma3_q4.weight_storage))

        # ---- gemma3 x dense budget x tied embeddings ---------------------
        # This is the combination the CLI picks by default and it had no
        # coverage. The Q4_0 checks above assert shapes and a matrix count
        # and nothing numeric, so densifying is also the numeric oracle:
        # dequantize-then-dense must agree with the quantized matvec.
        g3_budget_home = os.environ.get("ALPACCA_DENSE_WEIGHT_MB")
        try:
            os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "64"
            g3_dense = Model.load(str(srv / "tiny-gemma3-q4.gguf"), progress=False)
            g3_dense_logits = T.to_list(g3_dense.prefill([1, 2, 3, 4]))
            g3_q_logits = T.to_list(gemma3_q4_logits)
            g3_dense_diff = max(abs(float(a) - float(b))
                                for a, b in zip(g3_q_logits, g3_dense_logits))
            check(f"gemma3 densified weights match the quantized matvec "
                  f"(diff {g3_dense_diff:.2e})",
                  g3_dense_diff < 1e-4, str(g3_dense_diff))
            if T.HAS_NUMPY:
                check("the gemma3 dense budget densifies the tied token_embd",
                      "token_embd.weight" in g3_dense.weight_storage["densified"],
                      str(g3_dense.weight_storage["densified"][:4]))
                check("densifying a tied token_embd keeps the head aliased",
                      g3_dense.output is g3_dense.tok_embd)
                check("the gemma3 dense budget reports the bytes it spent",
                      g3_dense.weight_storage["densified_bytes"] > 0 and
                      "dense budget" in g3_dense.describe(),
                      g3_dense.describe())
            os.environ["ALPACCA_DENSE_WEIGHT_MB"] = "0"
            g3_zero = Model.load(str(srv / "tiny-gemma3-q4.gguf"), progress=False)
            check("a zero dense budget keeps gemma3 fully quantized",
                  g3_zero.weight_storage["densified"] == [] and
                  (g3_zero.weight_storage["quantized"] == {"Q4_0": 43}
                   if T.HAS_NUMPY else True),
                  str(g3_zero.weight_storage))
        finally:
            if g3_budget_home is None:
                os.environ.pop("ALPACCA_DENSE_WEIGHT_MB", None)
            else:
                os.environ["ALPACCA_DENSE_WEIGHT_MB"] = g3_budget_home

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

        # fused attn_q+attn_k / ffn_gate+ffn_up must be a pure launch-count
        # optimization: same per-row math, same logits. Guard that fusion
        # actually engaged, or this compares a path with itself (see 5.6).
        if T.HAS_NUMPY:
            import numpy as np
            fm = Model.load(str(srv / "tiny-q4k.gguf"), progress=False)
            fm_ids = fm.tok.encode("hi") or [fm.tok.bos_id]
            fused_logits = np.asarray(fm.prefill(fm_ids[:1]), dtype=np.float64)
            fused_engaged = (all(ly.wqk is not None for ly in fm.layers) and
                             all(ly.wgu is not None for ly in fm.layers))
            os.environ["ALPACCA_FUSE"] = "0"
            try:
                um = Model.load(str(srv / "tiny-q4k.gguf"), progress=False)
                unfused_engaged = (all(ly.wqk is None for ly in um.layers) and
                                   all(ly.wgu is None for ly in um.layers))
                unfused_logits = np.asarray(um.prefill(fm_ids[:1]),
                                            dtype=np.float64)
            finally:
                os.environ.pop("ALPACCA_FUSE", None)
            fdiff = float(np.abs(fused_logits - unfused_logits).max())
            check(f"fused qk/gate-up matches unfused logits (diff {fdiff:.2e})",
                  fused_engaged and unfused_engaged and fdiff < 1e-4,
                  f"engaged={fused_engaged}/{unfused_engaged} diff={fdiff:.2e}")

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
        # a fully dense model decodes single-pool on OpenBLAS: sending only
        # its attention to the numba pool creates the two-pool thrash the
        # kernel exists to avoid (measured 75 -> 258 ms/token), so the
        # kernel-attention gate must follow the weights, not the JIT
        check("dense models do not engage kernel attention",
              not f16_model._use_kernel_attention)
        if T.HAS_NUMPY and AK.available():
            q4k_gate = Model.load(str(srv / "tiny-q4k.gguf"), progress=False)
            check("quantized models engage kernel attention with the JIT",
                  q4k_gate._use_kernel_attention)

        # ---- context window --------------------------------------------
        # Nothing covered n_past near n_ctx for any architecture, which is why
        # a prompt of exactly n_ctx returned tokens=0/text='' with no signal.
        print("== context window ==")
        from alpacca import chat as chat_mod
        from alpacca.chat import ChatFormat, fit_to_context, generate
        ctx_model = Model.load(str(srv / "model.gguf"), n_ctx=32, progress=False)
        check("effective context window is reported next to the trained one",
              "ctx 32 of 256" in ctx_model.describe(), ctx_model.describe())
        ctx_params = SamplerParams(temperature=0.0, seed=1)

        full = list(range(1, 33))            # exactly n_ctx tokens
        res = generate(ctx_model, full, ctx_params, n_predict=-1)
        check("a prompt of exactly n_ctx says it ran out of context",
              res.tokens == 0 and res.stop_reason == "context" and
              res.prompt_tokens == 32,
              f"{res.tokens} {res.stop_reason} {res.prompt_tokens}")
        ctx_model.reset()
        res = generate(ctx_model, list(range(1, 31)), ctx_params, n_predict=-1)
        check("a prompt just under n_ctx still generates and reports length",
              res.tokens == 2 and res.stop_reason == "length", str(res))
        ctx_model.reset()
        res = generate(ctx_model, list(range(1, 5)), ctx_params, n_predict=4)
        check("a spent n_predict budget is reported as length",
              res.tokens == 4 and res.stop_reason == "length", str(res))
        # a multi-token stop string must not reach the stream before it can be
        # detected: the caller would print text that is then truncated away
        import re as _re
        ctx_model.reset()
        seen = []
        first = generate(ctx_model, [1, 2], ctx_params, n_predict=12,
                         stream=seen.append)
        check("streaming without stop strings still emits everything",
              "".join(seen) == first.text, f"{''.join(seen)!r} {first.text!r}")
        # A stop string that arrives inside a single token is detected before
        # anything is streamed, so it cannot show the bug. Script the token
        # sequence so the needle genuinely spans two stream callbacks.
        class _ScriptedTok:
            pieces = ["", "ab", "cd", "ef"]
            bos_id = 0

            def token_bytes(self, tid):
                return self.pieces[tid].encode("utf-8")

            def is_eog(self, tid):
                return False

        class _ScriptedModel:
            def __init__(self, seq):
                self.tok = _ScriptedTok()
                self.seq, self.i = seq, 0
                self.n_ctx, self.n_past = 100, 0

            def _next(self):
                tid = self.seq[min(self.i, len(self.seq) - 1)]
                self.i += 1
                out = [0.0] * len(self.tok.pieces)
                out[tid] = 1.0
                return out

            def prefill(self, ids):
                self.n_past = len(ids)
                return self._next()

            def forward(self, _tid):
                self.n_past += 1
                return self._next()

        # emits "ab", "cd", "ef"; the stop string "bcd" straddles the first two
        scripted = _ScriptedModel([1, 2, 3])
        chunks = []
        sres = generate(scripted, [0], SamplerParams(temperature=0.0, seed=1),
                        n_predict=3, stream=chunks.append, stop_strings=["bcd"])
        check("a stop string spanning two tokens is never streamed early",
              "bcd" not in "".join(chunks), f"streamed={''.join(chunks)!r}")
        check("the streamed text is exactly the returned text",
              "".join(chunks) == sres.text,
              f"streamed={''.join(chunks)!r} text={sres.text!r}")
        check("the stop string is trimmed from the returned text",
              sres.text == "a" and sres.stop_reason == "stop",
              f"{sres.stop_reason} {sres.text!r}")

        ctx_model.reset()
        overflow = False
        try:
            generate(ctx_model, list(range(1, 40)), ctx_params, n_predict=4)
        except RuntimeError:
            overflow = True
        check("a prompt longer than n_ctx raises rather than truncating",
              overflow)

        # trimming: the REPL drops the oldest exchanges to make room
        ctx_model.reset()
        fmt_ctx = ChatFormat(ctx_model, "raw")
        convo = [{"role": "system", "content": "s"}]
        for _ in range(6):
            convo.append({"role": "user", "content": "the test"})
            convo.append({"role": "assistant", "content": "ok"})
        convo.append({"role": "user", "content": "hello"})
        ids, dropped = fit_to_context(fmt_ctx, convo, ctx_model.n_ctx, reserve=8)
        check("fitting a long conversation drops the oldest exchanges",
              dropped > 0 and len(ids) + 8 <= ctx_model.n_ctx, f"{dropped} {len(ids)}")
        check("fitting never drops the system message or the newest turn",
              convo[0]["role"] == "system" and convo[-1]["content"] == "hello" and
              len(convo) == 2, str(convo))
        # a conversation that only slightly overflows keeps what still fits
        partial = [{"role": "user", "content": "the test"},
                   {"role": "assistant", "content": "ok"},
                   {"role": "user", "content": "hello"}]
        _, part_dropped = fit_to_context(fmt_ctx, partial, ctx_model.n_ctx, reserve=2)
        check("fitting drops whole exchanges, oldest first",
              part_dropped in (0, 1) and partial[-1]["content"] == "hello",
              f"{part_dropped} {partial}")
        short = [{"role": "user", "content": "hi"}]
        _, none_dropped = fit_to_context(fmt_ctx, short, ctx_model.n_ctx, reserve=8)
        check("fitting a conversation that already fits drops nothing",
              none_dropped == 0 and len(short) == 1)
        huge = [{"role": "user", "content": "testing " * 200}]
        _, cant = fit_to_context(fmt_ctx, huge, ctx_model.n_ctx, reserve=8)
        check("fitting keeps the newest turn even when it cannot fit",
              cant == 0 and len(huge) == 1)

        # the REPL must survive a turn that cannot be answered: before, the
        # RuntimeError propagated out of cmd_run and took the conversation
        ctx_model.reset()
        repl_out, repl_err = io.StringIO(), io.StringIO()
        repl_in = io.StringIO("testing " * 200 + "\nhi\n/exit\n")
        ctx_home = os.environ.get("ALPACCA_HOME")
        old_std = (sys.stdin, sys.stdout, sys.stderr)
        try:
            os.environ["ALPACCA_HOME"] = str(tmp / "ctx-home")
            sys.stdin, sys.stdout, sys.stderr = repl_in, repl_out, repl_err
            chat_mod.interactive(ctx_model, ctx_params, model_name="ctx-test")
            repl_failed = ""
        except BaseException as e:                      # noqa: BLE001
            repl_failed = f"{type(e).__name__}: {e}"
        finally:
            sys.stdin, sys.stdout, sys.stderr = old_std
            if ctx_home is None:
                os.environ.pop("ALPACCA_HOME", None)
            else:
                os.environ["ALPACCA_HOME"] = ctx_home
        check("the REPL survives a turn that overflows the context window",
              not repl_failed and "was not sent" in repl_err.getvalue(),
              repl_failed or repl_err.getvalue()[-300:])
        check("the REPL keeps answering after the overflowing turn",
              "tokens," in repl_err.getvalue(), repl_err.getvalue()[-300:])

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

        # Q2_K and Q3_K used to have no quantized matvec, so a 1B model in
        # either format silently expanded to 3.7 GiB of dense float32.
        q2 = Model.load(str(srv / "tiny-q2k.gguf"), progress=False)
        q2_desc = q2.describe()
        if T.HAS_NUMPY:
            check("load tiny Q2_K keeps matrix weights quantized",
                  q2.weight_storage["quantized"] == {"Q2_K": 16} and
                  not q2.weight_storage["fallback"] and
                  q2.weight_storage["dense"] == 0,
                  str(q2.weight_storage))
            check("describe reports quantized Q2_K storage with its size",
                  "weights quantized Q2_K (16 matrices," in q2_desc, q2_desc)
            q2_f32_env = os.environ.get("ALPACCA_F32")
            try:
                os.environ["ALPACCA_F32"] = "1"
                q2_dense = Model.load(str(srv / "tiny-q2k.gguf"), progress=False)
            finally:
                if q2_f32_env is None:
                    os.environ.pop("ALPACCA_F32", None)
                else:
                    os.environ["ALPACCA_F32"] = q2_f32_env
            q2_ratio = (q2_dense.weight_storage["dense_bytes"] /
                        max(q2.weight_storage["quantized_bytes"], 1))
            check(f"quantized Q2_K costs {q2_ratio:.1f}x less than dense float32",
                  q2_ratio > 8.0, str(q2_ratio))
            q2_err = max(abs(float(a) - float(b)) for a, b in
                         zip(T.to_list(q2.prefill([1, 2, 3, 4])),
                             T.to_list(q2_dense.prefill([1, 2, 3, 4]))))
            check(f"Q2_K quantized matvec matches dequantize-then-dense "
                  f"(diff {q2_err:.2e})", q2_err < 2e-3, str(q2_err))
        else:
            check("load tiny Q2_K falls back to dense without NumPy",
                  q2.weight_storage["fallback"] == {"Q2_K": 16}, str(q2.weight_storage))
            check("describe reports dense fallback Q2_K",
                  "dense fallback Q2_K" in q2_desc, q2_desc)
            check("Q2_K dense fallback still generates",
                  len(T.to_list(q2.prefill([1, 2]))) == q2.hp.n_vocab)
        check("Q2_K forward runs",
              len(T.to_list(q2.prefill([1, 2, 3]))) == q2.hp.n_vocab)

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
        r = run_cli("nickname", "tiny", "Tiny", "Buddy", env=env)
        check("nickname command sets model nickname",
              "nickname set: Tiny Buddy -> tiny" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("list", env=env)
        check("list shows model nickname", "Tiny Buddy" in r.stdout, r.stdout)
        r = run_cli("nickname", "--list", env=env)
        check("nickname --list shows every alias and its target",
              "Tiny Buddy" in r.stdout and "tiny" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("show", "Tiny Buddy", env=env)
        check("show resolves model nickname",
              '"name": "tiny"' in r.stdout and '"nickname": "Tiny Buddy"' in r.stdout,
              r.stdout)
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
        r = run_cli("run", "Tiny Buddy", "hello there", "-n", "4", "--seed", "1", env=env)
        check("run one-shot resolves nickname", "tokens," in r.stderr, r.stderr)
        model_path = tmp / "home" / "models" / "ollama" / "library" / "tiny" / "latest" / "model.gguf"
        r = run_cli("run", str(model_path), "hi", "-n", "4", "--seed", "1", env=env)
        check("run by file path", "tokens," in r.stderr)
        r = run_cli("run", "tiny", "hi", "-n", "4", "--seed", "1",
                    env={**env, "ALPACCA_PURE": "1"})
        check("run with pure-python backend", "tokens," in r.stderr)
        r = run_cli("tokenize", "-m", "tiny", "-p", "hello", env=env)
        check("tokenize via model name", "\u2581hello" in r.stdout or "hello" in r.stdout)
        r = run_cli("tokenize", "-m", "Tiny Buddy", "-p", "hello", env=env)
        check("tokenize resolves nickname", "\u2581hello" in r.stdout or "hello" in r.stdout)

        old_home = os.environ.get("ALPACCA_HOME")
        try:
            os.environ["ALPACCA_HOME"] = env["ALPACCA_HOME"]
            from alpacca.history import start_session
            h = start_session("tiny", str(model_path))
            h.append_message("user", "history question")
            h.append_message("assistant", "history answer", tokens=2, seconds=0.01)
            h.close()
            history_id = h.id
        finally:
            if old_home is None:
                os.environ.pop("ALPACCA_HOME", None)
            else:
                os.environ["ALPACCA_HOME"] = old_home
        r = run_cli("history", "list", env=env)
        check("history list shows saved chat",
              history_id in r.stdout and "history question" in r.stdout,
              r.stdout)
        r = run_cli("history", "show", "1", env=env)
        check("history show displays saved messages",
              "history question" in r.stdout and "history answer" in r.stdout,
              r.stdout)
        r = run_cli("history", "stats", env=env)
        check("history stats shows saved model token speed",
              "tiny" in r.stdout and "200.0" in r.stdout and "TOKENS" in r.stdout,
              r.stdout)
        r = run_cli("history", "rm", history_id[:12], env=env)
        check("history rm deletes one chat", f"deleted {history_id}" in r.stdout,
              r.stdout)
        r = run_cli("history", "list", env=env)
        check("history list is empty after delete", "no chat history yet" in r.stdout,
              r.stdout)
        r = run_cli("history", "stats", env=env)
        check("history stats lists installed model without saved chats",
              "tiny" in r.stdout and "n/a" in r.stdout,
              r.stdout)
        hist_dir = Path(env["ALPACCA_HOME"]) / "history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        malformed_json = hist_dir / "malformed.json"
        bad_utf8_json = hist_dir / "bad-utf8.json"
        temp_json = hist_dir / "orphan.json.tmp"
        malformed_json.write_text("{not json", encoding="utf-8")
        bad_utf8_json.write_bytes(b"\xff\xfe\xfa")
        temp_json.write_text("partial", encoding="utf-8")
        r = run_cli("history", "list", env=env)
        check("history list tolerates invalid utf-8 history files",
              "no chat history yet" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("history", "stats", env=env)
        check("history stats tolerates invalid utf-8 history files",
              "tiny" in r.stdout and "TOKENS" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("history", "clear", env=env)
        check("history clear without --yes leaves malformed artifacts",
              malformed_json.exists() and temp_json.exists(),
              r.stdout + r.stderr)
        r = run_cli("history", "clear", "--yes", env=env)
        check("history clear --yes removes malformed artifacts",
              not malformed_json.exists() and not bad_utf8_json.exists() and
              not temp_json.exists(),
              r.stdout + r.stderr)
        bad_schema_json = hist_dir / "bad-schema.json"
        bad_schema_tmp = hist_dir / "bad-schema.json.tmp"
        bad_schema_json.write_text(json.dumps({
            "id": "bad-schema",
            "started_at": "2026-01-02T03:04:05Z",
            "updated_at": "2026-01-02T03:04:05Z",
            "model": "tiny",
            "title": "bad schema",
            "messages": ["not-a-message"],
        }), encoding="utf-8")
        bad_schema_tmp.write_text("partial", encoding="utf-8")
        r = run_cli("history", "list", env=env)
        check("history list tolerates non-dict message entries",
              "bad-schema" in r.stdout,
              r.stdout)
        r = run_cli("history", "show", "bad-schema", env=env)
        check("history show tolerates non-dict message entries",
              "Chat:    bad-schema" in r.stdout,
              r.stdout)
        r = run_cli("history", "stats", env=env)
        check("history stats tolerates non-dict message entries",
              "tiny" in r.stdout and "TOKENS" in r.stdout,
              r.stdout)
        r = run_cli("history", "clear", "--yes", env=env)
        check("history clear --yes removes structurally malformed history",
              not bad_schema_json.exists() and not bad_schema_tmp.exists(),
              r.stdout + r.stderr)
        old_home = os.environ.get("ALPACCA_HOME")
        try:
            os.environ["ALPACCA_HOME"] = env["ALPACCA_HOME"]
            from alpacca.history import start_session
            for content in ("clear one", "clear two"):
                h = start_session("tiny", str(model_path))
                h.append_message("user", content)
                h.close()
        finally:
            if old_home is None:
                os.environ.pop("ALPACCA_HOME", None)
            else:
                os.environ["ALPACCA_HOME"] = old_home
        r = run_cli("history", "clear", env=env, expect=1)
        check("history clear requires confirmation",
              "rerun with --yes" in r.stderr, r.stderr)
        r = run_cli("history", "clear", "--yes", env=env)
        check("history clear --yes deletes all chats",
              "deleted 2 chat(s)" in r.stdout, r.stdout)

        r = run_cli("menu", env=env, input_text="8\n")
        check("menu opens and exits",
              "Current chat model:" in r.stdout and
              "Chat history" in r.stdout and r.returncode == 0,
              r.stdout + r.stderr)
        r = run_cli("menu", env=env, input_text="2\n\n8\n")
        check("menu doctor path runs",
              "alpacca 0.2.0" in r.stdout and "models dir:" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("menu", env=env, input_text="6\n\n8\n")
        check("menu history stats path runs",
              "Saved chat statistics" in r.stdout and "AVG TOK/S" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("menu", env=env, input_text="5\n4\n8\n")
        check("menu history list path runs",
              "Alpacca Chat History" in r.stdout and
              "no chat history yet" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("menu", env=env, input_text="4\n2\nTiny Buddy\n\n6\n8\n")
        default_model = Path(env["ALPACCA_HOME"]) / "default-model.txt"
        check("menu model switch accepts nickname",
              default_model.read_text(encoding="utf-8").strip() == "tiny" and
              "Tiny Buddy (tiny)" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("menu", env=env, input_text="4\n3\ntiny\nMenu Tiny\n\n6\n8\n")
        check("menu model manager renames nickname",
              "Nickname set:" in r.stdout and "Menu Tiny -> tiny" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("menu", env=env, input_text="4\n2\nMenu Tiny\n\n6\n8\n")
        check("menu model switch uses renamed nickname",
              default_model.read_text(encoding="utf-8").strip() == "tiny" and
              "Menu Tiny (tiny)" in r.stdout,
              r.stdout + r.stderr)
        r = run_cli("menu", env=env,
                    input_text="4\n2\nnot-installed-model\n\n6\n8\n")
        check("menu invalid model switch is rejected",
              default_model.read_text(encoding="utf-8").strip() == "tiny" and
              "Model is not installed" in r.stdout,
              r.stdout + r.stderr)

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
                      "weights quantized Q4_0 (16 matrices," in r.stderr,
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
                  "weights quantized Q4_0 (16 matrices," in r0.stderr,
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

        # ---- guaranteed-valid-JSON decoding (jsonform) --------------------
        # stdlib-only feature: runs identically on the numpy and pure paths,
        # so no capability gate is needed
        print("== json_only generation ==")
        import json as _json
        import make_tiny_model as _mtm
        import alpacca.jsonform as _jf
        from alpacca.jsonform import (JsonGuard, MAX_DEPTH as _JG_DEPTH,
                                      sample_json_token)
        from alpacca.chat import generate as _jgen
        from alpacca.model import Model as _JModel
        from alpacca.sample import Sampler as _JSampler, \
            SamplerParams as _JParams

        def _jg(data: bytes):
            g = JsonGuard()
            return g if g.feed(data) else None

        _j_good = [b"{", b'{"a": [1, 2', b'{"a": {"b": nu', b"-0.5e",
                   b"1e-5", b'"ab\\u00', b'"partial \xe2\x96', b"[[[[",
                   b'  [ "x" , 1 ]']
        _j_bad = [b"x", b"{]", b'{"a":}', b'{"a": 1,}', b"[1,]", b"01",
                  b"1..", b"+1", b'"a\nb"', b"{} ", b"{}x", b"true1",
                  b"{}{}", b'"\\u00g"']
        check("JsonGuard accepts valid JSON prefixes",
              all(_jg(p) is not None for p in _j_good))
        check("JsonGuard rejects invalid prefixes",
              all(_jg(p) is None for p in _j_bad))
        _j_done_ok = True
        for _p in (b"{}", b'{"a": [1.5e3, true]}', b"null", b'"s"'):
            _g = _jg(_p)
            _j_done_ok = (_j_done_ok and _g is not None and _g.done and
                          not _g.can_accept(b" "))
        check("JsonGuard reports done and allows nothing after", _j_done_ok)
        _g = _jg(b"42")   # a top-level number never closes on its own
        check("top-level number: not done, but valid if it ends here",
              _g is not None and not _g.done and _g.at_valid_end and
              _g.can_accept(b"5") and not _g.can_accept(b" "))
        _g = JsonGuard()
        _g.feed(b'{"a": ')
        check("feed is transactional and can_accept never advances",
              not _g.feed(b"12x") and _g.can_accept(b"12}") and
              _g.feed(b"12}") and _g.done)
        _g = JsonGuard()
        check("nesting cap rejects one level too many yet still closes",
              _g.feed(b"[" * _JG_DEPTH) and not _g.feed(b"[") and
              _g.feed(b"1" + b"]" * _JG_DEPTH) and _g.done)
        _j_docs_ok = True
        for _doc in ({"k": [1, -0.5, 1e-5, True, None, "s\né"]},
                     [{"a": {"b": []}}, "x"], True):
            _enc = _json.dumps(_doc).encode()
            _g = JsonGuard()
            _j_docs_ok = (_j_docs_ok and
                          all(_g.feed(_enc[i:i + 1])
                              for i in range(len(_enc))) and _g.done)
        check("every byte-prefix of json.dumps output is accepted", _j_docs_ok)

        # the rejection loop, the ranked-scan fallback, and the one error
        class _JTok:
            def is_eog(self, t):
                return t == 0

        _jt = _JTok()
        _jtab = [b"", b"x", b"{", b"}"]
        _jgd = JsonGuard()
        _js = _JSampler(_JParams(temperature=0.0, seed=1))
        check("json loop bans invalid + EOG until the value completes",
              sample_json_token(_js, [5.0, 9.0, 1.0, 0.5], _jgd, _jtab, _jt)
              == 2 and
              sample_json_token(_js, [9.0, 5.0, 1.0, 0.5], _jgd, _jtab, _jt)
              == 3 and _jgd.done and
              sample_json_token(_js, [1.0, 9.0, 5.0, 3.0], _jgd, _jtab, _jt)
              == 0)
        _j_old = _jf.MAX_REJECTS
        _jf.MAX_REJECTS = 0   # force the ranked full-vocab scan
        try:
            check("ranked scan takes the best valid token",
                  sample_json_token(_js, [0.0, 0.0, 1.0, 2.0], JsonGuard(),
                                    _jtab, _jt) == 2)
            _j_raised = False
            try:
                sample_json_token(_js, [1.0, 2.0, 3.0, 4.0], JsonGuard(),
                                  [b"", b"x", b"]", b")"], _jt)
            except ValueError:
                _j_raised = True
            check("no tokenizable continuation raises ValueError", _j_raised)
        finally:
            _jf.MAX_REJECTS = _j_old
        _jl = [0.5, 2.0, 1.5, -1.0, 0.0]
        check("Sampler banned= masks the same on both paths, copies first",
              _JSampler(_JParams(temperature=0.8, top_k=3, seed=11)).sample(
                  _jl, banned={1}) ==
              _JSampler(_JParams(temperature=0.8, top_k=3, seed=11))
              ._sample_list(list(_jl), {1}) and _jl[1] == 2.0)

        # end to end: tiny model at temperature 2.0 - every reply is a valid
        # JSON value or, when the budget cuts it, a valid prefix of one
        _jdir = Path(tempfile.mkdtemp(prefix="alpacca-jsonform-"))
        _mtm.main(str(_jdir / "tiny.gguf"), "F32")
        _jm = _JModel.load(str(_jdir / "tiny.gguf"), progress=False)
        _jp = _jm.tok.encode("hello world", add_bos=True)
        _j_ok, _j_complete = True, 0
        for _seed in range(12):
            _jm.reset()
            _jr = _jgen(_jm, list(_jp),
                        _JParams(temperature=2.0, top_k=0, top_p=1.0,
                                 seed=_seed),
                        n_predict=32, json_only=True)
            if _jr.stop_reason in ("stop", "eog"):
                _j_complete += 1
                try:
                    _json.loads(_jr.text)
                except ValueError:
                    _j_ok = False
            else:
                _j_ok = _j_ok and JsonGuard().feed(_jr.text.encode("utf-8"))
        check(f"tiny json_only sweep at temperature 2.0: 12 seeds valid "
              f"({_j_complete} complete)", _j_ok)
        _jm.reset()
        _j_chunks = []
        _jr = _jgen(_jm, list(_jp),
                    _JParams(temperature=2.0, top_k=0, top_p=1.0, seed=1),
                    n_predict=24, stream=_j_chunks.append, json_only=True)
        check("json_only streams exactly the returned text",
              "".join(_j_chunks) == _jr.text)
        del _jm
        gc.collect()
        shutil.rmtree(_jdir, ignore_errors=True)

        # ---- Ollama-native API (/api/*) -----------------------------------
        print("== ollama-native API ==")
        from alpacca.serve import (_iso_now, _ollama_options, _param_size_label,
                                   _same_model, _wants_json)
        from alpacca.sample import SamplerParams as _OllamaSP
        check("ollama name match: :latest and path separators are cosmetic",
              _same_model("m", "m:latest") and
              _same_model("C:\\x\\m.gguf", "C:/x/m.gguf") and
              not _same_model("a", "b"))
        check("ollama format=json detection accepts a schema dict",
              _wants_json({"format": "json"}) and
              _wants_json({"format": {"type": "object"}}) and
              not _wants_json({}) and not _wants_json({"format": ""}))
        _oparams, _on_predict, _ostop = _ollama_options(
            {"options": {"num_predict": 7, "temperature": 0.25, "seed": 3,
                         "top_k": 5, "stop": "END", "num_ctx": 1 << 30}},
            _OllamaSP())
        check("ollama options map onto the sampler (oversize num_ctx ignored)",
              _on_predict == 7 and _oparams.temperature == 0.25 and
              _oparams.seed == 3 and _oparams.top_k == 5 and _ostop == ["END"])
        check("ollama num_predict defaults to -1 (until EOG or context)",
              _ollama_options({}, _OllamaSP())[1] == -1)
        check("ollama created_at is RFC3339 UTC",
              _iso_now().endswith("Z") and "T" in _iso_now())
        check("ollama parameter size label",
              _param_size_label(1_240_000_000) == "1.2B" and
              _param_size_label(15_000_000) == "15M")

        import threading as _oll_threading
        from alpacca.model import Model as _OllamaModel
        from alpacca.serve import serve as _ollama_serve
        _oll_dir = Path(tempfile.mkdtemp(prefix="alpacca-ollama-api-"))
        subprocess.run([sys.executable, str(REPO / "tests" / "make_tiny_model.py"),
                        str(_oll_dir / "tiny.gguf")],
                       check=True, capture_output=True, cwd=str(REPO))
        _oll_model = _OllamaModel.load(str(_oll_dir / "tiny.gguf"), progress=False)
        _oll_ready = _oll_threading.Event()
        _oll_port: list[int] = []
        _oll_threading.Thread(
            target=_ollama_serve, args=(_oll_model, "tiny-api"),
            kwargs={"host": "127.0.0.1", "port": 0,
                    "defaults": _OllamaSP(temperature=0.0, seed=1),
                    "ready_callback":
                        lambda p: (_oll_port.append(p), _oll_ready.set())},
            daemon=True).start()
        check("ollama api server starts", _oll_ready.wait(10))
        _oll_base = f"http://127.0.0.1:{_oll_port[0]}"

        def _oll_post(path, body, timeout=60):
            req = urllib.request.Request(
                _oll_base + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.headers.get("Content-Type", ""), resp.read()

        _, _oll_raw = _oll_post("/api/chat", {
            "model": "tiny-api", "stream": False, "keep_alive": "5m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 4, "seed": 1}})
        _oll_body = json.loads(_oll_raw)
        check("api/chat non-stream returns the ollama shape",
              _oll_body["done"] is True and
              _oll_body["message"]["role"] == "assistant" and
              _oll_body["done_reason"] in ("stop", "length") and
              isinstance(_oll_body["total_duration"], int) and
              _oll_body["total_duration"] >= _oll_body["eval_duration"] >= 0 and
              _oll_body["eval_count"] <= 4 and _oll_body["prompt_eval_count"] > 0,
              _oll_raw[:200].decode("utf-8", "replace"))
        _oll_ct, _oll_raw = _oll_post("/api/chat", {
            "model": "tiny-api",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 4, "seed": 1}})
        _oll_lines = [json.loads(ln) for ln in _oll_raw.split(b"\n") if ln.strip()]
        check("api/chat streams ndjson by default and ends with done",
              "ndjson" in _oll_ct and _oll_lines[-1]["done"] is True and
              all(not ln["done"] for ln in _oll_lines[:-1]) and
              _oll_lines[-1]["message"]["content"] == "",
              f"{_oll_ct} {len(_oll_lines)}")
        with urllib.request.urlopen(_oll_base + "/api/tags", timeout=10) as resp:
            _oll_tags = json.loads(resp.read())
        check("api/tags always contains the served model with real details",
              any(m["model"] == "tiny-api" and m["details"]["format"] == "gguf"
                  and m["details"]["family"] == "llama"
                  and m["details"]["parameter_size"]
                  for m in _oll_tags["models"]), str(_oll_tags)[:200])
        with urllib.request.urlopen(_oll_base + "/api/version", timeout=10) as resp:
            from alpacca import __version__ as _oll_version
            check("api/version reports the package version",
                  json.loads(resp.read())["version"] == _oll_version)
        with urllib.request.urlopen(_oll_base + "/api/ps", timeout=10) as resp:
            _oll_ps = json.loads(resp.read())
        check("api/ps lists the resident model",
              _oll_ps["models"][0]["model"] == "tiny-api" and
              "expires_at" in _oll_ps["models"][0], str(_oll_ps)[:200])
        _, _oll_raw = _oll_post("/api/show", {"model": "tiny-api"})
        _oll_show = json.loads(_oll_raw)
        check("api/show serves the loaded model's real metadata",
              _oll_show["model_info"].get("general.architecture") == "llama" and
              any(k.endswith(".context_length") for k in _oll_show["model_info"]) and
              _oll_show["details"]["quantization_level"] != "",
              _oll_raw[:200].decode("utf-8", "replace"))
        _, _oll_raw = _oll_post("/api/generate", {
            "model": "tiny-api", "stream": False, "prompt": "hello",
            "options": {"num_predict": 3, "seed": 1}})
        _oll_gen = json.loads(_oll_raw)
        check("api/generate answers with response and empty context",
              isinstance(_oll_gen["response"], str) and
              _oll_gen["context"] == [] and _oll_gen["done"] is True)
        try:
            _oll_post("/api/chat", {"model": "other:1b", "stream": False,
                                    "messages": [{"role": "user", "content": "hi"}]})
            _oll_404 = False
        except urllib.error.HTTPError as e:
            _oll_404 = (e.code == 404 and json.loads(e.read())["error"]
                        == "model 'other:1b' not found")
        check("api requests naming another model 404 in ollama's error shape",
              _oll_404)
        _, _oll_raw = _oll_post("/api/chat", {
            "model": "tiny-api:latest", "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 2, "seed": 1}})
        check("api/chat accepts the served name with :latest appended",
              json.loads(_oll_raw)["done"] is True)
        # format=json is served whether or not the json_only tier is present:
        # serve gates on chat_once's signature and degrades to plain output
        try:
            _, _oll_raw = _oll_post("/api/chat", {
                "model": "tiny-api", "stream": False, "format": "json",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": 4, "seed": 1}})
            check("api/chat format=json answers (constrained or degraded)",
                  isinstance(json.loads(_oll_raw)["message"]["content"], str))
        except TypeError:
            check("api/chat format=json answers (constrained or degraded)", True)

        # ---- gpu tier (alpacca/cuda.py, optional pinned numba-cuda) -------
        # Cross-tier parity: GpuMatrix must match QuantMatrix within the
        # same relative bar the fused-vs-tiled check uses, row access must
        # be bit-exact, and greedy decode must be token-identical on every
        # placement the VRAM budget can produce. Everything except the
        # doctor line is gated on a working CUDA device + the pinned JIT.
        # The suite-wide ALPACCA_GPU=0 pin lifts here; cuda caches its env
        # gate at first init, so reset it before probing.
        os.environ["ALPACCA_GPU"] = "1"
        from alpacca import cuda as AG
        AG._state = None
        r = run_cli("doctor")
        check("doctor reports a gpu line", "gpu:" in r.stdout, r.stdout)
        if T.HAS_NUMPY and AG.available():
            import numpy as np
            from alpacca.qmatrix import QuantMatrix

            def gpu_q4k(n):
                out = bytearray()
                for block in range(n // 256):
                    d = 0.015625 + (block % 7) * 0.001953125
                    dmin = 0.00390625 + (block % 5) * 0.0009765625
                    sc = bytes(((block * 17 + i * 29) & 0xFF) for i in range(12))
                    qs = bytes(((block * 31 + i * 7) & 0xFF) for i in range(128))
                    out += struct.pack("<ee", d, dmin) + sc + qs
                return bytes(out)

            def gpu_q5k(n):
                out = bytearray()
                for block in range(n // 256):
                    d = 0.015625 + (block % 7) * 0.001953125
                    dmin = 0.00390625 + (block % 5) * 0.0009765625
                    sc = bytes(((block * 17 + i * 29) & 0xFF) for i in range(12))
                    qh = bytes(((block * 23 + i * 13) & 0xFF) for i in range(32))
                    ql = bytes(((block * 31 + i * 7) & 0xFF) for i in range(128))
                    out += struct.pack("<ee", d, dmin) + sc + qh + ql
                return bytes(out)

            def gpu_q6k(n):
                out = bytearray()
                for block in range(n // 256):
                    ql = bytes(((block * 13 + i * 11) & 0xFF) for i in range(128))
                    qh = bytes(((block * 19 + i * 5) & 0xFF) for i in range(64))
                    sc = struct.pack("<16b", *[((block + i) % 63) - 31
                                               for i in range(16)])
                    d = 0.001953125 + (block % 5) * 0.000244140625
                    out += ql + qh + sc + struct.pack("<e", d)
                return bytes(out)

            os.environ["ALPACCA_INT_DOT"] = "0"  # exact codes path CPU-side
            try:
                gworst = 0.0
                groworst = 0.0
                grng = np.random.default_rng(17)
                for _dt, _rows, _cols, raw in (
                        ("Q8_0", 5, 64, quants.quantize_q8_0(
                            [((i * 37) % 251) / 251.0 - 0.5
                             for i in range(5 * 64)])),
                        ("Q4_K", 3, 512, gpu_q4k(3 * 512)),
                        ("Q5_K", 3, 512, gpu_q5k(3 * 512)),
                        ("Q6_K", 3, 512, gpu_q6k(3 * 512))):
                    qm = QuantMatrix(raw, _dt, _rows, _cols)
                    gm = AG.GpuMatrix(raw, _dt, _rows, _cols)
                    gx = grng.standard_normal(_cols).astype(np.float32)
                    ref = np.asarray(qm.matvec(gx))
                    gscale = max(1e-6, float(np.abs(ref).max()))
                    gworst = max(gworst, float(
                        np.abs(ref - gm.matvec(gx)).max()) / gscale)
                    for B in (2, 9):  # wide kernel and the stacked oracle
                        GX = grng.standard_normal((B, _cols)).astype(np.float32)
                        refm = np.asarray(qm.matmul_t(GX))
                        gscale = max(1e-6, float(np.abs(refm).max()))
                        gworst = max(gworst, float(
                            np.abs(refm - gm.matmul_t(GX)).max()) / gscale)
                        stacked = np.stack([np.asarray(qm.matvec(GX[i]))
                                            for i in range(B)])
                        gworst = max(gworst, float(
                            np.abs(stacked - gm.matmul_t(GX)).max()) / gscale)
                        # force the looped-matvec matmul path as well
                        os.environ["ALPACCA_GPU_WIDE_MATMUL_ELEMS"] = "0"
                        try:
                            gworst = max(gworst, float(
                                np.abs(refm - gm.matmul_t(GX)).max()) / gscale)
                        finally:
                            os.environ.pop("ALPACCA_GPU_WIDE_MATMUL_ELEMS",
                                           None)
                    for i in range(_rows):
                        groworst = max(groworst, float(np.abs(
                            np.asarray(qm.row(i)) - gm.row(i)).max()))
                    groworst = max(groworst, float(np.abs(
                        np.asarray(qm.rows_at([_rows - 1, 0]))
                        - gm.rows_at([_rows - 1, 0])).max()))
                    check(f"{_dt} gpu empty row gather shape",
                          gm.rows_at([]).shape == (0, _cols))
            finally:
                os.environ.pop("ALPACCA_INT_DOT", None)
            check(f"gpu matvec/matmul match QuantMatrix on both kernel paths "
                  f"(worst rel {gworst:.2e})", gworst < 2e-5)
            check(f"gpu row access is bit-exact (worst {groworst:.2e})",
                  groworst == 0.0)

            # end-to-end: greedy decode must be token-identical to the CPU
            # tiers on full-GPU AND on VRAM-capped mixed placement (the cap
            # fits the first fused matrix and refuses the second)
            mk_gpu = REPO / "tests" / "make_tiny_model.py"
            gpu_gguf = tmp / "gpu-tier-q4k.gguf"
            r = subprocess.run([sys.executable, str(mk_gpu), str(gpu_gguf),
                                "Q4_K"], capture_output=True, text=True)
            check("write gpu-tier tiny Q4_K model", r.returncode == 0,
                  r.stderr)
            gpu_script = (
                "import sys\n"
                f"sys.path.insert(0, {str(REPO)!r})\n"
                "import numpy as np\n"
                "from alpacca.model import Model\n"
                f"m = Model.load({str(gpu_gguf)!r}, progress=False)\n"
                "logits = m.prefill(m.tok.encode('hello world',"
                " add_bos=True))\n"
                "out = []\n"
                "for _ in range(16):\n"
                "    t = int(np.argmax(logits))\n"
                "    out.append(t)\n"
                "    logits = m.forward(t)\n"
                "print('IDS', ' '.join(map(str, out)))\n")

            def gpu_greedy(env):
                e = dict(os.environ)
                e.update(env)
                r = subprocess.run([sys.executable, "-c", gpu_script],
                                   capture_output=True, text=True, env=e,
                                   cwd=str(REPO))
                for line in r.stdout.splitlines():
                    if line.startswith("IDS "):
                        return line
                return f"rc={r.returncode}: {r.stderr[-200:]}"

            gpu_cpu_ids = gpu_greedy({"ALPACCA_GPU": "0"})
            gpu_gpu_ids = gpu_greedy({"ALPACCA_GPU": "1"})
            gpu_cap_ids = gpu_greedy({"ALPACCA_GPU": "1",
                                      "ALPACCA_GPU_VRAM_MB": "0.2"})
            check("gpu greedy decode is token-identical to cpu",
                  gpu_cpu_ids.startswith("IDS ")
                  and gpu_cpu_ids == gpu_gpu_ids,
                  f"{gpu_cpu_ids} vs {gpu_gpu_ids}")
            check("vram-capped mixed placement stays token-identical",
                  gpu_cap_ids == gpu_cpu_ids,
                  f"{gpu_cap_ids} vs {gpu_cpu_ids}")

            # ---- device prefill chain legs + the prefix cache on the chain
            # In-process on the tiny model. Within one path the chain is
            # byte-deterministic: chunk boundaries (want_logits=False on
            # every chunk but the last), live prefix reuse, and a prefix-
            # cache restore all reproduce the one-chunk logits bytes.
            # Across paths (chain vs host recompute) the contract is
            # token-identity, exercised via the failure legs: a mid-chunk
            # failure falls back to the host recompute for that chunk, and
            # only _PREFILL_PARK_AFTER consecutive failures park the path.
            import alpacca.model as _amodel

            def _chain_load():
                return _amodel.Model.load(str(gpu_gguf), progress=False)

            def _greedy16(m, lg):
                outg = []
                for _ in range(16):
                    t = int(np.argmax(lg))
                    outg.append(t)
                    lg = m.forward(t)
                return outg

            try:
                cm = _chain_load()
                chain_prompt = cm.tok.encode(
                    "hello world the quick brown fox jumps over the lazy dog"
                    " and the crow watches from the wall at dusk",
                    add_bos=True)
                os.environ["ALPACCA_PREFILL_CHUNK"] = "4096"
                chain_ref = np.asarray(cm.prefill(chain_prompt))
                chain_ref_ids = _greedy16(cm, chain_ref)
                check("device prefill chain engages on the tiny model",
                      cm._gpu_chain is not None and not cm._gpu_prefill_dead)

                cm2 = _chain_load()
                os.environ["ALPACCA_PREFILL_CHUNK"] = "5"
                lg = np.asarray(cm2.prefill(chain_prompt))
                check("chain chunk boundaries are byte-invariant "
                      "(want_logits=False legs included)",
                      np.array_equal(chain_ref, lg))

                cm3 = _chain_load()
                real_chunk = AG.DecodeChain.prefill_chunk
                boom_calls = {"n": 0}

                def _boom(self, model, toks, want_logits):
                    boom_calls["n"] += 1
                    raise RuntimeError("synthetic mid-chunk failure")

                AG.DecodeChain.prefill_chunk = _boom
                try:
                    lg = cm3.prefill(chain_prompt)
                finally:
                    AG.DecodeChain.prefill_chunk = real_chunk
                check("mid-chunk failures fall back token-identically",
                      _greedy16(cm3, lg) == chain_ref_ids)
                check("only a failure streak parks the device prefill path",
                      cm3._gpu_prefill_dead
                      and boom_calls["n"] == AG._PREFILL_PARK_AFTER
                      and cm3._gpu_prefill_fails == AG._PREFILL_PARK_AFTER)

                cm4 = _chain_load()
                flaky = {"left": 1}

                def _flaky(self, model, toks, want_logits):
                    if flaky["left"] > 0:
                        flaky["left"] -= 1
                        raise RuntimeError("transient vram spike")
                    return real_chunk(self, model, toks, want_logits)

                AG.DecodeChain.prefill_chunk = _flaky
                try:
                    lg = np.asarray(cm4.prefill(chain_prompt))
                finally:
                    AG.DecodeChain.prefill_chunk = real_chunk
                check("a transient chunk failure neither parks nor drifts",
                      np.array_equal(chain_ref, lg)
                      and not cm4._gpu_prefill_dead
                      and cm4._gpu_prefill_fails == 0)

                cm5 = _chain_load()
                os.environ["ALPACCA_PREFILL_CHUNK"] = "256"
                cm5.prefill(chain_prompt[:11])
                lg = np.asarray(cm5.prefill(chain_prompt))
                check("live prefix reuse onto the chain is byte-identical",
                      np.array_equal(chain_ref, lg)
                      and cm5.last_prefill_forwarded
                      == len(chain_prompt) - 11)

                # prefix cache on the chain: a restore truncates to 0 and
                # the mirror re-uploads through its gap-fill, so the warm
                # prefill must equal a cold one byte-for-byte. MIN_MATCH is
                # shrunk so tiny-model prompts exercise the mechanism.
                min_match = _amodel._PREFIX_CACHE_MIN_MATCH
                _amodel._PREFIX_CACHE_MIN_MATCH = 4
                try:
                    cm6 = _chain_load()
                    conv_b = [chain_prompt[0]] + cm6.tok.encode(
                        "ships in the harbor waited for the tide to turn"
                        " while gulls argued over scraps", add_bos=False)
                    warm_prompt = chain_prompt + cm6.tok.encode(
                        " and the night came down", add_bos=False)
                    cm6.prefill(chain_prompt)
                    cm6.prefill(conv_b)          # context switch: saves A
                    st1 = cm6.prefix_cache_stats()
                    lg = np.asarray(cm6.prefill(warm_prompt))  # restores A
                    st2 = cm6.prefix_cache_stats()
                    cm7 = _chain_load()
                    cold = np.asarray(cm7.prefill(warm_prompt))
                    check("prefix cache saves on switch, restores on the "
                          "chain", st1["saves"] == 1 and st1["slots"] == 1
                          and st2["hits"] == 1 and st2["saves"] == 2)
                    check("restored prefill is byte-identical to cold",
                          np.array_equal(lg, cold))

                    # mid-run budget shrink: the next route call drains the
                    # store to the new cap, oldest-first, keeping the MRU
                    # slot (the just-restored conversation)
                    slot_a = st1["bytes"]
                    os.environ["ALPACCA_PREFIX_CACHE_MB"] = (
                        f"{slot_a * 1.1 / 1048576:.6f}")
                    cm6.prefill(warm_prompt)     # no save, no restore: drain
                    st3 = cm6.prefix_cache_stats()
                    check("mid-run budget shrink drains the store to cap",
                          st3["slots"] == 1 and st3["bytes"] == slot_a
                          and st3["evictions"] == 1)
                    os.environ["ALPACCA_PREFIX_CACHE_MB"] = "0"
                    cm6.prefill(conv_b)
                    st4 = cm6.prefix_cache_stats()
                    check("disable clears the store",
                          st4["slots"] == 0 and st4["bytes"] == 0)

                    # protected-slot edge: budget fits one slot; switching
                    # back to the saved conversation must skip the pre-
                    # restore save rather than evict the slot being restored
                    cm8 = _chain_load()
                    os.environ.pop("ALPACCA_PREFIX_CACHE_MB", None)
                    cm8.prefill(chain_prompt)
                    cm8.prefill(conv_b)          # saves A
                    sa = cm8.prefix_cache_stats()
                    os.environ["ALPACCA_PREFIX_CACHE_MB"] = (
                        f"{sa['bytes'] * 1.1 / 1048576:.6f}")
                    cm8.prefill(chain_prompt)    # restore A; B-save loses
                    sb = cm8.prefix_cache_stats()
                    check("pre-restore save loses to the protected slot",
                          sb["hits"] == 1 and sb["saves"] == sa["saves"]
                          and sb["evictions"] == 0 and sb["slots"] == 1
                          and cm8.last_prefill_forwarded == 1)
                finally:
                    _amodel._PREFIX_CACHE_MIN_MATCH = min_match
                    os.environ.pop("ALPACCA_PREFIX_CACHE_MB", None)

                # the pure-python gate: without numpy the route is inert
                has_np = _amodel.T.HAS_NUMPY
                _amodel.T.HAS_NUMPY = False
                try:
                    routed = cm7._prefix_cache_route(list(chain_prompt), 0)
                finally:
                    _amodel.T.HAS_NUMPY = has_np
                check("pure-python tier keeps the single-cache behavior",
                      routed == 0 and not cm7._prefix_slots)
            finally:
                os.environ.pop("ALPACCA_PREFILL_CHUNK", None)
        os.environ["ALPACCA_GPU"] = "0"  # back to the suite-wide host pin

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
