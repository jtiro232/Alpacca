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
import os
import shutil
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

        from alpacca.tokenizer import pretokenize
        toks = pretokenize("Hello there, world! It's 2026...\n  indented")
        check("BPE pretokenizer splits text", "".join(toks) == "Hello there, world! It's 2026...\n  indented",
              str(toks))

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
        for dtype, name in (("F32", "model.gguf"), ("Q4_0", "tiny-q4.gguf")):
            r = subprocess.run([sys.executable, str(mk), str(srv / name), dtype],
                               capture_output=True, text=True)
            check(f"write tiny {dtype} model", r.returncode == 0, r.stderr)

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
