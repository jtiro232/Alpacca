# Alpacca - command-line interface. MIT License. See LICENSE.
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .sample import SamplerParams
from .store import (LocalModel, find_local, human_size, list_models,
                    models_root, parse_model_ref, remove_model)

EXAMPLES = """\
model references:
  llama3.2:1b                      Ollama registry (registry.ollama.ai)
  ollama:user/model:tag            Ollama registry, user namespace
  hf:org/repo  |  org/repo         Hugging Face repo (best GGUF quant)
  hf:org/repo:Q4_K_M               Hugging Face repo, specific quant/file
  ./path/to/model.gguf             local GGUF file

examples:
  alpacca pull llama3.2:1b
  alpacca run llama3.2:1b                        # interactive chat
  alpacca run llama3.2:1b "why is the sky blue?" # one-shot
  alpacca serve llama3.2:1b --port 8080          # OpenAI-compatible API
"""


def _resolve_or_pull(name: str, auto_pull: bool = True) -> LocalModel:
    from .pull import pull_model
    ref = parse_model_ref(name)
    local = find_local(ref)
    if local is not None:
        return local
    if ref.source == "file":
        raise SystemExit(f"alpacca: model file not found: {ref.path}")
    if not auto_pull:
        raise SystemExit(f"alpacca: {ref.display()} is not installed "
                         f"(try `alpacca pull {name}`)")
    print(f"{ref.display()} is not installed yet - pulling it first", file=sys.stderr)
    return pull_model(ref)


def _sampler_params(args) -> SamplerParams:
    p = SamplerParams()
    if args.temp is not None:
        p.temperature = args.temp
    if args.top_k is not None:
        p.top_k = args.top_k
    if args.top_p is not None:
        p.top_p = args.top_p
    if args.repeat_penalty is not None:
        p.repeat_penalty = args.repeat_penalty
    if args.seed is not None:
        p.seed = args.seed
    return p


def _apply_manifest_defaults(local: LocalModel, args) -> None:
    """Model-supplied parameters (e.g. from an Ollama params layer) act as
    defaults; explicit flags win."""
    params = local.manifest.get("params") or {}
    if args.temp is None and "temperature" in params:
        args.temp = float(params["temperature"])
    if args.top_k is None and "top_k" in params:
        args.top_k = int(params["top_k"])
    if args.top_p is None and "top_p" in params:
        args.top_p = float(params["top_p"])
    if args.repeat_penalty is None and "repeat_penalty" in params:
        args.repeat_penalty = float(params["repeat_penalty"])
    if args.ctx == 0 and "num_ctx" in params:
        args.ctx = int(params["num_ctx"])
    if not args.system and local.manifest.get("system"):
        args.system = local.manifest["system"]


def _cgroup_limit_remaining_mb() -> float | None:
    """Remaining memory under a cgroup limit (containers), if any."""
    try:  # cgroup v2
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if raw != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            return max(0.0, (int(raw) - used) / (1024.0 * 1024.0))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        limit = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text())
        if limit < (1 << 60):  # v1 reports ~2^63 when unlimited
            used = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())
            return max(0.0, (limit - used) / (1024.0 * 1024.0))
    except (OSError, ValueError):
        pass
    return None


def _available_ram_mb() -> float | None:
    """Best-effort available physical RAM in MiB, standard library only."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    meminfo_mb = int(line.split()[1]) / 1024.0
                    cg = _cgroup_limit_remaining_mb()
                    return min(meminfo_mb, cg) if cg is not None else meminfo_mb
        elif sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32),
                            ("dwMemoryLoad", ctypes.c_uint32),
                            ("ullTotalPhys", ctypes.c_uint64),
                            ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64),
                            ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64),
                            ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / (1024.0 * 1024.0)
        elif sys.platform == "darwin":
            # macOS has no MemAvailable equivalent in the stdlib; use half
            # of physical RAM as a conservative stand-in
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return total / (1024.0 * 1024.0) / 2.0
    except Exception:
        return None
    return None


def _auto_dense_budget_mb(avail_mb: float, file_mb: float, n_ctx: int = 0) -> int:
    """Dense-weight budget (MiB) that fits beside the quantized residue,
    KV cache, and runtime baseline, with headroom kept free.

    Conservative on purpose: it reserves the full quantized size even
    though densified matrices never allocate their quantized form, and
    spends 85% of what is left. The KV/runtime reserve scales with
    explicitly requested context windows beyond the default 4096 clamp.
    """
    reserve = 1.2 * file_mb + 2048.0 * max(1.0, n_ctx / 4096.0)
    return max(0, int(0.85 * (avail_mb - reserve)))


def _maybe_auto_dense_budget(local: LocalModel, n_ctx: int = 0) -> None:
    """Default `alpacca run`/`serve` to the fastest storage this machine
    affords: size ALPACCA_DENSE_WEIGHT_MB from available RAM unless the
    user pinned it (any value - `0` keeps everything quantized). This is
    CLI policy; the library default (Model.load) stays opt-in."""
    from . import tensor
    if not tensor.HAS_NUMPY or os.environ.get("ALPACCA_F32"):
        return
    if os.environ.get("ALPACCA_DENSE_WEIGHT_MB") is not None:
        return
    avail = _available_ram_mb()
    if avail is None:
        print("alpacca: could not detect available RAM; keeping weights "
              "quantized (set ALPACCA_DENSE_WEIGHT_MB to choose a dense "
              "budget)", file=sys.stderr)
        return
    try:
        file_mb = local.model_path.stat().st_size / (1024.0 * 1024.0)
    except OSError:
        return
    budget = _auto_dense_budget_mb(avail, file_mb, n_ctx)
    if budget <= 0:
        return
    os.environ["ALPACCA_DENSE_WEIGHT_MB"] = str(budget)
    print(f"auto dense-weight budget: {budget} MiB "
          f"(~{avail:.0f} MiB RAM available; "
          f"set ALPACCA_DENSE_WEIGHT_MB=0 to keep weights quantized)",
          file=sys.stderr)


def _load_model(local: LocalModel, args):
    from .model import Model
    _maybe_auto_dense_budget(local, args.ctx)
    print(f"loading {local.model_path.name}...", file=sys.stderr)
    m = Model.load(str(local.model_path), n_ctx=args.ctx)
    print(m.describe(), file=sys.stderr)
    return m


def cmd_pull(args) -> int:
    from .pull import pull_model
    pull_model(parse_model_ref(args.model), force=args.force, verify=not args.no_verify)
    return 0


def cmd_list(_args) -> int:
    models = list_models()
    if not models:
        print("no models installed - try: alpacca pull llama3.2:1b")
        return 0
    width = max(4, max(len(m["name"]) for m in models))
    print(f"{'NAME':<{width}}  {'SOURCE':<8}  {'SIZE':<10}  PULLED")
    for m in models:
        print(f"{m['name']:<{width}}  {m['source']:<8}  "
              f"{human_size(m['size']):<10}  {m['pulled_at']}")
    return 0


def cmd_rm(args) -> int:
    rc = 0
    for name in args.models:
        ref = parse_model_ref(name)
        if remove_model(ref):
            print(f"removed {ref.display()}")
        else:
            print(f"alpacca: {ref.display()} is not installed", file=sys.stderr)
            rc = 1
    return rc


def cmd_show(args) -> int:
    import json
    ref = parse_model_ref(args.model)
    local = find_local(ref)
    if local is None:
        raise SystemExit(f"alpacca: {ref.display()} is not installed")
    print(json.dumps(local.manifest or {"model_file": str(local.model_path)}, indent=2))
    if local.dir:
        print(f"\nfiles in {local.dir}:")
        for f in sorted(local.dir.iterdir()):
            if f.is_file():
                print(f"  {f.name:<28} {human_size(f.stat().st_size)}")
    if args.metadata:
        from .gguf import GGUFFile
        with GGUFFile.open(local.model_path) as gf:
            print("\nGGUF metadata:")
            for k, v in gf.metadata.items():
                s = str(v)
                print(f"  {k} = {s[:80] + '...' if len(s) > 80 else s}")
    return 0


def cmd_run(args) -> int:
    local = _resolve_or_pull(args.model)
    _apply_manifest_defaults(local, args)
    model = _load_model(local, args)
    params = _sampler_params(args)

    from . import chat
    if args.prompt:
        prompt = " ".join(args.prompt)
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        res = chat.chat_once(model, messages, params, args.n_predict,
                             stream=lambda s: print(s, end="", flush=True))
        print()
        print(f"[{res.tokens} tokens, {res.tok_per_sec:.1f} tok/s]", file=sys.stderr)
        return 0
    chat.interactive(model, params, system=args.system, n_predict=args.n_predict)
    return 0


def cmd_serve(args) -> int:
    local = _resolve_or_pull(args.model)
    _apply_manifest_defaults(local, args)
    model = _load_model(local, args)
    from .serve import serve
    host = args.host or os.environ.get("ALPACCA_HOST", "127.0.0.1")
    port = args.port if args.port is not None else int(os.environ.get("ALPACCA_PORT", "8080"))
    serve(model, parse_model_ref(args.model).display(), host, port,
          defaults=_sampler_params(args))
    return 0


def cmd_doctor(_args) -> int:
    from . import tensor
    print(f"alpacca {__version__} (from-scratch python engine)")
    print(f"python:      {sys.version.split()[0]} ({sys.executable})")
    print(f"backend:     {tensor.backend_name()}"
          + ("  (optional accelerator active)" if tensor.HAS_NUMPY
             else "  (pip install numpy for big speedups)"))
    root = models_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        ok = "ok"
    except OSError as e:
        ok = f"NOT WRITABLE ({e})"
    print(f"models dir:  {root} ({ok})")
    n = len(list_models())
    print(f"installed:   {n} model(s)" + ("" if n else " - try `alpacca pull llama3.2:1b`"))
    return 0


def cmd_tokenize(args) -> int:
    local = _resolve_or_pull(args.model, auto_pull=False)
    from .gguf import GGUFFile
    from .tokenizer import Tokenizer
    with GGUFFile.open(local.model_path) as gf:
        tok = Tokenizer.from_gguf(gf.metadata)
    ids = tok.encode(args.text)
    for i in ids:
        print(f"{i:>8}  {ascii(tok.piece(i))}")
    return 0


def _add_model_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ctx", "-c", type=int, default=0, help="context window (tokens)")
    p.add_argument("--temp", type=float, default=None, help="sampling temperature")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repeat-penalty", type=float, default=None)
    p.add_argument("--seed", "-s", type=int, default=None)
    p.add_argument("--n-predict", "-n", type=int, default=-1,
                   help="max tokens to generate (-1 = until end)")
    p.add_argument("--system", "-sys", default="", help="system prompt")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to legacy code pages (cp1252); model output
    # is arbitrary UTF-8 and must never crash the CLI.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    ap = argparse.ArgumentParser(
        prog="alpacca",
        description="alpacca - LLMs in your terminal, implemented in pure Python",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", "-v", action="version",
                    version=f"alpacca {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("pull", help="download a model into ~/.alpacca/models")
    p.add_argument("model")
    p.add_argument("--force", "-f", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("run", help="chat with a model (one-shot if a prompt is given)")
    p.add_argument("model")
    p.add_argument("prompt", nargs="*")
    _add_model_flags(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="OpenAI-compatible API server")
    p.add_argument("model")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    _add_model_flags(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("list", aliases=["ls"], help="list installed models")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("rm", aliases=["remove"], help="remove installed models")
    p.add_argument("models", nargs="+")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("show", help="show a model's manifest and files")
    p.add_argument("model")
    p.add_argument("--metadata", action="store_true", help="dump GGUF metadata too")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("tokenize", help="show how text tokenizes for a model")
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-p", "--text", required=True)
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("doctor", help="check the installation")
    p.set_defaults(func=cmd_doctor)

    args = ap.parse_args(argv)
    if not args.command:
        ap.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError) as e:
        print(f"alpacca: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
