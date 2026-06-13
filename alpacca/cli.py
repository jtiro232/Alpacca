# Alpacca - command-line interface. MIT License. See LICENSE.
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .sample import SamplerParams
from .store import (LocalModel, alpacca_home, find_local, human_size, list_models,
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
  alpacca menu                                  # terminal app menu
  alpacca run llama3.2:1b                        # interactive chat
  alpacca run llama3.2:1b "why is the sky blue?" # one-shot
  alpacca history list                           # list saved chats
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
    from . import kernels
    if kernels.available():
        # fused quantized kernels read ~1.1-1.3 B/weight at native speed:
        # faster than dense BLAS (4 B/weight) AND ~3x less RAM, so the
        # fastest default is to keep everything quantized
        kernels.warmup()
        print(f"{kernels.status()}: keeping weights quantized "
              f"(fastest path, lowest RAM)", file=sys.stderr)
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
    # exact-fit first: if the machine can hold every densifiable matrix
    # (checked from the GGUF header against the residual quantized storage
    # and the KV cache), spend exactly that; otherwise fall back to the
    # conservative formula
    from .model import auto_budget_fit_mb
    budget = 0
    fit = auto_budget_fit_mb(str(local.model_path), n_ctx)
    if fit is not None:
        eligible_mb, fixed_mb = fit
        if eligible_mb > 0 and avail - 1024.0 >= eligible_mb + fixed_mb:
            budget = int(eligible_mb) + 1
    if budget <= 0:
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
    chat.interactive(model, params, system=args.system, n_predict=args.n_predict,
                     model_name=parse_model_ref(args.model).display(),
                     model_path=str(local.model_path))
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


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[:max(0, width - 3)].rstrip() + "..."


def _default_model_file() -> Path:
    return alpacca_home() / "default-model.txt"


def _read_default_model() -> str:
    try:
        value = _default_model_file().read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    if value:
        return value
    models = list_models()
    if models:
        return models[0]["name"]
    return "llama3.2:1b"


def _write_default_model(model_ref: str) -> None:
    path = _default_model_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model_ref.strip() + "\n", encoding="utf-8")


def _prompt_line(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _menu_pause() -> None:
    _prompt_line("Press Enter to continue...")


def _menu_error(e) -> None:
    print(f"alpacca: error: {e}", file=sys.stderr)


def _print_installed_models() -> None:
    print("Installed models:")
    cmd_list(argparse.Namespace())


def _menu_run_model() -> None:
    print("\nChat with a model\n")
    _print_installed_models()
    current = _read_default_model()
    print(f"\nCurrent chat model:\n  {current}\n")
    model_ref = _prompt_line("Model reference (blank = current): ").strip() or current
    print()
    args = argparse.Namespace(
        model=model_ref, prompt=[], ctx=0, temp=None, top_k=None, top_p=None,
        repeat_penalty=None, seed=None, n_predict=-1, system="")
    try:
        rc = cmd_run(args)
    except (RuntimeError, ValueError, SystemExit) as e:
        _menu_error(e)
        rc = 1
    if rc:
        _menu_pause()


def _menu_model_manager() -> None:
    while True:
        print("\nAlpacca Model Manager\n")
        _print_installed_models()
        print("\n1. Add/download a model")
        print("2. Switch chat model")
        print("3. Show model details")
        print("4. Delete an installed model")
        print("5. Back to main menu\n")
        choice = _prompt_line("Choose an option [1-5]: ").strip()
        if choice in ("", "5"):
            return
        if choice == "1":
            print("\nEnter any supported Alpacca model reference.")
            print("Examples: llama3.2:1b, qwen2.5:0.5b, "
                  "hf:NousResearch/Hermes-3-Llama-3.1-8B")
            model_ref = _prompt_line("Model reference (blank to cancel): ").strip()
            if model_ref:
                try:
                    cmd_pull(argparse.Namespace(model=model_ref, force=False,
                                                no_verify=False))
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()
        elif choice == "2":
            model_ref = _prompt_line(
                "New chat model, exactly as shown in NAME (blank to cancel): "
            ).strip()
            if not model_ref:
                continue
            try:
                local = find_local(parse_model_ref(model_ref))
            except ValueError as e:
                print(f"alpacca: error: {e}", file=sys.stderr)
                local = None
            if local is None:
                print("Model is not installed or the reference is invalid.")
                print("Use Add/download first, then switch to the installed model.")
                _menu_pause()
                continue
            _write_default_model(parse_model_ref(model_ref).display())
            print(f"Chat model set to:\n  {_read_default_model()}")
            _menu_pause()
        elif choice == "3":
            model_ref = _prompt_line("Model reference to inspect (blank to cancel): ").strip()
            if model_ref:
                try:
                    cmd_show(argparse.Namespace(model=model_ref, metadata=False))
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()
        elif choice == "4":
            model_ref = _prompt_line("Model reference to delete (blank to cancel): ").strip()
            if not model_ref:
                continue
            confirm = _prompt_line("Type DELETE to confirm: ")
            if confirm.upper() == "DELETE":
                try:
                    cmd_rm(argparse.Namespace(models=[model_ref]))
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()


def _menu_history() -> None:
    while True:
        print("\nAlpacca Chat History\n")
        _history_list()
        print("\n1. View a chat")
        print("2. Delete one chat")
        print("3. Delete all chat history")
        print("4. Back to main menu\n")
        choice = _prompt_line("Choose an option [1-4]: ").strip()
        if choice in ("", "4"):
            return
        if choice == "1":
            chat = _prompt_line("Chat number or ID to view (blank to cancel): ").strip()
            if chat:
                try:
                    _history_show(chat)
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()
        elif choice == "2":
            chat = _prompt_line("Chat number or ID to delete (blank to cancel): ").strip()
            if chat:
                try:
                    cmd_history(argparse.Namespace(history_command="rm", chats=[chat]))
                except (RuntimeError, ValueError, SystemExit) as e:
                    _menu_error(e)
                _menu_pause()
        elif choice == "3":
            confirm = _prompt_line("Type DELETE to delete all chat history: ")
            if confirm.upper() == "DELETE":
                cmd_history(argparse.Namespace(history_command="clear", yes=True))
                _menu_pause()


def _print_controls() -> None:
    print("\nAlpacca Controls\n")
    print("Core commands:")
    print("  alpacca menu")
    print("  alpacca list")
    print("  alpacca pull <model>")
    print("  alpacca run <model> [prompt text]")
    print("  alpacca serve <model> [--host HOST] [--port PORT]")
    print("  alpacca history list|show|stats|rm|clear --yes")
    print("  alpacca show <model> [--metadata]")
    print("  alpacca rm <model> [more models...]")
    print("  alpacca tokenize -m <model> -p \"text\"")
    print("  alpacca doctor")
    print("\nInteractive chat:")
    print("  Esc or /exit returns to the menu/caller")
    print("  /clear resets the current conversation")
    print("\nUseful environment variables:")
    print("  ALPACCA_HOME changes the model/history/default-model store")
    print("  ALPACCA_DENSE_WEIGHT_MB=0 keeps weights fully quantized")
    print("  ALPACCA_KERNELS=0 disables optional pinned JIT kernels")
    print("  ALPACCA_PURE=1 forces the standard-library backend")


def cmd_menu(_args) -> int:
    """Repo-owned local terminal app menu."""
    while True:
        print("\nAlpacca\n")
        _print_installed_models()
        print(f"\nCurrent chat model:\n  {_read_default_model()}\n")
        print("1. Chat with current or selected model")
        print("2. Alpacca doctor")
        print("3. Open Alpacca shell")
        print("4. Model manager")
        print("5. Chat history")
        print("6. Saved chat statistics")
        print("7. Controls tutorial")
        print("8. Exit\n")
        choice = _prompt_line("Choose an option [1-8]: ").strip()
        if choice in ("", "8"):
            return 0
        if choice == "1":
            _menu_run_model()
        elif choice == "2":
            cmd_doctor(argparse.Namespace())
            _menu_pause()
        elif choice == "3":
            os.system("cmd" if sys.platform == "win32"
                      else os.environ.get("SHELL", "sh"))
        elif choice == "4":
            _menu_model_manager()
        elif choice == "5":
            _menu_history()
        elif choice == "6":
            _history_stats()
            _menu_pause()
        elif choice == "7":
            _print_controls()
            _menu_pause()


def _history_list() -> int:
    from .history import list_chats
    chats = list_chats()
    if not chats:
        print("no chat history yet")
        return 0
    print(f"{'#':>3}  {'ID':<25}  {'STARTED':<20}  {'TURNS':>5}  {'MODEL':<32}  TITLE")
    for i, chat in enumerate(chats, 1):
        print(f"{i:>3}  {chat['id']:<25}  "
              f"{chat['started_at']:<20}  {chat['turns']:>5}  "
              f"{_clip(chat['model'], 32):<32}  {_clip(chat['title'], 72)}")
    return 0


def _history_show(selector: str) -> int:
    from .history import message_dicts, read_chat
    chat = read_chat(selector)
    print(f"Chat:    {chat['id']}")
    print(f"Started: {chat.get('started_at', '')}")
    if chat.get("ended_at"):
        print(f"Ended:   {chat['ended_at']}")
    print(f"Model:   {chat.get('model', '')}")
    if chat.get("model_path"):
        print(f"File:    {chat['model_path']}")
    if chat.get("title"):
        print(f"Title:   {chat['title']}")
    for msg in message_dicts(chat):
        role = msg.get("role", "?")
        created = msg.get("created_at", "")
        if role == "event":
            print(f"\n--- event {created} ---")
            print(msg.get("event", ""))
            continue
        print(f"\n--- {role} {created} ---")
        content = msg.get("content", "")
        if content:
            print(content)
        stats = []
        if "tokens" in msg:
            stats.append(f"{msg['tokens']} tokens")
        if isinstance(msg.get("seconds"), (int, float)):
            stats.append(f"{msg['seconds']:.2f}s")
        if stats:
            print(f"[{', '.join(stats)}]")
    return 0


def _history_stats() -> int:
    from .history import model_stats
    rows = model_stats()
    if not rows:
        print("no downloaded models or chat history yet")
        return 0
    print(f"{'MODEL':<40}  {'INST':<4}  {'CHATS':>5}  {'RESP':>5}  "
          f"{'TOKENS':>8}  {'SECONDS':>9}  {'AVG TOK/S':>9}")
    for row in rows:
        rate = f"{row['tok_per_sec']:.1f}" if row["responses"] else "n/a"
        print(f"{_clip(row['model'], 40):<40}  "
              f"{'yes' if row['installed'] else 'no':<4}  "
              f"{row['chats']:>5}  {row['responses']:>5}  "
              f"{row['tokens']:>8}  {row['seconds']:>9.2f}  {rate:>9}")
    return 0


def cmd_history(args) -> int:
    command = args.history_command or "list"
    if command in ("list", "ls"):
        return _history_list()
    if command == "show":
        return _history_show(args.chat)
    if command == "stats":
        return _history_stats()
    if command in ("rm", "delete"):
        from .history import list_chats, resolve_chat_entry
        chats = list_chats()
        targets = []
        seen = set()
        for sel in args.chats:
            chat = resolve_chat_entry(sel, chats)
            path = chat["path"]
            if path not in seen:
                targets.append(chat)
                seen.add(path)
        for chat in targets:
            try:
                chat["path"].unlink()
            except FileNotFoundError:
                pass
            print(f"deleted {chat['id']}")
        return 0
    if command == "clear":
        from .history import clear_history, list_chats
        count = len(list_chats())
        if not args.yes:
            if count == 0:
                print("no chat history to delete")
                return 0
            print(f"this will delete {count} chat(s); rerun with --yes to confirm",
                  file=sys.stderr)
            return 1
        deleted = clear_history()
        print(f"deleted {deleted} chat(s)")
        return 0
    raise ValueError(f"unknown history command: {command}")


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

    p = sub.add_parser("menu", help="open the local terminal app menu")
    p.set_defaults(func=cmd_menu)

    p = sub.add_parser("history", aliases=["hist"], help="manage saved chat history")
    hsub = p.add_subparsers(dest="history_command", metavar="<history-command>")
    hp = hsub.add_parser("list", aliases=["ls"], help="list saved chats")
    hp.set_defaults(func=cmd_history)
    hp = hsub.add_parser("show", help="show a saved chat")
    hp.add_argument("chat", help="chat number, full id, or unique id prefix")
    hp.set_defaults(func=cmd_history)
    hp = hsub.add_parser("stats", help="show read-only saved chat statistics")
    hp.set_defaults(func=cmd_history)
    hp = hsub.add_parser("rm", aliases=["delete"], help="delete saved chats")
    hp.add_argument("chats", nargs="+", help="chat number/id/prefix to delete")
    hp.set_defaults(func=cmd_history)
    hp = hsub.add_parser("clear", help="delete all saved chat history")
    hp.add_argument("--yes", action="store_true", help="confirm deletion")
    hp.set_defaults(func=cmd_history)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("doctor", help="check the installation")
    p.set_defaults(func=cmd_doctor)

    args = ap.parse_args(argv)
    if not args.command:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return cmd_menu(argparse.Namespace())
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
