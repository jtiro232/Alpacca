# Alpacca - local model store (~/.alpacca/models) and model references.
# MIT License. See LICENSE.
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


def alpacca_home() -> Path:
    env = os.environ.get("ALPACCA_HOME")
    if env:
        return Path(env)
    return Path.home() / ".alpacca"


def models_root() -> Path:
    return alpacca_home() / "models"


def _sanitize(part: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._+-]", "_", part)
    return out if out not in ("", ".", "..") else "_"


@dataclass
class ModelRef:
    """Parsed model reference.

    "llama3.2:1b"           -> ollama registry, library/llama3.2:1b
    "ollama:user/name:tag"  -> ollama registry, user namespace
    "hf:org/repo:Q4_K_M"    -> Hugging Face repo + quant/file selector
    "org/repo"              -> Hugging Face
    "./model.gguf"          -> local file
    """
    source: str   # "ollama" | "hf" | "file"
    ns: str = ""
    name: str = ""
    tag: str = ""
    path: Path | None = None

    def display(self) -> str:
        if self.source == "file":
            return str(self.path)
        if self.source == "hf":
            s = f"hf:{self.ns}/{self.name}"
            return f"{s}:{self.tag}" if self.tag else s
        s = self.name if self.ns == "library" else f"{self.ns}/{self.name}"
        return f"{s}:{self.tag}" if self.tag != "latest" else s

    def store_dir(self) -> Path:
        root = models_root()
        if self.source == "hf":
            return root / "hf" / _sanitize(self.ns) / _sanitize(self.name) / \
                   _sanitize(self.tag or "default")
        if self.source == "ollama":
            return root / "ollama" / _sanitize(self.ns) / _sanitize(self.name) / \
                   _sanitize(self.tag)
        raise ValueError("file references have no store directory")


def parse_model_ref(raw: str) -> ModelRef:
    s = raw.strip()
    if not s:
        raise ValueError("empty model name")

    looks_path = (s.startswith(("/", "./", "../", "~/")) or
                  (os.name == "nt" and (re.match(r"^[A-Za-z]:", s) or
                                        s.startswith((".\\", "..\\", "\\\\")))) or
                  (s.lower().endswith(".gguf") and Path(s).expanduser().exists()))
    if looks_path:
        return ModelRef(source="file", path=Path(s).expanduser())

    forced_hf = False
    for p in ("hf:", "hf.co/", "huggingface.co/", "https://huggingface.co/"):
        if s.startswith(p):
            s = s[len(p):]
            forced_hf = True
            break
    forced_ollama = False
    if not forced_hf and s.startswith("ollama:"):
        s = s[len("ollama:"):]
        forced_ollama = True

    # split a trailing :tag (no '/' after the colon)
    name_part, tag = s, ""
    colon = s.rfind(":")
    if colon != -1 and "/" not in s[colon:]:
        name_part, tag = s[:colon], s[colon + 1:]

    parts = name_part.split("/")
    if any(not p.strip() for p in parts):
        raise ValueError(f"invalid model reference: '{raw}'")

    if forced_hf or (not forced_ollama and len(parts) >= 2):
        if len(parts) != 2:
            raise ValueError(f"Hugging Face references look like org/repo[:quant] (got '{raw}')")
        return ModelRef(source="hf", ns=parts[0], name=parts[1], tag=tag)

    if len(parts) == 1:
        return ModelRef(source="ollama", ns="library", name=parts[0], tag=tag or "latest")
    if len(parts) == 2:
        return ModelRef(source="ollama", ns=parts[0], name=parts[1], tag=tag or "latest")
    raise ValueError(f"Ollama references look like [user/]name[:tag] (got '{raw}')")


@dataclass
class LocalModel:
    model_path: Path
    dir: Path | None = None
    manifest: dict = field(default_factory=dict)


def find_local(ref: ModelRef) -> LocalModel | None:
    if ref.source == "file":
        assert ref.path is not None
        return LocalModel(model_path=ref.path) if ref.path.exists() else None
    d = ref.store_dir()
    mf = d / "manifest.json"
    if not mf.exists():
        return None
    try:
        manifest = json.loads(mf.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_file = manifest.get("model_file", "")
    if not model_file or not (d / model_file).exists():
        return None
    return LocalModel(model_path=d / model_file, dir=d, manifest=manifest)


def write_manifest(d: Path, manifest: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    tmp.replace(d / "manifest.json")


def now_iso8601() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def list_models() -> list[dict]:
    out = []
    root = models_root()
    if not root.exists():
        return out
    for mf in sorted(root.rglob("manifest.json")):
        d = mf.parent
        try:
            manifest = json.loads(mf.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        size = manifest.get("size", 0)
        if not size:
            size = sum(f.stat().st_size for f in d.glob("*.gguf"))
        out.append({
            "name": manifest.get("name", d.name),
            "source": manifest.get("source", "?"),
            "size": int(size),
            "pulled_at": manifest.get("pulled_at", ""),
            "dir": d,
        })
    out.sort(key=lambda m: m["name"])
    return out


def remove_model(ref: ModelRef) -> bool:
    if ref.source == "file":
        raise ValueError("refusing to delete a raw file path; remove it yourself if intended")
    d = ref.store_dir()
    if not (d / "manifest.json").exists():
        return False
    for f in sorted(d.rglob("*"), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    d.rmdir()
    parent = d.parent
    while parent != models_root() and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent
    return True
