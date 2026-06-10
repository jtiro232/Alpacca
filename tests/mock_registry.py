#!/usr/bin/env python3
"""Offline mock of the Ollama registry + Hugging Face API, for smoke tests.

Serves the files of a directory as both:
  * an Ollama-style model:   /v2/<ns>/<name>/manifests/<tag>, /v2/.../blobs/...
    (model.gguf, params.json, system.txt, template.txt, license.txt,
     mmproj.gguf, adapter.gguf are mapped to their layer media types)
  * a Hugging Face repo:     /api/models/<org>/<repo>/tree/main,
                             /<org>/<repo>/resolve/main/<file>
    (repos whose name does not end in -GGUF return 404, to exercise
     alpacca's -GGUF fallback)

usage: mock_registry.py <serve_dir> <port_file>
Binds 127.0.0.1 on a free port and writes the port number to <port_file>.
"""
import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SERVE_DIR = Path(sys.argv[1])

MEDIA_TYPES = {
    "model.gguf":   "application/vnd.ollama.image.model",
    "mmproj.gguf":  "application/vnd.ollama.image.projector",
    "adapter.gguf": "application/vnd.ollama.image.adapter",
    "params.json":  "application/vnd.ollama.image.params",
    "system.txt":   "application/vnd.ollama.image.system",
    "template.txt": "application/vnd.ollama.image.template",
    "license.txt":  "application/vnd.ollama.image.license",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


FILES = {p.name: p for p in SERVE_DIR.iterdir() if p.is_file()}
DIGESTS = {sha256_file(p): p for p in FILES.values()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def send_bytes(self, data: bytes, ctype="application/octet-stream"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]

        m = re.match(r"^/v2/[^/]+/[^/]+/manifests/[^/]+$", path)
        if m:
            layers = []
            for name, media in MEDIA_TYPES.items():
                if name in FILES:
                    p = FILES[name]
                    layers.append({
                        "mediaType": media,
                        "digest": "sha256:" + sha256_file(p),
                        "size": p.stat().st_size,
                    })
            manifest = {
                "schemaVersion": 2,
                "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                           "digest": "sha256:" + "0" * 64, "size": 2},
                "layers": layers,
            }
            return self.send_bytes(json.dumps(manifest).encode(), "application/json")

        m = re.match(r"^/v2/[^/]+/[^/]+/blobs/sha256:([0-9a-f]{64})$", path)
        if m and m.group(1) in DIGESTS:
            return self.send_bytes(DIGESTS[m.group(1)].read_bytes())

        m = re.match(r"^/api/models/[^/]+/([^/]+)/tree/main$", path)
        if m:
            if not m.group(1).endswith("-GGUF"):
                self.send_error(404, "model not found (no -GGUF suffix)")
                return
            tree = [{
                "type": "file",
                "path": p.name,
                "size": p.stat().st_size,
                "lfs": {"oid": sha256_file(p), "size": p.stat().st_size},
            } for p in FILES.values() if p.suffix == ".gguf"]
            return self.send_bytes(json.dumps(tree).encode(), "application/json")

        m = re.match(r"^/[^/]+/([^/]+)/resolve/main/(.+)$", path)
        if m and m.group(1).endswith("-GGUF") and m.group(2) in FILES:
            return self.send_bytes(FILES[m.group(2)].read_bytes())

        self.send_error(404, "not found")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Path(sys.argv[2]).write_text(str(server.server_address[1]))
    server.serve_forever()


if __name__ == "__main__":
    main()
