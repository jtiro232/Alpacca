# Alpacca — OpenAI-compatible HTTP API on the standard library only.
# Endpoints: /health, /v1/models, /v1/chat/completions (incl. streaming),
# and a llama.cpp-style /completion. MIT License. See LICENSE.
from __future__ import annotations

import json
import socketserver
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import chat
from .model import Model
from .sample import SamplerParams


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):  # skip socket.getfqdn(), which can stall for seconds
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


def _params_from(body: dict, defaults: SamplerParams) -> SamplerParams:
    return SamplerParams(
        temperature=float(body.get("temperature", defaults.temperature)),
        top_k=int(body.get("top_k", defaults.top_k)),
        top_p=float(body.get("top_p", defaults.top_p)),
        repeat_penalty=float(body.get("repeat_penalty", defaults.repeat_penalty)),
        seed=int(body.get("seed", defaults.seed)),
    )


def serve(model: Model, model_name: str, host: str = "127.0.0.1", port: int = 8080,
          defaults: SamplerParams | None = None, ready_callback=None) -> None:
    defaults = defaults or SamplerParams()
    lock = threading.Lock()  # one generation at a time

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(f"[serve] {self.address_string()} {fmt % args}", file=sys.stderr)

        # -- helpers -----------------------------------------------------

        def send_json(self, obj, status=200):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        # -- routes ------------------------------------------------------

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self.send_json({"status": "ok"})
            if path == "/v1/models":
                return self.send_json({"object": "list", "data": [
                    {"id": model_name, "object": "model", "owned_by": "alpacca"}]})
            self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?")[0]
            body = self.read_body()
            if path == "/v1/chat/completions":
                return self.chat_completions(body)
            if path == "/completion":
                return self.completion(body)
            self.send_json({"error": "not found"}, 404)

        def completion(self, body: dict):
            prompt = str(body.get("prompt", ""))
            params = _params_from(body, defaults)
            n_predict = int(body.get("n_predict", body.get("max_tokens", 256)))
            stop = body.get("stop") or []
            with lock:
                model.reset()
                ids = model.tok.encode(prompt)
                res = chat.generate(model, ids, params, n_predict, stop_strings=stop)
            self.send_json({
                "content": res.text,
                "tokens_predicted": res.tokens,
                "timings": {"predicted_per_second": res.tok_per_sec},
                "model": model_name,
            })

        def chat_completions(self, body: dict):
            messages = body.get("messages") or []
            if not isinstance(messages, list) or not messages:
                return self.send_json({"error": "messages required"}, 400)
            params = _params_from(body, defaults)
            n_predict = int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)
            stop = body.get("stop") or []
            if isinstance(stop, str):
                stop = [stop]
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def chunk(obj):
                    payload = b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n"
                    self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")

                def piece(delta, finish=None):
                    chunk({"id": rid, "object": "chat.completion.chunk",
                           "created": created, "model": model_name,
                           "choices": [{"index": 0, "delta": delta,
                                        "finish_reason": finish}]})

                piece({"role": "assistant"})
                with lock:
                    model.reset()
                    res = chat.chat_once(model, messages, params, n_predict,
                                         stream=lambda s: piece({"content": s}),
                                         stop_strings=stop)
                piece({}, finish="stop")
                tail = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(tail):x}\r\n".encode() + tail + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                return

            with lock:
                model.reset()
                res = chat.chat_once(model, messages, params, n_predict, stop_strings=stop)
            self.send_json({
                "id": rid, "object": "chat.completion", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": res.text}}],
                "usage": {"completion_tokens": res.tokens},
            })

    httpd = _Server((host, port), Handler)
    actual_port = httpd.server_address[1]
    print(f"alpacca serving {model_name} on http://{host}:{actual_port} "
          f"(OpenAI-compatible: POST /v1/chat/completions)", file=sys.stderr)
    if ready_callback:
        ready_callback(actual_port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
