# Alpacca - chat formatting and generation loops.
# MIT License. See LICENSE.
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO

from .model import Model
from .sample import Sampler, SamplerParams
from .tokenizer import StreamDecoder

# Known chat formats, detected from the model's embedded chat template.
# Each entry: (needle in template, format name)
_FORMAT_NEEDLES = [
    ("<|start_header_id|>", "llama3"),
    ("<|im_start|>", "chatml"),
    ("<start_of_turn>", "gemma"),
    ("[INST]", "llama2"),
    ("<|user|>", "zephyr"),
]


def detect_format(metadata: dict) -> str:
    template = str(metadata.get("tokenizer.chat_template", ""))
    for needle, name in _FORMAT_NEEDLES:
        if needle in template:
            return name
    return "raw" if not template else "chatml"


@dataclass
class ChatFormat:
    """Renders a conversation into token ids for a given format."""
    model: Model
    name: str

    def _ids(self, text: str, add_bos: bool = False) -> list[int]:
        return self.model.tok.encode(text, add_bos=add_bos)

    def _special(self, piece: str) -> list[int]:
        tid = self.model.tok.token_id(piece)
        return [tid] if tid >= 0 else self._ids(piece)

    def render(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        tok = self.model.tok
        ids: list[int] = []
        if self.name == "llama3":
            if tok.bos_id >= 0:
                ids.append(tok.bos_id)
            for m in messages:
                ids += self._special("<|start_header_id|>")
                ids += self._ids(m["role"])
                ids += self._special("<|end_header_id|>")
                ids += self._ids("\n\n" + m["content"])
                ids += self._special("<|eot_id|>")
            if add_generation_prompt:
                ids += self._special("<|start_header_id|>")
                ids += self._ids("assistant")
                ids += self._special("<|end_header_id|>")
                ids += self._ids("\n\n")
            return ids

        if self.name == "chatml":
            for m in messages:
                ids += self._special("<|im_start|>")
                ids += self._ids(m["role"] + "\n")
                ids += self._ids(m["content"])
                ids += self._special("<|im_end|>")
                ids += self._ids("\n")
            if add_generation_prompt:
                ids += self._special("<|im_start|>")
                ids += self._ids("assistant\n")
            return ids

        if self.name == "gemma":
            for m in messages:
                role = "model" if m["role"] == "assistant" else "user"
                ids += self._special("<start_of_turn>")
                ids += self._ids(role + "\n" + m["content"])
                ids += self._special("<end_of_turn>")
                ids += self._ids("\n")
            if add_generation_prompt:
                ids += self._special("<start_of_turn>")
                ids += self._ids("model\n")
            return ids

        if self.name == "llama2":
            system = ""
            convo = []
            for m in messages:
                if m["role"] == "system":
                    system = m["content"]
                else:
                    convo.append(m)
            text = ""
            for i, m in enumerate(convo):
                if m["role"] == "user":
                    content = m["content"]
                    if system and i == 0:
                        content = f"<<SYS>>\n{system}\n<</SYS>>\n\n{content}"
                    text += f"[INST] {content} [/INST]"
                else:
                    text += f" {m['content']} "
            return self._ids(text, add_bos=True)

        if self.name == "zephyr":
            for m in messages:
                ids += self._special(f"<|{m['role']}|>")
                ids += self._ids("\n" + m["content"])
                if tok.eos_id >= 0:
                    ids.append(tok.eos_id)
            if add_generation_prompt:
                ids += self._special("<|assistant|>")
                ids += self._ids("\n")
            return ids

        # raw: plain completion with a simple convention
        text = ""
        for m in messages:
            prefix = {"system": "", "user": "User: ", "assistant": "Assistant: "}.get(m["role"], "")
            text += prefix + m["content"] + "\n"
        if add_generation_prompt:
            text += "Assistant:"
        return self._ids(text, add_bos=True)


@dataclass
class GenerationResult:
    text: str
    tokens: int
    seconds: float

    @property
    def tok_per_sec(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


def generate(model: Model, prompt_ids: list[int], params: SamplerParams,
             n_predict: int = -1, stream=None, stop_strings: list[str] | None = None
             ) -> GenerationResult:
    """Generate until EOG / n_predict / a stop string. `stream` is an
    optional callable receiving text fragments as they decode."""
    if not prompt_ids:
        if model.tok.bos_id < 0:
            raise ValueError("prompt produced no tokens and the tokenizer has no BOS token")
        prompt_ids = [model.tok.bos_id]
    sampler = Sampler(params)
    for t in prompt_ids:
        sampler.accept(t)
    logits = model.prefill(prompt_ids)

    dec = StreamDecoder(model.tok)
    emitted = 0
    n_tokens = 0
    t0 = time.time()
    budget = n_predict if n_predict and n_predict > 0 else (model.n_ctx - model.n_past)

    text = ""
    while n_tokens < budget and model.n_past < model.n_ctx:
        tid = sampler.sample(logits)
        sampler.accept(tid)
        n_tokens += 1
        if model.tok.is_eog(tid):
            break
        text += dec.feed(tid)
        if stop_strings:
            hit = next((s for s in stop_strings if s and s in text), None)
            if hit:
                text = text[:text.index(hit)]
                break
        if stream is not None and len(text) > emitted:
            stream(text[emitted:])
            emitted = len(text)
        if n_tokens >= budget:
            break
        logits = model.forward(tid)
    text += dec.flush()
    if stream is not None and len(text) > emitted:
        stream(text[emitted:])
    return GenerationResult(text=text, tokens=n_tokens, seconds=time.time() - t0)


def chat_once(model: Model, messages: list[dict], params: SamplerParams,
              n_predict: int = -1, stream=None,
              stop_strings: list[str] | None = None) -> GenerationResult:
    fmt = ChatFormat(model, detect_format(model.metadata))
    ids = fmt.render(messages)
    return generate(model, ids, params, n_predict, stream, stop_strings)


def _read_chat_line(prompt: str = "> ", stdin: TextIO | None = None,
                    stdout: TextIO | None = None) -> str | None:
    """Read one chat line. Return None when the user presses bare Escape."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    if stdin is not sys.stdin or stdout is not sys.stdout or not stdin.isatty():
        stdout.write(prompt)
        stdout.flush()
        line = stdin.readline()
        if line == "":
            raise EOFError
        line = line.rstrip("\r\n")
        return None if "\x1b" in line else line
    if sys.platform == "win32":
        return _read_chat_line_windows(prompt, stdout)
    return _read_chat_line_posix(prompt, stdin, stdout)


def _read_chat_line_windows(prompt: str, stdout: TextIO) -> str | None:
    import msvcrt

    stdout.write(prompt)
    stdout.flush()
    chars: list[str] = []
    while True:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()
            continue
        if ch == "\x1b":
            stdout.write("\n")
            stdout.flush()
            return None
        if ch in ("\r", "\n"):
            stdout.write("\n")
            stdout.flush()
            return "".join(chars)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x04":
            raise EOFError
        if ch in ("\b", "\x7f"):
            if chars:
                chars.pop()
                stdout.write("\b \b")
                stdout.flush()
            continue
        if ch == "\t" or ch >= " ":
            chars.append(ch)
            stdout.write(ch)
            stdout.flush()


def _read_chat_line_posix(prompt: str, stdin: TextIO,
                          stdout: TextIO) -> str | None:
    import select
    import termios
    import tty

    fd = stdin.fileno()
    old = termios.tcgetattr(fd)
    stdout.write(prompt)
    stdout.flush()
    chars: list[str] = []
    try:
        tty.setcbreak(fd)
        while True:
            ch = stdin.read(1)
            if ch == "\x1b":
                if select.select([stdin], [], [], 0.05)[0]:
                    stdin.read(1)
                    while select.select([stdin], [], [], 0.001)[0]:
                        stdin.read(1)
                    continue
                stdout.write("\n")
                stdout.flush()
                return None
            if ch in ("\r", "\n"):
                stdout.write("\n")
                stdout.flush()
                return "".join(chars)
            if ch == "\x04" and not chars:
                raise EOFError
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\b", "\x7f"):
                if chars:
                    chars.pop()
                    stdout.write("\b \b")
                    stdout.flush()
                continue
            if ch == "\t" or ch >= " ":
                chars.append(ch)
                stdout.write(ch)
                stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def interactive(model: Model, params: SamplerParams, system: str = "",
                n_predict: int = -1) -> None:
    fmt = ChatFormat(model, detect_format(model.metadata))
    print(f"alpacca chat - {model.describe()}", file=sys.stderr)
    print("press Esc or type /exit to return, /clear to reset the conversation\n",
          file=sys.stderr)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    while True:
        try:
            user = _read_chat_line("> ")
        except (EOFError, KeyboardInterrupt):
            print("", file=sys.stderr)
            return
        if user is None:
            print("(returning to main menu)", file=sys.stderr)
            return
        if user.strip() in ("/exit", "/quit", "/bye"):
            return
        if user.strip() == "/clear":
            messages = messages[:1] if system else []
            model.reset()
            print("(cleared)", file=sys.stderr)
            continue
        if not user.strip():
            continue
        messages.append({"role": "user", "content": user})
        ids = fmt.render(messages)
        res = generate(model, ids, params, n_predict,
                       stream=lambda s: print(s, end="", flush=True))
        print()
        print(f"[{res.tokens} tokens, {res.tok_per_sec:.1f} tok/s]", file=sys.stderr)
        messages.append({"role": "assistant", "content": res.text})
