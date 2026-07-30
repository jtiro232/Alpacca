# Alpacca - chat formatting and generation loops.
# MIT License. See LICENSE.
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
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

# Pieces that open a turn. If the model emits one it has started speaking as
# somebody else and its own reply is over. They are control tokens, so they
# decode to empty text and a stop *string* can never catch them - without this
# a runaway turn silently role-plays the user with no visible separator.
_FORMAT_TURN_STARTS = {
    "llama3": ("<|start_header_id|>",),
    "chatml": ("<|im_start|>",),
    "gemma": ("<start_of_turn>",),
    "zephyr": ("<|user|>", "<|system|>"),
}


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
    _id_cache: dict = field(default_factory=dict, init=False, repr=False)

    def _ids(self, text: str, add_bos: bool = False) -> list[int]:
        # role names and separators recur every message; on a tokenizer with
        # thousands of special pieces (Gemma 3) each encode of even "\n"
        # pays a fixed scan, so memoize the short constants. Copies out so a
        # caller can never mutate a cached entry.
        if len(text) <= 32:
            key = (text, add_bos)
            hit = self._id_cache.get(key)
            if hit is None:
                hit = self.model.tok.encode(text, add_bos=add_bos)
                self._id_cache[key] = hit
            return list(hit)
        return self.model.tok.encode(text, add_bos=add_bos)

    def _special(self, piece: str) -> list[int]:
        tid = self.model.tok.token_id(piece)
        return [tid] if tid >= 0 else self._ids(piece)

    def stop_tokens(self) -> set[int]:
        """Token ids that end the assistant's turn on top of the EOG set."""
        out = set()
        for piece in _FORMAT_TURN_STARTS.get(self.name, ()):
            tid = self.model.tok.token_id(piece)
            if tid >= 0:
                out.add(tid)
        return out

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
            # the gemma template opens with {{ bos_token }} and the models are
            # trained with <bos> at position 0; without it they answer empty
            if tok.bos_id >= 0:
                ids.append(tok.bos_id)
            # Gemma has no system turn: its template folds a leading system
            # message into the first user turn as a prefix, and raises rather
            # than emit two user turns in a row. Rendering it as its own turn
            # produced a sequence the model was never trained on.
            prefix = ""
            turns = messages
            if messages and messages[0].get("role") == "system":
                prefix = messages[0]["content"] + "\n\n"
                turns = messages[1:]
            for i, m in enumerate(turns):
                role = "model" if m["role"] == "assistant" else "user"
                ids += self._special("<start_of_turn>")
                # one encode call: the template concatenates these before the
                # tokenizer sees them, and splitting the call would tokenize
                # across a boundary the model never saw
                ids += self._ids(role + "\n" + (prefix if i == 0 else "") +
                                 m["content"].strip())
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
    prompt_tokens: int = 0
    # why generation ended: "eog" (the model finished), "stop" (a stop string
    # matched), "length" (the n_predict budget ran out) or "context" (no room
    # left in the context window). "context" with tokens == 0 is the caller's
    # signal that the prompt itself left nothing to generate into.
    stop_reason: str = "eog"

    @property
    def tok_per_sec(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


def generate(model: Model, prompt_ids: list[int], params: SamplerParams,
             n_predict: int = -1, stream=None, stop_strings: list[str] | None = None,
             stop_tokens: set[int] | None = None) -> GenerationResult:
    """Generate until EOG / n_predict / a stop string. `stream` is an
    optional callable receiving text fragments as they decode."""
    if not prompt_ids:
        if model.tok.bos_id < 0:
            raise ValueError("prompt produced no tokens and the tokenizer has no BOS token")
        prompt_ids = [model.tok.bos_id]
    sampler = Sampler(params)
    # only the last repeat_last_n tokens can ever remain in the penalty
    # window, so skip the rest rather than walk a 16k prompt to fill 64 slots
    for t in prompt_ids[-max(params.repeat_last_n, 1):]:
        sampler.accept(t)
    logits = model.prefill(prompt_ids)

    dec = StreamDecoder(model.tok)
    emitted = 0
    n_tokens = 0
    t0 = time.time()
    budget = n_predict if n_predict and n_predict > 0 else (model.n_ctx - model.n_past)

    text = ""
    reason = "eog"
    truncated = False
    # a stop string is only detectable once its last character arrives, so
    # hold back that much of the tail or the caller sees the beginning of it
    hold = max((len(s) for s in stop_strings or [] if s), default=1) - 1
    while True:
        if model.n_past >= model.n_ctx:
            reason = "context"   # checked first: no room beats no budget
            break
        if n_tokens >= budget:
            reason = "length"
            break
        tid = sampler.sample(logits)
        sampler.accept(tid)
        if model.tok.is_eog(tid):
            break  # never forwarded into the cache, so never counted either
        if stop_tokens and tid in stop_tokens:
            reason = "stop"   # a turn-start token: the reply is complete
            break
        n_tokens += 1
        piece = dec.feed(tid)
        text += piece
        if stop_strings and piece:
            # a match must involve the newly decoded piece - anything fully
            # inside older text was found on an earlier token - so search
            # only the tail the piece could participate in: a stop of length
            # L ending inside the piece starts at most L-1 (= hold) before it
            scan = max(0, len(text) - len(piece) - hold)
            tail = text[scan:]
            hit = next((s for s in stop_strings if s and s in tail), None)
            if hit:
                text = text[:scan + tail.index(hit)]
                reason = "stop"
                truncated = True
                break
        if stream is not None and len(text) - hold > emitted:
            stream(text[emitted:len(text) - hold])
            emitted = len(text) - hold
        if n_tokens >= budget:
            reason = "length"
            break
        logits = model.forward(tid)
    if truncated:
        # only a stop STRING invalidates the tail: those bytes are part of the
        # match. A stop token breaks before decoding, so its pending bytes are
        # ordinary text and still belong in the answer.
        dec.pending = b""
    else:
        text += dec.flush()
    if stream is not None and len(text) > emitted:
        stream(text[emitted:])
    return GenerationResult(text=text, tokens=n_tokens, seconds=time.time() - t0,
                            prompt_tokens=len(prompt_ids), stop_reason=reason)


#: tokens held back for the reply when trimming a conversation to fit
REPLY_RESERVE = 128


def fit_to_context(fmt: "ChatFormat", messages: list[dict], n_ctx: int,
                   reserve: int = REPLY_RESERVE) -> tuple[list[int], int]:
    """Render `messages`, dropping the oldest exchanges until the prompt leaves
    `reserve` tokens to answer in. Trims `messages` in place and returns the
    rendered ids plus how many exchanges were dropped.

    A leading system message is never dropped, and neither is the newest turn -
    if that alone does not fit there is nothing left to give up.
    """
    keep = 1 if messages and messages[0].get("role") == "system" else 0
    dropped = 0
    while True:
        ids = fmt.render(messages)
        if len(ids) + reserve <= n_ctx or len(messages) - keep <= 1:
            return ids, dropped
        del messages[keep]
        dropped += 1
        # a user turn and the reply it drew go together
        while len(messages) - keep > 1 and messages[keep].get("role") == "assistant":
            del messages[keep]


def chat_once(model: Model, messages: list[dict], params: SamplerParams,
              n_predict: int = -1, stream=None,
              stop_strings: list[str] | None = None) -> GenerationResult:
    fmt = ChatFormat(model, detect_format(model.metadata))
    ids = fmt.render(messages)
    return generate(model, ids, params, n_predict, stream, stop_strings,
                    stop_tokens=fmt.stop_tokens())


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
                n_predict: int = -1, model_name: str = "",
                model_path: str = "") -> None:
    fmt = ChatFormat(model, detect_format(model.metadata))
    description = model.describe()
    print(f"alpacca chat - {description}", file=sys.stderr)
    print("press Esc or type /exit to return, /clear to reset the conversation\n",
          file=sys.stderr)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    from .history import start_session
    history = start_session(model_name or description, model_path, system)
    history_disabled = False

    def record_history(method: str, *args, **kwargs) -> None:
        nonlocal history_disabled
        if history_disabled:
            return
        try:
            getattr(history, method)(*args, **kwargs)
        except OSError as e:
            history_disabled = True
            print(f"(chat history disabled: {e})", file=sys.stderr)

    try:
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
                record_history("append_event", "clear")
                print("(cleared)", file=sys.stderr)
                continue
            if not user.strip():
                continue
            messages.append({"role": "user", "content": user})
            record_history("append_message", "user", user)
            ids, dropped = fit_to_context(fmt, messages, model.n_ctx)
            if dropped:
                print(f"(dropped {dropped} earlier turn"
                      f"{'s' if dropped != 1 else ''} to fit the context window)",
                      file=sys.stderr)
            try:
                res = generate(model, ids, params, n_predict,
                               stream=lambda s: print(s, end="", flush=True),
                               stop_tokens=fmt.stop_tokens())
            except RuntimeError as e:
                # the prompt does not fit even on its own - stay in the REPL
                messages.pop()
                print(f"({e}; that message was too long, so it was not sent - "
                      f"/clear resets the conversation)", file=sys.stderr)
                continue
            print()
            if res.stop_reason == "context":
                print("(the context window is full - /clear resets the conversation)",
                      file=sys.stderr)
            print(f"[{res.tokens} tokens, {res.tok_per_sec:.1f} tok/s]", file=sys.stderr)
            messages.append({"role": "assistant", "content": res.text})
            record_history("append_message", "assistant", res.text,
                           tokens=res.tokens, seconds=res.seconds)
    finally:
        record_history("close")
