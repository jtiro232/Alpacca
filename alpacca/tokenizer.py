# Alpacca - tokenizers implemented from scratch: SentencePiece-style
# (Viterbi over piece scores, byte fallback) and byte-level BPE with a
# GPT-2/llama-3 style pre-tokenizer built on unicodedata (no regex deps).
# MIT License. See LICENSE.
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

SPM_SPACE = "\u2581"  # \u2581

# token types stored in tokenizer.ggml.token_type
TT_NORMAL, TT_UNKNOWN, TT_CONTROL, TT_USER_DEFINED, TT_UNUSED, TT_BYTE = 1, 2, 3, 4, 5, 6


def _gpt2_byte_encoder() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENC = _gpt2_byte_encoder()
_BYTE_DEC = {v: k for k, v in _BYTE_ENC.items()}


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def pretokenize(text: str) -> list[str]:
    """Split text the way GPT-2/llama-3 style BPE expects.

    Hand-rolled equivalent of the usual pattern
    (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\\r\\n L N]? L+ | N{1,3}
    |  ?[^\\s L N]+ [\\r\\n]* | \\s*[\\r\\n]+ | \\s+(?!\\S) | \\s+
    implemented with unicodedata so it needs no third-party regex engine.
    """
    out: list[str] = []
    i, n = 0, len(text)
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while i < n:
        ch = text[i]

        low = text[i:i + 3].lower()
        matched = next((c for c in contractions if low.startswith(c)), None)
        if matched:
            out.append(text[i:i + len(matched)])
            i += len(matched)
            continue

        # optional non-letter/number/newline prefix + letters
        if _is_letter(ch) or (ch not in "\r\n" and not _is_number(ch)
                              and i + 1 < n and _is_letter(text[i + 1])):
            j = i if _is_letter(ch) else i + 1
            k = j
            while k < n and _is_letter(text[k]):
                k += 1
            if k > j:
                out.append(text[i:k])
                i = k
                continue

        if _is_number(ch):
            k = i
            while k < n and k - i < 3 and _is_number(text[k]):
                k += 1
            out.append(text[i:k])
            i = k
            continue

        if not ch.isspace():
            # punctuation run, optionally preceded by one space, plus newlines
            k = i
            while k < n and not text[k].isspace() and not _is_letter(text[k]) \
                    and not _is_number(text[k]):
                k += 1
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue

        if ch == " " and i + 1 < n and not text[i + 1].isspace() \
                and not _is_letter(text[i + 1]) and not _is_number(text[i + 1]):
            # " ?" prefix of a punctuation run
            k = i + 1
            while k < n and not text[k].isspace() and not _is_letter(text[k]) \
                    and not _is_number(text[k]):
                k += 1
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue

        # whitespace run
        k = i
        while k < n and text[k].isspace():
            k += 1
        run = text[i:k]
        last_nl = max(run.rfind("\r"), run.rfind("\n"))
        if last_nl >= 0:
            out.append(run[:last_nl + 1])
            i += last_nl + 1
            continue
        if k < n and len(run) > 1:   # leave one space to attach to next word
            out.append(run[:-1])
            i = k - 1
            continue
        out.append(run)
        i = k
    return [t for t in out if t]


@dataclass
class Tokenizer:
    model: str                       # "llama" (SPM) or "gpt2" (BPE)
    pieces: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    types: list[int] = field(default_factory=list)
    piece_to_id: dict[str, int] = field(default_factory=dict)
    merge_ranks: dict[tuple[str, str], int] = field(default_factory=dict)
    bos_id: int = -1
    eos_id: int = -1
    unk_id: int = -1
    add_bos: bool = True
    add_space_prefix: bool = True
    eog_ids: set = field(default_factory=set)
    byte_ids: dict[int, int] = field(default_factory=dict)  # byte -> token id
    _max_piece_bytes: int = 1
    _piece_bytes: dict[bytes, int] = field(default_factory=dict)

    # ---- construction --------------------------------------------------

    @classmethod
    def from_gguf(cls, md: dict) -> "Tokenizer":
        model = str(md.get("tokenizer.ggml.model", "llama"))
        t = cls(model=model)
        t.pieces = list(md.get("tokenizer.ggml.tokens", []))
        t.scores = list(md.get("tokenizer.ggml.scores", [])) or [0.0] * len(t.pieces)
        t.types = list(md.get("tokenizer.ggml.token_type", [])) or [TT_NORMAL] * len(t.pieces)
        t.piece_to_id = {p: i for i, p in enumerate(t.pieces)}
        t.bos_id = int(md.get("tokenizer.ggml.bos_token_id", -1))
        t.eos_id = int(md.get("tokenizer.ggml.eos_token_id", -1))
        t.unk_id = int(md.get("tokenizer.ggml.unknown_token_id", -1))
        t.add_bos = bool(md.get("tokenizer.ggml.add_bos_token", model == "llama"))
        t.add_space_prefix = bool(md.get("tokenizer.ggml.add_space_prefix", True))

        for i, (p, tt) in enumerate(zip(t.pieces, t.types)):
            if tt == TT_BYTE and len(p) == 6 and p.startswith("<0x"):
                t.byte_ids[int(p[3:5], 16)] = i

        if t.eos_id >= 0:
            t.eog_ids.add(t.eos_id)
        for key in ("tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"):
            if key in md:
                t.eog_ids.add(int(md[key]))
        for name in ("<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>",
                     "<|endoftext|>", "<|end_of_text|>", "</s>"):
            if name in t.piece_to_id:
                t.eog_ids.add(t.piece_to_id[name])

        if model == "gpt2":
            merges = md.get("tokenizer.ggml.merges", []) or []
            t.merge_ranks = {}
            for rank, m in enumerate(merges):
                a, _, b = m.partition(" ")
                t.merge_ranks[(a, b)] = rank
        else:
            # SPM matches on UTF-8 bytes so byte-fallback composes correctly
            for p, i in t.piece_to_id.items():
                if t.types[i] in (TT_NORMAL, TT_USER_DEFINED):
                    b = p.encode("utf-8")
                    t._piece_bytes[b] = i
                    t._max_piece_bytes = max(t._max_piece_bytes, len(b))
        return t

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    def piece(self, token_id: int) -> str:
        return self.pieces[token_id] if 0 <= token_id < len(self.pieces) else ""

    def is_eog(self, token_id: int) -> bool:
        return token_id in self.eog_ids

    def token_id(self, piece: str) -> int:
        return self.piece_to_id.get(piece, -1)

    # ---- encoding -------------------------------------------------------

    def encode(self, text: str, add_bos: bool | None = None) -> list[int]:
        ids = self._encode_spm(text) if self.model == "llama" else self._encode_bpe(text)
        use_bos = self.add_bos if add_bos is None else add_bos
        if use_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        return ids

    def _encode_spm(self, text: str) -> list[int]:
        if not text:
            return []
        norm = text.replace(" ", SPM_SPACE)
        if self.add_space_prefix and not norm.startswith(SPM_SPACE):
            norm = SPM_SPACE + norm
        data = norm.encode("utf-8")
        n = len(data)
        NEG = -1e30
        best = [NEG] * (n + 1)
        back: list[tuple[int, int]] = [(-1, -1)] * (n + 1)
        best[0] = 0.0
        for i in range(n):
            if best[i] <= NEG:
                continue
            limit = min(self._max_piece_bytes, n - i)
            for ln in range(1, limit + 1):
                tid = self._piece_bytes.get(bytes(data[i:i + ln]))
                if tid is not None:
                    sc = best[i] + self.scores[tid]
                    if sc > best[i + ln]:
                        best[i + ln] = sc
                        back[i + ln] = (i, tid)
            bid = self.byte_ids.get(data[i])
            if bid is not None:
                sc = best[i] - 1e6  # byte fallback: heavily penalized
                if sc > best[i + 1]:
                    best[i + 1] = sc
                    back[i + 1] = (i, bid)
            elif best[i + 1] <= NEG and self.unk_id >= 0:
                best[i + 1] = best[i] - 1e7
                back[i + 1] = (i, self.unk_id)
        if best[n] <= NEG:
            return [self.unk_id] if self.unk_id >= 0 else []
        ids: list[int] = []
        pos = n
        while pos > 0:
            prev, tid = back[pos]
            ids.append(tid)
            pos = prev
        ids.reverse()
        return ids

    def _encode_bpe(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in pretokenize(text):
            word = [_BYTE_ENC[b] for b in chunk.encode("utf-8")]
            while len(word) > 1:
                best_rank = None
                best_i = -1
                for i in range(len(word) - 1):
                    r = self.merge_ranks.get((word[i], word[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank, best_i = r, i
                if best_i < 0:
                    break
                word[best_i:best_i + 2] = [word[best_i] + word[best_i + 1]]
            for piece in word:
                tid = self.piece_to_id.get(piece)
                if tid is not None:
                    ids.append(tid)
                else:  # last resort: per-character lookup
                    for ch in piece:
                        tid = self.piece_to_id.get(ch)
                        if tid is not None:
                            ids.append(tid)
        return ids

    # ---- decoding -------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes a token contributes to output (may be partial UTF-8)."""
        if not (0 <= token_id < len(self.pieces)):
            return b""
        p = self.pieces[token_id]
        tt = self.types[token_id]
        if tt in (TT_CONTROL, TT_UNKNOWN, TT_UNUSED):
            return b""
        if self.model == "llama":
            if tt == TT_BYTE and p.startswith("<0x"):
                return bytes([int(p[3:5], 16)])
            return p.replace(SPM_SPACE, " ").encode("utf-8")
        return bytes(_BYTE_DEC.get(ch, ord(" ")) for ch in p)


class StreamDecoder:
    """Incremental detokenizer that holds back partial UTF-8 sequences."""

    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.pending = b""

    def feed(self, token_id: int) -> str:
        self.pending += self.tok.token_bytes(token_id)
        # emit the longest prefix that is valid UTF-8
        for cut in range(len(self.pending), max(len(self.pending) - 4, -1), -1):
            try:
                text = self.pending[:cut].decode("utf-8")
                self.pending = self.pending[cut:]
                return text
            except UnicodeDecodeError:
                continue
        return ""

    def flush(self) -> str:
        text = self.pending.decode("utf-8", errors="replace")
        self.pending = b""
        return text
