# Alpacca — remaining fixes after the `alpacca-gemma3-615` audit

You are picking up a codebase mid-repair. A rigorous multi-agent audit of the
branch `alpacca-gemma3-615` (Gemma 3 text GGUF support + model nicknames) found
43 verified defects. The low-risk ones are **already fixed, committed and
pushed** — do not redo them. Your job is the rest.

## Setup

You are reading this from the repo, on branch `alpacca-gemma3-615`. Confirm the
fix commit is in your history before trusting §1:

```sh
git log --oneline -4
git cat-file -e 1109c5e^{commit} && echo "fix commit present"
```

Expected (the top SHA is this document's own commit and will differ):
```
<docs>  docs: audit handoff for the remaining Gemma 3 / nickname fixes
1109c5e fix: close audit findings in Gemma 3 chat, model store, and serve
426b43a Add Gemma 3 text GGUF support
d14f0b1 Add model nicknames to manager
```

If `1109c5e` is missing, stop and say so — §1 would be wrong and you would be
looking at unfixed code.

`main` is at `0223490` and is deliberately **not** merged with this branch,
because the Critical in Task 1 is still open. Do not merge or push to `main`.
Commit to `alpacca-gemma3-615` as you go; do not force-push. Delete this file
when the work is done.

The engine is dependency-free by design (pure Python, NumPy optional). Install
`numpy` for the fast path. `numba` is optional and was absent during the audit,
so `alpacca.kernels.available()` was False for every measurement quoted here.

---

## 0. Ground truth

Established empirically during the audit. Do not spend tokens re-deriving it —
except where Task 0 explicitly tells you to.

### The model this branch exists to support

The audit used a real Gemma-3-1B GGUF that is **not in this repo** and is not
available to you by default. Its metadata, which you should treat as the
reference for what a real Gemma 3 file looks like:

```
block_count=26  context_length=32768  embedding_length=1152
feed_forward_length=6912  head_count=4  head_count_kv=1
key_length=256  value_length=256  sliding_window=512
rope.freq_base=1e6  rope.freq_base_swa=1e4  rms_eps=1e-6
tokenizer: model="llama" (SPM), vocab 262144, bos=2 <bos>, eos=1 <eos>,
           <start_of_turn>=105, <end_of_turn>=106,
           add_bos_token=True, add_space_prefix=False
```

**Absent from that file** — these are the fallback paths production actually
takes, and none of them are covered by the test fixture (see Task 3a):
`attention.scale`, `attention.sliding_window_pattern`, `full_attention_interval`,
`vocab_size`, `rope.scaling.*`, `rope.dimension_count`, `embedding_scale`,
`final_logit_softcapping`, and `output.weight` (so embeddings are **tied**).

If you have network egress, you can fetch it — it is ~806 MB, loads in ~2 s and
decodes at ~1.4 tok/s fully quantized (use `ALPACCA_DENSE_WEIGHT_MB=0` to hold
RAM near 1 GB):

```sh
python3 -m alpacca pull hf:Andycurrent/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_GGUF
```

This is **optional but high value** — it is the only way to reproduce the
end-to-end checks. If you skip it, say so explicitly in your report and rely on
the tiny fixtures instead. Metadata-only reads (`GGUFFile.open` + `.metadata` /
`.tensors`) are cheap even for large files.

### Things verified CORRECT — do not "fix" these

- **The Gemma 3 forward-pass math.** Validated against an independent float64
  reference: max **2.1e-06** on NumPy, **2e-15** on the pure backend, across 6
  fixture configs × sequential/batch/chunked{1,2,3,4,5,7,256}. Order of
  operations, embed scale and its non-leakage into the tied head, per-head
  RMSNorm broadcasting, dual RoPE table selection, the `(i+1)%6` sliding rule,
  `kv_start`, the `-1e30` mask sentinel, GQA reshape layout, and softcap were
  each checked individually.
- **GGUF Gemma norm weights already have `+1` folded in** at conversion time
  (`blk.0.attn_norm.weight` mean +5.55). `T.rmsnorm(x, w) = x*inv*w` is correct.
  Do **not** add a `(1 + weight)`.
- **Tied output aliasing** (`m.output = m.tok_embd`, a Q8_0 QuantMatrix used for
  both row-gather and the logit matvec) is correct. Quantized matvec vs
  dequantize-then-GEMV on the 262144×1152 head: max abs diff 2.0e-05.
- **Chunked prefill / prefix reuse / cache truncation** under sliding-window
  attention: max divergence 1.37e-06 (float32 noise) across chunk sizes
  1,2,3,4,5,7,8,16,256.
- **No performance regression for non-Gemma architectures** from `426b43a`.
  A/B against the restored pre-commit package: llama3.2-1B NumPy decode
  797.6/815.0 ms/tok (old) vs 829.2/823.3 (new) — inside the old-vs-old spread;
  pure Python is marginally *faster*.

---

## 1. Already fixed in `1109c5e` (do not redo — but see Task 0b, they are unreviewed)

`git show 1109c5e` for the full diff.

| Area | Fix |
|---|---|
| `chat.py:77` | **gemma render now emits BOS** + applies the template's `\| trim` |
| `chat.py:165` | EOG token no longer counted in `res.tokens` / tok-s |
| `tokenizer.py:170` | name-based EOG additions gated on `TT_CONTROL`, so Gemma 3's USER_DEFINED `</s>` no longer halts generation |
| `store.py:271` | `resolve_model_input` returns file refs unchanged — fixes `alpacca rm ./NAME` deleting the wrong model, and `./missing.gguf` triggering a network pull |
| `store.py:180` | `_write_nicknames` uses a per-process `mkstemp` temp + `fsync` |
| `store.py:209` | new `_nicknames_lock()` (flock/msvcrt, best-effort) around every read-modify-write |
| `store.py:158` | `_read_nicknames` also catches `UnicodeDecodeError` |
| `store.py:149` | `_clean_nickname` replaces Unicode `Cc` control chars (kills ANSI escapes) |
| `store.py:316` | `set_model_nickname` no longer branches on its own error message; rejects path-shaped nicknames explicitly |
| `store.py:417` | `remove_model` never reports failure over alias bookkeeping |
| `cli.py:213` | `cmd_pull` resolves nicknames |
| `cli.py:226` | `cmd_list` clips the NICKNAME column |
| `cli.py:844` | `main()` catches `OSError` (also covers `urllib.error.URLError`) |
| `serve.py:131` | `RuntimeError` → JSON 400 on all routes; streaming closes cleanly with an error chunk + `[DONE]` |
| `serve.py:65` | `finish_reason` now `"length"` when the budget ran out; `usage` reports prompt/completion/total |
| `model.py:217` | `meta_required()` — missing/non-positive core dimensions raise a clear `ValueError` instead of `int(None)` |
| `model.py:269` | scalar-bool `sliding_window_pattern` no longer collapses to period 1 (which silently disabled SWA) |
| `model.py:1008` | `describe()` counts q/k-norm as shared `head_dim` vectors and counts an untied `output.weight` |
| `sample.py:22` | `_topk_indices` replaces the full 262k sort — **proven bit-identical** on both backends, 5× faster (44.2 → 8.9 ms) |

Evidence recorded at the time of that commit:
- `python3 tests/smoke.py` → **all 250 checks passed**
- `ALPACCA_PURE=1 python3 tests/smoke.py` → **all 179 checks passed**
- On the real model, `alpacca run <gemma3> "What is the capital of France?"`
  went from `Francia é tbe capan e tbe fcapiti al de Frane tbe capan.`
  to `The capital of France is **Paris**.`
- nickname concurrency (25 trials × 3 processes): invalid JSON 3/25 → 0/25,
  tracebacks 21/25 → 0/25, lost nicknames 24/25 → 0/25.

---

## 2. Your work, in priority order

### TASK 0 — Independently verify before you build on any of this

The audit and every fix in §1 were produced by a single agent that then tested
its own conclusions. That is not review. Do this first, and do it sceptically —
if something below is wrong, everything built on it inherits the error.

**0a. Re-derive the tokenizer diagnosis yourself. You are better placed to do
this than the agent that wrote it.** That agent had no independent tokenizer
library available and had to hand-port llama.cpp's algorithm from memory —
twice, by two sub-agents who could share a blind spot. If you have network
egress, you do not have that limitation:

```sh
pip install sentencepiece transformers    # or fetch llama.cpp's gguf-py / tokenizer
```

Independently establish (i) what `tokenizer.ggml.scores` actually means for a
Gemma vocab, and (ii) what llama.cpp's SPM tokenizer actually does. Then decide
whether Task 1's diagnosis holds. **Report disagreement loudly** rather than
working around it. This single check is the highest-value thing you can do.

**0b. Review commit `1109c5e`.** It passes both suites, but passing tests only
proves the tests did not catch a bug. Look hardest at:
- `sample.py::_topk_indices` — the first attempt was **wrong** (`argpartition`
  loses the tie-break at the cut; 964 mismatches) and the suite did **not** catch
  it. The current version is claimed bit-identical to
  `sorted(..., reverse=True)[:k]` on both backends. Re-prove that claim,
  including all-equal logits and `k >= n`.
- `store.py::_nicknames_lock` — best-effort locking that silently degrades to no
  lock. Check the failure path, the Windows `msvcrt` branch (never executed),
  and that no exception can leak a locked fd.
- `store.py::resolve_model_input` — now returns file refs unchanged. Confirm no
  caller depended on the old normalising behaviour (grep every call site).
- `serve.py` streaming error path — headers are already sent when the error
  fires; verify the chunked framing stays valid.
- `model.py::meta_required` — confirm it cannot reject a file that used to load.

**0c. Establish a baseline** before changing anything:
```sh
python3 tests/smoke.py
env PYTHONDONTWRITEBYTECODE=1 ALPACCA_PURE=1 python3 tests/smoke.py
```
Expect **250** and **179**. If you do not get those, stop and report — something
differs from what this packet assumes.

---

### TASK 1 — Critical: rewrite the SPM tokenizer (`alpacca/tokenizer.py:212`)

**The claimed defect** (verify it in 0a first). `Tokenizer._encode_spm` is a
unigram **Viterbi** that maximises the *sum* of `tokenizer.ggml.scores`. Gemma
stores **BPE merge ranks**, not log-probabilities: `score == -(id - 494)` for
92.5% of NORMAL tokens (236,249 of 255,474). Maximising a sum of ranks is
identical to *minimising the sum of token ids*, so it systematically prefers many
short low-id pieces over the one correct long piece.

```
▁capital  = -4785    vs  ▁c(-11) + ap(-176) + it(-15) + al(-20) = -222   → fragments win
▁the      = -12      vs  ▁t(-2)  + he(-3)                       = -5     → fragments win
```

llama.cpp selects `LLAMA_VOCAB_TYPE_SPM` for `tokenizer.ggml.model == "llama"`
(which this file sets) and does a **greedy highest-score-first bigram merge**
instead.

**Measured impact:**

| text | alpacca | canonical |
|---|---|---|
| `What is the capital of France?` | 14 tokens | 7 |
| `Write a short poem about the ocean.` | 14 | 8 |
| `user` / `model` (the chat role markers!) | `us`+`er` / `mod`+`el` | 1 each |
| 20k chars of README | 9855 | 6242 (**1.58×**) |

The model itself confirms the segmentation is off-distribution: 4.411 bits/byte
for alpacca's tokens vs 0.699 for the merge segmentation of the same string.

**What to do.** Replace `_encode_spm` with llama.cpp's `llm_tokenizer_spm`:

1. One symbol per UTF-8 *character* in a doubly-linked list (prev/next indices).
2. A max-heap of adjacent bigrams keyed on the merged piece's score; tie-break
   on the smaller left index.
3. Pop the best bigram; if either symbol has already been consumed, skip. Merge
   left+right into left, unlink right, then push the two new neighbour bigrams.
4. Emit surviving symbols; for any symbol not in the vocab, fall back to
   `byte_ids` per byte (and `unk_id` if there is no byte token).

This algorithm is correct for **both** rank-scored (Gemma) and log-prob-scored
(Llama 2) SPM vocabs, so it is not a Gemma special case — do not add an
arch-conditional branch.

**Constraints.**
- `add_space_prefix` (the real Gemma 3 file sets it **False**) and the
  `" " → "▁"` substitution must keep working.
- Must stay dependency-free and work on the pure-Python backend. Any reference
  library you install for verification must **not** become a runtime dependency.
- Keep `_piece_bytes` / `_max_piece_bytes` or remove them if they become dead.

**Required tests.**
- Golden id vectors for a Gemma 3 vocab, e.g.
  `encode("user", add_bos=False) == [2364]` and
  `encode("What is the capital of France?", add_bos=False) == [3689, 563, 506, 5279, 529, 7001, 236881]`.
  **Generate these from the authoritative tokenizer you validated in 0a**, not
  from this document — the values above came from a hand-port.
- A fixture-level test that a multi-character vocab entry encodes as ONE token
  when it exists. `tests/make_tiny_model.py:72` already writes rank-style scores
  (`-float(i+1)`), so a fixture reproduces the bug today with no model download.
- A round-trip test `decode(encode(s)) == s` over mixed ASCII/CJK/emoji.
- Confirm a BPE model (`tokenizer.ggml.model == "gpt2"`, e.g. llama3.2) is
  byte-for-byte unaffected — it takes `_encode_bpe`, a different path.

**Second-order effect to check.** `Model.prefill` reuses the KV cache by
comparing token prefixes (`model.py:960-966`). The model samples canonical ids
while `render()` currently re-encodes them non-canonically, which should break
prefix reuse at the first assistant content token and force a full re-prefill
every turn. This was reasoned from code, **not measured** — measure it before
and after your fix and report the number.

---

### TASK 2 — High: context-window management in the REPL (`alpacca/chat.py:158`)

**The defect.** With the default `n_predict=-1`, `generate` computes
`budget = model.n_ctx - model.n_past` *after* prefill, so the loop at
`chat.py:161` exits immediately when the budget is 0. `chat.interactive`
(`chat.py:339-346`) re-renders the **full** conversation every turn, never
trims, never resets, never checks remaining room, and does not catch the
`RuntimeError` that `Model.prefill` raises one turn later (`model.py:957`).

Reproduced (n_ctx=32 fixture — no model download needed): prompt of exactly
`n_ctx` → `tokens=0, text=''` with no error; `n_ctx-2` → silently truncated to 2
tokens; over → `RuntimeError` that propagates through `cmd_run` to
`cli.py:842-844`, printing one line and exiting 1 **with the in-memory
conversation lost**. From the menu it is caught at `cli.py:425-427`, but the
conversation is still gone and the user is never offered `/clear`.

Aggravated for Gemma 3: the model advertises `context_length=32768` but
`model.py:408` clamps the default to 4096, and Task 1's tokenizer defect burns
1.6× more tokens per character — so the ceiling arrives ~1.6× sooner on a model
whose headline feature is long context.

**What to do.**
- In `interactive`, before rendering: drop the oldest non-system message pairs
  until the rendered prompt leaves a usable reply budget (e.g. ≥128 tokens), and
  print `(dropped N earlier turns to fit the context window)`.
- Catch `RuntimeError` around `generate` in the REPL loop so the user can
  `/clear` instead of being ejected.
- Print effective `n_ctx` alongside `n_ctx_train` at load so the 32768→4096
  clamp is visible up front (`model.py:408`, surfaced via `describe()`).
- Consider making `generate` signal "no room" distinctly from "generated 0
  tokens" so callers can react.

**Required tests.** Nothing in `tests/` covers `n_past` near `n_ctx` for **any**
architecture (`rg -n n_ctx tests/smoke.py` returns nothing). Add: a `generate()`
test with a prompt of exactly `n_ctx` asserting a graceful result rather than
`tokens==0/text==''`; and a test that the REPL survives a context-overflow turn.

---

### TASK 3 — High-value test coverage (this is why the blockers shipped green)

The suite passes 250 checks and still missed two Critical defects. Fix the
structural cause, not just the symptoms. **None of this needs a model download.**

**3a. The gemma3 fixture takes the opposite branch from production.**
`tests/make_tiny_model.py:87-98` explicitly sets `gemma3.attention.scale`,
`attention.sliding_window_pattern` (as a **bool array**), `rope.scaling.type`,
`rope.scaling.factor`, and `final_logit_softcapping`. **No real Gemma 3 GGUF has
any of them** (see §0). So every fallback production relies on is untested:
- the `full_attention_period = 6` default and the `(i+1) % 6 != 0` rule
- the `attention.scale` fallback — and worse, the fixture's value is
  *numerically identical* to the code's fallback (`1/sqrt(head_dim)`), so the
  "metadata-driven attention scale" feature is verified by nothing
- the `n_layer == 62` 27B branch (`model.py:239`)
- the softcap-absent path

Add a **second** gemma3 fixture with none of those keys, mirroring the real
file's shape from §0, and assert the derived `Hyperparams` explicitly.

**3b. No test renders a chat for any format.** The only `render()` call
(`tests/smoke.py:1414-1415`) uses a llama fixture with no `chat_template`, so it
takes the `raw` path. Add `chat_template` + `<start_of_turn>`/`<end_of_turn>`
tokens to the gemma3 fixture, assert `detect_format() == "gemma"`, and assert
`render(...)[0] == tok.bos_id`. Generalise to every format whose template starts
with `bos_token`.

**3c. No gemma3 test exercises chunked prefill with `kv_start > 0`**
(`tests/smoke.py:738`), and the Q4_0 test (`smoke.py:747`) asserts only shapes
and a brittle matrix count (`{"Q4_0": 43}`) — nothing numeric.

**3d. The SWA parity test is a weak oracle** (`smoke.py:743`): batch and
sequential share the same silent `_rope_cos_swa is None` fallback, so the test
passes even if the SWA RoPE table were never built. Assert the sliding and
global RoPE tables actually differ.

**3e. gemma3 × dense-budget / tied-embedding densification** — the combination
the CLI selects by default — has zero coverage (`smoke.py:747`).

**3f.** `tests/bench.py` has no gemma3 fixture and `tests/acceptance.py` can
never run gemma3. Add `--arch` to `tests/make_bench_model.py` (it hardcodes
`GGUFWriter(path, "llama")` at line 91). `build.yml` runs only `smoke.py` and
`real_model_test.py` — consider adding the others.

---

### TASK 4 — Medium

| # | Defect | Location | Notes |
|---|---|---|---|
| 4a | Gemma system message becomes a second consecutive user turn; the real template folds system content into the **first user turn** as a prefix and forbids two user turns in a row | `chat.py:78` | **Contested** — see Risks. Measure before changing. |
| 4b | KV cache allocates full `n_ctx` for sliding-window layers: 1.35 GiB unreachable at 32k ctx | `model.py:424` | `np.zeros` is lazily mapped and the default clamp bounds waste to 154 MiB, so this is not urgent. **A naive ring buffer will corrupt prefix reuse** — see Risks. |
| 4c | Auto dense budget chosen from free RAM alone: 3814 MiB for a 768 MiB model (~3.3× RSS) for single-digit-to-11% decode; the tied 262144-row head is 30% of it | `cli.py:191` | Consider capping by measured benefit, or excluding the tied head from `_DENSIFY_TIERS`. |
| 4d | Q2_K/Q3_K GGUFs silently load every matrix as dense float32 (3.7 GiB for a 1B model); neither budget formula accounts for it | `model.py:316` | Real fix is adding Q2_K/Q3_K to `QUANT_GEOMETRY`/`np_unpack`. Minimum: warn loudly. |
| 4e | `StreamDecoder` stops emitting for the rest of the response after one unrecoverable UTF-8 byte — which also disables stop-string matching | `tokenizer.py:310` | |
| 4f | A multi-token stop string is leaked to the stream callback before it can be detected | `chat.py:168` | Hold back a tail of `max(len(s))-1` chars before emitting. |
| 4g | arch `gemma` (Gemma 1) is advertised as supported but runs the plain llama forward pass — no `sqrt(n_embd)` embed scale, SiLU instead of GELU | `model.py:280` | Pre-existing. Either implement it or stop advertising it in `SUPPORTED_ARCHES`/README. |
| 4h | A corrupt nicknames JSON is now survivable (fixed), but is still **silently discarded and permanently overwritten** by the next write | `store.py:158` | Warn to stderr and/or move the bad file aside as `.corrupt` instead of destroying it. |

### TASK 5 — Low

`model.py:239` 27B attention scale keys off `n_layer == 62` alone ·
`model.py:228` linear RoPE scaling now applies to **all** architectures (an
undeclared cross-arch behaviour change; also misses `factor`-without-`type`
files, where llama.cpp defaults to linear) ·
`model.py:222` unvalidated floats: `rope.freq_base=0` → all-NaN logits with no
error, `embedding_scale=0` → token-independent logits ·
`model.py:350` `tensor_vec` does no element-count validation, so a wrong-sized
norm vector loads cleanly and fails mid-generation with a cryptic broadcast
error · `model.py:791` `prefill` recomputes the 262144-row output head once per
256-token chunk and discards all but the last · `kernels.py:40` numba absent →
silent ~4× slower fallback with no hint to the user · `model.py:1014`
`_storage_description` reports matrix counts but never bytes ·
`cli.py:66` no default `<start_of_turn>` stop string for Gemma, and control
tokens decode to empty text, so a runaway turn silently role-plays the user ·
`store.py:196` nicknames the resolver will never honour are still advertised,
and orphaned entries are invisible with no way to enumerate them ·
`cli.py:255` `cmd_show` and `list_models` use different canonicalisations and
can disagree about a model's nickname · `cli.py:226` the NICKNAME column is
clipped but still sized with `len()`, so CJK/emoji still misalign (use a
display-width function).

### TASK 6 — Docs

`README.md:166-171` describes Gemma 3 support feature-by-feature. Re-check every
claim after your changes. Specifically: "Chat templates are detected from the
model's metadata" is true for *detection* but the gemma renderer is not faithful
to the template (see 4a). Also reconcile `SUPPORTED_ARCHES` with 4g, and check
the CLI's `EXAMPLES` block and `_print_controls()` for the same overstatements.

### TASK 7 — Improvements (non-bugs, do last)

Add Q2_K/Q3_K quantized support · expose the resolved Gemma 3 attention config
in `describe()` (sliding_window, period, attention_scale, and *which rule
fired*) so mis-detection is visible · wire up `store.list_nicknames()` (defined
at `store.py:187`, **zero call sites**) as `alpacca nickname --list` · convert
the `_rope_cos_swa is None` fallback (`model.py:663-664, 679-680`) into an
assertion — it is provably unreachable today but would silently rotate sliding
layers with the 1e6 global base · re-tune `_SMALL_MATVEC_ELEMS` (`qmatrix.py:55`;
the "small" path measured slower than einsum at every size on the audit box —
~1.6 ms/token across Gemma 3's 52 attn_k/v matvecs — but the crossover is
machine-specific, so re-measure on your hardware rather than blind-change).

---

## 3. Risks and traps

- **4a is contested.** Four auditors disagreed. Three graded it Medium, one
  High. Critically, the auditor who ran it end-to-end on the real model found
  the *faithful* template produced a **worse** answer on that particular
  fine-tune. The reference divergence is certain; the quality consequence is
  unproven in **both** directions. Measure before you change it, and do not
  claim "system prompts are ignored" without a fresh measurement.
- **4b: a naive SWA ring buffer will silently corrupt prefix reuse.**
  `_truncate_cache` (`model.py:450-460`) + `prefill` (`model.py:960-966`) can
  restart at position 100 after the cache reached 3000, where a `pos % window`
  slot would hold position 2659's K. Any KV fix must preserve absolute-position
  semantics or explicitly invalidate on truncation. Also, the batch path reaches
  `window + chunk - 1` rows, so a window-sized cache is too small at the default
  chunk of 256.
- **No 4B/12B/27B checkpoint was ever loaded.** The rope-scaling path (factor
  8.0, global layers only) and the 27B attention scale are untested against
  reality. Do not claim variant support you have not exercised.
- **The bool-array `sliding_window_pattern` polarity was never confirmed against
  an external converter** — only against this repo's own fixture. If the real
  convention is inverted, every layer's attention type flips on files carrying
  the key. Check `gguf-py` before relying on it. You may have network access the
  original auditor did not — use it.
- **No real converter is known to emit `gemma3.attention.scale`**, which is why
  the `n_layer == 62` heuristic is load-bearing. If a third-party converter ever
  writes it with `query_pre_attn_scalar` semantics (256) rather than the
  reciprocal (0.0625), `model.py:241` trusts it unconditionally and every
  attention softmax is catastrophically mis-scaled with no diagnostic.
- **The numba path was never exercised.** All perf numbers are numpy-only, against
  reference netlib BLAS. On threaded OpenBLAS the dense budget looks several
  times better than measured — re-measure before acting on 4c.
- **One pre-existing accounting bug is still open:** `auto_budget_fit_mb` counts
  a non-densified *tied* `token_embd` in neither `eligible` nor `residual`
  (~321–390 MiB unaccounted on a 1B model). The sibling `describe()` bug
  (−16.2% on the llama fixture) is already fixed in `1109c5e`.
- **Q4_K/Q6_K at 1152 columns fails `can_quantized_matvec`** (1152 % 256 = 128).
  The shipped Gemma 3 1B avoids this only because llama.cpp itself refuses
  K-quants on non-256-multiple rows. A third-party conversion forcing them would
  silently go fully dense (3.7 GiB) with only "dense fallback Q4_K" as the clue.
  Gemma 3 4B/12B/27B (`n_embd` 2560/3840/5376) are multiples of 256 and safe.
- **The `or default` idiom** at `model.py:219/220/223/224/232/241/257/270/279`
  swallows legitimate zeros. Currently harmless at every one of those sites; the
  harmful sites (no default at all) are already fixed via `meta_required`. If you
  add metadata keys to that block, the idiom will keep hiding zeros.

---

## 4. How to verify your work

Every change must keep both of these green:

```sh
python3 tests/smoke.py                                                # expect: all 250 checks passed
env PYTHONDONTWRITEBYTECODE=1 ALPACCA_PURE=1 python3 tests/smoke.py    # expect: all 179 checks passed
```

The counts should only go **up** as you add tests. If a count drops, you removed
coverage — say so and justify it.

If you pulled the real model (§0), this must keep producing a correct answer:

```sh
env ALPACCA_DENSE_WEIGHT_MB=0 python3 -m alpacca run \
  hf:Andycurrent/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_GGUF \
  "What is the capital of France?" -n 24 --seed 1 -c 256
# expect something equivalent to: The capital of France is **Paris**.
```

If you did not pull it, state that plainly in your report and note which claims
you therefore could not verify.

For Task 1 specifically, also report before/after token counts for the table in
§2 Task 1, and confirm a BPE model's output is byte-identical.

Report exact commands and outcomes. If a fix turns out to be wrong or a claim
above does not reproduce, say so plainly rather than working around it — several
findings here were downgraded or refuted during verification, and that is a
normal outcome. Do not merge to `main`.
