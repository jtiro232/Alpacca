# Alpaccaroo Qwen3.8-27B Hybrid Runtime Update Plan

- **Branch:** `qwen38-python`
- **Document status:** implementation plan; no Qwen3.8 runtime is claimed complete
- **Prepared:** 2026-08-19
- **Primary target:** [Unsloth Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- **Repository baseline:** commit `9d505cc148923d2f96a4f9ba7b1106aad66ed7b8`

## 1. Executive decision

Alpaccaroo can be extended to run the Qwen3.8-27B GGUF family while preserving its pure-Python architecture. The safe design is not to add a model name to the existing transformer switch. It is to add a separate, metadata-driven `qwen35` architecture backend that shares Alpaccaroo's tokenizer, tensor, quantization, sampling, CLI, and server infrastructure while owning its hybrid layer execution and state.

The target model combines two different layer types:

- Conventional full-attention layers with positional K/V cache.
- Gated DeltaNet recurrent layers with causal-convolution history and a recurrent matrix state.

That hybrid design is the architecture Alpaccaroo is currently missing. It also explains why a coding model is not categorically forbidden: "coding" is a learned capability, not a runtime architecture. Alpaccaroo can run a coding-capable model when it understands that model's tensor layout, forward equations, tokenizer/chat contract, state, and efficient kernels.

The recommended production target is:

```text
Model:                  exact Qwen3.8-27B GGUF whose header declares qwen35
Initial quantization:   Q4_K_M
Capability:             text generation and coding; one active sequence first
Correctness backend:    pure Python, then NumPy
Optimized CPU backend:  pinned Numba
GPU backend:            Numba-CUDA with weights kept in native packed GGUF form
Recurrent state:        float32
Attention K/V:          float32 first, float16 only after parity and stability tests
Practical 24 GiB goal:  4K-64K context depending on K/V type and measured headroom
Recovery:               periodic state checkpoint plus deterministic replay
Deferred initially:     vision, MTP acceleration, speculative decoding, multi-user batching
```

The implementation must follow a correctness-first ladder. "It generated plausible text" is not acceptance. A recurrent model can produce coherent output while its state update, layer schedule, RoPE sections, tokenizer, or gating is wrong. The branch is done only after deterministic parity against a pinned llama.cpp oracle, state/rollback tests, old-model regressions, and a practical 27B run all pass.

## 2. Purpose and portability

This document is intended to be executable on a different development computer. It deliberately does not assume the current computer's paths, GPU, installed packages, model files, or llama.cpp build. Every machine-specific value must be discovered and recorded before implementation begins.

The document uses these placeholders:

```text
<REPO>         Alpacca/Alpaccaroo checkout
<MODEL_DIR>    directory containing GGUF files
<TARGET_GGUF>  exact Qwen3.8-27B GGUF selected for release
<LLAMA_CLI>    pinned external llama.cpp executable used for comparison
```

The implementation branch should remain `qwen38-python` unless the maintainer explicitly renames it. Internally, use `qwen35` for the architecture identifier if and only if the target GGUF header declares `general.architecture = qwen35`. Product-facing text may call the model Qwen3.8. This distinction avoids hard-coding a marketing/version label into an established GGUF architecture contract.

## 3. Scope

### 3.1 Required in the first complete release

- Inspect and validate the exact target GGUF before allocating model memory.
- Recognize the metadata architecture and classify every layer from metadata/tensor evidence.
- Load every required recurrent and full-attention tensor with strict shape validation.
- Implement the Qwen hybrid equations in a dependency-free pure-Python reference backend.
- Implement the same architecture in NumPy without changing semantics.
- Add pinned Numba CPU kernels after pure/NumPy parity.
- Add Python/Numba-CUDA packed matrix storage and kernels for every matrix quantization present in the selected Q4_K_M artifact.
- Keep every committed implementation, kernel, utility, and test in Python source. No C, C++, Rust, Cython, custom DLL/SO, or custom compiled extension is added to Alpaccaroo.
- Add a dedicated hybrid CUDA execution chain; do not route Qwen3.8 through the existing conventional `DecodeChain`.
- Allocate K/V only for full-attention layers.
- Allocate causal-convolution and Gated DeltaNet state only for recurrent layers.
- Support token-by-token decode, chunked prefill, reset, snapshot, restore, branch, rollback, and replay.
- Preserve standard-library-only operation when `ALPACCAROO_PURE=1` is set.
- Preserve existing Llama, Mistral, Qwen2, Qwen3, StableLM, Gemma, and Gemma3 behavior.
- Match the target tokenizer and embedded chat template exactly for fixed fixtures.
- Expose backend, memory plan, unsupported features, and fallbacks clearly in CLI/model descriptions.
- Pass the focused tiny, 0.8B, exact 27B, state, regression, memory, and performance checks in this plan.

### 3.2 Explicitly deferred from the first release

- Vision input and multimodal projection.
- MTP/next-N-token acceleration, even if auxiliary MTP tensors are present.
- Speculative decoding.
- General multi-sequence continuous batching.
- Float16 recurrent state.
- Quantizers not present in the chosen release artifact unless already supported and tested.
- The model's maximum advertised context on hardware that cannot hold the required K/V state safely.
- An OpenAI-compatible API for serializing opaque recurrent state over the network.

Deferred features must fail loudly or report that they are disabled. They must never be silently accepted and ignored.

### 3.3 Python-only and verification boundary

This branch is Python-only. "Native packed CUDA" means that Python/Numba-CUDA kernels read native packed GGUF blocks directly; it does **not** authorize native extension source. Numba's existing runtime JIT tier is allowed because the repository remains Python source and the pure standard-library fallback remains functional.

llama.cpp may be invoked as an external, unmodified reference executable and its source may be read to implement the contract. Do not add or maintain a C/C++ oracle helper, patched llama.cpp fork, trace platform, evidence database, or release-verification framework in this branch.

Verification stays proportional to implementation:

- small deterministic Python unit vectors for the recurrent equations and state operations;
- one generated tiny hybrid GGUF;
- focused token/logit checks against an external llama.cpp reference;
- 0.8B smoke/parity before the exact 27B run;
- existing Alpaccaroo smoke regressions;
- focused memory/throughput checks for the packed CUDA target.

Test logs are sufficient. Do not build generalized verification infrastructure unrelated to running the model correctly.

## 4. Definition of done

The branch is complete only when all of the following pass:

1. The exact Hugging Face URL, revision, filename, byte size, SHA-256, GGUF metadata, tokenizer metadata, chat template, tensor manifest, and dtype census are recorded.
2. A pinned external llama.cpp executable/version is recorded and loads the exact file.
3. Tokenizer IDs and rendered chat-template bytes match the external reference exactly on the fixed corpus.
4. A deterministic tiny F32 hybrid GGUF is accepted by both llama.cpp and Alpaccaroo.
5. Pure Python passes the hand-computable primitive/layer/state vectors and the tiny model matches the external reference's deterministic output.
6. NumPy matches the pure backend and focused external comparisons.
7. All quantization formats advertised by the Qwen backend have explicit decode/matmul and parity tests.
8. Chunked prefill and token-by-token execution agree.
9. Reset, snapshot, restore, rollback, branching, prefix-slot switching, and independent-sequence tests pass.
10. A real 0.8B model passes before the 27B model is used as the primary debugging target.
11. A real 2B model passes if a compatible artifact is available; otherwise the exception is documented and the 0.8B plus synthetic coverage must be expanded.
12. The exact 27B artifact passes bounded deterministic CPU correctness tests.
13. The packed CUDA path passes numerical, state, failure-recovery, transfer, and memory gates.
14. The practical target configuration loads and generates a fixed coding corpus without silent CPU fallback, NaN/Inf, state corruption, or memory overrun.
15. The existing Alpaccaroo smoke suite and relevant real-model/server tests remain green.
16. Documentation clearly distinguishes implemented, experimental, and unsupported capabilities.

The following are specifically **not** proof of completion:

- The loader no longer rejects `qwen35`.
- The model prints readable text once.
- Greedy tokens happen to match on one short prompt.
- CUDA initializes but transfers large state every token.
- A Q4 file fits in system RAM but not in the GPU representation.
- Long context is advertised from metadata without a measured allocation and run.

## 5. Audited baseline and architectural gap

At the baseline commit, Alpaccaroo's principal execution path assumes each layer is approximately:

```text
RMSNorm
  -> Q/K/V projections
  -> RoPE
  -> grouped-query softmax attention over K/V cache
  -> output projection + residual
  -> RMSNorm
  -> SwiGLU feed-forward + residual
```

Relevant existing files include:

| Area | Existing file | Current limitation relevant to Qwen3.8 |
|---|---|---|
| Model registry, loader, state, forward | `alpaccaroo/model.py` | Conventional tensor roles and K/V on every layer; no recurrent state or `qwen35` backend |
| Tensor facade | `alpaccaroo/tensor.py` | Useful shared operations, but no sigmoid/softplus/L2/recurrent contracts |
| Quantized matrices | `alpaccaroo/qmatrix.py`, `alpaccaroo/quants.py` | Strong CPU base; GPU representation is not native packed GGUF blocks |
| CPU acceleration | `alpaccaroo/kernels.py` | Conventional attention and matvec; no causal conv/GDN and assumptions need head-dimension generalization |
| CUDA acceleration | `alpaccaroo/cuda.py` | Conventional chain, fixed attention geometry in key paths, no hybrid state; expanded weights are too large for a 27B Q4 target on 24 GiB |
| Chat/template | `alpaccaroo/chat.py` | Reusable only after exact Qwen tokenizer/template parity and state-aware truncation |
| Server | `alpaccaroo/serve.py` | Single serialized model is acceptable initially, but token IDs alone cannot reconstruct hybrid state cheaply |
| Prefix cache | model internals | Stores/restores K/V; recurrent state would be stale or missing |
| Tests | `tests/smoke.py`, `tests/real_model_test.py`, `tests/acceptance.py` | Good regression base; needs deterministic hybrid fixtures and oracle parity |

The existing conventional path must remain a fast, stable specialization. Qwen support should be an architecture backend selected at load time, not dozens of recurrent conditionals inside every standard-model loop.

## 6. Target architecture contract

### 6.1 Model identity must be discovered, not guessed

The linked repository is named Qwen3.8, while compatible GGUF implementations may declare the established architecture string `qwen35`. The loader must treat the GGUF as authoritative. Before coding against dimensions, the inspector must report:

- `general.architecture`
- model name, source, quantizer, and file type
- block count and any next-N prediction block count
- embedding and feed-forward dimensions
- query/KV head counts and explicit key/value dimensions
- context and RoPE metadata
- recurrent layer list or full-attention interval
- every `qwen35.ssm.*` value
- RoPE dimension sections
- tokenizer pre-tokenizer, special IDs, vocabulary size, merges/tokens, and chat template
- tensor names, shapes, dtypes, offsets, and byte sizes

If the architecture is not supported by the implemented schema, loading stops before large allocation. Do not alias an unknown architecture to `qwen35` based only on the filename.

### 6.2 Expected dense 27B profile, subject to header verification

Planning estimates use this likely profile:

| Property | Planning value |
|---|---:|
| Main layers | 64 |
| Recurrent layers | 48 |
| Full-attention layers | 16 |
| Hidden width | 5120 |
| FFN width | 17408 |
| Query heads | 24 |
| K/V heads | 4 |
| Full-attention head dimension | 256 |
| Recurrent convolution kernel | 4 |
| Recurrent key groups | 16 |
| Recurrent value/time-step heads | 48 |
| Recurrent state dimension | 128 |
| Recurrent inner width | 6144 |
| Mixed convolution width | 10240 |

These values are validation expectations, not constants. The backend must derive shapes from metadata and tensors and produce a precise mismatch error.

Layer scheduling must follow this precedence:

1. Use explicit `qwen35.attention.recurrent_layers` metadata when present.
2. If it is absent, use `qwen35.full_attention_interval`; default the interval to `4` only when the interval metadata is also absent.
3. For zero-based main layer `i`, mark it full attention exactly when `(i + 1) % interval == 0`; mark the other main layers recurrent.
4. Mark appended `nextn_predict_layers` as non-recurrent and outside the main text stack.
5. Cross-check the result against each layer's tensor roles and reject contradictions.

### 6.3 GGUF metadata contract

The initial backend should understand at least:

```text
general.architecture
<arch>.block_count
<arch>.embedding_length
<arch>.feed_forward_length
<arch>.attention.head_count
<arch>.attention.head_count_kv
<arch>.attention.key_length
<arch>.attention.value_length
<arch>.attention.layer_norm_rms_epsilon
<arch>.context_length
<arch>.rope.freq_base
qwen35.rope.dimension_sections
qwen35.attention.recurrent_layers
qwen35.full_attention_interval
qwen35.ssm.conv_kernel
qwen35.ssm.inner_size
qwen35.ssm.state_size
qwen35.ssm.time_step_rank
qwen35.ssm.group_count
qwen35.nextn_predict_layers
```

Metadata array types and integer widths must be validated. Missing optional metadata may use only a documented upstream-compatible fallback. Required state dimensions have no guessed fallback.

For main inference, calculate:

```text
main_layer_count = block_count - nextn_predict_layers
```

MTP layers are inventoried and excluded from the main stack only when the metadata and tensor manifest agree. Initial text generation should report MTP as present-but-disabled.

Valid appended MTP tensors must not be mislabeled as corruption. Inventory the appended dense attention/FFN block roles plus `nextn.eh_proj [2*hidden, hidden]`, `nextn.enorm [hidden]`, `nextn.hnorm [hidden]`, optional `nextn.embed_tokens [hidden, vocab]`, optional `nextn.shared_head_head [hidden, vocab]`, and optional `nextn.shared_head_norm [hidden]`. Cross-check exact GGUF orientation against the pinned contract. They remain disabled for the initial text path.

### 6.4 Tensor-role contract

Global tensors:

```text
token_embd.weight
output_norm.weight
output.weight                 optional only when tied embeddings are valid
```

Every main layer:

```text
blk.N.attn_norm.weight
blk.N.post_attention_norm.weight
blk.N.ffn_gate.weight
blk.N.ffn_up.weight
blk.N.ffn_down.weight
```

Full-attention layer roles:

```text
blk.N.attn_q.weight           or an explicitly supported combined form
blk.N.attn_k.weight
blk.N.attn_v.weight
blk.N.attn_output.weight
blk.N.attn_q_norm.weight
blk.N.attn_k_norm.weight
```

For Qwen35 full-attention layers, `blk.N.attn_q.weight` is the joint Q+gate projection. Its output width is `2 * query_heads * head_dim`, with each head laid out as a Q segment and gate segment at a stride of `2 * head_dim`. Apply Q normalization and RoPE only to Q. Apply sigmoid to the gate and multiply it with the attention result. Do not require a separate full-attention `attn_gate` tensor.

Recurrent-layer roles:

```text
blk.N.attn_qkv.weight
blk.N.attn_gate.weight
blk.N.ssm_conv1d.weight
blk.N.ssm_dt.bias
blk.N.ssm_a
blk.N.ssm_alpha.weight
blk.N.ssm_beta.weight
blk.N.ssm_norm.weight
blk.N.ssm_out.weight
```

The exact target manifest decides whether additional biases or naming variants are legal. Each accepted variant needs a fixture.

Important converter rule: compatible GGUF conversion has already transformed and reordered recurrent tensors. The implementation must record the exact target-converter contract. In the audited Qwen35 path this includes `ssm_a = -exp(A_log)`, splitting source `in_proj_qkvz` into GGUF QKV and Z/gate tensors, grouped-to-tiled V-head reordering across QKV/Z/alpha/beta/convolution/output roles, `dt_bias` naming conversion, convolution-kernel squeeze, and the required recurrent norm-weight adjustment. Alpaccaroo consumes the GGUF representation exactly once. The `% key_head_count` mapping below applies to the post-converter tiled GGUF order, not raw Hugging Face order.

### 6.5 Full-attention forward contract

For a full-attention layer, preserve the upstream operation order:

```text
residual = x
x_norm = rms_norm(x, attn_norm)
q_and_gate = project_q(x_norm)
k = project_k(x_norm)
v = project_v(x_norm)
q = q_norm(q)
k = k_norm(k)
q, k = qwen_mrope(q, k, positions, dimension_sections)
attention = causal_gqa(q, k_cache, v_cache)
attention = attention * sigmoid(attention_gate)
x = residual + project_output(attention)
residual = x
x_norm = rms_norm(x, post_attention_norm)
x = residual + ffn_down(silu(ffn_gate(x_norm)) * ffn_up(x_norm))
```

The exact Q/gate split, reshaping, scaling, K/V append order, and MRoPE position convention must be validated against the oracle. The likely full-attention head dimension is 256, so any kernel specialized for 128 must dispatch to a correct 256-capable implementation.

Use Qwen35 **interleaved MRoPE (IMROPE)**, not contiguous MRoPE or ordinary one-dimensional RoPE. For text tokens, provide four position lanes `[p, p, p, 0]` and metadata sections such as `[11, 11, 10, 0]`, then apply the interleaved section pattern used by the pinned `GGML_ROPE_TYPE_IMROPE` implementation.

### 6.6 Recurrent forward contract

Use conceptual layouts:

```text
x:            [batch, tokens, embedding]
q/k:          [batch, tokens, key_heads, state_dim]
v/z:          [batch, tokens, value_heads, state_dim]
conv state:   [batch, conv_kernel - 1, mixed_conv_width]
delta state:  [batch, value_heads, value_dim, key_dim]
```

For the expected dense model:

```text
key_heads    = group_count = 16
value_heads  = time_step_rank = 48
state_dim    = 128
inner_size   = value_heads * state_dim = 6144
conv_width   = 2 * key_heads * state_dim + inner_size = 10240
```

Validate the Qwen35 structural constraints before allocation:

```text
key_dim == value_dim == state_size
inner_size % value_heads == 0
value_heads % key_heads == 0
conv_width == inner_size + 2 * key_heads * state_size
```

The upstream physical recurrent-state order is `[state_dim, state_dim, value_heads, sequence]`. Alpaccaroo may use a batch-first logical view, but the transpose/layout boundary must be explicit and tested rather than implied by generic `key_dim`/`value_dim` axes.

The decomposed recurrent order is:

1. Apply attention RMSNorm.
2. Project the mixed Q/K/V input.
3. Project the recurrent output gate `z`.
4. Calculate beta and alpha/time-step inputs.
5. Append mixed Q/K/V input to causal-convolution history.
6. Apply depthwise causal convolution and SiLU.
7. Split Q, K, and V with exact target ordering.
8. L2-normalize Q and K.
9. Map key heads to value heads using the upstream grouping rule.
10. Calculate beta with sigmoid.
11. Calculate time step with softplus and combine it with the already-converted `ssm_a`.
12. Decay and update the Gated DeltaNet state sequentially by token.
13. Read the updated state with Q and scale by `1 / sqrt(state_dim)`.
14. Apply learned recurrent RMSNorm and multiply by `SiLU(z)`.
15. Apply `ssm_out`, residual, post-attention RMSNorm, SwiGLU FFN, and the second residual.

Numerical definitions to freeze in tests:

```text
l2(x) denominator = max(sqrt(sum(x_i * x_i)), epsilon)
beta              = sigmoid(beta_projection)
dt                = softplus(alpha_projection + dt_bias)
log_decay         = dt * ssm_a
decay             = exp(log_decay)
```

Do not substitute `sqrt(sum_sq + epsilon)` for the L2 denominator without authoritative source/model proof. Do not replace the post-GDN `SiLU(z)` gate with sigmoid.

For one value head and one token, with state orientation `[value_dim, key_dim]`:

```text
A_bar      = A_previous * exp(log_decay)
prediction = A_bar @ k
delta      = beta * (v - prediction)
A_new      = A_bar + outer(delta, k)
output     = (A_new @ q) / sqrt(state_dim)
```

Key-head mapping is interleaved in the audited contract:

```text
key_head_for_value_head = value_head_index % key_head_count
```

Do not implement contiguous group mapping unless the target oracle proves a different layout.

A mandatory orientation-lock unit test should use a zero 2x2 state, `k=[1,0]`, `v=[2,3]`, `beta=0.5`, and `q=[1,0]`. Before output scaling, the updated state must be `[[1,0],[1.5,0]]` and the state read must be `[1,1.5]`. This catches swapped outer-product operands and transposed state reads.

During prefill, recurrence is causal and stateful across tokens. The initial correct implementation scans the token dimension sequentially while vectorizing batch and heads. Chunk-level parallel algorithms may be added only after exact equivalence to the sequential reference.

## 7. Proposed software architecture

### 7.1 Design principles

- Preserve `alpaccaroo.model.Model` as the public facade where practical.
- Select an architecture backend once during load; avoid checking the architecture inside every low-level operation.
- Keep pure Python as the semantic reference and an operational fallback.
- Make NumPy, Numba, and CUDA implement the same named contracts.
- Separate immutable weights from mutable sequence state.
- Make state type explicit per layer.
- Treat GPU placement as a planned allocation, not a greedy side effect.
- Fail closed on unknown tensors, shapes, quantizers, layer schedules, and unsupported features.
- Keep existing standard-model objects and hot loops stable behind a compatibility adapter.
- Keep a small optional debug callback before fusion so a failing optimized kernel can be compared to the pure Python path.

### 7.2 Recommended module map

The exact filenames may be adjusted during the initial refactor, but ownership should remain clear:

```text
alpaccaroo/
  architecture.py        architecture protocol, registry, and dispatch
  weights.py             shared GGUF tensor loading and strict role validation
  memory.py              layer memory specs, live state, snapshots, replay/checkpoints
  qwen35.py              config, tensor roles, layer objects, pure runner/backend
  qwen35_ops.py          dependency-free recurrent/full-attention reference primitives
  qwen35_numpy.py        NumPy implementation of the same operations
  qwen35_kernels.py      optional pinned Numba CPU kernels and dispatch
  packed_gpu.py          packed GGUF GPU matrix abstraction and Python memory planner
  qwen35_cuda.py         dedicated hybrid chain and Numba-CUDA kernels

tests/
  qwen35_fixture.py      deterministic fixture definitions and expected contracts
  make_qwen35_model.py   tiny hybrid GGUF generator
  qwen35_test.py         focused primitive, layer, state, and backend tests

docs/
  ALPACCAROO_QWEN38_UPDATE_PLAN.md
  QWEN35_RUNTIME.md       user-facing capability and operation documentation
```

Do not place all Qwen code into `model.py`, `kernels.py`, or `cuda.py`. Existing modules may expose stable extension hooks, but the recurrent implementation and hybrid CUDA chain should remain separate enough to test and disable independently.

### 7.3 Architecture backend contract

The backend needs equivalent operations to:

```python
class ArchitectureBackend(Protocol):
    architecture: str

    def validate_manifest(self, gguf) -> ModelManifest: ...
    def load_weights(self, gguf, manifest, policy) -> ModelWeights: ...
    def new_state(self, *, context, sequences, backend) -> ModelState: ...
    def forward_token(self, token_id, state, *, trace=None) -> Logits: ...
    def prefill(self, token_ids, state, *, chunk_size, trace=None) -> Logits: ...
    def estimate_memory(self, context, sequences, placement) -> MemoryPlan: ...
    def describe(self) -> dict: ...
```

The existing standard transformer should be represented by a compatibility backend without a behavioral rewrite in the first architecture-registry commit. Qwen35 then becomes the first backend with heterogeneous layer memory.

### 7.4 Shared tensor contract

Add only generally useful primitives to `tensor.py`, such as:

- stable sigmoid
- stable softplus
- L2 normalization with an explicit epsilon convention
- typed reshape/view helpers
- backend-neutral trace naming

Keep Gated DeltaNet equations in `qwen35_ops.py` until a second architecture proves they belong in a more general module. This avoids turning the common tensor facade into an architecture-specific dumping ground.

### 7.5 Dependency policy

The core package remains dependency-free and contains Python source only. Optional extras retain the current tiering:

```text
pure standard library
  -> optional NumPy
  -> optional pinned Numba CPU
  -> optional pinned Numba-CUDA
```

The pure implementation should favor `array('f')`, `memoryview`, `struct`, and explicit loops over millions of Python float objects. It is a correctness path, not a promise that 27B pure inference is fast. If Numba-CUDA cannot meet the practical target, report that blocker; do not quietly introduce a native extension.

## 8. Hybrid memory and state architecture

### 8.1 Why the current K/V abstraction is insufficient

A conventional K/V cache can be truncated by discarding rows after a token position. A recurrent Gated DeltaNet state cannot. Its matrix at position `N` is a function of the entire prefix. Changing `n_past` without reconstructing that state creates silent corruption.

Qwen state must therefore be represented as heterogeneous per-layer memory:

```text
ModelState
  position
  token_ids
  model_fingerprint
  schema_version
  generation
  layer_states
    FullAttentionState
      key[position, kv_head, key_dim]
      value[position, kv_head, value_dim]
      logical_length
    RecurrentState
      conv_history[conv_kernel - 1, mixed_width]
      delta_matrix[value_head, value_dim, key_dim]
      logical_position
```

The old standard-model K/V buffers should be wrapped in a compatibility `FullAttentionState`; the first refactor must demonstrate that this adapter does not alter old behavior or storage.

### 8.2 Layer memory descriptors

Derive an immutable descriptor for each layer:

```python
LayerMemorySpec(
    layer_index=...,
    kind="attention" | "recurrent" | "none",
    kv_heads=...,
    key_dim=...,
    value_dim=...,
    conv_channels=...,
    conv_kernel=...,
    recurrent_key_heads=...,
    recurrent_value_heads=...,
    recurrent_key_dim=...,
    recurrent_value_dim=...,
)
```

The descriptors are part of the model fingerprint and snapshot schema. A snapshot must be rejected if any descriptor differs, even if the token IDs match.

### 8.3 Required state operations

The backend-neutral state API must support:

```text
reset()
snapshot()
restore(snapshot)
clone()
truncate(position)
checkpoint(position)
replay(token_ids)
invalidate(component or generation)
memory_bytes()
describe()
```

GPU implementations additionally need:

```text
begin_step_or_chunk()
commit_step_or_chunk()
abort_step_or_chunk()
checkpoint_to_host()
restore_to_device()
```

The public model API can continue exposing `reset`, `prefill`, `forward`, `forward_batch`, and `truncate`. Internally, all five operate on `ModelState` rather than directly on `cache_k`, `cache_v`, and `n_past`.

### 8.4 Ownership and aliasing rules

- Mutable live state has exactly one execution owner.
- A stored snapshot is logically immutable.
- The initial CPU implementation deep-copies K/V, convolution history, DeltaNet matrices, token IDs, and metadata.
- Prefix slots must never share writable buffers with live state.
- Restore validates the model fingerprint, schema version, descriptors, shapes, dtype, and position before publishing any change.
- Restore is atomic from the caller's perspective: prepare and validate all buffers, then swap the complete state.
- Copy-on-write is a later optimization and requires aliasing tests before activation.
- A model/quantization/adapter revision change invalidates every prior snapshot.

### 8.5 Truncation, rollback, and branching

For attention-only state, truncation may slice K/V rows. For hybrid state:

1. Find the nearest complete checkpoint at or before the target token.
2. Restore its attention, convolution, and DeltaNet components together.
3. Replay tokens from that checkpoint to the requested position.
4. Compare the reconstructed state/logits to a fresh replay in tests.
5. If no checkpoint exists, reset and replay from token zero.

Never restore K/V without recurrent state, restore the DeltaNet matrix without convolution history, or change the logical position alone.

Branching uses complete snapshots:

```text
run prefix P
snapshot P
run suffix A and save logits/state
restore P
run suffix B
restore P
run suffix A again
require the same logits and final state as the first A run
```

This is a hard regression test because a K/V-only implementation will often appear functional but fail it.

### 8.6 Prefix cache design

A prefix-slot key should include:

```text
model fingerprint
architecture/schema version
weights and adapter identity
token tuple
positioning and RoPE configuration
memory representation version
backend-independent numerical mode where relevant
```

Each slot records its full byte cost and participates in byte-budgeted LRU eviction. A recurrent snapshot is roughly 150 MiB per sequence for the expected 27B dimensions before K/V, so retaining many slots is expensive. The first correct release may use a small default slot count or checkpoints plus replay. It must report the cost instead of hiding it.

### 8.7 Host/device authority reconciliation

Logical state semantics must be backend-neutral, but physical authority differs:

- Pure, NumPy, and Numba CPU: host memory is authoritative after every successful step.
- CUDA while actively decoding: device memory is authoritative between explicit checkpoints. Copying about 150 MiB of recurrent state to the host every token is prohibited.
- The host holds a last-known-good complete checkpoint plus token history after that checkpoint.
- On successful checkpoint, copy state to pinned host buffers and advance the checkpoint generation atomically.
- On a CUDA error that may have partially mutated state, discard the device generation, restore the latest complete host checkpoint, and replay the recorded suffix on a selected safe backend.
- Do not attempt to continue from a partially updated device matrix based only on a token counter.

This model combines correctness with performance: host checkpoints provide recovery, while device-resident live state avoids catastrophic per-token transfers.

Start with checkpoints every 128 tokens and make the interval configurable in the range 128-512 after fault-injection and performance measurements. Branch creation, arbitrary rollback, or explicit snapshot may force an immediate checkpoint and should report its cost.

### 8.8 Server semantics

The initial release may preserve the current one-model/one-lock serialized server. Prefix snapshots can switch conversations safely, but a token list alone is not an opaque recurrent-state handle. Therefore:

- Keep conversation text/history separate from model state snapshots.
- Do not claim that an API `context` token array resumes hybrid state without replay.
- Do not expose process-specific GPU pointers or mutable snapshots.
- Add persistent session handles only after lifecycle, byte limits, model fingerprinting, eviction, and replay behavior are designed.

## 9. Memory model and hardware planning

### 9.1 Exact formulas

For one sequence, recurrent DeltaNet state bytes are:

```text
n_recurrent_layers * value_heads * value_dim * key_dim * state_bytes
```

Expected float32 calculation:

```text
48 * 48 * 128 * 128 * 4 = 150,994,944 bytes = 144 MiB
```

Convolution history bytes are:

```text
n_recurrent_layers * (conv_kernel - 1) * mixed_conv_width * state_bytes
48 * 3 * 10,240 * 4 = 5,898,240 bytes = 5.625 MiB
```

Expected total recurrent state is therefore about **149.625 MiB per active sequence**, independent of context length.

Full-attention K/V bytes are:

```text
2 * n_full_attention_layers * context * kv_heads * head_dim * kv_bytes
```

Expected cost per token:

```text
float32: 2 * 16 * 4 * 256 * 4 = 131,072 bytes = 128 KiB/token
float16: 2 * 16 * 4 * 256 * 2 =  65,536 bytes =  64 KiB/token
```

| Context | Full-attention K/V F32 | Full-attention K/V F16 |
|---:|---:|---:|
| 4K | 512 MiB | 256 MiB |
| 8K | 1 GiB | 512 MiB |
| 16K | 2 GiB | 1 GiB |
| 32K | 4 GiB | 2 GiB |
| 64K | 8 GiB | 4 GiB |
| 128K | 16 GiB | 8 GiB |
| 262K | about 32 GiB | about 16 GiB |

All estimates must be recomputed from the actual header. The allocator must compare estimated and actual bytes, including alignment, workspace, JIT, logits, activations, snapshots, and driver reserve.

### 9.2 Packed-weight requirement

The existing CUDA path expands quantized blocks to approximately one byte or more per weight plus scale/minimum storage. A 27B model then requires roughly 28-33 GiB for weights alone. It cannot provide a full-GPU Q4 target on a 24 GiB card.

The production CUDA path must keep model matrices in native packed GGUF block form. A Q4_K_M file is a mixed quantization, not only Q4_K. A representative target may contain Q4_K, Q5_K, Q6_K, Q8_0, and F32 tensors. The final packed GPU path must support every **matrix** dtype in the selected file or place an entire measured compute group/layer on CPU. It may not silently expand a few large tensors and exceed the plan.

Direct packed-block storage is a release blocker for practical 27B GPU inference.

### 9.3 Planning sizes by quantization

Use the file itself for exact bytes. These are only selection ranges:

| Quantization | Approximate packed 27B size | Planning interpretation |
|---|---:|---|
| Q3 family | 12.8-13.5 GiB | More context headroom, larger quality tradeoff |
| Q4_K_M | 16-17 GiB | Recommended first production target |
| Q5_K_M | 18.5-19.5 GiB | Higher quality, constrained 24 GiB context/headroom |
| Q6_K | 21-22 GiB | Usually too tight on 24 GiB after state/workspace |
| Q8_0 | 27+ GiB | Requires larger VRAM or CPU/offload |

For a planning example with 16.2 GiB packed weights, 1.5 GiB driver/JIT/workspace reserve, and 0.15 GiB recurrent state:

| Context | Approx. total with F32 K/V | Approx. total with F16 K/V |
|---:|---:|---:|
| 4K | 18.35 GiB | 18.10 GiB |
| 8K | 18.85 GiB | 18.35 GiB |
| 16K | 19.85 GiB | 18.85 GiB |
| 32K | 21.85 GiB | 19.85 GiB |
| 64K | 25.85 GiB | 21.85 GiB |
| 128K | 33.85 GiB | 25.85 GiB |

On a nominal 24 GiB NVIDIA card, default to a measured allocation ceiling around 22.5-23.0 GiB rather than consuming every reported byte.

### 9.4 Hardware/quantization decision matrix

| Target hardware | Recommended first target | Expected limitations |
|---|---|---|
| NVIDIA GPU below 16 GiB | CPU/partial-offload development; tiny/0.8B parity | Full 27B GPU residency is not a realistic release target |
| NVIDIA 16 GiB | Q3 experimentation or CPU hybrid | Tight workspace/context; measure transfers before accepting |
| NVIDIA 24 GiB | Q4_K_M packed CUDA, F32 recurrent, then validated F16 K/V | Practical 4K-64K range; exact cap depends on file and reserve |
| NVIDIA 32 GiB | Q4/Q5 and more context | Still use planner; 128K may remain expensive |
| NVIDIA 48 GiB | Q5/Q6, potentially Q8 with offload | Better headroom; quant kernel coverage still required |
| 64+ GiB accelerator/unified memory | Q6/Q8 and longer context trials | Bandwidth and backend support remain acceptance factors |
| No NVIDIA CUDA device | Pure/NumPy/Numba CPU | Correctness supported; 27B throughput may be impractical |
| AMD/other GPU | CPU path unless a separately scoped backend is added | Numba-CUDA plan does not imply ROCm support |

System RAM should hold the mapped file, CPU-side metadata/state, checkpoints, test processes, and an external reference run. For Q4 27B development, 64 GiB RAM is recommended; 32 GiB may be too tight depending on mapping and simultaneous processes. Preserve at least 100 GiB free disk for GGUF variants and temporary test output; 200 GiB is safer during multi-quant work.

## 10. Backend and acceleration architecture

### 10.1 Explicit selection ladder

Backend selection must be visible and deterministic:

```text
Qwen35 packed CUDA
  -> pinned Numba CPU if CUDA is unavailable, over budget, unsupported, or parked
  -> NumPy if Numba is unavailable or disabled
  -> pure standard library if NumPy is unavailable or ALPACCAROO_PURE=1
```

Preserve and test existing controls:

```text
ALPACCAROO_PURE=1
ALPACCAROO_KERNELS=0
ALPACCAROO_KERNELS=force
ALPACCAROO_GPU=0
ALPACCAROO_GPU=force
ALPACCAROO_GPU_VRAM_MB=<limit>
```

Add Qwen-specific controls only when necessary, document them centrally, and print the resolved backend in focused test output. A fallback must emit one clear reason and remain queryable; it must not quietly satisfy a GPU test on CPU.

### 10.2 NumPy path

The NumPy implementation should use contiguous float32 activations and recurrent state, existing quantized matrix interfaces, and K/V arrays only for full-attention layers. Initially favor transparent intermediate arrays and named trace points over fusion.

Acceptance requirements:

- Pure and NumPy agree at every named intermediate.
- Snapshot/restore is bitwise-equivalent where operations are copied and tolerance-equivalent after recomputation.
- Recurrent state allocation does not grow with context.
- Measured allocation is within 5% of the planner, excluding explicitly reported allocator overhead.
- Chunk sizes 1, 2, 4, 16, and 64 agree within frozen tolerances.

### 10.3 Numba CPU path

Add architecture-specific kernels after NumPy parity:

```text
qwen35_conv_step
qwen35_normalize_qk
qwen35_beta_decay
qwen35_gdn_step
qwen35_recurrent_layer
qwen35_full_attention_decode
qwen35_full_attention_prefill
qwen35_state_copy_restore
```

Parallelize across independent value heads and sequences. Avoid opening a parallel region for tiny elementwise operations. Fuse recurrent state traversal where profiling shows repeated 3 MiB state walks per layer/token. Generalize attention to head dimension 256 and online softmax without materializing a complete score matrix.

Hard checks:

- No Numba object mode.
- Numba diagnostics confirm the nopython JIT path rather than Python object mode.
- Pinned supported Numba version by default.
- No OpenBLAS/Numba oversubscription.
- Faster one-token decode than NumPy on the target CPU.
- Numerically inside optimized-CPU tolerances.

### 10.4 Packed GGUF matrix layer in Python/Numba-CUDA

`packed_gpu.py` should retain:

- raw packed quantized blocks
- block geometry
- packed scales/minima/high bits as defined by each GGUF type
- original shape, dtype, offset, and byte count
- device allocation and alignment
- matvec and batched-matmul dispatch capability

Implement and validate target dtypes in manifest order, typically:

1. Q4_K packed matvec.
2. Q4_K batched GEMM/prefill.
3. Q5_K equivalents.
4. Q6_K equivalents.
5. Q8_0 equivalents.
6. F16/BF16/F32 dense fallback kernels as required by actual matrices.

Every kernel needs block-level deterministic tests against Alpaccaroo's CPU decoder and the oracle. Unpacking the entire matrix as a convenience path is allowed only in small tests, never in the accepted 27B allocation.

### 10.5 Dedicated hybrid CUDA chain

`qwen35_cuda.py` should own:

- packed weight placement by complete layer/compute group
- full-attention K/V only for full layers
- device causal-convolution history
- device DeltaNet matrices
- head-dimension-256 full attention with online softmax
- Q/K normalization, sigmoid, softplus, decay, and GDN update kernels
- recurrent gated RMSNorm and SiLU gate
- SwiGLU and residual sequencing
- device state generation/transaction tracking
- periodic pinned-host checkpoints
- restore/replay after failure
- byte/transfer/launch/synchronization counters

Do not add recurrent special cases to the conventional `DecodeChain`. Reuse well-defined generic kernels or allocators when correct, but keep state and launch sequencing separate.

Assign each recurrent value-head state matrix to one logical execution owner so the update needs no atomics. Measure register pressure, occupancy, global-memory traffic, and state traversal. Correct decomposed kernels come before fusion.

### 10.6 Placement planner

Replace greedy per-matrix GPU upload for this architecture with an explicit plan:

1. Inventory exact packed bytes and supported kernels.
2. Reserve driver/JIT/workspace and configured safety margin.
3. Reserve recurrent state, K/V, activations, logits, and checkpoints.
4. Group matrices into embeddings, each complete recurrent layer, each complete full-attention layer, each FFN group, final norm, and output projection.
5. Score groups by transfer cost and token frequency.
6. Select groups without exceeding the hard budget.
7. Print and serialize the plan before allocation.
8. Verify actual allocations and abort/fallback on unexplained overrun.

Avoid individual projection ping-pong. If a required large matrix dtype is unsupported on CUDA, prefer a complete CPU layer or CPU backend until the packed kernel exists. The final performance release should not alternate CPU/GPU within every recurrent layer.

### 10.7 CUDA transfer hard limits

Instrument host-to-device and device-to-host bytes per token/chunk. The accepted decode path must satisfy:

- No approximately 150 MiB recurrent-state copy per token.
- No whole K/V-cache copy per token.
- No repeated full-matrix quantized unpack.
- No hidden synchronization after every small operation unless measured and justified.
- Routine boundary traffic is limited to token/position inputs, small metadata, logits/sampling data, and amortized checkpoint traffic.

## 11. Target-computer intake and branch setup

### 11.1 Record the machine before choosing a target

On Windows PowerShell, capture:

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsArchitecture
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum
Get-Volume | Select-Object DriveLetter, FileSystem, SizeRemaining, Size
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
python --version
git --version
```

On Linux, capture:

```bash
uname -a
lscpu
free -h
df -h
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
python3 --version
git --version
```

Also record CUDA toolkit/driver compatibility, Python executable path, RAM speed if known, storage type, and whether memory overclocking is enabled in the implementation notes.

If `nvidia-smi` is absent, the implementation can continue through pure/NumPy/CPU phases. Do not plan a CUDA acceptance milestone until a supported NVIDIA environment is available.

### 11.2 Verify repository identity

The audited checkout's `origin` was `https://github.com/jtiro232/Alpacca.git` even though its local directory was named Alpaccaroo. The target machine must not assume directory name equals repository identity.

Run:

```powershell
git clone https://github.com/jtiro232/Alpacca.git
Set-Location Alpacca
git remote -v
git status --short --branch
git rev-parse HEAD
git log -1 --oneline
```

If the repository already exists, inspect its status first. Preserve unrelated user changes; do not reset or clean them. Use a fresh clone or worktree if the existing checkout is dirty.

### 11.3 Create or retrieve the feature branch

If this plan branch has already been published:

```powershell
git fetch origin
git switch --track origin/qwen38-python
```

If it has not:

```powershell
git fetch origin
git switch main
git pull --ff-only
git switch -c qwen38-python
```

Record the base commit in the implementation notes. If the branch is rebased, rerun the affected focused tests.

### 11.4 Establish a baseline before Qwen changes

From the checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python tests/smoke.py
```

Then create separate optional environments or reinstall extras as appropriate:

```powershell
python -m pip install -e ".[kernels]"
python tests/smoke.py
python -m pip install -e ".[gpu]"
python tests/smoke.py
```

On Linux, activate with `source .venv/bin/activate` and use `python3` if required.

Save logs, duration, Python/package versions, and the names of skipped GPU tests. A pre-existing failure must be understood and recorded before implementation; it must not be attributed to Qwen work later.

### 11.5 Model download policy

- Start with synthetic fixtures and the smallest compatible real GGUF.
- Pin Hugging Face revision/commit rather than relying on a mutable branch head.
- Download only the selected quant file, not every variant.
- Record SHA-256 immediately.
- Keep large models outside Git.
- Never commit access tokens, cache paths containing secrets, or model binaries.
- Store the expected filename/hash in a small manifest committed to the branch.

Candidate real-model ladder:

1. [Unsloth Qwen3.5-0.8B GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) or another verified architecture-compatible 0.8B artifact.
2. A verified 2B architecture-compatible GGUF if available.
3. The exact [Unsloth Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) release artifact.

Architecture compatibility must be proven from the header; a similar name is not enough.

## 12. Implementation roadmap

The phases below are ordered by dependency. Each gate is a focused stop/go check tied to code introduced in that phase. Keep commits small enough to review and bisect. Do not combine a correctness refactor, a new architecture, and a performance kernel in one commit.

### Phase 0 - Freeze the baseline and target pins

**Objective:** start from a known repository and exact model/reference versions without creating a verification subsystem.

**Work:**

1. Record repository URL, branch, base commit, Python, OS, CPU, RAM, GPU, driver, CUDA, and optional package versions in the implementation notes.
2. Run the existing pure, default/NumPy, pinned-kernel, and GPU smoke configurations that the target machine supports.
3. Record the selected model repository revision, exact GGUF filename, byte size, and SHA-256 in `docs/QWEN35_RUNTIME.md`.
4. Record the external llama.cpp version/commit used for comparisons.
5. Pin locale, UTF-8 behavior, thread counts, and deterministic sampling settings in focused model tests.

**Gate:** existing smoke tests pass, the target GGUF is uniquely identified, and baseline failures are zero.

**Suggested commit:** `docs: pin qwen target and implementation baseline`

### Phase 1 - Build the read-only GGUF identity inspector

**Objective:** understand the exact artifact before the runtime claims support or allocates large buffers.

**Work:**

1. Reuse the existing GGUF parser to emit deterministic JSON.
2. Include all architecture, recurrent, RoPE, tokenizer, and template metadata.
3. Enumerate tensors with layer index, role candidate, dimensions, dtype, offset, packed bytes, and alignment.
4. Classify each layer from explicit metadata and tensor roles.
5. Report contradictions, missing tensors, unused tensors, and unsupported auxiliary stacks.
6. Calculate exact weight bytes by dtype and logical parameter count.
7. Calculate state/KV bytes for requested context, sequence count, and dtype.
8. Calculate a conservative CUDA plan without allocating the model.
9. Hash the file in a streaming fashion and record source URL/revision supplied by the user or manifest.

**Tests:**

- Tiny known GGUF metadata round-trip.
- Malformed arrays, wrong integer types, duplicate tensor roles, offset overlap, and shape contradictions.
- Architecture name mismatch stops inspection with a precise reason.
- Schedule metadata versus tensor-role contradiction is detected.
- Planner formulas match hand-calculated small fixtures.

**Gate:** inspector produces a complete manifest for the selected 0.8B and 27B files; no tensor is silently unclassified.

**Suggested commit:** `feat: add strict qwen35 gguf manifest inspector`

### Phase 2 - Pin an external llama.cpp reference

**Objective:** have a known-good external runtime for focused token and output comparisons without adding non-Python tooling to this branch.

**Work:**

1. Pin an existing llama.cpp release/commit that demonstrably loads the exact target.
2. Use its unmodified `llama-cli`, tokenizer utility, or server as an external process.
3. Run deterministic CPU comparisons first with no flash attention and F32 K/V.
4. Compare exact token IDs, greedy continuations, and available log-probability/logit outputs on a short fixed corpus.
5. Read the pinned Qwen35/DeltaNet source for operation ordering and layout.
6. If a numerical mismatch cannot be localized with the tiny Python fixture, use a temporary local debug build outside this repository; do not commit or maintain it as Alpaccaroo tooling.
7. Record the reference version, command, model SHA, and settings in the focused test log.

**Deterministic oracle configuration for initial correctness:**

```text
CPU only
one sequence
context 1024 initially
threads 1
batch threads 1
flash attention disabled
K/V F32
temperature 0
top-k 1
top-p 1
penalties disabled
explicit BOS behavior
explicit token positions
logits/log-probabilities requested where the unmodified reference exposes them
```

**Gate:** the external reference loads the exact model and reproduces the same deterministic short run twice.

**Suggested commit:** `docs: pin qwen35 llama cpp reference settings`

### Phase 3 - Introduce architecture dispatch without changing old models

**Objective:** make the model facade delegate architecture-owned behavior while preserving standard-model behavior.

**Work:**

1. Add an `ArchitectureBackend` protocol/registry.
2. Extract only genuinely shared GGUF matrix/vector loading into `weights.py`.
3. Wrap the existing standard transformer path as the default backend.
4. Make `Model` delegate load, state creation, forward, prefill, memory estimate, and description.
5. Keep existing standard layer representation and hot loops intact initially.
6. Ensure unsupported architectures still fail before large allocation.
7. Add a Qwen35 descriptor that can validate/describe a manifest but cannot yet execute; report `execution not implemented` clearly.

**Regression tests:**

- Existing model fixtures load with identical parameter counts and descriptions except for intentionally versioned fields.
- Existing pure/NumPy outputs and K/V behavior are unchanged.
- Existing prefix-cache and server tests pass.
- Unsupported architecture and unexpected-tensor errors remain strict.

**Gate:** complete smoke suite is green before any recurrent equations are merged.

**Suggested commits:**

```text
refactor: add architecture backend registry
refactor: share strict gguf weight loading
feat: register non-executable qwen35 descriptor
```

### Phase 4 - Add generic memory with a conventional compatibility adapter

**Objective:** establish correct heterogeneous state semantics before Qwen execution mutates state.

**Work:**

1. Implement `LayerMemorySpec`, `ModelState`, `FullAttentionState`, snapshot metadata, generations, and byte accounting.
2. Wrap current K/V arrays/lists in a compatibility adapter.
3. Migrate `reset`, `truncate`, prefix save/restore, cache clear, and memory description to the generic interface.
4. Remove external direct manipulation of private `_prefix_slots` in benchmarks/tests by adding public operations.
5. Make snapshot validation and restore atomic.
6. Add byte-budgeted immutable prefix slots.
7. Keep standard attention truncation efficient.
8. Add architecture/schema/model fingerprinting to cache keys.

**Tests:**

- Existing model output and K/V contents before/after adapter match the baseline.
- Snapshot mutation cannot affect live state and vice versa.
- Wrong model/schema/shape snapshots are rejected without altering live state.
- Restore failure is atomic.
- Prefix-slot LRU bytes and eviction are correct.
- Reset semantics preserve or clear slots exactly as documented.

**Gate:** all old state, prefix, server, context trimming, and GPU invalidation tests pass; no Qwen code is needed to prove the abstraction.

**Suggested commits:**

```text
feat: add backend-neutral model memory state
refactor: route transformer kv and prefixes through memory state
test: lock snapshot ownership and atomic restore semantics
```

### Phase 5 - Implement and freeze primitive math

**Objective:** remove ambiguity from the scalar equations before assembling layers.

**Work:**

1. Implement dependency-free stable sigmoid and softplus.
2. Implement RMSNorm and L2 normalization with explicit epsilon semantics.
3. Implement SiLU, gated RMSNorm, causal depthwise convolution, Q/K head mapping, and one-token GDN update.
4. Implement sequential multi-token GDN scan.
5. Implement Qwen four-section MRoPE for text positions from the pinned reference.
6. Add pure conventional GQA capable of arbitrary head dimension.
7. Add named trace callbacks without forcing array copies when tracing is off.
8. Freeze float32 casting/accumulation points based on oracle comparison.

**Required deterministic vector classes:**

- sigmoid/softplus extreme values and zero.
- L2 vector below, at, and above epsilon threshold.
- causal convolution with nonzero history across two chunks.
- interleaved key-head mapping where contiguous grouping yields a different answer.
- orientation-lock 2x2 GDN vector from Section 6.6.
- nonzero previous-state decay/prediction/correction.
- two-token scan where token two depends on token one.
- learned RMSNorm multiplied by `SiLU(z)`.
- MRoPE positions zero, one, and a larger position across section boundaries.
- online attention against a direct softmax reference.

**Gate:** pure primitive vectors pass hand-computable expectations and the tiny model stays within the F32 tolerance used for focused external comparisons.

**Suggested commit:** `feat: add pure qwen35 recurrent and mrope primitives`

### Phase 6 - Create a deterministic tiny hybrid GGUF

**Objective:** debug architecture and state quickly without a 27B iteration cycle.

**Recommended fixture dimensions:**

```text
embedding width          256
main layers                4
layers 0-2                 recurrent
layer 3                    full attention
query heads                4
K/V heads                  2
attention head dimension  64
FFN width                 512
context                   256
state dimension            16
key groups                  2
value/time heads            4
conv kernel                 4
recurrent inner width      value_heads * state_dim
```

The primary fixture deliberately uses unequal recurrent key/value-head counts so the target's tiled/interleaved head mapping and converter ordering are exercised. An equal-head fixture may exist only as a simpler control.

**Work:**

1. Generate deterministic F32 values from a documented seed and algorithm.
2. Include every required global, recurrent, full-attention, norm, and FFN tensor.
3. Include tokenizer metadata sufficient for controlled token-ID tests, or separate tensor and tokenizer fixtures cleanly.
4. Add an optional tied/untied output variant.
5. Validate the GGUF first with the pinned llama.cpp build.
6. Emit a checked-in small manifest and expected tensor shapes.
7. Keep generated binary artifacts small enough for repository policy or generate them deterministically during tests.

**Negative fixtures:**

- recurrent tensor missing
- wrong convolution width
- incorrect gate projection width
- schedule/tensor contradiction
- second MTP layer when only one is supported
- unknown quant dtype
- vision tensors requested in text-only mode

**Gate:** llama.cpp and inspector both accept the valid fixture and reject the intentionally invalid variants for the expected reason.

**Suggested commit:** `test: generate deterministic tiny qwen35 hybrid gguf`

### Phase 7 - Establish tokenizer and chat-template parity

**Objective:** ensure both runtimes receive exactly the same tokens before interpreting model differences.

**Work:**

1. Inventory target tokenizer type, `tokenizer.ggml.pre`, vocabulary, merges, special tokens, BOS/EOS/padding IDs, and embedded template.
2. Extend Alpaccaroo only where the target contract is not already supported.
3. Freeze exact UTF-8 prompt bytes, rendered chat bytes, and expected token IDs.
4. Test parsing-special and literal-special-token modes explicitly.
5. Test template modes such as `enable_thinking=true/false` if present.
6. In model parity tests, render once and feed identical token IDs to both runtimes; do not let each runtime independently render a prompt.

**Fixed tokenizer corpus:**

```text
""
"Hello"
"Hello, world!\n"
"The quick brown fox jumps over the lazy dog."
"中文 café — naïve 😀"
"e\u0301 versus é"
"Write a Python function that reverses a linked list."
"Find the bug in this Python function and explain the fix."
"Do not modify this literal: <|im_start|>user"
```

Chat fixtures include system+user, user-only, multi-turn, empty system, empty content, code blocks, literal special tokens, and the prompt ending immediately before the assistant generation marker.

**Gate:** exact rendered-byte and token-ID equality on all fixtures. There is no tolerance for tokenizer differences.

**Suggested commit:** `feat: match qwen35 tokenizer and chat template contract`

### Phase 8 - Implement pure recurrent layers

**Objective:** execute recurrent layers transparently in the standard-library backend.

**Work:**

1. Define immutable recurrent weights and mutable `RecurrentState` separately.
2. Load and shape-check mixed QKV, gate, convolution, dt bias, A, alpha, beta, norm, output, and shared FFN tensors.
3. Preserve GGUF converter ordering and transformed `ssm_a` values exactly.
4. Implement single-token decode using `array('f')`/memoryviews and explicit loops.
5. Implement sequential prefill with state continuity across chunk calls.
6. Emit named traces at every projection, activation, split, state update, norm, residual, and FFN boundary.
7. Detect NaN/Inf with layer/token/tensor context in test mode.
8. Implement reset, deep snapshot, and restore for recurrent state.

**Gate:** one recurrent layer and three consecutive recurrent layers match the oracle at every named trace and produce identical state checksums after replay within F32 tolerances.

**Suggested commit:** `feat: execute qwen35 recurrent layers in pure python`

### Phase 9 - Implement pure full-attention Qwen layers

**Objective:** execute the target's full-attention layers rather than reusing an approximately similar Qwen path.

**Work:**

1. Load and validate Q/gate, K, V, Q/K norm, output, post-attention norm, and FFN roles.
2. Implement exact Q/gate split and reshape.
3. Apply Q/K norm and four-section MRoPE.
4. Append K/V only to the full-attention layer state.
5. Implement causal GQA and sigmoid attention gate.
6. Preserve output/residual/norm/FFN order.
7. Support head dimension 256 generically rather than as a global constant.
8. Trace pre-gate attention, gate, gated attention, output, residual, and FFN.

**Gate:** isolated full-attention layer and the recurrent-to-full boundary match oracle traces; recurrent layers allocate zero ordinary K/V rows.

**Suggested commit:** `feat: execute qwen35 full attention in pure python`

### Phase 10 - Assemble the complete pure Qwen35 backend

**Objective:** run the tiny hybrid model end to end with complete state semantics.

**Work:**

1. Dispatch each layer by the validated schedule.
2. Apply token embedding, all main layers, output norm, tied/untied output projection, and logits.
3. Implement token-by-token and chunked prefill through the same semantic path.
4. Add model-level reset, snapshot, restore, truncate/replay, branch, prefix-slot switch, and independent state instances.
5. Report MTP/vision status explicitly.
6. Integrate with sampling, chat, CLI, and serialized server calls without optimizing.
7. Ensure tracing off has minimal semantic impact and tracing on is bounded/configurable.

**Gate:** tiny F32 model passes intermediate/logit parity, exact greedy sequence, chunk equivalence, reset, snapshot/restore, rollback, A->B->A branch, and two independent sequences.

**Suggested commits:**

```text
feat: assemble pure qwen35 hybrid backend
feat: add hybrid snapshot rollback and replay
test: prove qwen35 branch and chunk state equivalence
```

### Phase 11 - Lock the pure reference behavior

**Objective:** freeze the correctness reference before vectorization changes reduction order.

**Work:**

1. Keep focused tests for the primitive equations, one recurrent layer, one full-attention layer, and the tiny end-to-end model.
2. Compare a small set of named intermediates only where they localize likely layout/order errors.
3. Compare short greedy output and available logits/log-probabilities with the external reference.
4. Test empty prompt behavior, explicit BOS/EOS, and unconsumed target tensors.

**Gate:** the focused Python tests pass and become the reference for NumPy/Numba/CUDA.

**Suggested commit:** `test: lock pure qwen35 reference behavior`

### Phase 12 - Implement the NumPy backend

**Objective:** create a practical correctness backend without changing architecture semantics.

**Work:**

1. Implement NumPy forms of every Qwen primitive and layer.
2. Keep recurrent state float32 and contiguous.
3. Use vectorized batch/head dimensions with a sequential token scan.
4. Allocate K/V only for full-attention layers.
5. Add snapshot/restore and prefix slots with explicit deep copies.
6. Reuse quantized matrix APIs without permanently unpacking entire models.
7. Compare named traces to pure and oracle.
8. Add measured allocation reporting.

**Gate:** pure/NumPy/oracle parity, state semantics, and all old-model NumPy regressions pass. NumPy is measurably faster than pure on the tiny model.

**Suggested commit:** `feat: add numpy qwen35 hybrid backend`

### Phase 13 - Prove synthetic quantization coverage

**Objective:** distinguish architecture errors from quantized-matmul differences and avoid advertising phantom dtype support.

**Work:**

1. Generate tiny fixture variants for every dtype the backend claims.
2. At minimum cover the selected 27B artifact's matrix types, expected to include Q4_K, Q5_K, Q6_K, Q8_0, and float types.
3. Test block decode, row dot, matvec, batched matmul, complete layer, and final logits.
4. Record top-logit margin so a near-tie is visible.
5. Treat Q8_K/IQ or other unimplemented types as unsupported even if a registry name exists elsewhere.
6. Freeze empirical tolerances by dtype after comparing the same accumulation strategy.

**Starting tolerance ceilings for investigation, not automatic waivers:**

| Path/type | Absolute | Relative |
|---|---:|---:|
| F32 primitive/intermediate | `2e-5` | `2e-4` |
| F32 final logits | `5e-5` | `5e-4` |
| Q8_0 final logits | `5e-3` | `5e-4` |
| Q5_K/Q6_K final logits | `1e-2` | `1e-3` |
| Q4_K final logits | `2e-2` | `2e-3` |

The test should compare to an oracle running the **same quantized file**. Greedy equality is required on the bounded corpus unless a documented top-logit near-tie is reviewed and a stronger distribution comparison is added.

**Gate:** every advertised dtype has passing tests; unknown/unsupported dtypes fail before execution.

**Suggested commit:** `test: cover qwen35 target quantization formats`

### Phase 14 - Validate compatible 0.8B and 2B real models

**Objective:** prove that the implementation generalizes beyond hand-selected tiny dimensions before debugging 27B.

**Work:**

1. Pin and hash the 0.8B artifact.
2. Run tokenizer/chat, short prompt logits, code prompts, Unicode, chunk sizes, reset, save/restore, and branching.
3. Test contexts 1, 2, 8, 32, and 128 tokens and at least 512 generated tokens for recurrent stability when practical.
4. Repeat on a verified 2B artifact, adding multi-turn and independent-sequence cases.
5. Record peak RSS, mapped bytes, state bytes, throughput, and fallback status.
6. Fix architecture assumptions exposed by different dimensions; do not special-case the tiny or 27B shapes.

**Gate:** exact tokenizer parity, frozen logit tolerances, exact greedy corpus, no NaN/Inf, chunk equivalence, and state replay pass. If a compatible 2B artifact is unavailable, record that fact and expand 0.8B dimensional/quant fixture coverage before proceeding.

**Suggested commits:**

```text
test: validate qwen35 0.8b oracle parity
test: validate qwen35 2b oracle parity
```

### Phase 15 - Add pinned Numba CPU acceleration

**Objective:** create a usable CPU fallback and optimized correctness path.

**Work:**

1. Compile the primitive kernels listed in Section 10.3 in nopython mode.
2. Fuse convolution/split/normalization and GDN traversals only after isolated parity.
3. Implement online full attention for head dimension 256.
4. Add tuned but bounded parallelism across value heads.
5. Add chunked prefill kernels after token-scan parity.
6. Reuse existing pinned Numba policy and diagnostics.
7. Confirm nopython signatures and run the focused benchmark.

**Gate:** optimized CPU error stays within `atol=1e-4`, `rtol=1e-3` for intermediates unless a stricter frozen dtype-specific limit applies; it is faster than NumPy; no object mode or oversubscription; old kernel tests remain green.

**Suggested commits:**

```text
feat: accelerate qwen35 recurrence with pinned numba
feat: add numba qwen35 online full attention
perf: tune qwen35 cpu prefill without changing semantics
```

### Phase 16 - Implement packed GPU matrices in Python/Numba-CUDA

**Objective:** make the 27B Q4 memory target physically possible.

**Work:**

1. Add the Python packed-device-matrix abstraction and exact byte accounting.
2. Upload native GGUF blocks without universal int8 expansion.
3. Implement Q4_K matvec and tiled batched GEMM, then target-manifest Q5_K, Q6_K, Q8_0, and dense types.
4. Test odd row counts, alignment boundaries, block tails allowed by GGUF, and large dimensions.
5. Compare every kernel to CPU quantized row-dot and oracle layer outputs.
6. Report unsupported matrix types before allocation planning.
7. Prove packed allocations match file tensor bytes plus documented alignment/metadata overhead.

**Gate:** all large matrices in the chosen Q4_K_M target have an accepted packed GPU path or an explicit whole-layer placement; the complete planned weight allocation fits the target budget without expanded copies.

**Suggested commits:**

```text
feat: retain native gguf blocks in gpu matrices
feat: add packed q4_k cuda matvec and gemm
feat: cover target q5_k q6_k and q8_0 gpu matrices
```

### Phase 17 - Implement Qwen hybrid Numba-CUDA kernels

**Objective:** execute both layer types and mutable state entirely on device during active decoding.

**Work:**

1. Implement causal convolution update with device history.
2. Implement Q/K L2 normalization and exact interleaved head mapping.
3. Implement stable beta sigmoid, softplus time step, decay, GDN update, and state read.
4. Implement learned recurrent RMSNorm times `SiLU(z)`.
5. Implement full attention with arbitrary supported head dimension, including 256, online softmax, gate, and K/V append.
6. Integrate packed projections, residuals, norms, SwiGLU, and output.
7. Start decomposed; add fusion only after the focused tiny-model checks still pass.
8. Measure kernel time, launch count, and temporary allocation during optimization; do not build a general profiling framework.

**Gate:** tiny, 0.8B, and selected 27B-layer traces meet initial CUDA `atol=1e-3`, `rtol=2e-3` ceilings, preserve top-k on fixed prompts, and maintain correct state across at least 512 tokens on a smaller model.

**Suggested commits:**

```text
feat: add qwen35 cuda recurrent state kernels
feat: add qwen35 cuda 256-wide full attention
feat: compose qwen35 cuda layer execution
```

### Phase 18 - Add the transactional hybrid CUDA chain

**Objective:** manage placement, state lifetime, checkpoints, failures, and fallback correctly.

**Work:**

1. Build `Qwen35CudaChain` separately from conventional `DecodeChain`.
2. Allocate only planned packed weights and layer-specific state.
3. Track host checkpoint generation, device generation, current token position, and dirty transaction.
4. Keep device state resident between tokens.
5. Checkpoint to pinned host memory at the configured interval and on explicit snapshot.
6. On a simulated mid-token failure, discard uncertain device state, restore the latest host checkpoint, replay, and compare to a clean CPU continuation.
7. Park a failed CUDA chain so repeated calls do not repeatedly corrupt/retry.
8. Expose planned/actual VRAM, backend, checkpoint interval, replay count, transfer bytes, and fallback reason.
9. Focus failure tests on unsupported dtype, initialization/allocation failure, one mid-token failure, snapshot/rollback, and branch switching.

**Gate:** no large state/KV transfer per token; all fault-injection continuations equal clean execution; peak VRAM stays under configured limit; fallback is explicit and state-correct.

**Suggested commits:**

```text
feat: add transactional qwen35 cuda chain
feat: checkpoint and replay hybrid device state
test: prove qwen35 cuda failure rollback and fallback
```

### Phase 19 - Final 27B bounded correctness gate

**Objective:** prove the exact release artifact before optimizing or raising context claims.

**Initial conservative CPU/oracle configuration:**

```text
context             1024
sequences              1
threads                1
batch threads          1
flash attention      off
K/V                   F32
sampling            greedy
penalties             off
prefill chunks       1, 2, 4, 16, 64, 256
```

**Work:**

1. Compare exact token IDs and prompt bytes.
2. Compare selected logits/log-probabilities when exposed, greedy tokens, and recurrent checksums at focused checkpoints.
3. Run one-token decode, several short prefills, recurrent/full boundary prompts, reset, snapshot/restore, rollback, and branch tests.
4. Record runtime and peak host memory; a slow correctness run is acceptable here.
5. Repeat bounded tests on packed CUDA with explicit confirmation that no CPU fallback occurred.
6. Run a fixed coding corpus and compare deterministic tokens to the oracle.

**Gate:** all correctness/state criteria pass for the exact SHA-256 artifact. No unexplained fallback, skipped layer, unknown tensor, NaN/Inf, or silent context reduction.

**Suggested commit:** `test: validate exact qwen38 27b bounded parity`

### Phase 20 - Performance, memory, and context qualification

**Objective:** turn a correct implementation into a practical one and make context claims measured rather than assumed.

**Benchmark matrix:**

```text
backends:       NumPy, Numba CPU, packed CUDA, pinned llama.cpp reference
contexts:       1, 2, 4, 8, 16, 32K; add 64K only when memory plan allows
prefill chunks: 1, 8, 32, 64, 128, 256
sequences:      1 first; 2 only after state isolation
metrics:        load time, prompt tok/s, decode tok/s, first-token latency,
                peak RAM/VRAM, state/KV/weight/workspace bytes, transfer bytes,
                launches/token, synchronizations/token, replay/checkpoint cost
```

**Targets:**

- First accepted CUDA milestone: at least 50% of pinned llama.cpp throughput on the same model/settings/hardware.
- Optimization milestone: approach 80% or better where Numba-CUDA permits.
- Peak VRAM on a 24 GiB target: at or below the configured 22.5-23.0 GiB ceiling.
- No performance claim if the run silently falls back or spends most time transferring/unpacking.
- Float16 K/V may be enabled only after short/long parity, rollback, and stability tests.

**Optimization order:**

1. Eliminate repeated unpack/copy/synchronization.
2. Improve packed projection kernels.
3. Fuse recurrent state passes with trace-verified equivalence.
4. Tune online attention and K/V layout.
5. Tune chunked prefill.
6. Tune output projection placement.
7. Evaluate graph capture or persistent-kernel techniques only if supported reliably.

**Gate:** publish a measured hardware-specific configuration table. Do not equate metadata maximum context with supported practical context.

**Suggested commits:**

```text
perf: remove qwen35 cuda transfer and unpack bottlenecks
perf: tune qwen35 recurrent and attention kernels
docs: publish measured qwen38 hardware and context matrix
```

### Phase 21 - Product integration, CI, and documentation

**Objective:** make support understandable and keep it from regressing.

**Work:**

1. Update CLI/model descriptions with architecture, layer counts, state bytes, K/V bytes, requested context, selected backend, placement, actual VRAM, checkpoint policy, and unsupported features.
2. Add precise load errors for architecture identity, shape, dtype, MTP/vision requests, memory plan, and missing kernels.
3. Make chat context fitting invoke valid checkpoint/replay semantics when dropping messages.
4. Keep server serialized initially and state its limitations.
5. Add optional environment variables for real model paths; never require large downloads in offline CI.
6. Run synthetic pure and NumPy Qwen tests in default CI.
7. Run Numba tests where the pinned extra is installed.
8. Run focused CUDA tests on a supported GPU runner, including one recovery test and memory accounting.
9. Run manual 0.8B and 27B checks with hash verification before release.
10. Document model selection, download, inspect-only mode, expected memory, backend controls, context limits, and troubleshooting.
11. Add a public prefix-cache clear method and avoid tests/tools reaching into private slots.

**Gate:** offline CI is green; optional real/GPU jobs report explicit pass/skip reasons; user documentation agrees with measured results.

**Suggested commits:**

```text
feat: expose qwen35 runtime and memory diagnostics
test: add qwen35 ci and optional real-model gates
docs: document qwen38 text runtime and limitations
```

### Phase 22 - Final branch check

**Objective:** confirm the implementation works without creating a separate release-verification product.

**Work:**

1. Run the existing smoke suite.
2. Run the focused Qwen tiny, state, quantization, 0.8B, and exact 27B tests on each claimed backend.
3. Confirm the practical packed-CUDA configuration fits and generates without fallback.
4. Audit unused target tensors, warnings, and documented deferred features.
5. Update user documentation with the measured configuration and limitations.

**Gate:** Section 4 is satisfied and the branch's claimed backends all pass their focused tests.

**Suggested commit:** `chore: finish qwen38 runtime branch`

## 13. Critical-path and parallel work guidance

The dependency-critical sequence is:

```text
identity/manifest
  -> pinned oracle
  -> architecture dispatch
  -> generic state
  -> primitive math
  -> tiny fixture
  -> pure recurrent + full attention
  -> pure parity
  -> NumPy and target quant parity
  -> small real models
  -> Numba CPU
  -> packed GPU matrices
  -> hybrid CUDA chain
  -> exact 27B correctness
  -> performance/context qualification
  -> release closure
```

After the tiny fixture schema is frozen, these side streams can proceed in parallel with disjoint ownership:

- Tokenizer/chat fixtures.
- Focused external-reference comparisons.
- Generic state tests.
- Packed quant block-kernel prototypes.
- User documentation.
- Existing-model regression expansion.

Do not parallelize different implementations of the same unsettled equation. Freeze the pure contract first, then let optimized backend owners implement against it.

## 14. Focused correctness checks

### 14.1 Oracle hierarchy

Use three small, practical references:

1. **Hand-computable primitive vectors** for orientation, indexing, epsilon, and operation-order mistakes.
2. **The tiny Python-generated hybrid GGUF** for layer/state/backend equivalence.
3. **An unmodified pinned llama.cpp executable** for tokenizer, deterministic output, available logits/log-probabilities, and performance comparisons.

Once pure Python agrees with these references, optimized Python backends compare to the pure implementation. No separate trace or evidence platform is required.

### 14.2 Debug checkpoints

Keep a lightweight optional Python callback capable of capturing only these useful boundaries when a test fails:

```text
layer.N.attn_norm
layer.N.linear_attn_qkv_mixed
layer.N.conv_output_raw
layer.N.conv_output_silu
layer.N.state_before
layer.N.new_state
layer.N.attn_output
layer.N.Qcur_normed
layer.N.Kcur_normed
layer.N.attn_pregate
layer.N.attn_gated
layer.N.ffn_out
model.result_output
```

The callback is off by default, stores nothing permanently, and is not a public tracing system. Tests compare arrays in memory and print the first failing stage.

### 14.3 Error comparison

Focused array comparisons need max absolute/relative error, the worst index/values, and a finite-value check. Output comparisons need exact token IDs plus top-logit margin where logits are available. If recurrent error grows with token count, compare state before/after the update and verify float32 casting and operation order.

### 14.4 Acceptance tolerance policy

Initial ceilings:

| Comparison | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Hand/pure F32 primitive | `1e-6` where arithmetic order is identical | `1e-6` |
| Pure/NumPy vs decomposed F32 oracle | `2e-5` | `2e-4` |
| F32 final logits | `5e-5` | `5e-4` |
| Optimized Numba CPU intermediates | `1e-4` | `1e-3` |
| Initial CUDA intermediates | `1e-3` | `2e-3` |

Quantized final-logit starting ceilings are in Phase 13. Freeze actual limits from representative fixtures. Never relax a tolerance only to make a failure green; require a written numerical explanation, long-state stability evidence, and token/ranking evidence.

### 14.5 State test matrix

Every backend must pass the applicable rows:

| Scenario | Required assertion |
|---|---|
| Fresh run twice | identical logits and state |
| Reset then prompt B | equals a fresh model running B |
| Snapshot/restore | continuation repeats exactly in greedy mode |
| A -> B -> A branch | both A runs have matching logits and state |
| Truncate to checkpoint | equals fresh replay to target |
| Truncate between checkpoints | restore+replay equals fresh replay |
| Chunk sizes 1/2/4/16/64/full | logits/state within frozen tolerance |
| Full layer state | K/V grows exactly once per token |
| Recurrent layer state | no conventional K/V allocated; conv/delta update once per token |
| Two isolated states | interleaving does not cross-contaminate |
| Prefix slot eviction | all state bytes released; no alias remains |
| Wrong snapshot fingerprint | restore rejected atomically |
| CPU after GPU checkpoint | continuation agrees |
| CUDA failure mid-layer | checkpoint+replay agrees with clean run |
| CUDA failure during checkpoint | previous checkpoint remains valid |
| Context trim in chat | rebuilt state agrees with freshly rendered trimmed prompt |

### 14.6 Fixed generation corpus

Use deterministic prompts that exercise both ordinary text and the requested coding capability:

```text
P00: "Hello"
P01: multilingual and combining Unicode corpus
P02: explain why the sky appears blue in two sentences
P03: write a pure Python function to reverse a singly linked list
P04: find and fix an off-by-one error in a supplied loop
P05: implement binary search and state its invariant
P06: refactor a small state machine without changing behavior
P07: emit JSON matching a supplied schema
P08: multi-turn request to revise earlier code
P09: prompt containing literal Qwen special-token text
P10: long repeated structure crossing several full-attention intervals
```

Check token IDs and logits before evaluating code quality. Coding evaluation can be added after parity and should use deterministic unit tests in an isolated temporary directory; it is not an architecture oracle.

### 14.7 Regression matrix for existing Alpaccaroo models

At minimum retain coverage for:

- Llama.
- Mistral.
- Qwen2.
- Qwen3 conventional attention.
- StableLM.
- Gemma and Gemma3.
- Pure, NumPy, pinned Numba, and optional CUDA tiers.
- Context reset and trimming.
- Prefix cache and slot switching.
- Server serialization and chat rendering.
- Existing quantized tiny fixtures.
- GPU budget cap, failure fallback, and chain invalidation.

A Qwen failure may not be fixed by weakening common shape/tensor validation or changing standard-model cache semantics without old-model evidence.

## 15. Failure, fallback, and rollback policy

### 15.1 Loader failures

Stop before model allocation for:

- unsupported `general.architecture`
- target identity mismatch
- missing required metadata
- schedule/tensor contradiction
- missing/duplicate tensor roles
- shape mismatch
- overlapping or invalid GGUF offsets
- unsupported required quant dtype
- unsupported requested MTP/vision feature

Messages should identify the key/tensor, expected contract, actual value, and supported alternative.

### 15.2 Memory-plan failures

Before GPU allocation:

- compare required bytes to configured budget and safety reserve
- report largest groups and context/KV contribution
- suggest a smaller context, F16 K/V only if validated, lower quant, CPU backend, or larger GPU
- do not start a partial greedy upload that is guaranteed to fail

If actual allocation exceeds the plan beyond documented alignment/workspace tolerance, release allocations, report the discrepancy, and fall back safely. Do not progressively consume all VRAM with repeated attempts.

### 15.3 Unsupported CUDA operations

An unsupported matrix type or layer kernel selects a precomputed whole-layer/whole-backend fallback. It must not reinterpret bytes, switch dtype silently, or alternate individual projections across PCIe every token.

### 15.4 Runtime CUDA failures

On initialization failure, leave host state untouched and select CPU before generation. On a launch/synchronization error after state mutation may have begun:

1. Mark the device generation uncertain and park the chain.
2. Retain the last complete host checkpoint.
3. Recreate a clean CPU or device state from that checkpoint.
4. Replay recorded tokens after the checkpoint.
5. Continue only after replay state matches the expected generation.
6. Emit one structured fallback record.

Do not copy uncertain recurrent matrices back or continue by decrementing a position counter.

### 15.5 Numerical failures

NaN/Inf in test or validation mode must stop with architecture, backend, layer, token, named stage, and state generation. Production fallback from a numerical error is allowed only if the last checkpoint is finite and replay on a safer backend is validated. Repeated numerical failure becomes a hard error.

### 15.6 User cancellation and partial prefill

Define cancellation boundaries. The initial design should commit state only after a complete token or chunk transaction. If cancellation arrives mid-kernel/chunk, abort to the last committed generation. This makes server cancellation and timeout behavior deterministic.

## 16. Risk register

| Risk | Impact | Detection | Mitigation/exit condition |
|---|---|---|---|
| Qwen3.8 label does not match `qwen35` schema | Wrong implementation selected | Gate 0 manifest | Stop; add a new descriptor only from authoritative source |
| GGUF converter already transformed/reordered recurrent tensors | Double transform causes plausible but wrong state | primitive/layer trace | Consume GGUF layout once; fixture transformed values explicitly |
| MRoPE sections interpreted like ordinary RoPE | Full-attention drift | named Q/K trace at several positions | Copy pinned position/section contract and freeze vectors |
| Gate projection split wrong | Plausible output with wrong logits | shapes and pre/post-gate traces | Validate exact projection width and split axes |
| Delta state orientation or head mapping wrong | Recurrent corruption | 2x2 hand vector and interleaved-head fixture | Freeze `[value,key]` and `% key_heads` contract |
| Softplus/decay/reduction precision differs | Long-context drift | state error curve by token | F32 state, stable math, cast points from oracle |
| Recurrent snapshot omits conv or delta state | Branch/rollback corruption | A->B->A and rollback tests | Complete immutable snapshot schema |
| GPU copies recurrent state each token | Severe slowdown | transfer counters | Device-authoritative active state + periodic checkpoints |
| GPU expands Q4 weights | 27B cannot fit 24 GiB | planned/actual allocation | Native packed blocks; no expanded release path |
| Q4_K_M contains unsupported mixed dtypes | Partial offload/ping-pong | manifest dtype census | Implement all target matrix dtypes or whole-layer fallback |
| Existing attention assumes head dim 128 | Wrong/crashing full layers | dimension-generic fixtures | Dedicated 256-capable online attention |
| Numba-CUDA cannot achieve needed kernel behavior | Performance/reliability limit | focused benchmark and kernel check | Report the Python-only blocker; do not introduce a native extension |
| Prefix snapshots consume excessive RAM | OOM with many conversations | per-slot byte accounting | Small byte-budgeted cache, checkpoint+replay, explicit limits |
| Float16 recurrent state is unstable | Long-term quality corruption | 512+ token state parity | Keep F32 until an independent release gate passes |
| 262K metadata context is treated as practical support | OOM/misleading claim | planner + measured matrix | Publish hardware-specific tested caps only |
| MTP/vision tensors silently ignored | False capability claim | unused tensor/feature audit | Explicit inventory and fail/disabled report |
| Old models regress under generic state refactor | Broad product regression | baseline and complete smoke suite | Compatibility adapter first, bisection-friendly commits |
| External reference has a backend-specific bug | Bad comparison target | repeat on CPU and compare hand vectors/source contract | Do not make one external output the sole math specification |
| Tokenizer/template mismatch blamed on model math | Wasted debugging | Gate 1 exact IDs/bytes | Freeze token fixtures before logit comparisons |
| Dirty target checkout contaminates branch | Lost/unrelated changes | `git status` intake | Fresh clone/worktree; never reset user work |

The largest feasibility risk is the packed Numba-CUDA implementation. If it cannot meet correctness or practical performance on the target computer, the original 27B goal remains incomplete and the blocker must be reported. Do not solve that blocker by adding C, C++, Rust, Cython, a custom compiled extension, or a substituted external runtime; changing the Python-only constraint requires a separate user decision.

## 17. Completion checklist

### Artifact and external reference

- [ ] Exact target URL and pinned revision recorded.
- [ ] Exact filename, bytes, and SHA-256 recorded.
- [ ] GGUF architecture, metadata, tensor, dtype, tokenizer, and template manifests complete.
- [ ] External llama.cpp version/commit and commands recorded.
- [ ] Short deterministic reference outputs reproduced.

### Architecture and loader

- [ ] Architecture dispatch preserves all standard backends.
- [ ] Layer schedule is metadata-driven and tensor-cross-checked.
- [ ] Every required recurrent and full-attention tensor is shape-checked.
- [ ] Converted recurrent tensors are not transformed twice.
- [ ] Unknown tensors and unsupported auxiliary stacks are explicit.

### Pure correctness

- [ ] Stable sigmoid, softplus, L2, RMSNorm, SiLU, convolution, MRoPE, GQA, and GDN vectors pass.
- [ ] Orientation and interleaved-head tests pass.
- [ ] Tiny F32 hybrid fixture passes focused layer/state/output checks.
- [ ] Pure standard-library mode has no optional import requirement.
- [ ] Token/chunk/state semantics pass.

### Tokenizer and chat

- [ ] Exact prompt bytes and token IDs pass fixed corpus.
- [ ] BOS/EOS/padding and special parsing are exact.
- [ ] Chat roles, multi-turn, Unicode, special literals, and thinking modes are frozen.

### State and prefix behavior

- [ ] K/V exists only for full layers.
- [ ] Convolution and DeltaNet state exist only for recurrent layers.
- [ ] Reset equals fresh execution.
- [ ] Snapshot/restore and A->B->A pass.
- [ ] Rollback via nearest checkpoint+replay passes.
- [ ] Snapshot validation/restore is atomic.
- [ ] Prefix byte budgeting/eviction releases all buffers.
- [ ] Two independent sequences do not share state.

### Optimized CPU and quantization

- [ ] NumPy matches pure/oracle.
- [ ] Every advertised target dtype has block, matmul, layer, and logit tests.
- [ ] 0.8B real-model gate passes.
- [ ] Optional 2B check passes when a compatible artifact is readily available.
- [ ] Numba is nopython, faster than NumPy, and within tolerance.

### GPU and practical 27B

- [ ] Target matrix dtypes remain native packed on GPU.
- [ ] No full-model expanded quantized copy exists.
- [ ] Dedicated hybrid CUDA chain is used.
- [ ] Head-dimension-256 full attention passes.
- [ ] Device state remains resident between checkpoints.
- [ ] No full recurrent/KV copy per token.
- [ ] One simulated device failure restores/replays correctly.
- [ ] Planner and actual VRAM agree within documented overhead.
- [ ] Peak stays under configured safety ceiling.
- [ ] Exact 27B bounded parity passes on CPU and packed CUDA.
- [ ] Fixed coding corpus passes deterministic oracle comparison.

### Regression, product, and release

- [ ] Existing smoke suite passes in clean environment.
- [ ] Existing architecture/cache/server/GPU tests pass.
- [ ] CLI reports architecture, backend, memory, context, fallbacks, and deferred features.
- [ ] Offline CI requires no large downloads.
- [ ] Optional real/GPU jobs use pinned hashes and explicit skip reasons.
- [ ] Performance/context table is measured on named hardware.
- [ ] No required focused test is silently skipped.

## 18. How this architecture helps future models

This work improves Alpaccaroo beyond Qwen3.8 when the abstractions remain narrow and measured:

- **Architecture registry:** future models can own tensor roles and forward/state logic without destabilizing the conventional transformer loop.
- **Heterogeneous memory:** future recurrent, state-space, sliding-window, hybrid-attention, or expert-routing models can describe memory per layer instead of pretending everything is K/V.
- **Snapshots and replay:** Alpaccaroo's state-machine use case gains deterministic branch/rollback semantics for all architectures.
- **Manifest-first loading:** exact shapes, dtypes, tensor consumption, and memory are known before expensive allocation.
- **Packed GPU matrices:** every future quantized model can avoid the memory penalty of expanded GPU weights.
- **Placement planner:** hardware selection becomes based on total weights+state+K/V+workspace rather than model file size alone.
- **Small debug checkpoints:** pure, NumPy, Numba, and CUDA can be compared at the few boundaries that localize layout errors.
- **Transactional acceleration:** device failure can recover from a complete state generation instead of leaving hidden corruption.
- **Repeatable model bring-up:** adding a model follows the same small-fixture-to-real-model path instead of relying on a one-off text demo.

These benefits do not mean one generic kernel should run every architecture. Common contracts should improve planning, state management, testing, and dispatch; specialized hot loops should remain specialized. Existing models should pay no recurring branch or state cost for features they do not use.

## 19. Fresh-machine execution runbook

This is the concise operational path after the branch is available. Adjust only paths and pinned revisions; do not skip gates.

### 19.1 Clone and branch

```powershell
git clone https://github.com/jtiro232/Alpacca.git
Set-Location Alpacca
git fetch origin
git switch --track origin/qwen38-python
git status --short --branch
git rev-parse HEAD
```

### 19.2 Create environments and run baseline

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
$env:ALPACCAROO_GPU = '0'
python tests/smoke.py
```

Then install/test the pinned optional tiers in sequence (use separate virtual environments only when diagnosing dependency conflicts):

```powershell
python -m pip install -e ".[kernels]"
python tests/smoke.py
python -m pip install -e ".[gpu]"
$env:ALPACCAROO_GPU = '1'
python tests/smoke.py
```

Use a clean shell between environment-control matrices when module import caches could retain a backend decision.

### 19.3 Prepare the external reference

Install or copy the pinned, unmodified llama.cpp release executable outside the repository. Record its version and path, then prove it loads the selected GGUF:

```powershell
$LlamaCli = 'C:\Tools\llama.cpp\llama-cli.exe'
$TargetGguf = 'D:\Models\Qwen3.8-27B-Q4_K_M.gguf'
& $LlamaCli --version
& $LlamaCli -m $TargetGguf -p 'Hello' -n 8 --temp 0 --top-k 1
```

Do not build or commit an oracle helper. On Linux, set `$LlamaCli`/`$TargetGguf` equivalents in the shell and invoke the prebuilt `llama-cli` binary.

### 19.4 Inspect and hash models before load

The finished branch should expose a command equivalent to:

```powershell
$TargetGguf = 'D:\Models\Qwen3.8-27B-Q4_K_M.gguf'
python -m alpaccaroo inspect $TargetGguf --architecture-contract --context 4096
Get-FileHash -Algorithm SHA256 -LiteralPath $TargetGguf
```

If the exact CLI spelling changes during implementation, update this document and CI together. The required behavior is strict, allocation-free manifest and memory-plan output.

### 19.5 Run focused checks from small to large

The finished test tooling should provide focused commands equivalent to:

```powershell
$QwenSmallGguf = 'D:\Models\Qwen3.5-0.8B-Q4_K_M.gguf'
$TargetGguf = 'D:\Models\Qwen3.8-27B-Q4_K_M.gguf'
python tests/make_qwen35_model.py
python tests/qwen35_test.py --model tiny --backend pure
python tests/qwen35_test.py --model tiny --backend numpy
python tests/qwen35_test.py --model $QwenSmallGguf --backend numpy
python tests/qwen35_test.py --model $TargetGguf --backend numba --bounded
python tests/qwen35_test.py --model $TargetGguf --backend cuda --bounded --require-backend cuda
```

`--require-backend` or an equivalent assertion is essential so a CUDA acceptance job cannot pass after CPU fallback.

### 19.6 Run practical memory/performance qualification

Before generation, inspect the planned allocation at each context. Start with 4K, then 8K, 16K, 32K, and only then 64K. Monitor:

```powershell
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv -l 1
```

Run Alpaccaroo and llama.cpp sequentially, not simultaneously, for comparable peak VRAM. Use identical model hash, prompt token IDs, context, K/V dtype, batch, and sampling. Record warm-up separately from steady state.

### 19.7 Final branch commands

```powershell
git status --short
python tests/smoke.py
python tests/real_model_test.py
python tests/acceptance.py
git diff --check
git log --oneline --decorate origin/main..HEAD
```

The existing real-model and acceptance scripts remain regression checks, not substitutes for `qwen35_test.py`. A required target model/backend that is unavailable does not count as a pass.

## 20. Source references

Primary external references to pin/recheck during implementation:

- [Unsloth Qwen3.8-27B GGUF repository](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [Qwen hybrid Gated DeltaNet architecture overview](https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d)
- [llama.cpp Qwen35 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)
- [llama.cpp DeltaNet base implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/delta-net-base.cpp)
- [llama.cpp recurrent memory implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory-recurrent.cpp)
- [llama.cpp public API](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)
- [llama.cpp recurrent state rollback test](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-recurrent-state-rollback.cpp)
- [Unsloth Qwen3.5-0.8B GGUF repository](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)

Use permalinks to the pinned llama.cpp commit in the implementation documentation; the moving `master` links above are navigation aids only.

## 21. Final implementation recommendation

Build this branch as an architecture and state-system upgrade whose first client is Qwen3.8-27B:

1. Prove artifact identity and tokenizer first.
2. Establish a pinned external llama.cpp reference and hand-computable Python vectors.
3. Add architecture dispatch and heterogeneous state without changing old-model behavior.
4. Implement exact pure recurrent and full-attention paths.
5. Freeze parity on a tiny model and then 0.8B/2B.
6. Add NumPy and pinned Numba without changing the contract.
7. Keep native packed GGUF weights on CUDA; otherwise the practical 24 GiB target is impossible.
8. Use a dedicated hybrid device chain with resident state, checkpoints, and replay.
9. Prove the exact 27B SHA at bounded context before performance work.
10. Publish only measured hardware/context claims and preserve explicit fallbacks.

This route preserves Alpaccaroo's Python-pure identity while adding the architecture that modern hybrid models require. It also produces reusable architecture dispatch, state, packed-weight, planning, tracing, and verification foundations for future models without forcing the existing conventional transformer path to become slower or more fragile.
