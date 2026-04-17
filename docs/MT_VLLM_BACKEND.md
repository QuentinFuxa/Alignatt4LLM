# Gemma MT through vLLM — design notes (PLAN Phases 0–5)

Status: **shipped**, validated end-to-end on one clip (Phase 5). Quality/latency numbers in [`RESULTS.md`](RESULTS.md).

## What this backend is

`mt_backend_name = "gemma_vllm_alignatt"` is the active MT path: a vLLM engine plus an engine-native MT AlignAtt observer.

| Surface | File |
|---|---|
| MT backend | `gemma_vllm_mt_backend.py` — `VLLMAlignAttGemmaMTBackend(BaseMTBackend)` |
| MT worker | `gemma_vllm_mt_worker.py` — `GemmaMTAlignAttWorker(VLLMGPUWorker)` |
| MT observer | `gemma_vllm_mt_observer.py` — `_MTPromptDecodeQKTensorObserver` + reconstruction |
| MT dispatcher | `cascade_mt_backend.build_mt_backend()` — Phase 0 |
| Probe harness | `run_mt_backend_parity.py` (historical filename, vLLM-only) |

## Dispatcher and runtime surface (Phase 0)

ASR remains a runtime axis; MT no longer does. The shipped constants in `cascade_runtime.py` are:

```python
VALID_ALIGNMENT_BACKEND_NAMES = ("qwen_forced", "gemma_onepass_qk_fast", "gemma_vllm_qk_fast")
STABLE_ALIGNMENT_BACKEND_NAMES = ("qwen_forced", "gemma_onepass_qk_fast")

VALID_MT_BACKEND_NAMES = ("gemma_vllm_alignatt",)
STABLE_MT_BACKEND_NAMES = ("gemma_vllm_alignatt",)
```

`CascadeRuntimeConfig.mt_backend_name` defaults to the vLLM MT path. `LoadedModelBundle.ensure_mt_backend()` still fingerprints the MT engine so runtime knob changes rebuild safely when needed.

## Minimal backend (Phase 1)

First iteration: render the same prompt package as Transformers MT, call `llm.generate(prompt_token_ids=..., sampling_params=...)` for determinism, decode the draft, return `MTBackendResult`.

Gotchas fixed:

- **Trailing EOS leaking into draft text**: vLLM includes the stop-triggering token in `completion.token_ids` when stopping fires via `stop_token_ids`; Transformers `.generate(...)` stops *before* EOS. Fix: after generation, pop any trailing ids in `tokenizer.all_special_ids` before decoding.
- **KV cache underflow at 0.3 GPU utilization**: Gemma 4 E4B weights are 15.28 GiB (audio tower bundled) — 0.3 × 40 GiB = 12 GiB doesn't fit. Default is now `mt_vllm_gpu_memory_utilization = 0.5`.
- **Cross-allocator contamination**: running Transformers MT and vLLM MT back-to-back in one Python process reliably runs vLLM out of KV-cache memory, even after `gc.collect() + torch.cuda.empty_cache()`. The PyTorch caching allocator does not return fragmented blocks to the driver. The parity harness works around this by running each backend in its own subprocess.

## Engine-native AlignAtt observer (Phase 2)

The backbone design copies the ASR-side tensor-observer pattern (`gemma_vllm_alignment_backend._AudioQKTensorObserver` + `gemma_vllm_worker.GemmaAlignAttWorker`) with two MT-specific deltas:

### 1. Capture K at *every* prompt position, not just an audio span

The ASR observer captures K only inside the audio embedding span (positions `[audio_prompt_start, audio_prompt_end)`). MT provenance needs the full 4-way partition:

| Region | Positions |
|---|---|
| `source_accessible` | source token positions within the accessible frontier |
| `source_inaccessible` | source token positions beyond the frontier but in the prompt |
| `non_source_prompt` | all other prompt positions (system, instructions, history, scaffolding) |
| `suffix` | decode positions (the draft itself) |

So the MT observer stores K over the full prompt (`max_prompt_tokens = gemma_max_model_len`).

### 2. Capture K at decode positions too

`softmax(Q @ [prompt_K | decode_K])` requires both keys to get the suffix mass. The ASR observer only captures Q at decode; the MT observer adds `decode_k_buffer` parallel to `decode_q_buffer`. Everything stays in fixed-shape tensor buffers with `scatter_add_` writes — no `nonzero`, no Python conditionals — so the capture path is compatible with `torch.compile` and `cudagraph=full`.

### 3. Worker bootstrap before engine build

`GemmaMTAlignAttWorker.load_model()` installs `install_global_gemma4_attention_mt_patch()` (which patches `Gemma4Attention.forward`) **before** `super().load_model()`, then reads a JSON bootstrap payload from the `CASCADE_MT_ALIGNATT_OBSERVER_BOOTSTRAP` env var to configure per-layer observers. `compile_or_warm_up_model()` is deferred until `prepare_mt_observer` is called with a real prompt length so the compiled graph is built *with* the observer attached. Matches the ASR worker's pattern directly.

### 4. Reconstruction

`reconstruct_mt_attention_rows` in `gemma_vllm_mt_observer.py`:

1. For each selected `(layer, head)` AlignAtt head, compute `prompt_logits = Q @ prompt_K^T` and `suffix_logits = Q @ decode_K^T`, scale by `attn.scaling` (= 1.0 on Gemma 4).
2. Apply causal mask on suffix, concatenate, softmax over the full row.
3. Slice source rows at `source_token_positions` → per-token row `(n_heads, n_source_positions)` for argmax-based alignment.
4. Sum-reduce into 4-way provenance, average across heads.

Sliding-window mask is a no-op at `gemma_max_model_len=1024` (Gemma 4 sliding window = 512 is larger than any prompt we handle); a full implementation of the window mask is kept for the Transformers qk_fast path (`cascade_mt_backend.apply_window_mask_to_prompt_logits`).

## Gemma-4 architectural quirks the observer has to know about

Picked up while debugging Phase 2/3 provenance divergence:

- Layer types alternate: `['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', ...]` — full attention every 6 layers (indices 5, 11, 17, 23, 29, 35, 41).
- **Per-layer-type head_dim**: `head_dim = 256` on sliding layers, `global_head_dim = 512` on full-attention layers. Observer reads `attn.head_dim` per-layer so buffer shape is correct either way.
- **Per-layer-type RoPE**: `sliding_attention` uses default RoPE with `rope_theta = 10000`, `full_attention` uses `proportional` RoPE with `rope_theta = 1_000_000` and `partial_rotary_factor = 0.25`. vLLM's `get_rope` handles this via a Gemma4-specific kernel.
- **KV sharing** from layer `first_kv_shared_layer_idx = 24` through 41. The patched forward gates K capture behind `not self.is_kv_shared_layer`. None of the top-K MT AlignAtt heads land in the shared range for en→de (layers 5, 6, 10, 11, 17, 20), so this doesn't bite us today but the guard is there if a future heads file picks a layer ≥ 24.

## Semantic contract (Phase 3)

`VLLMAlignAttGemmaMTBackend.translate()` plugs into the existing runtime by producing an `MTBackendResult` with the same stop-reason vocabulary as the Transformers backend:

- `alignatt:source_frontier` — the next drafted token would align beyond the accessible frontier
- `alignatt:rewind` — the aligned source position jumped backward past `translation_alignatt_rewind_threshold`
- `alignatt:provenance_weak` — `translation_alignatt_min_source_mass` gate fires
- `alignatt:observer_empty` — observer produced no rows (defensive; should not occur in steady state)

The policy loop runs over the **trimmed** draft (after dropping trailing stop tokens). Observer rows and provenance are sliced to that length so `alignatt_metadata` is tokenized against the same count the Transformers backend reports. This was the last semantic fix before parity — earlier the policy ran over the raw observer sequence which included the trailing `<end_of_turn>` and skewed provenance means by one token.

## What is *not* implemented

- **Prefix caching.** `mt_vllm_enable_prefix_caching = False` by default, per PLAN Phase 1. The host-side prompt observer cache from the ASR-side vLLM backend has not been ported; until it is, prefix caching could silently drop prompt-side K capture.
- **Shared observer with the ASR Gemma path.** The ASR and MT observers patch `Gemma4Attention.forward` to different capture functions and cannot coexist in a single process. PLAN's target pair is `qwen_forced` ASR + `gemma_vllm_alignatt` MT, so this is a non-issue in practice.

## Gates

- The vLLM MT backend is the active default in `STABLE_MT_BACKEND_NAMES`.
- `build_mt_backend(... mt_backend_name="unknown")` raises `ValueError`.
- Observer sizing is checked at `translate()` time: if `compute_max_tokens(...)` would ask for more tokens than the observer was configured for, the backend raises rather than silently truncate. The test `test_mt_vllm_observer_max_decode_covers_all_runtime_caps` pins the invariant used at `load()` time.

## Validation harness

- `run_mt_backend_parity.py`: isolated per-prompt MT probe harness for MT-side semantics and observer regressions.
