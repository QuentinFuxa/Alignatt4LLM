# Troubleshooting — vLLM cascade gotchas

Operational issues that cost hours during Phase 1–5 and that a future agent should not rediscover the hard way.

## vLLM torch.compile cache collision between ASR and MT observers

**Symptom:** after switching between `gemma_vllm_qk_fast` (ASR-side observer) and `gemma_vllm_alignatt` (MT-side observer) in the same environment, vLLM engine startup crashes with

```
KeyError: '_alignatt_audio_qk_tensor_observer'
```

inside `gemma4.Gemma4Model.forward` at AOT compile time.

**Cause:** vLLM stores compiled forward artefacts under `~/.cache/vllm/torch_compile_cache/`. Both observers patch `Gemma4Attention.forward`, but the compile hash doesn't always change between observer variants — so an old compiled artefact baked against the ASR observer's attribute name gets reloaded when the MT observer is active.

**Fix:**

```bash
rm -rf ~/.cache/vllm/torch_compile_cache
```

before switching observer variants. Phase 5 will want a principled cache-invalidation hook keyed to observer identity, not a manual `rm`.

## Cross-allocator GPU memory contamination (TF + vLLM in one process)

**Symptom:** loading a Transformers model (e.g. the Transformers MT backend) and then a vLLM engine in the same Python process runs the vLLM engine out of KV-cache memory on a 40 GiB A100, even after `del backend; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()`.

Specific vLLM error:

```
INFO Available KV cache memory: -4.7 GiB
ValueError: No available memory for the cache blocks. Try increasing
`gpu_memory_utilization` when initializing the engine.
```

**Cause:** PyTorch's caching allocator holds fragmented blocks that it does not return to the driver. vLLM measures *driver-visible* free memory at engine init time, which looks smaller than what's actually free according to PyTorch.

**Fix:** subprocess isolation per backend when they need to share a GPU. `run_mt_backend_parity.py` does this by spawning one child per backend with `subprocess.run(...)`; the orchestrator only computes agreement by reading back each child's JSON.

## Gemma 4 E4B weights + vLLM = don't under-size `gpu_memory_utilization`

**Symptom:** same `Available KV cache memory: -4.7 GiB` error on a fresh process with nothing else on the GPU.

**Cause:** Gemma 4 E4B loads **15.28 GiB** of weights (the audio tower is bundled even for text-only MT). At `gpu_memory_utilization = 0.3` on a 40 GiB card, target = 12 GiB < weight footprint.

**Fix:** `mt_vllm_gpu_memory_utilization = 0.5` (default since Phase 3). Qwen ASR on vLLM adds another ~8 GiB at its default `asr_gpu_memory_utilization = 0.2`; total ≈ 28 GiB which fits comfortably.

## Trailing `<end_of_turn>` token leaking into vLLM MT drafts

**Symptom:** vLLM MT generates one more token than the Transformers MT backend on the same prompt, and the extra token decodes to `"<turn|>"` appearing in the draft text (e.g. `"... von der Fudan University.<turn|>"`).

**Cause:** when generation stops via `stop_token_ids`, vLLM includes the stop-triggering token in `completion.token_ids`. Transformers' `.generate(...)` stops *before* emitting EOS.

**Fix:** `VLLMAlignAttGemmaMTBackend.translate()` strips any trailing ids in `tokenizer.all_special_ids` before decoding the text. See `gemma_vllm_mt_backend.py` around the `raw_ids` → `trimmed_ids` loop.

## `translation_alignatt_inaccessible_ms` has ~zero latency effect

**Symptom:** increasing `--translation-alignatt-inaccessible-ms` from 0 to 2000 leaves `LongYAAL (CA)` unchanged (1650 → 1650 ms on `OiqEWDVtWk.wav`).

**Cause:** `inaccessible_ms` only masks source tokens as "not yet accessible" during partial MT acceptance. The cascade scheduler already waits on commits/finals, so the mask doesn't bite — whatever word the extra latency would delay is blocked anyway by the higher-level commit rule.

**Fix:** the clean latency knob is `--chunk-ms`. On `OiqEWDVtWk.wav` (en→de), empirical slope is ~7 ms CA per ms of chunk_ms over 450–850 ms. See `docs/RESULTS.md` for the calibration table.

## Stopping stray runaway processes

```bash
# after a crash / interrupt, this usually cleans up:
pkill -f 'run_simulstream_compare.py|run_simulstream_batch.py|VLLM::EngineCore'
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # expect 0 MiB
```

If the GPU shows non-zero memory with no Python processes, a Jupyter kernel or a stale vLLM worker may still be holding it. Check `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.

## Environments

- `.venv-inference` — everything streaming/inference. Used by `run_simulstream_batch.py`, `run_simulstream_compare.py`, `run_mt_backend_parity.py`.
- `.venv-evaluation` — OmniSTEval + XCOMET-XL. Used by `evaluate_cascade_outputs.py`. Do **not** mix: COMET has a Lightning stack that conflicts with vLLM's.

## Reloading models is expensive

Per `AGENTS.md`: loading ASR + MT takes ~5 minutes. If the target is a parameter sweep on one clip, batch the sweep into a single `run_simulstream_batch.py` invocation so the engine is hot across iterations. For multi-backend comparisons, subprocess-per-backend is mandatory (see the cross-allocator issue above), but within one backend the load cost can be amortized.
