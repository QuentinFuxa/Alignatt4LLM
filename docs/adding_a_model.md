# Adding a New LLM (Bring Your Own Model)

AlignAtt4LLM reconstructs target-to-source attention from selected decoder heads
captured at runtime inside vLLM, then commits only the target prefix supported by
accessible source. That machinery is model-agnostic. Porting it to another
decoder-only LLM means supplying two model-specific things and reusing everything
else.

The worked reference is `src/alignatt4llm/mt/qwen_vllm_backend.py` (Qwen2.5,
`qwen_vllm_alignatt`). Copy it.

## What is generic vs. what you supply

The generic base lives in `src/alignatt4llm/vllm_qk/`:

- `spec.py` — `VLLMAttentionSpec`: which vLLM attention class to patch, the
  attributes its `forward` exposes, and how to build the patched forward.
- `patch.py` — `make_standard_decoder_patched_forward` (the standard no-QK-norm
  attention shape), the spec-driven patch installer, and the stub/configure
  wiring. Reuses the proven `alignatt::capture_mt_qk` custom op, the per-layer
  observer buffers, and `reconstruct_mt_attention_rows`.
- `worker.py` — `BaseQKObserverWorker`: the full vLLM warmup/compile-deferral
  lifecycle. Your worker is a subclass that sets `spec`.

You supply only:

1. a `VLLMAttentionSpec` for your model's attention class, and
2. a thin `BaseMTBackend` subclass (reusing the AlignAtt `translate` loop), and
3. calibrated heads for your model and language direction.

## Steps

1. **Find your vLLM attention class.** e.g.
   `vllm.model_executor.models.qwen2.Qwen2Attention`. Confirm its `forward`
   exposes the standard `qkv_proj`, `q_size`, `kv_size`, `num_heads`,
   `num_kv_heads`, `head_dim`, `rotary_emb`, `attn`, `o_proj`.

2. **Write a `VLLMAttentionSpec`.** Use `make_standard_decoder_patched_forward`
   unless your attention applies **per-head QK-norm before the rotary**
   (Gemma, Qwen3 do; Llama, Qwen2 do not). If it does, write a norm-aware
   forward that norms `q`/`k` before `rotary_emb` and captures the post-norm,
   post-rotary tensors — otherwise the reconstructed attention is wrong.

3. **Subclass the backend.** Inherit `MiLMMTVLLMMTBackend`, set `backend_name`,
   `model_family`, `context_config_attr`, point `worker_cls` at your worker
   (canonical `alignatt4llm.mt.<your>_vllm_worker.<Worker>` path — not the
   legacy `cascade.` shim), and choose a prompt contract (reuse
   `BaseMTBackend.render_prompt_*` for chat-template models).

4. **Subclass the worker.** `class YourWorker(BaseQKObserverWorker): spec = YOUR_SPEC`.

5. **Register in 5 places.** In `src/alignatt4llm/runtime.py`: add to
   `VALID_MT_BACKEND_NAMES`; add a model snapshot/id constant; add a branch in
   `mt_model_name_for_backend()`; add a branch in `alignatt_heads_path_for()`;
   extend the set in `mt_backend_fingerprint()`. And add a dispatch branch in
   `build_mt_backend()` in `src/alignatt4llm/mt/base.py`.

6. **Calibrate heads.** Head artifacts are model- and direction-specific. Do not
   reuse another model's heads except as a diagnostic. Run:
   ```bash
   python data/alignatt_heads/detect_translation_heads.py \
     --model <hf-id> --direction en-de
   ```
   It writes `translation_heads_<safe-model-name>_<src>-<tgt>.json` (the
   `<safe-model-name>` replaces `/` and `.` with `_`). The loader reads
   `token_alignment_heads[:top_k]`, each `{layer, head, ts}`.

## Validate

Most of the wiring is unit-testable without a GPU (see
`tests/test_qwen_mt.py`): registration dispatch, model/heads-path resolution,
and `VLLMAttentionSpec` shape. The attention **capture and reconstruction are
only meaningful on a GPU**, so always validate end-to-end on real hardware:

```bash
.venv-inference/bin/alignatt-batch \
  --inputs <local.wav> --target de \
  --mt-backend-name <your>_vllm_alignatt \
  --trace-attention --output-dir outputs/<your>_smoke
.venv-evaluation/bin/alignatt-eval --output-dir outputs/<your>_smoke
```

`--trace-attention` is the fastest sanity check: it prints, per draft token,
which source token it attends to and the accessible/inaccessible mass. If the
attention does not track the source as words arrive, your patched forward is
capturing the wrong tensors (a common cause: missing QK-norm).

## Notes & limits

- New MT backends keep `mt_vllm_enforce_eager=True` by default: cudagraph replay
  can corrupt the observer payload.
- The observer assumes standard contiguous grouped-query attention; non-standard
  KV grouping needs care in the head→KV-head mapping.
- The vLLM attention-class import path and its internal attribute names are
  vLLM-version-internal. `assert_supported_attention_module` fails loudly at
  install time if a version bump renames them.
- Single vLLM worker only (tensor-parallel > 1 is unsupported).
