For each change, examine the existing system and redesign it into the most elegant solution that would have emerged if the change had been a foundational assumption from the start.

## qwen3asr_gemma_cascade.py notes

- The ASR part runs with `qwen_asr` (vLLM-backed), while Gemma uses Transformers+AlignAtt in this code path.
- Use the repo environment `.venv-inference`.

### Important assumptions that are now encoded in the code

- `qwen_asr` is patched at runtime via `patch_qwen_asr_for_transformers5.py`.
- Extra runtime monkey-patches are still required for this stack:
  - `Qwen3ASRConfig.get_text_config`
  - `_qwen3_asr_default_rope_init`
- Models are loaded inside `load_models()`, not at import time, to avoid long startup/reload issues.
- The script uses local Hugging Face snapshot paths instead of Hub model ids to avoid flaky network `HEAD`/`504` issues during startup.

### Current stable settings

- ASR:
  - `Qwen3-ASR-1.7B`
  - `gpu_memory_utilization=0.2`
- Gemma:
  - `gemma-4-E4B-it`
  - `max_model_len=1024`
  - `transformers` inference defaults (`device=cuda:0`, `dtype=bfloat16`)
- These values were tuned to fit ASR + Gemma on one A100 40GB.

### If it breaks again

- First verify GPU is clean:
  - `nvidia-smi`
- If old test processes are still alive (ASR vLLM engine):
  - `pkill -f 'qwen3asr_gemma_cascade.py|VLLM::EngineCore'`
- If local Hugging Face snapshot hashes changed, update the three snapshot paths in `qwen3asr_gemma_cascade_core.py`.
- If someone removes the runtime monkey-patches, the old `thinker_config` / RoPE crashes will likely come back.

### Confidence

- The main compatibility fixes are already in the code, so we should not need to rediscover the same `qwen_asr` bugs again.
- The most likely future pain point is environment drift:
  - different `qwen_asr` / `transformers` / `torch` versions
  - missing local snapshots
  - different GPU memory budget

## Kernel use
- There is the .venv-inference KERNEL that should be running.
- Reloading in memory the ASR and MT model takes 5 minutes. So please do not do that except if necessary. If the models are loaded in memory, reuse it. if you have to restart, please justify it.
