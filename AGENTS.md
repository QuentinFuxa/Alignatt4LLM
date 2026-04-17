Welcome on the A100 Machine, dedicated for your needs !

For each change, examine the existing system and redesign it into the most elegant solution that would have emerged if the change had been a foundational assumption from the start.

## Defensible Research Code

- We want the final system to be clean, generalizable, and defensible in a paper.
- The codebase is very recent and almost nothing should be treated as settled or sacred yet.
- We are in a pure experimentation phase: be proactive, be willing to rethink foundations, and do not be overly conservative about existing code.
- Do not add hardcoded lexical substitutions, phrase tables, dataset-specific rewrites, content-aware string repairs, or other ad hoc heuristics just to make examples look better.
- Do not smuggle in special-case behavior for known failures or benchmark artifacts.
- Prefer principled architectural changes, generic normalization, well-scoped model/runtime improvements, and explicitly measurable mechanisms.
- If a shortcut is not something we could defend honestly in a paper, it is forbidden here.
- No "screugneugneu" adjustments: no hacky tuning whose only purpose is to patch a specific example without a general justification.
- Do not overproduce low-value tests. We are experimenting, so tests shopleauld protect real invariants, critical regressions, or reusable mechanisms, not add weight for its own sake.
- Avoid test bloat, hyper-granular assertion noise, and large test scaffolding for temporary or exploratory changes.
- Do not hesitate to remove, replace, or redesign code that is poorly conceived. During this phase, strong cleanup and bold simplification are encouraged when they improve the system.d
- What we want here, is, more generally, break alignatt for LLMs. Goal is to write a paper. That's a challenge, and that justify the investigation you should deep dive in. The more interessting and clever/replicable/solid/ implementaiton, the happier i am
- **Never emit `deleted_tokens` or `deleted_string`. Streaming output is append-only: once target text has been emitted, we forbid rewriting the past.**

## Runtime notes
- The active runtime lives in `cascade_runtime.py`.
- `qwen3asr_gemma_cascade_core.py` is only a temporary compatibility shim consumed by historical scripts under `scripts/`; no active code imports it.
- We now have **vLLM on both sides** (ASR and MT). The recommended combination is `qwen_forced` ASR + `gemma_vllm_alignatt` MT. See `docs/RUNTIME_ARCHITECTURE.md` for the full matrix.
- Use the repo environment `.venv-inference`.

### Supported ASR / alignment backends

- `qwen_forced` = `Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B` (vLLM) — **stable, default**
- `gemma_onepass_qk_fast` = Gemma 4 ASR + audio AlignAtt `qk_fast` in one Transformers pass — stable, experimental
- `gemma_vllm_qk_fast` = Gemma 4 ASR via vLLM + engine-native audio observer (worker_cls + tensor buffers, cudagraph=full by default) — experimental

### Supported MT backends

- `gemma_vllm_alignatt` = Gemma-4-E4B MT through vLLM + engine-native MT AlignAtt observer — **sole supported MT backend**

MT now has one supported path. See `docs/MT_VLLM_BACKEND.md`.

Everything else is historical or archived. `hybrid_*` and `gemma_two_pass` are no longer active backends. `eager` remains acceptable only for explicit calibration / debug tooling.

### Important assumptions that are now encoded in the code

- `qwen_asr` is patched only when the Qwen backend is actually loaded, via `patch_qwen_asr_for_transformers5.py`.
- Extra runtime monkey-patches are still required for this stack:
  - `Qwen3ASRConfig.get_text_config`
  - `_qwen3_asr_default_rope_init`
- Models are loaded lazily through `LoadedModelBundle.load()`, not at import time, to avoid long startup/reload issues.
- The script uses local Hugging Face snapshot paths instead of Hub model ids to avoid flaky network `HEAD`/`504` issues during startup.
- Mutable streaming state now belongs to `CascadeSession`, not to module-level globals.

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
  - `pkill -f 'run_simulstream_compare.py|run_simulstream_batch.py|VLLM::EngineCore'`
- If local Hugging Face snapshot hashes changed, update the three snapshot paths in `cascade_runtime.py`.
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
- Running the cascade is also expensive in wall-clock time even when the models are already hot in memory. Treat full streaming evaluations as costly experiments, not cheap probes.
- Do not launch multiple audios or broad benchmark sweeps until the current objective has already been reached, or very nearly reached, on a single audio.
- SimulStream is the canonical inference path.
- The canonical single-audio validation loop is `run_simulstream_compare.py` on `tmp/alignatt_smoke18.wav`.
- The two backend runs in that comparison must stay sequential and isolated: do not keep Qwen ASR + Gemma ASR + Gemma MT all resident together.
- Preferred workflow:
  - first validate the idea on one audio
  - first clip for this repo cleanup is `tmp/alignatt_smoke18.wav`
  - then iterate on that one audio until the mechanism behaves as intended
  - only then scale out to multiple audios or full benchmark runs

- Simulstream is the recommanded framework to run inference.
- Omnisteval the recommanted framwork to run evaluation.

## Where to find things

- `PLAN.md` — current plan (short; history archived).
- `DECISIONS.md` — append-only session-level decision log.
- `docs/RUNTIME_ARCHITECTURE.md` — ASR + MT axes, module map, session lifecycle.
- `docs/MT_VLLM_BACKEND.md` — experimental `gemma_vllm_alignatt` design.
- `docs/RESULTS.md` — consolidated quality/latency numbers and the `chunk_ms` calibration curve.
- `docs/TROUBLESHOOTING.md` — GPU / vLLM gotchas (compile cache, utilization, trimming).
- `docs/archive/` — historical design docs.
- `docs/reference/` — upstream model cards + referenced papers/implementations.
- `scripts/` — dated research scripts, preserved for reference only.
