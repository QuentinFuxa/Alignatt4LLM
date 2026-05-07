# System

## Runtime shape

- Canonical inference path: `run_simulstream_batch.py` + `cascade.simulstream_processor.CascadeAlignAttProcessor`
- Active runtime state lives in `cascade/runtime.py`
- Submission presets live only in `cascade/submission.py`

## Supported backends

- ASR `qwen_forced` — stable default
- ASR `gemma_vllm_qk_fast` — sole Gemma AlignAtt ASR path, sub-1 s CU-LongYAAL
  operating point. Standalone entrypoint: `gemma_asr_low_latency.py`.
- MT `gemma_vllm_alignatt` — submitted and stable default; uses
  `google/gemma-4-E4B-it` with Gemma-specific AlignAtt heads
- MT `milmmt_vllm_alignatt` — experimental improvement route; uses
  `xiaomi-research/MiLMMT-46-4B-v0.1` with MiLMMT-specific AlignAtt heads and
  must be selected explicitly
- MT partial acceptance uses frontier and confidence gating only; there is no
  MT anti-rewind threshold because legitimate target-side reorderings make it
  a poor fit for EN->ZH streaming translation.

## Data layout

- `data/devset/` — tracked development set, references, and PDFs
- `dev-set/` — compatibility alias to the tracked development set
- `data/testset/` — local untracked evaluation set
- `data/alignatt_heads/` — tracked runtime head payloads
- `data/smoke/` — tiny reproducible smoke fixtures

## Extra context

- Extra-context code now lives in `cascade/paper_context/`
- Default is still `paper_context_mode=off`
- Paper artifacts stay in `data/paper_artifacts/`
- Extra-context presets are not part of the maintained DockerHub submission surface

## Operational notes

- Use `.venv-inference` for runtime work and `.venv-evaluation` for OmniSTEval/XCOMET
- Reuse hot models whenever possible; full reloads are expensive
- Compare backends sequentially on one GPU
- First-line smoke test: `run_simulstream_compare.py --wav data/smoke/alignatt_smoke18.wav`
