# System

## Runtime shape

- Canonical inference path: `run_simulstream_batch.py` + `cascade.simulstream_processor.CascadeAlignAttProcessor`
- Active runtime state lives in `cascade/runtime.py`
- Submission presets live only in `cascade/submission.py`

## Supported backends

- ASR `qwen_forced` — stable default
- ASR `gemma_onepass_qk_fast` — stable experimental
- ASR `gemma_vllm_qk_fast` — experimental
- MT `gemma_vllm_alignatt` — only maintained MT backend

## Data layout

- `data/devset/` — tracked development set, references, and PDFs
- `data/testset/` — local untracked evaluation set
- `data/alignatt_heads/` — tracked runtime head payloads
- `data/smoke/` — tiny reproducible smoke fixtures

## Extra context

- Extra-context code now lives in `cascade/paper_context/`
- Default is still `paper_context_mode=off`
- Paper artifacts stay in `data/paper_artifacts/`

## Operational notes

- Use `.venv-inference` for runtime work and `.venv-evaluation` for OmniSTEval/XCOMET
- Reuse hot models whenever possible; full reloads are expensive
- Compare backends sequentially on one GPU
- First-line smoke test: `run_simulstream_compare.py --wav data/smoke/alignatt_smoke18.wav`
