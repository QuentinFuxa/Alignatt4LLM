# System

## Runtime Shape

- Canonical inference path: `run_simulstream_batch.py` plus
  `cascade.simulstream_processor.CascadeAlignAttProcessor`
- Active runtime state: `cascade/runtime.py`
- Active preset source of truth: `cascade/presets.py`
- Active presets: `gemma_low_latency` and `gemma_high_latency`

## Backends

- ASR `qwen_forced` — stable default.
- ASR `gemma_vllm_qk_fast` — Gemma AlignAtt ASR research path; standalone
  entrypoint: `gemma_asr_low_latency.py`.
- MT `gemma_vllm_alignatt` — stable Gemma baseline route using
  `google/gemma-4-E4B-it` with Gemma-specific AlignAtt heads.
- MT `milmmt_vllm_alignatt` — active MiLMMT improvement route using
  `xiaomi-research/MiLMMT-46-4B-v0.1` with MiLMMT-specific AlignAtt heads.

Both Gemma and MiLMMT MT routes share the same Gemma-family vLLM Q/K observer
and AlignAtt acceptance machinery. The MT acceptance policies are `alignatt`
and `cut_last_target_units`.

## Data Layout

- `data/devset/` — tracked development set and references
- `dev-set/` — compatibility alias to the tracked development set
- `data/testset/` — local untracked evaluation set
- `data/alignatt_heads/` — tracked runtime head payloads
- `data/smoke/` — tiny reproducible smoke fixtures

## Operational Notes

- Use `.venv-inference` for runtime work and `.venv-evaluation` for scoring.
- Reuse hot models whenever possible; full reloads are expensive.
- Compare backends sequentially on one GPU.
- Validate one clip before launching a broader sweep.
- First-line smoke test: `run_simulstream_compare.py --wav data/smoke/alignatt_smoke18.wav`

The Docker submission surface is historical and is no longer present in the
active tree. The paper source and generated PDF are not vendored on the public
branch; use the arXiv record at https://arxiv.org/abs/2606.03967. The archived
submission record is `docs/archive/2026-05-submission.md`.
