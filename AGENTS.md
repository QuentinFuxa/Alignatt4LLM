Welcome on the A100 machine.

For each change, prefer the solution we could defend in a paper: clean,
general, measurable, and not benchmark-specific.

## Core rules

- Do not add hardcoded lexical repairs, phrase tables, or dataset-specific hacks.
- Prefer principled runtime or architecture changes over local patches.
- Do not overproduce tests; protect real invariants and reusable mechanisms.
- Streaming output is append-only: never emit `deleted_tokens` or `deleted_string`.

## Active runtime

- Main runtime: `cascade/runtime.py`
- Preferred stack: `qwen_forced` ASR + `gemma_vllm_alignatt` MT
- Inference env: `.venv-inference`
- Evaluation env: `.venv-evaluation`

## Supported backends

- ASR `qwen_forced` — stable default
- ASR `gemma_onepass_qk_fast` — stable experimental
- ASR `gemma_vllm_qk_fast` — experimental
- MT `gemma_vllm_alignatt` — sole supported MT backend

## Operational constraints

- Model reloads are expensive; reuse hot bundles whenever possible.
- Treat full streaming runs as costly experiments, not cheap probes.
- Validate on one clip before launching any broader sweep.
- Canonical single-audio validation loop: `run_simulstream_compare.py` on `data/smoke/alignatt_smoke18.wav`.
- Keep backend comparisons sequential and isolated on a single GPU.

## Data and docs

- Tracked dev-set: `data/devset/`
- Local untracked test-set: `data/testset/`
- AlignAtt heads: `data/alignatt_heads/`
- Smoke fixtures: `data/smoke/`
- Current docs: `docs/system.md`, `docs/results.md`, `docs/status.md`, `docs/submission.md`
- Historical notes: `docs/archive/2026-04-history.md`
