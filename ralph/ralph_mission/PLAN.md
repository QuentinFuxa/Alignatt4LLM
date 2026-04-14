## Focus Hypothesis IDs

- `h_obj1_reproducible_single_audio_eval_loop`

## Blocked Or Frozen IDs

- `h_bootstrap_first_bounded_slice`
- `h_obj2_prompt_only_quality_latency_tuning`

## Why This Is Not A Revisit

This keeps the same active hypothesis but advances to the next bounded runtime
step after the artifact contract landed. The repo now has repo-local
entrypoints for inference and evaluation, so the next iteration is to replace
offline smoke-test proof with a real `ccpXHNfaoy.wav` baseline run, not to
reopen any frozen branch.

## Goal

Produce the first real `outputs/cascade_v1/` bundle for `ccpXHNfaoy.wav` and
replace the current offline-only smoke test with runtime proof from the split
`.venv-inference` and `.venv-evaluation` workflow.

## Scope

- code surface:
  `run_cascade_baseline.py`, `evaluate_cascade_outputs.py`,
  `qwen3asr_gemma_cascade_core.py`, kernel helper scripts, and only the
  smallest fixes required by the first real persisted run
- required inputs:
  `ralph_mission/EXISTING.md`, `test-set/audio/ccpXHNfaoy.wav`,
  `outputs/cascade_v1/`, local HF snapshots, `.venv-inference`,
  `.venv-evaluation`
- explicit non-goals:
  Objectif 2 prompt tuning, model swaps, broad refactors, synthetic-only proof
  presented as final evidence, or silent kernel restarts

## Tasks

1. Reuse a live `.venv-inference` kernel if one exists; otherwise justify the
   one-time model load and run `run_cascade_baseline.py` for
   `ccpXHNfaoy.wav`.
2. Run `evaluate_cascade_outputs.py` from `.venv-evaluation` on the emitted
   `hypothesis.jsonl`, and record whether `Unbabel/XCOMET-XL` is available as a
   real score or an explicit blocker.
3. Keep `h_obj2_prompt_only_quality_latency_tuning` frozen until the real
   baseline bundle and evaluation outputs exist and the repo is clean again.

## Done When

- `outputs/cascade_v1/` contains real inference artifacts from
  `ccpXHNfaoy.wav`
- the evaluation bundle contains `BLEU`, `CHRF`, `LongYAAL CU`,
  `LongYAAL CA`, and either a real `XCOMETXL` score or explicit blocker
  evidence
- `h_obj1_reproducible_single_audio_eval_loop` remains the only active focus
