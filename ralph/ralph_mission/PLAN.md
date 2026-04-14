## Focus Hypothesis IDs

- `h_obj1_reproducible_single_audio_eval_loop`

## Blocked Or Frozen IDs

- `h_bootstrap_first_bounded_slice`
- `h_obj2_prompt_only_quality_latency_tuning`

## Why This Is Not A Revisit

This keeps the same active hypothesis and uses a new runtime artifact rather
than reopening any frozen branch. The repo now has a first real
`outputs/cascade_v1/` bundle plus a real offline evaluation bundle, and those
artifacts expose two fresh runtime facts: `translation.de.txt` is only a short
prefix while `transcript.en.txt` is complete, and `XCOMETXL` now fails with an
explicit local-cache blocker instead of an opaque evaluator crash.

## Goal

Diagnose and remove the prefix-only final translation failure in the real
`ccpXHNfaoy.wav` baseline, then rerun the single-audio bundle and offline
evaluation without broadening scope beyond Objectif 1.

## Scope

- code surface:
  `qwen3asr_gemma_cascade_core.py`, `run_cascade_baseline.py`,
  `evaluate_cascade_outputs.py`, `outputs/cascade_v1/`, and only the
  smallest instrumentation or config fixes required to explain and fix the
  incomplete final translation
- required inputs:
  `ralph_mission/EXISTING.md`, `test-set/audio/ccpXHNfaoy.wav`,
  `outputs/cascade_v1/`, local HF snapshots, `.venv-inference`,
  `.venv-evaluation`
- explicit non-goals:
  Objectif 2 prompt-only tuning, model swaps, network downloads for XCOMET,
  broad notebook refactors, or silent multi-run GPU restarts

## Tasks

1. Use the persisted real bundle under `outputs/cascade_v1/` to trace why the
   final translation is prefix-only despite a full ASR transcript; check
   `max_new_tokens`, `gemma_max_model_len`, prompt truncation, and the
   stream-versus-final aggregation path before changing behavior.
2. Implement the smallest runtime fix that yields a materially complete final
   `translation.de.txt` for `ccpXHNfaoy.wav` without breaking the artifact
   contract or the explicit offline `XCOMETXL` blocker capture.
3. Justify one more model load only if artifact inspection alone is
   insufficient, rerun the real baseline once, reevaluate offline, and keep
   `h_obj2_prompt_only_quality_latency_tuning` frozen until the corrected
   baseline is persisted.

## Done When

- the next real `outputs/cascade_v1/translation.de.txt` is no longer just the
  short prefix captured in this iteration
- the rerun evaluation bundle contains refreshed `BLEU`, `CHRF`,
  `LongYAAL CU`, `LongYAAL CA`, and the current `XCOMETXL` status
- `h_obj1_reproducible_single_audio_eval_loop` remains the only active focus
