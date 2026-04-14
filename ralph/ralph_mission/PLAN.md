## Focus Hypothesis IDs

- `h_obj2_prompt_only_quality_latency_tuning`

## Blocked Or Frozen IDs

- `h_bootstrap_first_bounded_slice`
- `h_obj1_reproducible_single_audio_eval_loop`

## Why This Is Not A Revisit

This reopens `h_obj2_prompt_only_quality_latency_tuning` under the allowed
token `new_runtime_artifact`. The corrected `outputs/cascade_v1/` bundle now
contains a full 709-word German translation plus refreshed offline metrics
(`BLEU=40.3126`, `CHRF=68.5453`, `LongYAAL CU=2638.8114`) instead of the old
62-word prefix artifact, so the next bounded step is prompt-only
quality/latency tuning rather than another Objective 1 rescue pass.

## Goal

Reduce `LongYAAL CU` below `2s` on `ccpXHNfaoy.wav` with exactly one prompt-only
or lightweight-context variant, while keeping the corrected baseline contract
and avoiding any non-authorized runtime changes.

## Scope

- code surface:
  `qwen3asr_gemma_cascade_core.py`, `qwen3asr_gemma_cascade_notebook.py`,
  `outputs/cascade_v1/`, and only prompt/context/tail-trim logic allowed by
  Objective 2
- required inputs:
  `ralph_mission/EXISTING.md`, `outputs/cascade_v1/`,
  `test-set/audio/ccpXHNfaoy.wav`, `.venv-inference`, `.venv-evaluation`
- explicit non-goals:
  model swaps, non-prompt architectural rewrites, silent kernel restarts,
  network downloads for `XCOMETXL`, or broad evaluation refactors

## Tasks

1. Inspect the corrected `outputs/cascade_v1/stream_updates.jsonl` and
   `scores.tsv` to isolate the latency pattern that still keeps
   `LongYAAL CU=2638.8114 ms`.
2. Implement one bounded Objective 2 variant limited to prompt wording,
   previous-sentence reinjection, or conservative tail trimming, and keep
   `h_obj1_reproducible_single_audio_eval_loop` out of focus except for the
   explicit `XCOMETXL` external blocker.
3. If no `.venv-inference` kernel is alive, justify one model reload, rerun the
   real baseline once, reevaluate offline, and compare the refreshed metrics to
   the current locked baseline (`BLEU=40.3126`, `CHRF=68.5453`,
   `LongYAAL CU=2638.8114`, `XCOMETXL=NA`).

## Done When

- a single committed Objective 2 variant either pushes `LongYAAL CU` below
  `2s` or produces a bounded negative result that explains why the tested
  prompt-only slice failed
- the refreshed bundle under `outputs/cascade_v1/` preserves a materially
  complete final translation and a fresh offline evaluation bundle
- `h_obj2_prompt_only_quality_latency_tuning` remains the only active focus
