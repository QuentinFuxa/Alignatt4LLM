## Focus Hypothesis IDs

- `h_obj2_prompt_only_quality_latency_tuning`

## Blocked Or Frozen IDs

- `h_bootstrap_first_bounded_slice`
- `h_obj1_reproducible_single_audio_eval_loop`
- `h_obj2_freeze14_emission_annex`

## Why This Is Not A Revisit

`h_obj2_prompt_only_quality_latency_tuning` never left `active`. The new
`freeze14` emission replay is now parked in `h_obj2_freeze14_emission_annex`,
while `outputs/cascade_v1/` has been restored as the canonical raw-passthrough
baseline and `outputs/cascade_v1_emit_freeze14/` remains comparison evidence
only.

## Goal

Run exactly one real prompt/context experiment against the restored baseline,
seeking a material quality gain while staying compatible with the sub-`2s`
LongYAAL CU gate that the annex replay showed is reachable.

## Scope

- code surface:
  `qwen3asr_gemma_cascade_core.py`, `qwen3asr_gemma_cascade_notebook.py`,
  `outputs/cascade_v1/`, and `outputs/cascade_v1_emit_freeze14/` for
  comparison only
- required inputs:
  `ralph/ralph_mission/EXISTING.md`, `outputs/cascade_v1/`,
  `outputs/cascade_v1_emit_freeze14/`, `test-set/audio/ccpXHNfaoy.wav`,
  `.venv-inference`, `.venv-evaluation`
- explicit non-goals:
  revisiting the `freeze14` emission branch, model swaps, online `XCOMETXL`
  downloads, or broad latency-metric refactors

## Tasks

1. Compare the restored baseline bundle `outputs/cascade_v1/` against the
   annex replay `outputs/cascade_v1_emit_freeze14/` only to identify one
   prompt/context change that targets quality rather than metric-gating.
2. Implement one bounded prompt/context variant and keep the raw/emitted
   timeline separation available, but do not promote `freeze14` into the live
   default path.
3. If no `.venv-inference` kernel is alive, justify one model reload, rerun
   once, reevaluate offline, and compare the refreshed real variant against the
   restored raw baseline (`BLEU=40.3126`, `CHRF=68.5453`,
   `LongYAAL CU=2638.8114`) and the annex replay (`LongYAAL CU=1940.9677`
   without quality gain).

## Done When

- one committed prompt/context variant either improves quality materially while
  remaining bounded on latency, or yields a bounded negative result with fresh
  real metrics
- `outputs/cascade_v1/` stays the canonical baseline unless the new live
  experiment earns promotion
- `h_obj2_freeze14_emission_annex` remains out of focus
