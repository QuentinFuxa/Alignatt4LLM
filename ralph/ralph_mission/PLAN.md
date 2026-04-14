## Focus Hypothesis IDs

- `h_obj2_prompt_only_quality_latency_tuning`

## Blocked Or Frozen IDs

- `h_bootstrap_first_bounded_slice`
- `h_obj1_reproducible_single_audio_eval_loop`
- `h_obj2_freeze14_emission_annex`
- `h_obj2_context1_terminology_guard_annex`

## Why This Is Not A Revisit

`h_obj2_prompt_only_quality_latency_tuning` never left `active`. The live
`context1_terminology_guard` run now lives in
`h_obj2_context1_terminology_guard_annex`, while the older `freeze14` emission
replay remains parked in `h_obj2_freeze14_emission_annex`. The next slice
isolates prompt wording without reopening either frozen branch.

## Goal

Run exactly one prompt-only wording experiment that tries to keep the quality
gain found by `context1_terminology_guard` while removing the latency cost of
`max_history_utterances=1`.

## Scope

- code surface:
  `cascade_translation_variants.py`, `qwen3asr_gemma_cascade_core.py`,
  `qwen3asr_gemma_cascade_notebook.py`, `run_cascade_baseline.py`,
  `outputs/cascade_v1/`, and
  `outputs/cascade_v1_context1_terminology_guard/` for comparison only
- required inputs:
  `ralph/ralph_mission/EXISTING.md`, `outputs/cascade_v1/`,
  `outputs/cascade_v1_context1_terminology_guard/`,
  `test-set/audio/ccpXHNfaoy.wav`, `.venv-inference`, `.venv-evaluation`
- explicit non-goals:
  revisiting the `freeze14` emission branch, rerunning
  `context1_terminology_guard`, model swaps, online `XCOMETXL` downloads, or
  broad latency-metric refactors

## Tasks

1. Derive one prompt-only variant from the new translation-variant registry so
   the wording gains can be tested without previous-utterance reinjection.
2. If no `.venv-inference` kernel is alive, justify one model reload and run
   exactly once into a new output dir that does not replace
   `outputs/cascade_v1/`.
3. Reevaluate offline and compare the new prompt-only run against the canonical
   baseline (`BLEU=40.3126`, `CHRF=68.5453`, `LongYAAL CU=2638.8114`) and the
   frozen `context1_terminology_guard` annex
   (`BLEU=40.6694`, `CHRF=69.8671`, `LongYAAL CU=2339.1869`).

## Done When

- one committed prompt-only live run either beats the baseline on quality and
  clears the latency gate, or records a bounded negative result distinct from
  the `context1_terminology_guard` annex
- `outputs/cascade_v1/` stays the canonical baseline unless the new live
  experiment earns promotion
- `h_obj2_freeze14_emission_annex` and
  `h_obj2_context1_terminology_guard_annex` remain out of focus
