## Focus Hypothesis IDs

- `h_obj2_prompt_only_quality_latency_tuning`

## Blocked Or Frozen IDs

- `h_bootstrap_first_bounded_slice`
- `h_obj1_reproducible_single_audio_eval_loop`
- `h_obj2_freeze14_emission_annex`
- `h_obj2_context1_terminology_guard_annex`
- `h_obj2_prompt_only_terminology_guard_annex`

## Why This Is Not A Revisit

`h_obj2_prompt_only_quality_latency_tuning` never left `active`. The next slice
uses the `new_runtime_artifact`
`outputs/cascade_v1_prompt_only_terminology_guard/` to test one deterministic
emission replay, while `h_obj2_freeze14_emission_annex`,
`h_obj2_context1_terminology_guard_annex`, and
`h_obj2_prompt_only_terminology_guard_annex` stay parked out of focus.

## Goal

Replay exactly one conservative emission policy over the new prompt-only live
bundle to see whether the remaining `218.5736 ms` `LongYAAL CU` gap can be
closed without losing the small `CHRF` win already achieved over the canonical
baseline.

## Scope

- code surface:
  `cascade_emission.py`, `reemit_cascade_outputs.py`,
  `evaluate_cascade_outputs.py`,
  `outputs/cascade_v1_prompt_only_terminology_guard/`, and one fresh replay
  output dir
- required inputs:
  `ralph/ralph_mission/EXISTING.md`, `outputs/cascade_v1/`,
  `outputs/cascade_v1_emit_freeze14/`,
  `outputs/cascade_v1_prompt_only_terminology_guard/`, `.venv-evaluation`
- explicit non-goals:
  another live vLLM rerun, revisiting `context1_terminology_guard`, changing
  prompt wording again, online `XCOMETXL` downloads, or broad metric refactors

## Tasks

1. Replay exactly one deterministic emission policy from
   `outputs/cascade_v1_prompt_only_terminology_guard/` into a fresh output dir,
   using the existing replay path rather than another live run.
2. Evaluate the replay offline and compare it against the raw prompt-only
   bundle, the canonical baseline, and the old baseline `freeze14` annex.
3. Promote only if the replay clears `LongYAAL CU < 2s` while keeping the
   prompt-only final translation quality profile intact; otherwise record one
   more bounded negative result.

## Done When

- one committed replay result shows whether
  `prompt_only_terminology_guard + conservative emission` can satisfy the
  active success gate
- no additional ASR/Gemma reload occurs in the iteration
- frozen annexes remain out of focus unless a later plan explicitly reopens one
  via `new_runtime_artifact`
