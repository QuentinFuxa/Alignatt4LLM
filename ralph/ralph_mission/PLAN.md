## Focus Hypothesis IDs

- `h_obj2_prompt_only_quality_latency_tuning`

## Blocked Or Frozen IDs

- `h_bootstrap_first_bounded_slice`
- `h_obj1_reproducible_single_audio_eval_loop`
- `h_obj2_freeze14_emission_annex`
- `h_obj2_context1_terminology_guard_annex`
- `h_obj2_prompt_only_terminology_guard_annex`
- `h_obj2_prompt_only_freeze14_emission_annex`

## Why This Is Not A Revisit

`h_obj2_prompt_only_quality_latency_tuning` never left `active`. The next slice
uses the `new_runtime_artifact`
`outputs/cascade_v1_prompt_only_terminology_guard_emit_freeze14/` as a bounded
negative control: we already know pure `freeze14` clears `LongYAAL CU < 2s`,
but only with pathological `LongAL`/`LongLAAL`/`LongDAL` CU values, so the
follow-up is to test exactly one less brittle replay rule without reopening any
frozen annex.

## Goal

Find out whether one milder offline emission policy can keep the prompt-only
final translation intact, stay under `2s` on `LongYAAL CU`, and avoid the
orders-of-magnitude CU latency blow-up seen in the `freeze14` replay annex.

## Scope

- code surface:
  `cascade_emission.py`, `reemit_cascade_outputs.py`,
  `evaluate_cascade_outputs.py`,
  `outputs/cascade_v1_prompt_only_terminology_guard/`,
  `outputs/cascade_v1_prompt_only_terminology_guard_emit_freeze14/`, and one
  fresh replay output dir
- required inputs:
  `ralph/ralph_mission/EXISTING.md`,
  `outputs/cascade_v1/`,
  `outputs/cascade_v1_prompt_only_terminology_guard/`,
  `outputs/cascade_v1_prompt_only_terminology_guard_emit_freeze14/`,
  `.venv-evaluation`
- explicit non-goals:
  another live vLLM rerun, prompt wording changes, revisiting
  `context1_terminology_guard`, online `XCOMETXL` downloads, or a sweep over
  multiple emission rules

## Tasks

1. Implement exactly one bounded emission rule that is less brittle than pure
   `freeze_major_tail_rewrites` and that keeps replay provenance explicit.
2. Replay that rule once on
   `outputs/cascade_v1_prompt_only_terminology_guard/` into a fresh output dir,
   then evaluate it offline.
3. Compare the new replay against the raw prompt-only bundle and the
   `prompt_only_freeze14` annex; only promote it if `LongYAAL CU` stays below
   `2s` and `LongAL`/`LongLAAL`/`LongDAL` CU fall back to a sane range.

## Done When

- one committed replay result shows whether the prompt-only bundle admits a
  sub-`2s` emission policy that is not just a narrow latency hack
- no additional ASR/Gemma reload occurs in the iteration
- all frozen annexes remain out of focus unless a later plan explicitly
  reopens one via `new_runtime_artifact`
