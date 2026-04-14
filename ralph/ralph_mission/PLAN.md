## Focus Hypothesis IDs

- `h_obj2_prompt_only_quality_latency_tuning`

## Blocked Or Frozen IDs

- `h_bootstrap_first_bounded_slice`
- `h_obj1_reproducible_single_audio_eval_loop`
- `h_obj2_freeze14_emission_annex`
- `h_obj2_context1_terminology_guard_annex`
- `h_obj2_prompt_only_terminology_guard_annex`
- `h_obj2_prompt_only_freeze14_emission_annex`
- `h_obj2_prompt_only_nonexpanding_emission_annex`

## Why This Is Not A Revisit

`h_obj2_prompt_only_quality_latency_tuning` never left `active`. The next slice
uses the `new_runtime_artifact`
`outputs/cascade_v1_prompt_only_terminology_guard_emit_nonexpanding14/` as a
second bounded negative control next to the existing `freeze14` annex: the
offline replay space is now bracketed by one path that clears `LongYAAL CU`
pathologically and one path that keeps `LongAL`/`LongLAAL`/`LongDAL` sane but
still misses the `<2s` gate, so the useful next step is a single upstream live
prompt-only variant rather than reopening any frozen replay branch.

## Goal

Find out whether one new prompt-only live variant can reduce partial-segment
translation churn at the source, preserve or improve useful quality, and clear
`LongYAAL CU < 2s` without relying on replay-only emission hacks.

## Scope

- code surface:
  `cascade_translation_variants.py`,
  `qwen3asr_gemma_cascade_core.py`,
  `qwen3asr_gemma_cascade_notebook.py`,
  `run_cascade_baseline.py`,
  `outputs/cascade_v1_prompt_only_terminology_guard/`,
  `outputs/cascade_v1_prompt_only_terminology_guard_emit_freeze14/`,
  `outputs/cascade_v1_prompt_only_terminology_guard_emit_nonexpanding14/`, and
  one fresh live output dir
- required inputs:
  `ralph/ralph_mission/EXISTING.md`,
  `.venv-inference`,
  `.venv-evaluation`,
  the persistent kernel if it exists, or one explicitly justified reload if no
  reusable kernel is alive
- explicit non-goals:
  another offline emission sweep, reopening frozen emission annexes, revisiting
  `context1_terminology_guard`, online `XCOMETXL` downloads, or a broad prompt
  matrix

## Tasks

1. Add exactly one prompt-only variant that tightens partial-segment
   continuation behavior without reintroducing previous-utterance context.
2. Reuse the persistent `.venv-inference` kernel if it exists; otherwise
   justify exactly one reload, run one real bundle into a fresh output dir, and
   evaluate it offline.
3. Compare the new live variant against the canonical baseline, the current
   prompt-only live annex, and the two replay annexes; only promote it if the
   quality profile remains useful and `LongYAAL CU` clears `2s` without a broad
   CU latency pathology.

## Done When

- one committed live variant shows whether upstream prompt changes can beat the
  current prompt-only bundle more honestly than replay-only emission policies
- no frozen branch is reopened without an allowed token
- the iteration records clearly whether a kernel was reused or one reload was
  unavoidable
