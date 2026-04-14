## Focus Hypothesis IDs

- `h_obj1_reproducible_single_audio_eval_loop`

## Blocked Or Frozen IDs

- `h_obj2_prompt_only_quality_latency_tuning`

## Why This Is Not A Revisit

This is the first real mission framing after bootstrap. The human objective is
now explicit, so Ralph is not reopening a frozen branch; it is replacing the
placeholder mission with one bounded active baseline and one intentionally
frozen follow-up branch.

## Goal

Establish Objectif 1 as the active bounded mission slice: one reproducible
En->DE single-audio cascade run plus evaluation outputs under
`outputs/cascade_v1/`, while preserving the persistent `.venv-inference`
kernel workflow.

## Scope

- code surface:
  `qwen3asr_gemma_cascade_core.py`, `qwen3asr_gemma_cascade_notebook.py`,
  `pyproject.toml`, `setup_inference_qwen_asr_vllm.sh`, kernel helper scripts,
  and the smallest new output plumbing needed for `outputs/cascade_v1/`
- required inputs:
  `ralph_mission/EXISTING.md`, repo rules, `test-set/audio/ccpXHNfaoy.wav`,
  local HF snapshots, `.venv-inference`, `.venv-evaluation`
- explicit non-goals:
  Objectif 2 experiments, broad refactors, model swaps, environment rebuilds
  without evidence, and anything that forces unnecessary model reloads

## Tasks

1. Turn Objectif 1 into a concrete artifact contract: define what inference and
   evaluation outputs must exist in `outputs/cascade_v1/`.
2. Make the baseline path executable from the persistent `.venv-inference`
   kernel and from `.venv-evaluation` without rediscovering the stack.
3. Record remaining blockers precisely if the baseline cannot yet be completed,
   then keep Objectif 2 frozen until the repo is clean and the baseline commit
   exists.

## Done When

- `h_obj1_reproducible_single_audio_eval_loop` remains the only active focus
- Ralph has a concrete path to produce inference and evaluation artifacts under
  `outputs/cascade_v1/`
- `h_obj2_prompt_only_quality_latency_tuning` stays frozen until Objectif 1 is
  clean and the repo is clean
