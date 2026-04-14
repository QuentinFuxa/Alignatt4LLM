# Ralph State

Derived from `ralph_mission/RALPH_HYPOTHESES.json`. Do not edit by hand.

## Goal
- Finalize a reproducible En->DE single-audio cascade workflow with persistent kernel reuse and evaluation artifacts under outputs/cascade_v1.
- Keep prompt-only quality and latency tuning explicitly gated behind a clean Objectif 1 baseline and a clean repo state.
- Track which runtime branch is active, frozen, or superseded so Ralph iterations stay bounded and auditable.

## Active Focus
- `h_obj2_prompt_only_quality_latency_tuning` [status=`active` kind=`runtime_candidate` priority=`primary`] Tune prompt-only cascade variants under LongYAAL CU below 2 seconds | talks=objective2; delays=prompt_only_latency_quality; slice=Explore only prompt changes, previous-sentence reinjection, and conservative tail trimming after a clean baseline exists; commit every experiment separately.

## Open Runtime Candidates
- `h_obj2_prompt_only_quality_latency_tuning` [status=`active` kind=`runtime_candidate` priority=`primary`] Tune prompt-only cascade variants under LongYAAL CU below 2 seconds | talks=objective2; delays=prompt_only_latency_quality; slice=Explore only prompt changes, previous-sentence reinjection, and conservative tail trimming after a clean baseline exists; commit every experiment separately.

## Frozen Or Falsified Branches
- `h_bootstrap_first_bounded_slice` [status=`superseded` kind=`runtime_candidate` priority=`annex`] Bootstrap the first bounded mission slice | talks=bootstrap; delays=bootstrap; slice=Replace the placeholders with the first real bounded slice for the mission.
- `h_obj2_freeze14_emission_annex` [status=`frozen_annex` kind=`runtime_candidate` priority=`annex`] Keep the freeze14 emission replay as an annex rather than the promoted path | talks=objective2; delays=prompt_only_latency_quality; slice=Replay a deterministic emission policy over the captured raw translation stream without reloading models, and only promote it if the broader latency profile stays sane.

## Paper-Only Claims
- none

## Reopen Conditions
- `h_bootstrap_first_bounded_slice` -> new_talk, new_delay, new_runtime_artifact, human_override
- `h_obj2_freeze14_emission_annex` -> new_runtime_artifact, human_override

## Recent Iterations
- `objective2_emission_freeze14_annex` `workspace` | ids=h_obj2_prompt_only_quality_latency_tuning, h_obj2_freeze14_emission_annex | Separated raw versus emitted translation timelines, restored the canonical baseline bundle, and parked a freeze14 emission replay in an annex after it lowered LongYAAL CU below 2 seconds without improving quality.
- `objective1_incremental_translation_unlock_objective2` `workspace` | ids=h_obj1_reproducible_single_audio_eval_loop, h_obj2_prompt_only_quality_latency_tuning | Replaced full-transcript Gemma truncation with utterance-incremental translation, reran the real baseline to a full final translation plus refreshed offline metrics, and reopened Objective 2 because the corrected runtime artifact now exists.
- `objective1_real_bundle_blockers` `workspace` | ids=h_obj1_reproducible_single_audio_eval_loop | Produced the first real outputs/cascade_v1 bundle and offline evaluation for ccpXHNfaoy.wav, hardened evaluation against XCOMET failures, and exposed that the persisted final translation is still only a short prefix.
- `objective1_artifact_contract` `workspace` | ids=h_obj1_reproducible_single_audio_eval_loop | Landed a versioned outputs/cascade_v1 artifact contract, notebook-safe baseline entrypoints, and a repo-local OmniSTEval driver for the single-audio path without reloading the model stack.
- `mission_objectives_bootstrap` `workspace` | ids=h_bootstrap_first_bounded_slice, h_obj1_reproducible_single_audio_eval_loop, h_obj2_prompt_only_quality_latency_tuning | Replaced the blank mission bootstrap with a concrete Objective 1 baseline branch and a frozen Objective 2 prompt-only tuning branch.
