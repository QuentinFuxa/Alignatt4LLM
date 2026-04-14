# Ralph State

Derived from `ralph_mission/RALPH_HYPOTHESES.json`. Do not edit by hand.

## Goal
- Finalize a reproducible En->DE single-audio cascade workflow with persistent kernel reuse and evaluation artifacts under outputs/cascade_v1.
- Keep prompt-only quality and latency tuning explicitly gated behind a clean Objectif 1 baseline and a clean repo state.
- Track which runtime branch is active, frozen, or superseded so Ralph iterations stay bounded and auditable.

## Active Focus
- `h_obj1_reproducible_single_audio_eval_loop` [status=`active` kind=`runtime_candidate` priority=`primary`] Lock a reproducible single-audio cascade and evaluation loop | talks=objective1; delays=single_audio_baseline; slice=Define and execute the minimal reproducible baseline that writes inference outputs and evaluation metrics for ccpXHNfaoy.wav under outputs/cascade_v1/ without unnecessary model reloads.

## Open Runtime Candidates
- `h_obj1_reproducible_single_audio_eval_loop` [status=`active` kind=`runtime_candidate` priority=`primary`] Lock a reproducible single-audio cascade and evaluation loop | talks=objective1; delays=single_audio_baseline; slice=Define and execute the minimal reproducible baseline that writes inference outputs and evaluation metrics for ccpXHNfaoy.wav under outputs/cascade_v1/ without unnecessary model reloads.

## Frozen Or Falsified Branches
- `h_bootstrap_first_bounded_slice` [status=`superseded` kind=`runtime_candidate` priority=`annex`] Bootstrap the first bounded mission slice | talks=bootstrap; delays=bootstrap; slice=Replace the placeholders with the first real bounded slice for the mission.
- `h_obj2_prompt_only_quality_latency_tuning` [status=`frozen_annex` kind=`runtime_candidate` priority=`secondary`] Tune prompt-only cascade variants under LongYAAL CU below 2 seconds | talks=objective2; delays=prompt_only_latency_quality; slice=Explore only prompt changes, previous-sentence reinjection, and conservative tail trimming after a clean baseline exists; commit every experiment separately.

## Paper-Only Claims
- none

## Reopen Conditions
- `h_bootstrap_first_bounded_slice` -> new_talk, new_delay, new_runtime_artifact, human_override
- `h_obj2_prompt_only_quality_latency_tuning` -> new_runtime_artifact, human_override

## Recent Iterations
- `objective1_real_bundle_blockers` `workspace` | ids=h_obj1_reproducible_single_audio_eval_loop | Produced the first real outputs/cascade_v1 bundle and offline evaluation for ccpXHNfaoy.wav, hardened evaluation against XCOMET failures, and exposed that the persisted final translation is still only a short prefix.
- `objective1_artifact_contract` `workspace` | ids=h_obj1_reproducible_single_audio_eval_loop | Landed a versioned outputs/cascade_v1 artifact contract, notebook-safe baseline entrypoints, and a repo-local OmniSTEval driver for the single-audio path without reloading the model stack.
- `mission_objectives_bootstrap` `workspace` | ids=h_bootstrap_first_bounded_slice, h_obj1_reproducible_single_audio_eval_loop, h_obj2_prompt_only_quality_latency_tuning | Replaced the blank mission bootstrap with a concrete Objective 1 baseline branch and a frozen Objective 2 prompt-only tuning branch.
- `bootstrap` `bootstrap` | ids=h_bootstrap_first_bounded_slice | Initialized a blank Ralph mission scaffold with one active bootstrap hypothesis.
