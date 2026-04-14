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
- `h_obj2_context1_terminology_guard_annex` [status=`frozen_annex` kind=`runtime_candidate` priority=`annex`] Keep the one-utterance terminology-guard run as a measured annex, not the promoted runtime | talks=objective2; delays=prompt_only_latency_quality; slice=Run exactly one live prompt/context variant with one previous utterance plus stricter terminology guidance, and only promote it if the quality gain survives the sub-2s LongYAAL CU gate.
- `h_obj2_prompt_only_terminology_guard_annex` [status=`frozen_annex` kind=`runtime_candidate` priority=`annex`] Keep the prompt-only terminology-guard run as a measured annex, not the promoted runtime | talks=objective2; delays=prompt_only_latency_quality; slice=Run exactly one live prompt-only terminology-guard variant without previous-utterance context, and only promote it if the quality profile survives while LongYAAL CU clears 2 seconds.
- `h_obj2_prompt_only_freeze14_emission_annex` [status=`frozen_annex` kind=`runtime_candidate` priority=`annex`] Keep the prompt-only freeze14 emission replay as an annex rather than the promoted path | talks=objective2; delays=prompt_only_latency_quality; slice=Replay exactly one deterministic freeze14 emission policy over the prompt-only live bundle without another vLLM run, and only promote it if the broader latency profile stays sane alongside the sub-2s LongYAAL win.
- `h_obj2_prompt_only_nonexpanding_emission_annex` [status=`frozen_annex` kind=`runtime_candidate` priority=`annex`] Keep the prompt-only nonexpanding emission replay as an annex rather than the promoted path | talks=objective2; delays=prompt_only_latency_quality; slice=Replay exactly one deterministic nonexpanding major-rewrite emission policy over the prompt-only live bundle without another vLLM run, and only promote it if it keeps the broader CU latency family sane while also clearing the sub-2s gate.

## Paper-Only Claims
- none

## Reopen Conditions
- `h_bootstrap_first_bounded_slice` -> new_talk, new_delay, new_runtime_artifact, human_override
- `h_obj2_freeze14_emission_annex` -> new_runtime_artifact, human_override
- `h_obj2_context1_terminology_guard_annex` -> new_runtime_artifact, human_override
- `h_obj2_prompt_only_terminology_guard_annex` -> new_runtime_artifact, human_override
- `h_obj2_prompt_only_freeze14_emission_annex` -> new_runtime_artifact, human_override
- `h_obj2_prompt_only_nonexpanding_emission_annex` -> new_runtime_artifact, human_override

## Recent Iterations
- `objective2_prompt_only_emission_nonexpanding_annex` `workspace` | ids=h_obj2_prompt_only_quality_latency_tuning, h_obj2_prompt_only_nonexpanding_emission_annex | Added a nonexpanding major-rewrite replay policy for the prompt-only bundle, then parked the derived artifact in an annex after it kept the broader CU latency profile sane but still missed the sub-2s LongYAAL gate.
- `objective2_prompt_only_emission_freeze14_annex` `workspace` | ids=h_obj2_prompt_only_quality_latency_tuning, h_obj2_prompt_only_freeze14_emission_annex | Replayed a freeze14 emission policy over the prompt-only live bundle, recorded explicit replay provenance, and parked the sub-2s result in an annex because the broader CU latency metrics still behaved pathologically.
- `objective2_prompt_only_terminology_guard_annex` `workspace` | ids=h_obj2_prompt_only_quality_latency_tuning, h_obj2_prompt_only_terminology_guard_annex | Added one prompt_only_terminology_guard live variant, reran the real bundle once, and parked it in an annex after it improved latency versus baseline but still missed the sub-2s gate and lost BLEU.
- `objective2_context1_terminology_guard_annex` `workspace` | ids=h_obj2_prompt_only_quality_latency_tuning, h_obj2_context1_terminology_guard_annex | Added named translation variants, ran one real context1_terminology_guard experiment, and parked it in an annex after it improved BLEU/CHRF and LongYAAL CU but still missed the sub-2s gate.
- `objective2_emission_freeze14_annex` `workspace` | ids=h_obj2_prompt_only_quality_latency_tuning, h_obj2_freeze14_emission_annex | Separated raw versus emitted translation timelines, restored the canonical baseline bundle, and parked a freeze14 emission replay in an annex after it lowered LongYAAL CU below 2 seconds without improving quality.
