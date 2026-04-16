# Delegated Agent Plan: Audit, De-risk, and Rebuild the Gemma ASR/Aligner Evaluation

## Mission

You are not being asked to "make Gemma look good."

You are being asked to determine, rigorously and defensibly:

1. whether Gemma free-run ASR is actually being used correctly
2. whether the current negative ASR conclusion is fair
3. whether the Gemma attention-based aligner result is real and reproducible
4. whether the current hybrid path is honestly measured
5. what the clean next architecture should be

This plan replaces the earlier broad exploration plan with a **critical execution plan** based on the current implementation and its weaknesses.


## Current Bottom Line

The implementation appears to have achieved something interesting on the **alignment** side, but the **ASR** side is not yet on solid ground.

At the moment, the strongest defensible claim is:

- Gemma attention may provide a usable forced-alignment signal.

At the moment, the following claims are **not yet strong enough**:

- Gemma free-run ASR has been benchmarked fairly and still fails.
- The hybrid cascade numbers cleanly reflect Gemma alignment rather than hidden fallback to Qwen timings.
- The reported runtime path is using the same calibrated head bundle that produced the best offline numbers.

Your first responsibility is to make the experiment honest.


## Hard Requirements

Follow these project constraints:

1. No heuristic lexical repairs.
2. No content-aware timestamp nudges.
3. No ad hoc fixes to make specific examples look better.
4. No broad benchmark sweeps until the single-audio story is clean.
5. Reuse hot models whenever possible.
6. Do not restart or reload expensive models unless necessary and justified.
7. Prefer structural fixes, explicit diagnostics, and measurable comparisons.


## What You Must Read First

Read these before changing any code:

1. [AGENTS.md](/home/fuxa/cascade_simultaneous/AGENTS.md)
2. [CLAUDE.md](/home/fuxa/cascade_simultaneous/CLAUDE.md)
3. [PLAN_RESULT_IMPLEMENTATION.md](/home/fuxa/cascade_simultaneous/PLAN_RESULT_IMPLEMENTATION.md)
4. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)
5. [alignment_backend.py](/home/fuxa/cascade_simultaneous/alignment_backend.py)
6. [qwen_alignment_backend.py](/home/fuxa/cascade_simultaneous/qwen_alignment_backend.py)
7. [gemma_alignment_probe.py](/home/fuxa/cascade_simultaneous/gemma_alignment_probe.py)
8. [hybrid_alignment_backend.py](/home/fuxa/cascade_simultaneous/hybrid_alignment_backend.py)
9. [run_alignment_single_audio.py](/home/fuxa/cascade_simultaneous/run_alignment_single_audio.py)
10. [run_streaming_stability.py](/home/fuxa/cascade_simultaneous/run_streaming_stability.py)
11. [Qwen3_aligner.md](/home/fuxa/cascade_simultaneous/Qwen3_aligner.md)
12. [gemma_stt.py](/home/fuxa/cascade_simultaneous/gemma_stt.py)

Then consult the official Gemma docs again, specifically for:

1. canonical audio prompt ordering
2. canonical `generate()` usage
3. audio length limits
4. model class / processor expectations


## Critical Findings You Must Treat As Open Problems

These are not optional follow-ups. These are the core unresolved issues.

### 1. Wrong runtime default for the head bundle

The current config default points to:

- `assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en.json`

But the best reported results come from:

- `assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json`

This is a major integrity issue.

Until fixed, "what the cascade can do today" is ambiguous.

### 2. Hybrid evaluation is not auditable

The hybrid backend falls back to Qwen timings when Gemma alignment fails.

That is acceptable behavior for robustness.

What is not acceptable is that current harnesses do not clearly expose:

- how often fallback happened
- on which ticks
- for what reason
- whether reported hybrid metrics partially reflect Qwen timings

Until this is visible, hybrid quality and stability numbers are not trustworthy.

### 3. Gemma ASR fairness is unresolved

The current negative ASR conclusion may be directionally right, but the evaluation is not yet tight enough.

The implementation currently mixes:

- a manual greedy decoding loop
- prompt-template changes
- message-order changes
- language handling that is partially ignored
- cookbook and non-cookbook paths

You must isolate these variables and benchmark them cleanly.

### 4. Reported alignment robustness is still narrow

The current "generalization" evidence is promising, but weak:

- same model
- same language
- same speaker
- same talk

This is not enough to claim robustness.

### 5. Long-audio behavior is unsafe

The implementation notes already admit silent truncation risk around the 30-second audio cap.

This must be fixed before any real deployment or cascade claim.


## Deliverables

You should aim to produce the following deliverables.

### Required

1. A corrected and auditable Gemma alignment runtime path.
2. A fair Gemma ASR benchmark harness.
3. A short written verdict on whether Gemma ASR was previously misused.
4. A short written verdict on whether the Gemma aligner result is real.
5. A revised recommendation for the architecture:
   - Gemma-only
   - hybrid
   - or stop

### Strongly preferred

1. Artifact bundles for all key comparisons.
2. A fallback audit report.
3. A small multi-clip robustness table.
4. A list of exactly which claims are defensible and which are not.


## Success Criteria

The task is successful only if the following are true:

1. The default runtime path matches the best calibrated method, or fails loudly if not configured.
2. Hybrid metrics explicitly report Gemma-vs-fallback usage.
3. Gemma ASR is benchmarked through a clean and controlled comparison matrix.
4. Any negative conclusion about Gemma ASR is based on fair evidence.
5. Long-audio truncation is either prevented or handled explicitly.
6. The final recommendation is based on measured evidence, not optimism or pessimism.


## Work Plan

## Phase 0: Freeze The Story Before Changing It

Objective:

- establish the exact current state and avoid drifting into undocumented re-interpretation

Tasks:

1. Read `PLAN_RESULT_IMPLEMENTATION.md` closely.
2. Extract all reported claims into a checklist:
   - offline MAE
   - streaming drift
   - free-run ASR failure
   - cross-content generalization
   - hybrid readiness
3. For each claim, mark:
   - directly supported by code and artifacts
   - partially supported
   - unsupported or ambiguous
4. Preserve current artifact paths and filenames.

Exit criterion:

- you have a one-page audit note stating exactly what the repo currently claims


## Phase 1: Fix Experimental Integrity Before Chasing More Results

Objective:

- make the runtime and measurement paths honest

Tasks:

1. Correct the runtime default head path.
   - Either set the default to the forced-calibrated bundle.
   - Or require explicit selection and raise if the wrong bundle would be used silently.
2. Propagate backend diagnostics through all relevant harnesses.
3. In the hybrid backend and all evaluation scripts, log:
   - whether Gemma alignment succeeded
   - whether fallback to Qwen timings occurred
   - reason for fallback
   - head bundle used
   - offset used
4. Ensure all reports include this metadata.

Important:

- Do not hide fallback behind “robustness.”
- If a metric partially comes from fallback, the report must say so.

Exit criterion:

- every hybrid result can be decomposed into Gemma-aligned ticks vs fallback ticks


## Phase 2: Build A Fair Gemma ASR Benchmark Harness

Objective:

- answer the specific question: “Was Gemma ASR actually used correctly?”

This is the most important unresolved question.

You need a **controlled matrix** that isolates implementation choices.

Compare at least the following dimensions:

1. Decoding path
   - official `model.generate()`
   - current manual greedy token-by-token loop
2. Prompt ordering
   - audio-before-text
   - text-before-audio
3. Prompt wording
   - cookbook “original language”
   - explicit English transcription wording
4. Model class entrypoint
   - canonical auto class used by current docs
   - any alternate auto class only if it truly resolves differently
5. Input casting path
   - cookbook-style `.to(device)`
   - any manual float casting only if needed and justified
6. Candidate Gemma checkpoints
   - current E4B-it
   - optionally E2B or another plausible audio-capable Gemma checkpoint if available locally

Tasks:

1. Build one script dedicated only to free-run Gemma transcription fairness.
2. Reuse one short audio first.
3. Produce exact outputs and WER/CER against a trusted reference.
4. Save all prompt variants and outputs to JSON.

Important:

- This must be a transcription harness, not an alignment harness pretending to be one.
- Do not mix free-run ASR benchmarking with alignment head calibration.

Exit criterion:

- you can state, with evidence, whether Gemma ASR was misused previously


## Phase 3: Revalidate The Attention Aligner Under The Correct Prompting Contract

Objective:

- determine whether the reported alignment result survives after cookbook-consistent prompting

The current implementation notes already admit a critical mismatch:

- the strong forced-alignment numbers were obtained under the old ordering
- the free-run path now uses the cookbook ordering

You must resolve this.

Tasks:

1. Recalibrate heads under the current intended prompting contract.
2. Re-estimate the global offset.
3. Compare:
   - old ordering / old calibration
   - new ordering / new calibration
4. Report:
   - MAE
   - median
   - P90
   - monotonicity
   - streaming drift

Important:

- Do not keep an old calibration just because it scores better if the runtime path has changed.
- Runtime and evaluation must match.

Exit criterion:

- the best reported alignment number corresponds to the actual intended runtime configuration


## Phase 4: Make The Hybrid Evaluation Honest

Objective:

- determine the actual performance of the deployable hybrid architecture

Tasks:

1. Run the hybrid backend on one audio with full fallback logging.
2. Quantify:
   - fallback rate by tick
   - fallback rate by word
   - fallback reasons
3. Report hybrid stability both:
   - including fallback
   - excluding fallback-only ticks if meaningful
4. Clearly label whether reported numbers reflect:
   - pure Gemma alignment behavior
   - mixed Gemma/Qwen behavior

Important:

- The hybrid path may still be the right answer.
- But if so, its value must be described honestly:
  - “replace the external forced aligner most of the time”
  - not “Gemma aligner solved it” unless that is truly what the logs show

Exit criterion:

- the hybrid result is auditable and reproducible


## Phase 5: Fix Long-Audio Safety

Objective:

- prevent silent invalid behavior

Tasks:

1. Identify the true effective audio cap for the Gemma path.
2. Add an explicit guard in:
   - free-run Gemma transcription
   - teacher-forced Gemma alignment
3. Decide on one policy:
   - fail loudly
   - explicit truncation with metadata
   - or a principled chunk/window mechanism
4. Document the chosen policy in code and in the report.

Important:

- Silent truncation is not acceptable.
- If chunking is introduced, it must be principled and measurable.

Exit criterion:

- there is no silent over-length behavior left in the Gemma path


## Phase 6: Run The Minimum Robustness Check

Objective:

- determine whether the alignment result is local or real

Do not do a broad sweep.

Use a **small but deliberately diverse** set:

1. the original short clip
2. one different segment from the same talk
3. at least one different speaker or accent if available
4. optionally one cleaner clip and one harder clip

Tasks:

1. Re-run the best calibrated aligner on this small set.
2. Save all artifact bundles.
3. Report:
   - word-end MAE
   - median
   - P90
   - monotonicity
   - drift
4. Note whether the same head cluster remains best.

Important:

- The purpose is not to maximize numbers.
- The purpose is to learn whether Layer 23 + one scalar offset is genuinely robust or just local.

Exit criterion:

- you can describe robustness honestly without overclaiming


## Phase 7: Reassess The Architecture Decision

Objective:

- recommend the correct next architecture based on evidence

At this point, choose one of the following and justify it cleanly.

### Option A: Gemma free-run ASR is actually viable

Choose this only if:

- the fair ASR benchmark shows competitive transcription quality
- and the aligner remains usable

### Option B: Hybrid is the correct architecture

Choose this if:

- Qwen remains clearly better for transcription
- Gemma alignment remains useful
- fallback rate is acceptable
- the hybrid path is operationally honest and stable

### Option C: Stop or downgrade the Gemma aligner line

Choose this if:

- the aligner result collapses under correct prompting
- or it depends too heavily on fallback
- or it does not generalize enough to justify the added complexity

Exit criterion:

- a short final recommendation note exists with evidence, tradeoffs, and confidence level


## Specific Questions You Must Answer

Your work is not complete until you can answer these clearly.

1. Was the previous Gemma ASR evaluation fair?
2. Does `generate()` agree with the manual greedy loop on the same prompt?
3. Does prompt ordering materially change transcription quality?
4. Does explicit language wording materially change transcription quality?
5. Is the current reported aligner result attached to the runtime path actually used in the cascade?
6. What percentage of hybrid ticks use Gemma timings vs fallback timings?
7. Does the forced-calibrated head bundle remain best under the final intended prompt contract?
8. Is the alignment signal robust beyond one speaker in one talk?
9. What is the cleanest architecture we can defend in a paper right now?


## Metrics

Use these metrics at minimum.

### For ASR fairness

1. WER
2. CER
3. exact transcript output snapshots
4. prompt-template metadata
5. decode-path metadata

### For alignment

1. word-end MAE
2. median absolute error
3. P90
4. monotonicity
5. offset value used

### For streaming

1. mean drift stdev
2. median drift stdev
3. mean drift range
4. backward jumps
5. time-to-stable

### For hybrid integrity

1. fallback rate per tick
2. fallback rate overall
3. reasons for fallback
4. metrics with and without fallback contribution


## Testing Strategy

## Stage A: Single-Audio Cleanup

Do this first.

Goals:

- correct the runtime wiring
- audit fallback
- benchmark Gemma ASR fairly

Do not move on until the single-audio story is coherent.

## Stage B: Small Robustness Set

Only after Stage A is clean.

Goals:

- check whether the alignment signal persists across a few diverse clips

## Stage C: Full Cascade Comparison

Only after Stage B supports the method.

Goals:

- compare Qwen baseline vs audited hybrid on one talk
- only then consider broader evaluation


## Suggested File-Level Responsibilities

You will likely need to touch these files.

### Likely modifications

1. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)
   - fix default head bundle or force explicit selection
   - expose runtime config honestly
2. [hybrid_alignment_backend.py](/home/fuxa/cascade_simultaneous/hybrid_alignment_backend.py)
   - make fallback diagnostics explicit and durable
3. [gemma_alignment_probe.py](/home/fuxa/cascade_simultaneous/gemma_alignment_probe.py)
   - separate clean ASR benchmarking path from alignment path
   - add explicit long-audio guard
   - align runtime path with benchmarked calibration
4. [run_alignment_single_audio.py](/home/fuxa/cascade_simultaneous/run_alignment_single_audio.py)
   - preserve metadata for reproducible comparisons
5. [run_streaming_stability.py](/home/fuxa/cascade_simultaneous/run_streaming_stability.py)
   - include backend diagnostics and fallback accounting

### Likely additions

1. a dedicated Gemma ASR fairness benchmark script
2. a concise audit results note
3. possibly a small helper for metric aggregation


## Artifact Requirements

For every key comparison, save:

1. wav path
2. clip start/end or duration
3. prompt variant
4. decode path
5. model path
6. transcript text
7. word timings if applicable
8. backend diagnostics
9. fallback metadata if applicable

Do not rely on prose summaries alone.


## What Not To Do

Do not:

1. claim Gemma ASR failure without the fair comparison matrix
2. claim hybrid success without fallback accounting
3. use the old calibration with a new runtime prompt contract without revalidation
4. quietly keep the wrong default bundle
5. patch poor ASR with heuristic string repairs
6. run broad evaluation before the single-audio story is fixed


## Final Output Expected From You

At the end of this task, produce:

1. a short audit summary
2. a list of code changes made
3. a table for Gemma ASR fairness
4. a table for Gemma aligner validity
5. a hybrid fallback audit
6. a final recommendation:
   - proceed with Gemma-only
   - proceed with hybrid
   - or stop

The final recommendation should be blunt, honest, and paper-defensible.


## Immediate First Steps

Do these first, in order:

1. verify the current default head bundle vs the forced-calibrated bundle
2. instrument fallback visibility in the hybrid path and harnesses
3. build the fair Gemma ASR comparison matrix on one short clip
4. re-calibrate the aligner under the intended final prompt contract
5. only then decide whether to keep pushing Gemma ASR or to formalize the hybrid architecture

This is the shortest path to truth.
