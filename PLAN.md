# Current Plan: Audited Hybrid Gemma-Aligner Path

## Mission

Take the project from:

- "interesting alignment result with some experimental ambiguity"

to:

- "audited, defensible hybrid front-end with clear failure accounting"

The current evidence no longer supports spending primary effort on
Gemma-only free-run ASR. The current evidence does support continuing on
the **hybrid** path:

- Qwen3-ASR for transcript text
- Gemma attention for source timings

The next work should make that hybrid result honest, robust enough to
judge, and useful for the full cascade.


## Current Status

These points should now be treated as established unless new evidence
contradicts them.

### Established

1. The runtime integrity issues were addressed.
   - default head bundle now points to the forced-calibrated file
   - missing calibrated bundle now fails loudly
   - hybrid fallback is now exposed in diagnostics
   - streaming harness now records fallback usage
   - long-audio silent truncation is now guarded

2. The fair Gemma ASR check was done on `smoke18`.
   - `generate()` and the manual greedy loop agree under matched settings
   - prompt-order and wording changes were tested
   - the bad free-run Gemma ASR result on that clip does **not** appear to be a simple implementation bug

3. The strongest aligner result remains the teacher-forced, text-first path.
   - the forced-calibrated text-first bundle is the correct runtime bundle for the current aligner
   - the audio-first forced-alignment ablation is materially worse

4. The most defensible current architecture is hybrid.
   - Gemma-only ASR is not the right next objective
   - audited hybrid is the right next objective


## Read First

Before doing more work, read:

1. [AGENTS.md](/home/fuxa/cascade_simultaneous/AGENTS.md)
2. [CLAUDE.md](/home/fuxa/cascade_simultaneous/CLAUDE.md)
3. [PLAN_RESULT_IMPLEMENTATION.md](/home/fuxa/cascade_simultaneous/PLAN_RESULT_IMPLEMENTATION.md)
4. [PLAN_AUDIT_NOTE.md](/home/fuxa/cascade_simultaneous/PLAN_AUDIT_NOTE.md)
5. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)
6. [gemma_alignment_probe.py](/home/fuxa/cascade_simultaneous/gemma_alignment_probe.py)
7. [hybrid_alignment_backend.py](/home/fuxa/cascade_simultaneous/hybrid_alignment_backend.py)
8. [run_streaming_stability.py](/home/fuxa/cascade_simultaneous/run_streaming_stability.py)
9. [run_gemma_asr_fairness.py](/home/fuxa/cascade_simultaneous/run_gemma_asr_fairness.py)


## Non-Goals

Do not spend time on:

1. further prompt tinkering for `gemma-4-E4B-it` free-run ASR on the current clip set
2. heuristic text repair for Gemma ASR
3. broad benchmark sweeps before the hybrid runtime pass is clean
4. pretending the project is Gemma-only when the evidence currently points to hybrid


## Remaining Open Questions

These are now the real unresolved issues.

### 1. What is the actual fallback rate of the hybrid path?

We now have the instrumentation, but we do not yet have the decisive
runtime answer on a real talk:

- how many ticks use Gemma timings?
- how many fall back to Qwen timings?
- why?

This is the most important next measurement.

### 2. Does the aligner generalize beyond the current very narrow evidence?

The current evidence is still too local:

- one main clip
- one same-talk continuation
- same language

We need a tiny but deliberately diverse robustness set.

### 3. What is the real downstream impact on the full cascade?

Front-end alignment metrics are promising, but the actual question is:

- what happens to the translation cascade?

We still need one honest end-to-end comparison:

- `qwen` baseline
- vs `hybrid_qwen_asr_gemma_aligner`

### 4. Do we want stricter failure behavior in evaluation mode?

The hybrid backend currently catches Gemma exceptions and falls back.
That is good for robustness, but can hide implementation bugs during
research evaluation.

We likely need a strict mode for evaluation.

### 5. Should the audio cap be derived from model/processor config?

The explicit cap is much better than silent truncation, but it is still
effectively a fixed E4B assumption. That should ideally be derived from
the active model configuration.


## Success Criteria For The Next Iteration

The next iteration is successful only if all of the following are true:

1. We know the hybrid fallback rate on one real talk.
2. We know whether fallback is rare, common, or dominant.
3. We have a small robustness table across 3–5 clips.
4. We have one cascade-level comparison against the Qwen baseline.
5. We can state clearly whether the hybrid path is worth adopting as the research baseline.


## Work Plan

## Phase 1: Tighten Evaluation Hygiene

Objective:

- make the runtime evaluation path maximally trustworthy before running expensive experiments

Tasks:

1. Add or verify a strict evaluation mode for the hybrid backend.
   - expected runtime failures may still be handled explicitly
   - unexpected implementation bugs should be able to surface loudly
2. Improve diagnostics where needed so every hybrid tick can be audited.
3. If practical, derive the Gemma audio cap from model/processor config rather than a fixed constant.

Exit criterion:

- evaluation mode cannot silently hide important Gemma-side failures


## Phase 2: Run The Hybrid Fallback Audit

Objective:

- answer whether the hybrid path is truly using Gemma alignment often enough to matter

Tasks:

1. Run the streaming stability harness on one real talk using the hybrid backend.
2. Collect:
   - `fallback_aware_ticks`
   - `gemma_used_ticks`
   - `fallback_ticks`
   - `fallback_rate`
   - `fallback_reasons`
3. Save the artifact bundle and summarize the result.

Key decision rule:

- If fallback is frequent enough that the hybrid result is mostly Qwen timings, we should not overclaim Gemma’s contribution.

Exit criterion:

- we have one auditable fallback report on a real talk


## Phase 3: Small Robustness Check

Objective:

- determine whether the forced-calibrated Gemma aligner is local or real

Use a small set only:

1. `smoke18`
2. one different segment from the same talk
3. one or two different speakers/accents if available
4. optionally one cleaner and one harder clip

Tasks:

1. Reuse the current forced-calibrated runtime path.
2. Measure:
   - word-end MAE
   - median
   - P90
   - monotonicity
   - streaming drift if practical
3. Record whether the same head bundle remains acceptable without retuning.

Important:

- do not recalibrate per clip just to make the numbers look nice
- the point is to judge transfer, not to optimize every example

Exit criterion:

- we know whether the current aligner setup is robust enough to keep


## Phase 4: One Cascade-Level Comparison

Objective:

- determine the actual downstream value of the hybrid front-end

Tasks:

1. Run one talk with:
   - `alignment_backend_name = "qwen"`
   - `alignment_backend_name = "hybrid_qwen_asr_gemma_aligner"`
2. Compare downstream translation behavior using the metrics already used in this project.
3. Inspect whether any delta is explained by:
   - better timing
   - worse timing
   - fallback behavior
   - instability

Exit criterion:

- we have one honest end-to-end comparison instead of only front-end proxy metrics


## Phase 5: Architecture Decision

Objective:

- decide what should become the working research baseline

Choose one of the following.

### Option A: Adopt hybrid as the baseline

Choose this if:

- fallback rate is acceptable
- robustness is decent
- the cascade-level result is competitive enough

### Option B: Keep hybrid as exploratory only

Choose this if:

- the aligner is interesting but too brittle
- fallback is too frequent
- or downstream benefit is too small

### Option C: Re-open Gemma-only work later

Only choose this if new evidence emerges.

Do not treat Gemma-only ASR as the main line of work right now.

Exit criterion:

- a short recommendation note exists with confidence and tradeoffs


## Concrete Commands To Run Next

These are the natural next expensive runs.

### 1. Hybrid fallback audit on one talk

Use the streaming harness with the forced-calibrated bundle and save the report.

Goal:

- quantify real Gemma-vs-fallback usage

### 2. Tiny robustness set

Run the current forced-calibrated aligner on 3–5 clips.

Goal:

- find out whether the current result is robust or local

### 3. One full cascade comparison

Run one talk through:

- Qwen baseline
- audited hybrid

Goal:

- decide whether the hybrid path matters downstream


## Expected Deliverables

At the end of the next iteration, produce:

1. one fallback audit note
2. one small robustness table
3. one cascade comparison note
4. one final recommendation:
   - adopt hybrid as the baseline
   - or keep it exploratory


## Final Guidance

The project has crossed an important threshold:

- the main uncertainty is no longer "did we accidentally misuse Gemma ASR?"

The main uncertainty is now:

- "is the hybrid front-end materially useful, robust enough, and honest enough to become the baseline?"

That is the next question to answer.
