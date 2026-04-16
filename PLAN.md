# PLAN.md
# Validate and Harden the Two-Pass Full-Gemma Path

## 1. Mission — STATUS: Phases 1-2, 4-6 Complete

### Completed

1. **Phase 1**: `run_gemma_two_pass_validation.py` rewritten as artifact-producing harness
2. **Phase 2**: Backend-level comparison on smoke18 — saved to `tmp/two_pass_validation/`
3. **Phase 4**: Harder clip (rxrToXvRyM_first18) — saved to `tmp/two_pass_validation/`
4. **Phase 5**: Stale repo claims cleaned in 4 files
5. **Phase 6**: Decision — current structure is adequate (see below)

6. **Phase 3**: Cascade-level comparison on smoke18 — saved to `tmp/two_pass_validation/`

### Results Summary

#### smoke18 (backend-level comparison)

| Metric | Two-Pass Gemma | Hybrid (Qwen+Gemma) |
|--------|---------------:|---------------------:|
| Word count | 35 | 35 |
| Runtime (s) | 7.9 | 1.2 |
| Monotonicity | 0.922 | 0.959 |
| Mean timing diff | — | 0.089s |
| Max timing diff | — | 0.680s |

Transcripts nearly identical. Two-pass: "Si Yuan" (correct). Hybrid: "Siyu Yuan" (correct).

#### smoke18 (cascade-level comparison)

| Metric | Two-Pass Gemma | Hybrid (Qwen+Gemma) |
|--------|---------------:|---------------------:|
| Updates | 12 | 13 |
| ASR | "Si Yuan", "While distinct script knowledge..." | "Siyu Yuan", "Distinguishing script knowledge..." |
| Translation | Truncated earlier (shorter DE output) | Longer DE output, closer to complete |

Both cascades produce sensible ASR and German translations. The hybrid
cascade produces slightly more complete output (13 vs 12 updates) and
better-preserved title semantics ("Distinguishing" vs "While distinct").
The two-pass cascade is viable but the ASR quality difference propagates
into MT: minor transcript variations cause different translation phrasing.

**Rule C consideration**: the bottleneck is not integration instability —
both paths run cleanly. The two-pass path produces slightly weaker
cascade output because of the ASR quality gap (entity spelling, title wording).

#### rxrToXvRyM_first18 (harder clip, two-pass only)

- 47 words, 4.9s runtime, monotone timestamps spanning 0.6–17.5s
- Transcript correct semantically; entity noise: "Mara" for "Myra", "Sándor Musch" for "Esin Durmus", "Dandrowski" for "Dan Jurafsky"
- Failure mode is **mild entity noise**, not semantic degradation or alignment breakdown

#### Decision (per PLAN.md Decision Rules)

**Rule A applies**: Two-pass full-Gemma works cleanly on smoke18.

**Rule B partially applies**: rxrToXvRyM_first18 shows entity-name noise (WER ~0.26)
but no hallucination, no alignment breakdown, and sensible timings. The result
is mixed but plausible — not a clear degradation.

**Phase 6 decision**: Keep current structure. `GemmaTwoPassAlignmentBackend` is a
clean 76-line wrapper. No deeper refactor needed until a real coupling bug appears.

### Artifacts

- `tmp/two_pass_validation/two_pass_smoke18.json`
- `tmp/two_pass_validation/hybrid_smoke18.json`
- `tmp/two_pass_validation/comparison_backend_smoke18.json`
- `tmp/two_pass_validation/comparison_cascade_smoke18.json`
- `tmp/two_pass_validation/cascade_two_pass_smoke18/` (full cascade output)
- `tmp/two_pass_validation/cascade_hybrid_smoke18/` (full cascade output)
- `tmp/two_pass_validation/two_pass_rxrToXvRyM_first18.json`


## 2. Current Ground Truth

Treat these as the current working facts unless you find a concrete contradiction in code or saved artifacts.

### 2.1 The old Gemma ASR failure story was mostly a measurement artifact

From the latest ablation work:

- [run_gemma_asr_fairness.py](/home/fuxa/cascade_simultaneous/run_gemma_asr_fairness.py)
- [ITERATION_RESULT.md](/home/fuxa/cascade_simultaneous/ITERATION_RESULT.md)
- [gemma_asr_fairness_ablation_smoke18.json](/home/fuxa/cascade_simultaneous/tmp/alignment_research/gemma_asr_fairness_ablation_smoke18.json)
- [gemma_asr_fairness_ablation_rxrToXvRyM_first18.json](/home/fuxa/cascade_simultaneous/tmp/alignment_research/gemma_asr_fairness_ablation_rxrToXvRyM_first18.json)

The main result is:

1. `attn_implementation="eager"` is the dominant cause of the catastrophic old ASR results
2. default attention gives much better transcripts
3. the earlier strong "domain mismatch" story is no longer defensible in its old form

### 2.2 A two-pass full-Gemma path now exists

The relevant implementation files are:

1. [gemma_two_pass_frontend.py](/home/fuxa/cascade_simultaneous/gemma_two_pass_frontend.py)
2. [gemma_alignment_probe.py](/home/fuxa/cascade_simultaneous/gemma_alignment_probe.py)
3. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)
4. [run_gemma_two_pass_validation.py](/home/fuxa/cascade_simultaneous/run_gemma_two_pass_validation.py)

### 2.3 What is still missing

The current implementation is promising, but not yet fully defensible as a repo-level result because:

1. the validation evidence is mostly written into `ITERATION_RESULT.md`
2. there is no strong saved JSON artifact for the new two-pass validation itself
3. the comparison so far is backend-level on one clip, not yet a clearly saved cascade-level comparison
4. stale text in older notes and core docstrings still conflicts with the new conclusion

So the next goal is not "invent something new".
The next goal is to make the current story coherent and auditable.


## 3. Design Principle for This Iteration

Assume the architectural insight is already correct:

1. default attention for Gemma ASR
2. eager attention for Gemma forced alignment
3. two passes when we want a full-Gemma front-end with timings

Do not spend this iteration rediscovering that.

Instead, make the implementation and evidence good enough that another researcher could inspect the repo and understand:

1. what the two-pass path is
2. how to run it
3. what evidence we have for it
4. how it compares to the hybrid path on one controlled example


## 4. Hard Constraints

### 4.1 Research integrity

1. No lexical repair hacks.
2. No content-specific transcript fixes.
3. No benchmark-specific substitutions.
4. No rhetorical overclaiming.
5. If a result is only shown on one clip, say that clearly.

### 4.2 Cost discipline

1. Do not do broad sweeps.
2. Do not run many full cascade evaluations.
3. Validate on one short clip first.
4. Add one harder follow-up clip only after the first result is saved cleanly.
5. Avoid restarting expensive hot models unless necessary.

### 4.3 Scope discipline

This iteration is about validation, comparison, and repo coherence.

Not the focus:

1. discovering new Gemma heads
2. broad robustness benchmarking
3. large-scale translation-quality studies
4. major paper writing
5. deep refactoring unless the current structure blocks validation


## 5. Files to Read First

### 5.1 Core instructions

1. [AGENTS.md](/home/fuxa/cascade_simultaneous/AGENTS.md)
2. [CLAUDE.md](/home/fuxa/cascade_simultaneous/CLAUDE.md)

### 5.2 New implementation and result files

1. [gemma_two_pass_frontend.py](/home/fuxa/cascade_simultaneous/gemma_two_pass_frontend.py)
2. [gemma_alignment_probe.py](/home/fuxa/cascade_simultaneous/gemma_alignment_probe.py)
3. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)
4. [run_gemma_two_pass_validation.py](/home/fuxa/cascade_simultaneous/run_gemma_two_pass_validation.py)
5. [ITERATION_RESULT.md](/home/fuxa/cascade_simultaneous/ITERATION_RESULT.md)

### 5.3 Existing comparison and hybrid context

1. [hybrid_alignment_backend.py](/home/fuxa/cascade_simultaneous/hybrid_alignment_backend.py)
2. [run_hybrid_audit.py](/home/fuxa/cascade_simultaneous/run_hybrid_audit.py)
3. [phase4_cascade_comparison.json](/home/fuxa/cascade_simultaneous/tmp/hybrid_audit/phase4_cascade_comparison.json)
4. [phase5_recommendation.md](/home/fuxa/cascade_simultaneous/tmp/hybrid_audit/phase5_recommendation.md)

### 5.4 Older notes that may now be stale

1. [PLAN_RESULT_IMPLEMENTATION.md](/home/fuxa/cascade_simultaneous/PLAN_RESULT_IMPLEMENTATION.md)
2. [PLAN_AUDIT_NOTE.md](/home/fuxa/cascade_simultaneous/PLAN_AUDIT_NOTE.md)


## 6. Primary Goal

Create a saved, auditable validation package for the two-pass full-Gemma path on one short audio.

That package should include:

1. transcript
2. word timings
3. diagnostics
4. runtime
5. comparison against hybrid on the same clip
6. enough metadata to reproduce the run

Right now the repo has the implementation but not yet the clean artifact story.


## 7. Secondary Goal

Run one real cascade-level comparison between:

1. `alignment_backend_name="gemma_two_pass"`
2. `alignment_backend_name="hybrid_qwen_asr_gemma_aligner"`

on the same short audio and save the outputs in a form that can be inspected later.

This comparison does not need to be broad.
It needs to be honest, saved, and interpretable.


## 8. Step-by-Step Work Plan

## Phase 1: Strengthen the validation script into an artifact-producing harness

### Objective

Turn [run_gemma_two_pass_validation.py](/home/fuxa/cascade_simultaneous/run_gemma_two_pass_validation.py) from a console-oriented script into a proper validation harness.

### Tasks

1. Make it write JSON artifacts to `tmp/...`.
2. Save at least:
   - clip id or wav path
   - transcript text
   - word list with times
   - diagnostics
   - runtime seconds
   - backend name
   - attention mode per pass
3. If comparison mode is enabled, also save the hybrid result in the same artifact or in a paired artifact.
4. Make the artifact schema obvious and stable enough for later comparison.

### Exit criterion

There is at least one saved JSON artifact for a two-pass run, not just printed output and prose.


## Phase 2: Run one saved backend-level comparison on `smoke18`

### Objective

Produce the cleanest first evidence on the easiest short clip already used in this line of work.

### Recommended clip

1. `tmp/alignatt_smoke18.wav`

### Tasks

1. Run the two-pass frontend and save the artifact.
2. Run the hybrid frontend on the same audio and save the artifact.
3. Save a direct comparison summary containing:
   - transcript text
   - word count
   - timing differences where comparable
   - runtime
   - diagnostics

### Important note

This is not yet the real final benchmark.
It is the first clean saved validation artifact.

### Exit criterion

The repo contains a machine-readable two-pass-vs-hybrid comparison on `smoke18`.


## Phase 3: Run one actual cascade-level comparison on the same short audio

### Objective

Move beyond backend-level comparison and check whether the real cascade behaves sensibly with the new frontend.

### Tasks

1. Run the actual cascade once with `alignment_backend_name="gemma_two_pass"`.
2. Run the actual cascade once with `alignment_backend_name="hybrid_qwen_asr_gemma_aligner"`.
3. Save outputs in a comparable way.
4. Record only the most relevant facts:
   - final transcript
   - final translation
   - number of updates
   - rough delay/runtime stats if already available
   - any obvious behavioral differences

### Discipline

Do this on one short audio only.
Do not fan out.

### Exit criterion

You have one real saved cascade-level comparison that can support or weaken the claim that the two-pass path is viable in practice.


## Phase 4: Probe one harder clip

### Objective

Check whether the new path still looks serious outside the most favorable short example.

### Recommended harder clip

1. `tmp/rxrToXvRyM_first18.wav`

### Tasks

1. Run the two-pass frontend on the harder clip.
2. Save the artifact.
3. If cost is manageable, compare once against hybrid at backend level.
4. Focus on whether the failure mode is:
   - mild entity noise
   - real semantic degradation
   - alignment breakdown
   - runtime instability

### Exit criterion

We know whether the two-pass path still looks plausible on a less forgiving clip.


## Phase 5: Clean the stale repo story

### Objective

Bring the repo’s written claims back into alignment with the corrected Gemma evidence.

### Files likely needing updates

1. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)
2. [PLAN_RESULT_IMPLEMENTATION.md](/home/fuxa/cascade_simultaneous/PLAN_RESULT_IMPLEMENTATION.md)
3. [PLAN_AUDIT_NOTE.md](/home/fuxa/cascade_simultaneous/PLAN_AUDIT_NOTE.md)
4. [ITERATION_RESULT.md](/home/fuxa/cascade_simultaneous/ITERATION_RESULT.md)

### What to fix

1. remove or qualify stale statements that say Gemma ASR is simply unreliable on this audio
2. distinguish clearly between:
   - old eager-tainted ASR result
   - corrected default-attention ASR result
   - current status of the two-pass full-Gemma path
3. avoid claiming broader robustness than we have actually shown

### Exit criterion

A reader no longer encounters conflicting stories in code comments and notes.


## Phase 6: Decide whether a deeper refactor is now justified

### Objective

Only after the saved evidence exists, decide whether the current structure is good enough or whether a deeper split is worth doing.

### Question to answer

Should we now split Gemma ASR runtime and Gemma alignment runtime into more explicit classes, or is the current implementation already clean enough for the next research step?

### Guidance

Do not refactor for aesthetics alone.
Refactor only if the current coupling between `transcribe()` and the alignment backend still creates a real risk of confusion or future misuse.

### Exit criterion

You can justify either:

1. keep current structure for now
2. do one deeper explicit runtime split next


## 9. Expected Deliverables

### Code deliverables

1. improved [run_gemma_two_pass_validation.py](/home/fuxa/cascade_simultaneous/run_gemma_two_pass_validation.py) that saves artifacts
2. any minimal supporting changes needed to make saved comparison artifacts clean
3. documentation cleanup in the most relevant stale files

### Artifact deliverables

1. one saved two-pass validation artifact on `smoke18`
2. one saved backend-level comparison artifact on `smoke18`
3. one saved cascade-level comparison artifact on `smoke18`
4. one saved two-pass artifact on `rxrToXvRyM_first18`

### Documentation deliverables

One concise updated result note stating:

1. what was run
2. what artifacts were produced
3. whether two-pass Gemma still looks serious after real saved comparisons
4. what should happen next


## 10. Decision Rules

### Rule A

If the two-pass path looks good on the saved backend-level and cascade-level `smoke18` comparisons, keep it as a serious mainline research path.

### Rule B

If it works well on `smoke18` but weakens sharply on `rxrToXvRyM_first18`, state that clearly and frame the conclusion as mixed rather than general.

### Rule C

If the backend-level result is good but the real cascade-level result is awkward or unstable, say the bottleneck is integration/runtime behavior rather than Gemma ASR quality.

### Rule D

If hybrid remains clearly better in practice, the conclusion should be:

1. hybrid remains the baseline
2. two-pass full Gemma is promising but not yet the best deployment tradeoff
3. that conclusion is now based on corrected evidence rather than the old eager-tainted ASR result


## 11. Suggested Execution Order

Keep the iteration narrow and disciplined.

1. upgrade the validation script to save artifacts
2. save the two-pass `smoke18` result
3. save the backend-level `smoke18` comparison versus hybrid
4. save one actual cascade-level `smoke18` comparison
5. run one harder two-pass clip
6. clean stale docs
7. decide whether a deeper class split is the next iteration

Do not scale beyond that unless the mechanism is already clearly working.


## 12. Final Standard

At the end of this iteration, another researcher should be able to inspect the repo and understand:

1. that a two-pass full-Gemma path exists
2. what saved evidence supports it
3. how it compares to the hybrid path on one real short example
4. what remains uncertain
5. what the next rational step should be

That is the bar for this iteration.
