# PLAN.md
# Next Agent Brief: Build the Clean Two-Pass Full-Gemma Front-End

## 1. Mission

The last iteration resolved the core Gemma ASR discrepancy.

Gemma E4B ASR is not broadly broken on our conference clips.
The previous catastrophic ASR result was mainly caused by evaluating Gemma
through an eager-attention path that is appropriate for alignment extraction
but harmful for free-run transcription.

The next task is therefore not another fairness audit.
The next task is to turn that finding into the clean architecture that should
have existed from the start:

1. use Gemma with default attention for ASR
2. use Gemma with eager attention for forced alignment
3. expose this as a principled two-pass full-Gemma front-end
4. compare it honestly against the existing hybrid baseline on one audio

This iteration is about redesigning the front-end around the newly-correct
understanding of Gemma, not about piling on more diagnosis.


## 2. Current Ground Truth

Treat the following as established unless you find a concrete implementation
error in the new ablation harness.

### 2.1 Gemma ASR discrepancy is resolved

From the latest ablations:

- [run_gemma_asr_fairness.py](/home/fuxa/cascade_simultaneous/run_gemma_asr_fairness.py)
- [ITERATION_RESULT.md](/home/fuxa/cascade_simultaneous/ITERATION_RESULT.md)
- [gemma_asr_fairness_ablation_smoke18.json](/home/fuxa/cascade_simultaneous/tmp/alignment_research/gemma_asr_fairness_ablation_smoke18.json)
- [gemma_asr_fairness_ablation_rxrToXvRyM_first18.json](/home/fuxa/cascade_simultaneous/tmp/alignment_research/gemma_asr_fairness_ablation_rxrToXvRyM_first18.json)

The main result is:

1. `attn_implementation="eager"` destroys free-run Gemma ASR on these clips
2. default attention produces much better transcripts
3. audio filepath vs numpy array does not materially matter here
4. greedy vs sampled does not materially change the main conclusion

### 2.2 Alignment backend still legitimately needs eager attention

The current Gemma alignment path uses eager attention because the recorder
needs materialized attention weights.

That means the repo currently conflates two different runtime regimes:

1. ASR regime: default attention
2. alignment regime: eager attention

This conflation is the architectural smell we now need to remove.

### 2.3 Updated architectural opportunity

A clean full-Gemma front-end is now plausible via two passes:

1. Pass A: Gemma free-run ASR with default attention
2. Pass B: Gemma forced alignment with eager attention, using the transcript
   from Pass A

This removes the Qwen ASR dependency while keeping word-level timing.


## 3. The Design Goal

Build the front-end we would design if we had known from day one that:

1. Gemma ASR wants default attention
2. Gemma alignment extraction wants eager attention
3. those are different responsibilities and should not share one muddy runtime path

The resulting design should be:

1. explicit
2. auditable
3. reusable
4. defensible in a paper
5. easy to compare against the hybrid baseline

Do not just patch the old code until it works.
Separate responsibilities cleanly.


## 4. Hard Constraints

### 4.1 Research integrity

1. No lexical repairs.
2. No content-aware transcript fixes.
3. No hardcoded substitutions for names or titles.
4. No benchmark-specific heuristics.
5. No narrative claims that outrun the evidence.

### 4.2 Cost discipline

1. Do not restart hot models unless necessary.
2. Do not run broad sweeps.
3. Validate on one audio first.
4. Treat full cascade runs as expensive.

### 4.3 Scope discipline

This iteration is about making the architecture correct.

Not the focus:

1. new alignatt-head discovery
2. broad robustness campaigns
3. paper writing
4. many-audio benchmarking

Those can come after the two-pass front-end exists and behaves correctly on one audio.


## 5. Files to Read First

### 5.1 Instructions and context

1. [AGENTS.md](/home/fuxa/cascade_simultaneous/AGENTS.md)
2. [CLAUDE.md](/home/fuxa/cascade_simultaneous/CLAUDE.md)

### 5.2 New discrepancy resolution

1. [ITERATION_RESULT.md](/home/fuxa/cascade_simultaneous/ITERATION_RESULT.md)
2. [run_gemma_asr_fairness.py](/home/fuxa/cascade_simultaneous/run_gemma_asr_fairness.py)
3. [standalone_gemma_asr_test.py](/home/fuxa/cascade_simultaneous/standalone_gemma_asr_test.py)
4. [gemma_asr_fairness_ablation_smoke18.json](/home/fuxa/cascade_simultaneous/tmp/alignment_research/gemma_asr_fairness_ablation_smoke18.json)
5. [gemma_asr_fairness_ablation_rxrToXvRyM_first18.json](/home/fuxa/cascade_simultaneous/tmp/alignment_research/gemma_asr_fairness_ablation_rxrToXvRyM_first18.json)

### 5.3 Existing alignment/front-end code

1. [gemma_alignment_probe.py](/home/fuxa/cascade_simultaneous/gemma_alignment_probe.py)
2. [alignment_backend.py](/home/fuxa/cascade_simultaneous/alignment_backend.py)
3. [hybrid_alignment_backend.py](/home/fuxa/cascade_simultaneous/hybrid_alignment_backend.py)
4. [qwen_alignment_backend.py](/home/fuxa/cascade_simultaneous/qwen_alignment_backend.py)
5. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)

### 5.4 Older notes that now need reinterpretation

1. [PLAN_RESULT_IMPLEMENTATION.md](/home/fuxa/cascade_simultaneous/PLAN_RESULT_IMPLEMENTATION.md)
2. [PLAN_AUDIT_NOTE.md](/home/fuxa/cascade_simultaneous/PLAN_AUDIT_NOTE.md)


## 6. Main Architectural Problem to Solve

The current code path was effectively built around one Gemma object trying to
serve two conflicting roles:

1. transcript generator
2. attention recorder for alignment

That is the wrong abstraction.

The clean redesign should make the split explicit.

### Desired conceptual split

1. `GemmaAsrBackend`
   - owns free-run transcription
   - uses default attention
   - does not expose alignment internals

2. `GemmaForcedAlignmentBackend` or equivalent role inside the existing Gemma backend
   - owns transcript-to-audio forced alignment
   - uses eager attention
   - does not pretend to be the best path for free-run ASR

3. `GemmaTwoPassFrontend` or equivalent integration path
   - runs ASR first
   - runs forced alignment second
   - returns text plus word timing in the same contract expected by the cascade

You do not have to use exactly these class names.
But the role separation should be this clean.


## 7. Primary Goal

Implement a principled two-pass Gemma front-end that can replace the current
Qwen-ASR-plus-Gemma-align front-end for at least one controlled experiment.

The two-pass path should:

1. produce Gemma transcript with default attention
2. produce word timings by Gemma forced alignment with eager attention
3. expose outputs in the same structural form the cascade already expects


## 8. Secondary Goal

Compare the new full-Gemma two-pass front-end against the current hybrid
baseline on one audio, focusing on:

1. transcript quality
2. timing behavior
3. whether the cascade logic still functions correctly
4. rough cost implications

Do not over-benchmark.
One audio is enough for this iteration if the mechanism is clear.


## 9. Step-by-Step Work Plan

## Phase 1: Refactor the Gemma responsibilities cleanly

### Objective

Remove the conceptual mistake that one Gemma path should serve both ASR and
alignment under the same attention configuration.

### Tasks

1. Inspect how Gemma is currently loaded and used in [gemma_alignment_probe.py](/home/fuxa/cascade_simultaneous/gemma_alignment_probe.py).
2. Identify the minimum clean factoring needed to separate:
   - default-attention ASR
   - eager-attention forced alignment
3. Refactor toward explicit responsibility boundaries rather than scattered flags.

### Design requirement

The code should make it hard to accidentally benchmark free-run ASR through the
eager alignment path again.

### Exit criterion

The code structure clearly distinguishes ASR runtime and alignment runtime.


## Phase 2: Build the two-pass full-Gemma path

### Objective

Create a real reusable path, not just a notebook-style experiment.

### Tasks

1. Implement the ASR pass using the correct default-attention multimodal path.
2. Implement the forced-alignment pass using the existing eager-attention machinery.
3. Connect them so a transcript from pass 1 feeds pass 2.
4. Return a result object compatible with the cascade’s existing expectations:
   - transcript text
   - per-word or per-token timings
   - any needed diagnostics

### Important requirement

Do not duplicate large chunks of loosely-related inference code if you can
factor shared prompt/input handling cleanly.

### Exit criterion

There is one callable full-Gemma two-pass front-end that can run on a single clip.


## Phase 3: Integrate it at the cascade boundary

### Objective

Make the new front-end usable inside the actual cascade code path without
breaking the abstraction.

### Tasks

1. Inspect how the current cascade expects ASR outputs and timing information in [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py).
2. Add the minimum integration needed to plug in the two-pass Gemma front-end.
3. Keep the existing hybrid path available for comparison.
4. Make the runtime choice explicit and auditable.

### Exit criterion

The cascade can be run, at least in a controlled way, with either:

1. hybrid front-end
2. two-pass full-Gemma front-end


## Phase 4: Validate on one audio

### Objective

Show that the new front-end actually behaves like a viable front-end and not
just a code refactor.

### Tasks

Pick one short audio first.
Recommended starting point:

1. `tmp/alignatt_smoke18.wav`

On that audio, evaluate:

1. free-run Gemma transcript quality
2. forced-alignment output quality
3. whether the produced timings look structurally sane
4. whether the cascade consumes them correctly

### Important discipline

This is not the moment for a broad benchmark.
If the mechanism fails here, fix the mechanism before expanding.

### Exit criterion

The one-audio run demonstrates a functioning two-pass full-Gemma front-end.


## Phase 5: Compare against hybrid on one audio

### Objective

Answer the only comparison question that matters right now:

Is the new full-Gemma two-pass path good enough to be taken seriously against
`Qwen ASR + Gemma aligner`?

### Tasks

Compare on one audio:

1. hybrid front-end
2. full-Gemma two-pass front-end

Focus on:

1. transcript quality
2. timing quality or stability indicators
3. whether downstream cascade behavior stays sensible
4. rough runtime overhead of the second pass

### Exit criterion

You can write one honest paragraph saying whether full Gemma is:

1. already competitive enough to keep pursuing
2. promising but currently too expensive
3. still clearly behind hybrid in practice


## 10. Non-Goals for This Iteration

Unless the two-pass front-end is already working cleanly, do not spend major effort on:

1. new head search
2. multi-speaker robustness campaigns
3. translation-quality benchmark sweeps
4. prompt tinkering
5. paper polish


## 11. Expected Deliverables

### Code deliverables

1. cleanly factored Gemma ASR vs Gemma forced-alignment runtime separation
2. one implemented two-pass full-Gemma front-end path
3. minimal cascade integration for controlled comparison

### Artifact deliverables

1. one result artifact for the two-pass full-Gemma run
2. one comparison artifact versus hybrid on the same audio
3. enough diagnostics to show which attention mode was used in each pass

### Documentation deliverables

Update the most appropriate result note with:

1. what was implemented
2. whether the two-pass full-Gemma path worked
3. how it compared to hybrid on the one-audio check
4. what the next recommendation is

Prefer updating an existing note over scattering conclusions across many files.


## 12. Decision Rules

### Rule A

If the two-pass full-Gemma front-end works cleanly and the outputs are strong,
reopen full Gemma as a serious mainline research path.

### Rule B

If it works but is clearly slower or more awkward than hybrid, say so honestly.
It can still be a valuable no-Qwen alternative.

### Rule C

If it fails because the forced-alignment pass or cascade integration is fragile,
do not hide that behind ASR quality claims.
State that the bottleneck is integration or runtime cost, not Gemma ASR itself.

### Rule D

If hybrid remains better, the conclusion should now be:

1. hybrid wins on practicality
2. not because Gemma ASR was fundamentally bad
3. but because the two-pass full-Gemma path is not yet the best deployment tradeoff


## 13. Suggested Execution Order

Keep the iteration narrow.

1. refactor Gemma runtime responsibilities
2. implement the two-pass front-end
3. plug it into the cascade boundary
4. run one audio
5. compare once against hybrid
6. write the updated recommendation

Do not scale before step 4 succeeds.


## 14. Final Standard

At the end of this iteration, another researcher should be able to inspect the
repo and understand:

1. why Gemma ASR and Gemma alignment require different attention modes
2. how the code now reflects that separation cleanly
3. how to run a full-Gemma two-pass front-end
4. whether that front-end is competitive enough to justify further work

That is the bar for this iteration.
