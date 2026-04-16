# PLAN.md
# Next Agent Brief: Build a Small Dedicated Aligner on Frozen Gemma Audio Features

## 1. Mission

We now have two important facts:

1. Gemma ASR itself is viable when run with default attention.
2. Gemma's current alignment path works, but it depends on running the full model in eager attention mode, which is expensive and not the right long-term deployment story.

The next step is to build the most promising Qwen-independent successor to the current eager-attention aligner:

**a small dedicated transcript-conditioned aligner on top of frozen Gemma audio features.**

This should become the main non-Qwen research path.

The objective is not to make Gemma attention a little prettier.
The objective is to replace the expensive eager second pass with a purpose-built aligner that is:

1. faster than the current eager Gemma forced-alignment pass
2. at least as stable or better in timestamps
3. fully independent from Qwen at inference time
4. principled and defensible in a paper


## 2. Why This Is the Right Next Design

The repo now has three relevant front-end stories:

1. `qwen`
   - practical baseline
   - Qwen ASR + Qwen aligner
2. `hybrid_qwen_asr_gemma_aligner`
   - Qwen ASR + Gemma eager-attention alignment
   - strong practical research baseline
3. `gemma_two_pass`
   - Gemma ASR + Gemma eager-attention forced alignment
   - Qwen-independent, but expensive

The current weakness of `gemma_two_pass` is not that Gemma ASR is unusable.
The weakness is that alignment still requires the full Gemma model in eager mode.

That suggests the clean next move:

- keep Gemma ASR
- replace eager alignment with a lightweight dedicated aligner trained on Gemma-side representations

This is more promising than continuing to rely on raw LM attention forever.


## 3. Target Outcome

By the end of this iteration, we want the repo to contain the first working version of:

- a **Gemma-feature aligner**
- trained or at least scaffolded to predict transcript-to-audio timing
- usable as a drop-in replacement for the current eager Gemma alignment pass in an offline alignment setting

The first success criterion is not full streaming deployment.
The first success criterion is:

1. given audio + transcript
2. produce monotone word timings
3. with a clean architecture and measurable offline quality


## 4. What Exactly To Build

Build **Design 1** from the previous discussion:

### Core design

1. Run Gemma's audio front-end / multimodal encoder stack to obtain frozen audio features.
2. Feed those audio features, plus the known transcript, into a small dedicated aligner network.
3. Predict aligned positions or word-end times for transcript units.
4. Project them to monotone word timings.

### Important scope choice

This aligner is **not** Gemma ASR.
It is **not** a separate speech recognizer.
It is a transcript-conditioned forced aligner.

So its input is:

1. audio
2. known transcript

And its output is:

1. word timings or subword timings
2. aggregated into the repo's existing `WordAlignment` format


## 5. Hard Constraints

### 5.1 Research integrity

1. No lexical repair rules.
2. No content-specific transcript fixes.
3. No hand-coded word timing heuristics tied to specific examples.
4. No hidden fallback to Qwen at inference time in the new path.
5. If you bootstrap training labels from Qwen, that is acceptable for training only, but must be stated clearly.

### 5.2 Architectural cleanliness

1. Keep the aligner small and clearly separate from Gemma itself.
2. Do not tangle training scaffolding into the runtime path.
3. Make it obvious which parts are frozen Gemma and which parts are the new aligner.
4. Preserve the current backends; add a new experimental path rather than destabilizing the existing ones.

### 5.3 Cost discipline

1. Do not launch large training sweeps immediately.
2. Start with one or a few clips just to prove the pipeline works.
3. Prefer proving the representation and target pipeline before spending time on optimization.
4. Treat full cascade runs as out of scope until offline alignment works.


## 6. Files to Read First

### 6.1 Core repo instructions

1. [AGENTS.md](/home/fuxa/cascade_simultaneous/AGENTS.md)
2. [CLAUDE.md](/home/fuxa/cascade_simultaneous/CLAUDE.md)

### 6.2 Current Gemma alignment implementation

1. [gemma_alignment_probe.py](/home/fuxa/cascade_simultaneous/gemma_alignment_probe.py)
2. [gemma_two_pass_frontend.py](/home/fuxa/cascade_simultaneous/gemma_two_pass_frontend.py)
3. [alignment_backend.py](/home/fuxa/cascade_simultaneous/alignment_backend.py)
4. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)

### 6.3 Existing evaluation harnesses and artifacts

1. [run_alignment_single_audio.py](/home/fuxa/cascade_simultaneous/run_alignment_single_audio.py)
2. [run_gemma_two_pass_validation.py](/home/fuxa/cascade_simultaneous/run_gemma_two_pass_validation.py)
3. [ITERATION_RESULT.md](/home/fuxa/cascade_simultaneous/ITERATION_RESULT.md)
4. [two_pass_smoke18.json](/home/fuxa/cascade_simultaneous/tmp/two_pass_validation/two_pass_smoke18.json)
5. [two_pass_rxrToXvRyM_first18.json](/home/fuxa/cascade_simultaneous/tmp/two_pass_validation/two_pass_rxrToXvRyM_first18.json)
6. [comparison_backend_smoke18.json](/home/fuxa/cascade_simultaneous/tmp/two_pass_validation/comparison_backend_smoke18.json)
7. [phase3_robustness_summary.json](/home/fuxa/cascade_simultaneous/tmp/hybrid_audit/phase3_robustness_summary.json)

### 6.4 Existing teacher/timestamp sources

1. [qwen_alignment_backend.py](/home/fuxa/cascade_simultaneous/qwen_alignment_backend.py)
2. [smoke18_qwen_teacher.json](/home/fuxa/cascade_simultaneous/tmp/alignment_research/smoke18_qwen_teacher.json)
3. any other existing teacher bundles already produced in `tmp/alignment_research/`


## 7. Recommended First Architecture

Start simple.

### 7.1 Feature source

Use **frozen Gemma audio features**.

You need to decide which feature tap is the cleanest first target.
Recommended order of preference:

1. final audio encoder features before they are mixed into the text model
2. audio token embeddings after projection into the shared model space
3. if needed, a small stack of layerwise features, but only if clearly justified

Prefer one feature source first.
Do not start with multi-layer fusion unless the single-feature baseline clearly fails.

### 7.2 Transcript units

Start with **token- or subword-level alignment targets** that can be aggregated to words.

Why:

1. easier to supervise densely
2. easier to align with the existing `aggregate_token_timings_to_words` style logic
3. less fragile than predicting whole-word spans from scratch at the first attempt

### 7.3 Aligner model

Start with a **small transcript-conditioned aligner**, for example:

1. transcript embedding layer
2. 2-4 layers of cross-attention or lightweight transformer blocks
3. output head that scores audio positions for each transcript token

The key property is monotone alignability, not model cleverness.

### 7.4 Output representation

Recommended first target:

- predict a discrete distribution over audio positions for each transcript token

Then:

1. choose peak or expected position
2. enforce monotonicity
3. convert positions to seconds
4. aggregate token timings to words

This is the simplest first version that matches the existing pipeline.


## 8. Supervision Strategy

### 8.1 Recommended first supervision

Bootstrap from existing timestamp teachers.

Use one of these, in this order:

1. Qwen forced-aligner timestamps
2. current Gemma eager forced-alignment timestamps

Recommended first choice:

- **Qwen forced-aligner timestamps** as training targets

Why:

1. they are already available in the repo
2. they are cleaner than raw eager-attention peaks
3. they let the new aligner learn a stable target even if Gemma attention is noisy

### 8.2 Important rule

Using Qwen timestamps for **training supervision** is acceptable.
Using Qwen at **inference time** in the new backend is not.

State that distinction clearly in code and notes.

### 8.3 First training data regime

Start tiny.

1. prove the pipeline on 1 clip
2. then 3-5 clips
3. only then worry about a larger supervision set

The first milestone is pipeline correctness, not benchmark breadth.


## 9. What the Agent Must Produce

## Phase 1: Build the feature extraction pipeline

### Objective

Make Gemma audio features available cleanly for alignment training and inference.

### Tasks

1. Identify the correct point in the Gemma stack to extract frozen audio features.
2. Build a reusable function or module that returns:
   - audio features
   - feature-frame timing scale
   - any needed metadata for mapping positions to seconds
3. Save one artifact for one clip showing:
   - feature shape
   - audio duration
   - feature-step duration or equivalent

### Exit criterion

We can reproducibly extract Gemma audio features for one clip and map feature indices back to time.


## Phase 2: Build the first dedicated aligner module

### Objective

Create a small aligner network that consumes frozen audio features plus transcript tokens.

### Tasks

1. Add a new module for the aligner.
2. Keep it small and explicit.
3. Define its input contract and output contract clearly.
4. Implement inference for one audio+transcript pair.

### Recommended output contract

For each transcript token:

1. predicted audio position
2. confidence or score if easy to expose

### Exit criterion

There is a working model object that can run forward on one `(audio_features, transcript)` pair.


## Phase 3: Build the supervision and training harness

### Objective

Train the first version of the dedicated aligner on a tiny dataset.

### Tasks

1. Build a tiny dataset generator from existing repo artifacts or teacher outputs.
2. Use Qwen timestamps as first supervision.
3. Train on a very small number of examples first.
4. Save:
   - model checkpoint
   - training config
   - a small JSON summary of loss and settings

### Important note

Do not spend this iteration on fancy hyperparameter search.
The first question is whether the concept works at all.

### Exit criterion

A small trained aligner checkpoint exists and can be run offline.


## Phase 4: Convert predictions into `WordAlignment`

### Objective

Make the new aligner useful inside the existing repo abstractions.

### Tasks

1. Convert token-level predictions into time values.
2. Enforce monotonicity.
3. Aggregate to words.
4. Return the existing `AlignmentResult` / `WordAlignment` contract.

### Exit criterion

The new aligner can produce a standard repo-compatible alignment result for one known transcript.


## Phase 5: Offline evaluation on one clip

### Objective

Get the first honest quality read.

### Recommended first clip

1. `tmp/alignatt_smoke18.wav`

### Compare against

1. Qwen teacher timestamps
2. current Gemma eager forced alignment

### Metrics to report

1. MAE
2. median error
3. P90 error
4. monotonicity
5. inference runtime

### Exit criterion

We know whether the first dedicated aligner is at least in the right regime.


## Phase 6: Small extension to a few clips

### Objective

Check whether the result is real and not just clip memorization.

### Tasks

1. Evaluate on 3-5 short clips max.
2. Keep the report compact.
3. Focus on whether quality and runtime are promising enough to justify another iteration.

### Exit criterion

We can tell whether this is a live research path or a dead end.


## 10. Integration Goal for Later, Not First

Do **not** try to fully integrate into the streaming cascade in the first implementation pass.

The correct order is:

1. offline forced alignment works
2. repo-compatible outputs exist
3. quality/runtime are promising
4. then add a new runtime backend using the dedicated aligner

If you integrate too early, debugging will become much harder.


## 11. Suggested New Files

You do not have to use exactly these names, but the separation should look like this.

1. `gemma_audio_features.py`
   - feature extraction from frozen Gemma audio path
2. `gemma_feature_aligner.py`
   - small dedicated aligner model
3. `run_gemma_feature_aligner_train.py`
   - tiny training harness
4. `run_gemma_feature_aligner_eval.py`
   - offline evaluation harness
5. optional small note documenting the first result

Keep training/eval code separate from runtime backend code.


## 12. Deliverables

### Code deliverables

1. frozen Gemma audio feature extraction utility
2. small dedicated aligner module
3. tiny training harness
4. offline evaluation harness
5. repo-compatible alignment conversion path

### Artifact deliverables

1. one saved feature-inspection artifact
2. one saved trained-checkpoint summary
3. one offline evaluation artifact on `smoke18`
4. one small multi-clip evaluation artifact

### Documentation deliverables

One concise result note stating:

1. what feature source was used
2. what supervision was used
3. what the first runtime and MAE numbers are
4. whether this path is promising enough to continue


## 13. Decision Rules

### Rule A

If the dedicated aligner is clearly faster than eager Gemma alignment and lands in the same approximate MAE regime, this becomes the new main Qwen-independent alignment path.

### Rule B

If it is much faster but somewhat worse, that is still promising.
The next iteration should focus on improving quality.

### Rule C

If it is accurate but not materially faster, the path is still interesting, but the justification becomes architectural cleanliness rather than runtime advantage.

### Rule D

If it is both slower and worse than eager Gemma alignment, stop and reassess before integrating it further.


## 14. Final Standard

At the end of this iteration, another researcher should be able to inspect the repo and understand:

1. how Gemma audio features are extracted
2. how the new dedicated aligner works
3. what supervision it used
4. how it performs offline on a few clips
5. whether it is a credible successor to the eager Gemma alignment pass

That is the bar for this iteration.
