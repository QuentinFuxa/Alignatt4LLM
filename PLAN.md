# Gemma-Only Aligner Research Plan

## Mission

Build and evaluate a **Gemma-only source-timestamp alignment mechanism** that can replace the current dependency on `Qwen3-ForcedAligner-0.6B` in the streaming cascade.

The target is not "some timestamps." The target is a mechanism we could defend in a paper:

- principled
- architecture-aware
- measurable
- compatible with streaming partial updates
- useful for downstream AlignAtt-controlled emission

The end goal is a clean front-end stack where Gemma provides:

- source transcription
- source timing / alignment signal
- enough streaming stability to support the current latency-aware MT cascade


## Read This First

This plan is written for a specialized agent working inside this repository.

Before changing code, the agent should internalize the following constraints:

- We are in a research phase, so redesign is allowed when it improves the system.
- We do not want ad hoc lexical repairs, punctuation-specific hacks, or benchmark-tuned exceptions.
- We want a mechanism we could explain honestly in a methods section.
- Full model reloads are expensive.
- Broad evaluation sweeps are expensive.
- The preferred workflow is:
  1. validate on one audio
  2. iterate until the mechanism behaves correctly
  3. only then scale out


## Documents To Read, In Order

### Local project documents

Read these first:

1. [AGENTS.md](/home/fuxa/cascade_simultaneous/AGENTS.md)
2. [CLAUDE.md](/home/fuxa/cascade_simultaneous/CLAUDE.md)
3. [assets/alignatt_doc/E4B_ALIGNATT_CASCADE_DESIGN.md](/home/fuxa/cascade_simultaneous/assets/alignatt_doc/E4B_ALIGNATT_CASCADE_DESIGN.md)
4. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)
5. [Qwen3_aligner.md](/home/fuxa/cascade_simultaneous/Qwen3_aligner.md)
6. [gemma_stt.py](/home/fuxa/cascade_simultaneous/gemma_stt.py)

Then inspect supporting local AlignAtt material referenced by the design doc:

1. `assets/alignatt_doc/alignatt_markdown.md`
2. `assets/alignatt_doc/alignatt_whipser.py`

The specialized agent should treat the current cascade code and design doc as the ground truth for:

- what timestamps are used for
- which invariants matter
- what "success" means in the full system

### External documents

The agent should read these before proposing architecture changes:

1. Gemma audio capability docs
   - https://ai.google.dev/gemma/docs/capabilities/audio
2. Gemma Hugging Face inference docs
   - https://ai.google.dev/gemma/docs/core/huggingface_inference
3. Gemma model cards and processor/model implementation on Hugging Face / Transformers
   - verify how audio tokens are represented
   - verify whether the model uses pure causal self-attention over audio+text tokens or a different multimodal block structure
4. Qwen3 ASR and Qwen forced aligner docs / model cards
   - especially the forced aligner interface and assumptions
5. Whisper / AlignAtt references
   - the original AlignAtt writeup or code used in this repo
   - Whisper timestamp / attention alignment literature if needed
6. Any Gemma architecture or multimodal internals references needed to understand:
   - hidden states
   - attention maps
   - token/audio packing
   - generation-time access to attentions


## Core Background: What The Current System Actually Needs

The current system uses Qwen for more than "ASR text."

It uses Qwen outputs to obtain:

1. a partial transcript for the current tail audio
2. word-level timestamps for that partial transcript
3. sentence-end cut times for committed source segments
4. timestamp-based source accessibility frontiers for the downstream AlignAtt MT policy

This is the crucial distinction.

The problem is not merely:

- "Can Gemma transcribe audio?"

The real problem is:

- "Can Gemma provide a stable, word-level temporal alignment signal suitable for partial streaming commitment and downstream source-frontier control?"

That is the actual research target.


## What Must Be Preserved

Any Gemma-only aligner proposal must be judged against the needs of the current cascade.

At minimum, it must support:

1. Word-level or unit-level end timestamps for partial hypotheses.
2. Stable enough timestamps to cut committed utterances.
3. A source accessibility frontier compatible with the current AlignAtt MT pipeline.
4. Partial-update behavior that does not create pathological churn.
5. A monotone-enough notion of progress that the system can emit text safely.

If a proposed method only gives:

- full-utterance timestamps after the fact
- coarse clip-level timing
- unstable attention maps with no calibration

then it is not yet sufficient.


## High-Level Research Hypothesis

The most promising research path is:

- derive a timestamp signal from **Gemma's own internal representations**
- calibrate and validate it against text-audio alignment targets
- expose it through a reusable alignment backend

The likely candidates are:

1. attention-based alignment from generated text tokens to audio token positions
2. lightweight learned alignment heads on top of frozen Gemma hidden states
3. distillation from the current Qwen aligner into a Gemma-based alignment module

The plan should investigate them in that order of increasing complexity, but with a strict rule:

- do not keep a weak method just because it is simpler
- prefer the cleanest method that actually satisfies the downstream invariants


## Explicit Non-Goals

Do not solve this by:

- punctuation heuristics standing in for alignment
- lexical anchoring rules
- hand-written phrase tables
- dataset-specific repairs
- content-aware timestamp nudges
- forcing the MT side to compensate for missing alignment quality

Also do not quietly redefine success from:

- "usable streaming aligner"

to:

- "produces some plausible-looking timestamps"


## Success Criteria

The project is successful only if all of the following are eventually true.

### Alignment quality

- Word or unit timestamps are meaningfully correlated with actual speech timing.
- Boundary error is low enough to support clean utterance cutting.
- Timestamps remain usable on partial updates, not only on final transcripts.

### Streaming behavior

- The aligner produces stable enough progress across repeated partial decodes.
- Timestamp drift under incremental context extension remains bounded.
- Prefix churn is low enough not to destroy downstream emission quality.

### Downstream usefulness

- The source accessibility frontier built from Gemma-only timings supports the current AlignAtt cascade.
- Translation emission remains monotone enough to be usable.
- Latency-quality tradeoffs are at least competitive with the current setup, or the failure mode is clearly characterized.

### Research defensibility

- The method can be described as a generic mechanism.
- The method does not depend on ad hoc examples or benchmark patches.
- The evaluation is clean and reproducible.


## Deliverables

The specialized agent should aim to produce the following deliverables.

### Required deliverables

1. A short design note describing the final Gemma-only aligner method.
2. A clean alignment backend interface in code.
3. An experimental harness for single-audio evaluation.
4. A metrics script or notebook for timestamp and streaming stability evaluation.
5. A downstream cascade integration path behind a flag.
6. A failure analysis note if the approach does not work.

### Strongly preferred deliverables

1. Saved diagnostic artifacts:
   - attention maps
   - predicted alignments
   - source/token timing traces
   - partial-update traces over time
2. A comparison table against:
   - Qwen aligner baseline
   - no-aligner or coarse baseline
3. A clear ablation story.


## Proposed Work Breakdown

## Phase 0: Orientation And Reproducibility

Objective:

- understand the current codepath and isolate the minimum alignment contract Gemma must satisfy

Tasks:

1. Read the documents listed above in the specified order.
2. Trace the current alignment data flow in [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py):
   - ASR call
   - timestamp normalization
   - utterance boundary cut
   - source frontier construction
   - downstream MT use
3. Write down the exact data structure currently consumed by the cascade:
   - text
   - per-word start/end timestamps
   - current audio time
   - accessible unit count
4. Identify the narrowest interface that a Gemma aligner must implement.

Exit criterion:

- a clear interface spec exists for "alignment backend output"


## Phase 1: Build A Clean Alignment Abstraction

Objective:

- decouple alignment from Qwen so experiments are possible without contaminating the cascade

Tasks:

1. Introduce an internal abstraction such as:
   - `ASRBackend`
   - `AlignmentBackend`
   - or a combined `SpeechFrontEndResult` with pluggable providers
2. Ensure the cascade does not assume the aligner is Qwen-specific.
3. Preserve the current Qwen path as the baseline implementation.
4. Add a Gemma experimental path behind a clear runtime flag.

Important:

- This refactor should be structural, not cosmetic.
- The alignment representation should be generic:
  - source text units
  - timestamps
  - confidence or diagnostics
  - partial/final mode metadata

Exit criterion:

- the current Qwen setup still exists as a baseline backend, and Gemma experiments can plug into the same contract


## Phase 2: Establish The Strong Baselines

Objective:

- know exactly what must be beaten or matched

Tasks:

1. Define the baseline front ends:
   - `Qwen ASR + Qwen aligner`
   - `Gemma ASR + Qwen aligner`
   - `Gemma ASR only` with no timestamp-aware frontier, if useful as a negative control
2. On one carefully chosen audio, collect:
   - transcript text
   - word timestamps
   - utterance cut points
   - partial-update traces
3. Save artifacts for later comparison.
4. Measure downstream effects on the existing cascade.

Why this phase matters:

- It separates "Gemma transcription quality" from "Gemma alignment quality."
- It prevents the project from confusing ASR failures with alignment failures.

Exit criterion:

- there is a single-audio baseline bundle that can be reused for every later comparison


## Phase 3: Gemma Internal Instrumentation

Objective:

- determine what internal signal is even available for alignment

Tasks:

1. Inspect the Gemma Hugging Face implementation and verify:
   - exact model class being used
   - how audio is encoded
   - whether audio is converted into discrete tokens, embeddings, or projected frames
   - where those representations enter the model
   - how generated text attends to the audio-derived context
2. Verify what can be extracted during generation:
   - per-layer attentions
   - per-head attentions
   - hidden states
   - logits over time
3. Confirm whether there is a stable mapping from audio encoder positions to real time.
4. Determine whether there is a direct conversion from:
   - audio token index
   - frame index
   - latent chunk index
   to milliseconds.

Critical question:

- Are there audio positions with usable temporal semantics, or only a pooled/opaque conditioning representation?

If the answer is "opaque only," then a pure attention-argmax aligner is much less plausible, and a learned head may be required earlier.

Exit criterion:

- a technical note exists explaining which internal Gemma tensors are candidates for alignment


## Phase 4: Attention-Probing Prototype

Objective:

- test the lightest plausible Gemma-only aligner first

Method hypothesis:

- generated transcription tokens may attend in a structured way to temporally ordered audio representations
- selected heads and layers may reveal monotone alignment similar in spirit to AlignAtt

Tasks:

1. Implement a diagnostic path that runs Gemma transcription while collecting attentions.
2. For each generated text token:
   - isolate its attention to audio-derived positions only
   - aggregate over candidate heads/layers
   - compute an aligned audio position
3. Convert aligned audio positions into milliseconds.
4. Aggregate token-level alignments into word-level timestamps.
5. Save visualizations:
   - layer/head heatmaps
   - token-to-time traces
   - monotonicity plots

Initial aggregation recipes to try:

1. raw per-head argmax on audio positions
2. mean over heads, then argmax
3. selected-head average with median filtering on the source axis
4. cumulative or monotonic smoothing over token positions

This phase is exploratory, but still principled:

- no manual per-example rules
- no lexical correction
- no punctuation-only alignment shortcuts

Exit criterion:

- evidence for or against usable attention-derived alignment on one audio


## Phase 5: Head Selection And Calibration

Objective:

- turn noisy attention probing into a reproducible alignment method if Phase 4 shows promise

Tasks:

1. Search for heads/layers that yield the strongest monotone audio-text correspondence.
2. Define a generic selection criterion, such as:
   - monotonicity score
   - correlation with external alignment labels
   - word boundary consistency
3. Decide whether head selection is:
   - fixed globally
   - fixed per language
   - fixed per model size
4. Apply generic smoothing / filtering if justified:
   - median filter
   - monotonic regression
   - local non-decreasing constraint
5. Calibrate audio-position to milliseconds if required.

Important constraint:

- If a calibration step is used, it must be generic and measurable.
- Do not hide poor alignment under hand-tuned post-processing.

Exit criterion:

- a fully specified attention-based alignment recipe exists, with fixed head/layer selection and a documented scoring rationale


## Phase 6: Lightweight Learned Aligner On Top Of Gemma

Objective:

- move to a stronger method if raw attention is insufficient

This phase becomes the main path if:

- attention-based alignment is too noisy
- the time mapping is unstable
- partial-update consistency is too weak

Candidate clean methods:

1. Frozen Gemma + alignment probe
   - input:
     - Gemma hidden states at audio positions
     - Gemma hidden states at generated text positions
   - output:
     - token-to-audio alignment distribution
     - or token end-time prediction
2. Frozen Gemma + monotonic alignment head
   - enforce monotone structure directly in the predictor
3. Distillation head
   - train on targets produced by Qwen forced aligner, possibly filtered for confidence

Tasks:

1. Define the prediction target:
   - token-aligned frame index
   - word end time
   - token-to-frame distribution
2. Decide the supervision source:
   - Qwen forced aligner pseudo-labels
   - external forced-alignment labels if available
   - manual spot checks for calibration
3. Build a minimal train/eval dataset.
4. Keep the model lightweight and modular.
5. Prefer frozen Gemma with a small trainable probe before considering end-to-end finetuning.

Research preference:

- A small, explicit alignment head on top of frozen Gemma is more defensible than an opaque pile of heuristics.

Exit criterion:

- there is a learned Gemma-based aligner prototype that produces explicit timestamp predictions


## Phase 7: Partial Streaming Stability

Objective:

- test the property that matters most for the cascade: repeated partial updates

Tasks:

1. Simulate the current streaming pattern:
   - repeatedly extend available audio
   - rerun partial transcription/alignment
   - compare the predicted timings across updates
2. Record:
   - transcript prefix stability
   - word timestamp drift
   - utterance cut drift
   - frontier accessible-unit drift
3. Measure whether the aligner supports stable commitment under:
   - sentence growth
   - clause completion
   - punctuation appearance
4. Identify failure modes such as:
   - large backward jumps
   - jitter on the last few words
   - unstable cut points near punctuation
   - early or late unlocking of source units

Exit criterion:

- the agent knows whether the Gemma-only aligner is stable enough for streaming use, not just offline timestamping


## Phase 8: Downstream Cascade Integration

Objective:

- determine whether the aligner is actually good enough where it matters

Tasks:

1. Plug the Gemma-only aligner into the current source frontier builder.
2. Reuse the existing AlignAtt MT stack unchanged as much as possible.
3. Compare against the Qwen baseline on one audio first.
4. Evaluate:
   - emitted translation stability
   - latency to first useful target words
   - target churn
   - pathological truncation
   - sentence boundary behavior
5. Only after a convincing single-audio result, expand to more clips.

This phase is essential:

- A front-end aligner that looks reasonable in isolation but fails to support source-frontier control is not yet a success.

Exit criterion:

- the agent can show either:
  - usable downstream behavior
  - or a precise reason the method fails


## Phase 9: Ablations And Paper-Quality Evidence

Objective:

- produce a defensible empirical story

Required comparisons:

1. `Qwen ASR + Qwen aligner`
2. `Gemma ASR + Qwen aligner`
3. `Gemma ASR + Gemma attention aligner`
4. `Gemma ASR + Gemma learned aligner`
5. Optional negative control:
   - `Gemma ASR` with no timestamp-aware alignment

Required ablations:

1. head selection on vs off
2. smoothing on vs off
3. monotonic constraint on vs off
4. learned head with and without Qwen pseudo-label distillation
5. partial-update evaluation vs final-only evaluation

Required outputs:

1. tables
2. a few clean visualizations
3. error analysis on representative cases


## Metrics

The specialized agent should not rely on one metric.

Use at least the following metric families.

### Alignment metrics

1. Word end-time MAE
2. Word end-time median absolute error
3. P90 / P95 timestamp error
4. Boundary detection error for utterance cuts
5. Correlation between predicted and reference timing order

### Streaming stability metrics

1. Prefix stability across partial updates
2. Timestamp drift for already-seen words
3. Number and magnitude of backward timing jumps
4. Accessible frontier churn
5. Time-to-stable-word metric

### Downstream cascade metrics

1. Emitted target prefix stability
2. Latency to first accepted translation material
3. Over-eager emission rate
4. Over-conservative blocking rate
5. Human-readable qualitative trace on one clip

### Research diagnostics

1. Head monotonicity score
2. Alignment sharpness / entropy
3. Calibration error from latent position to milliseconds
4. Failure mode counts by category


## Reference Labels And Testing Strategy

## Stage A: Single-Audio Deep Dive

This is the default mode until the mechanism works.

Tasks:

1. Pick one audio clip with:
   - clean speech
   - enough sentence structure to test boundaries
   - enough length to observe partial updates
2. Produce a complete artifact bundle:
   - waveform
   - final transcript
   - Qwen word timestamps
   - Gemma transcript
   - partial-update traces
   - candidate Gemma alignments
3. Diagnose failures manually.

This stage is where most iteration should happen.

## Stage B: Small Controlled Set

Only after Stage A is convincing.

Tasks:

1. Expand to a small, diverse set:
   - different speakers
   - different rates
   - punctuation-rich and punctuation-poor cases
   - at least some harder segments
2. Keep the set small enough for fast iteration.

## Stage C: Broader Evaluation

Only when the mechanism is already close to usable.

Tasks:

1. Run the broader comparison table.
2. Prepare paper-ready figures and examples.


## Where Supervision Should Come From

The cleanest supervision hierarchy is:

1. external human-labeled forced alignment data if available
2. Qwen forced aligner pseudo-labels
3. manual spot checks for sanity, not for per-example tuning

If using Qwen pseudo-labels:

- treat them as teacher labels, not ground truth
- quantify agreement and disagreement
- do not overclaim "true" alignment quality from teacher-matching alone


## Recommended Research Order

This is the recommended order of attack.

1. Decouple alignment backend from Qwen.
2. Build baseline comparisons on one audio.
3. Instrument Gemma internals.
4. Try attention-based alignment first.
5. If attention is weak, move quickly to a frozen-Gemma learned aligner.
6. Evaluate streaming stability before scaling out.
7. Integrate into the full cascade only after the aligner is credible in isolation.

This order keeps the project honest and avoids wasting time polishing a mechanism that cannot satisfy the true contract.


## Concrete Questions The Agent Must Answer

The project is not complete until the specialized agent can answer these clearly.

1. What exactly are Gemma's audio-time-bearing internal positions?
2. Can generated transcript tokens be aligned reliably to those positions?
3. Are there specific layers/heads with stable monotone alignment behavior?
4. Can word-level end timestamps be recovered with low enough error?
5. Are partial-update timestamps stable enough for utterance cutting?
6. Does the resulting source frontier behave well enough for AlignAtt MT?
7. If the answer is no, where exactly does the failure happen?
   - transcription instability
   - weak attention semantics
   - poor time calibration
   - streaming drift
   - downstream integration mismatch


## Suggested Code Organization

The agent should prefer explicit modularity.

Suggested components:

1. `alignment_backend.py`
   - shared interfaces / dataclasses
2. `gemma_alignment_probe.py`
   - attention extraction and diagnostics
3. `gemma_alignment_calibration.py`
   - generic calibration logic if needed
4. `gemma_alignment_model.py`
   - lightweight learned head or probe
5. `alignment_eval.py`
   - metrics and artifact generation
6. `alignment_debug_artifacts.py`
   - plots / trace dumping

The exact filenames may differ, but the responsibilities should stay separated.


## Suggested Artifact Schema

The agent should persist enough information for later analysis.

For each run, save:

1. source audio id
2. transcript text
3. token list
4. word list
5. predicted token timestamps
6. predicted word timestamps
7. reference timestamps if available
8. layer/head diagnostics
9. partial-update index
10. current visible audio duration
11. any confidence or entropy signals

This matters because the hardest part of alignment work is failure diagnosis.


## Risks And Likely Failure Modes

The agent should expect these failure modes.

### Architecture risk

Gemma audio internals may not expose temporally localizable positions strongly enough for direct attention alignment.

### Generation risk

Generated token attention may be diffuse or semantically useful without being temporally faithful.

### Streaming risk

A method that works for final transcripts may be too unstable for repeated partial updates.

### Supervision risk

Teacher-distilled alignments may replicate Qwen biases rather than uncover genuine Gemma structure.

### Integration risk

Even decent timestamps may not be precise enough for source-frontier control in the downstream MT policy.


## Decision Rules

The specialized agent should make crisp go/no-go decisions.

### Continue with raw attention if:

- head/layer patterns are clearly structured
- time calibration is straightforward
- partial-update drift is manageable

### Escalate to a learned aligner if:

- raw attention is noisy
- monotonicity is poor
- word timing is unstable
- downstream frontier behavior is unusable

### Stop the Gemma-only path if:

- Gemma internals do not provide enough temporal structure
- even a learned lightweight probe cannot support streaming stability
- the method becomes too contrived to defend cleanly

If the project stops, the agent should document the negative result clearly. That is still useful.


## What A Strong Final Outcome Looks Like

A strong outcome would be:

- Gemma transcribes audio
- Gemma provides alignment via a generic attention-based or lightweight learned mechanism
- the cascade no longer depends on Qwen's external forced aligner
- the system remains compatible with the current timestamp-based source frontier
- the method is simple enough to explain and strong enough to evaluate rigorously

A weaker but still publishable outcome could be:

- raw Gemma attention is insufficient
- a small learned aligner on top of frozen Gemma works
- the result clarifies what multimodal LLM internals do and do not provide for streaming alignment


## Immediate First Actions For The Specialized Agent

If starting from scratch, do these first:

1. Read the local documents in the order listed above.
2. Write down the exact current alignment contract from the cascade code.
3. Refactor the code so alignment is a backend rather than a Qwen assumption.
4. Build the one-audio baseline artifact bundle.
5. Instrument Gemma attentions and verify whether there is a meaningful audio-position axis.

Do not start with broad benchmark runs.
Do not start with heuristics.
Do not start by tuning the MT side.
Start by isolating the alignment problem and making it observable.


## Final Reminder

This project should be approached as a serious systems-and-representation question:

- can a multimodal LLM's own internals support streaming-quality source alignment?

That is the paper-worthy question.

The implementation should make that question sharper, not blurrier.

When everything is correctly done, you can stop the Ralph loop with 'I meet all the Success Criterias !'