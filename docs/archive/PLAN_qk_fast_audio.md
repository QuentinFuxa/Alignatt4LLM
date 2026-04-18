# PLAN_qk_fast_audio.md
# Dedicated Agent Brief: Port `qk_fast` AlignAtt to Gemma Audio Alignment Under `sdpa`

## 1. Mission

Build the **audio-side analogue** of the MT `qk_fast` AlignAtt probe.

The goal is to remove the current dependence on `eager` attention for Gemma audio-text forced alignment by reconstructing the needed source-attention rows under `sdpa`, using the same general strategy already used on the MT side.

This is the highest-priority next experiment because it is the cleanest, most paper-defensible path:

1. no separate trained aligner required
2. no Qwen supervision required at inference time
3. no eager-attention runtime requirement if it works
4. closest to the real AlignAtt idea: extract alignment from the model's own internals efficiently

The current trained feature aligner remains a fallback path, not the preferred one.


## 2. Critical Runtime Constraint

**The GPU is currently being used. Do not run anything.**

That means:

1. do not load Gemma
2. do not run evaluation scripts
3. do not generate teacher artifacts
4. do not benchmark
5. do not run training
6. do not run any model-backed smoke tests

This iteration is a **static implementation pass only**.

You should:

1. think carefully
2. read the relevant code
3. design the cleanest solution
4. implement the code paths
5. leave clear hooks and notes for the later runtime validation pass

But **run nothing**.


## 3. Why This Path Matters

Current situation:

1. Gemma ASR works with default attention / `sdpa`
2. Gemma raw-attention forced alignment currently works only because the code forces `eager`
3. MT AlignAtt already has a `qk_fast` path under `sdpa`

That strongly suggests the following possibility:

- the audio alignment task may also be recoverable under `sdpa` by reconstructing the selected head rows from layer inputs + KV information, instead of reading `attn_weights` directly

If true, this would likely be the best Qwen-independent path:

- more elegant than the trained feature aligner
- more direct than the current eager-attention audio aligner
- likely faster than the current eager audio aligner
- much easier to defend than “we trained a tiny student to imitate Qwen timestamps”


## 4. Current Ground Truth

Treat these as established working facts.

### 4.1 Gemma audio alignment today

Current Gemma audio forced alignment lives in:

- [cascade/alignment/gemma_transformers_asr_backend.py](/home/fuxa/cascade_simultaneous/cascade/alignment/gemma_transformers_asr_backend.py)

The current implementation:

1. teacher-forces the transcript into the assistant turn
2. identifies the audio-token span
3. captures transcript-token attention into the audio span
4. uses calibrated alignment heads + offset to convert peaks into word timings

Today, this path uses `eager` because `SelectedAttentionRecorder` expects actual attention weights to be materialized.

### 4.2 Gemma ASR attention modes

Also in [cascade/alignment/gemma_transformers_asr_backend.py](/home/fuxa/cascade_simultaneous/cascade/alignment/gemma_transformers_asr_backend.py):

1. ASR is now run with default attention (`sdpa`-like path)
2. eager attention was shown to damage free-run ASR badly

So the current split is:

1. ASR -> default attention
2. alignment -> eager attention

The whole point of this project is to remove the second requirement if possible.

### 4.3 MT already has a `qk_fast` path

The MT backend in:

- [cascade/mt/base.py](/home/fuxa/cascade_simultaneous/cascade/mt/base.py)

already supports a fast AlignAtt probe under `sdpa` using:

1. layer-input capture
2. prompt KV snapshot reuse
3. query/key reconstruction
4. selected-head row extraction without materialized full attentions

This is the key precedent.

### 4.4 Qwen uses a separate aligner on purpose

Qwen does **not** rely on “alignment heads from Qwen ASR” in the shipped repo path.
It uses a dedicated aligner model:

- [qwen_cascade/alignment/base.py](/home/fuxa/cascade_simultaneous/qwen_cascade/alignment/base.py)
- [Qwen3_aligner.md](/home/fuxa/cascade_simultaneous/Qwen3_aligner.md)

So if we can make the Gemma `qk_fast` path work, that would be a genuinely novel and stronger Gemma-native story.


## 5. Files You Must Read First

Read these before changing anything.

### 5.1 Repo instructions

1. [AGENTS.md](/home/fuxa/cascade_simultaneous/AGENTS.md)
2. [CLAUDE.md](/home/fuxa/cascade_simultaneous/CLAUDE.md)

### 5.2 Audio aligner implementation

1. [cascade/alignment/gemma_transformers_asr_backend.py](/home/fuxa/cascade_simultaneous/cascade/alignment/gemma_transformers_asr_backend.py)
2. [gemma_two_pass_frontend.py](/home/fuxa/cascade_simultaneous/gemma_two_pass_frontend.py)
3. [cascade/alignment/base.py](/home/fuxa/cascade_simultaneous/cascade/alignment/base.py)
4. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)

### 5.3 MT fast-path AlignAtt implementation

1. [cascade/mt/base.py](/home/fuxa/cascade_simultaneous/cascade/mt/base.py)

Pay special attention to:

1. `SelectedLayerInputRecorder`
2. `_temporary_attention_implementation(...)`
3. `_run_model(...)`
4. `_probe_source_attention_rows_qk_fast(...)`
5. `extract_source_attention_rows_per_token_from_fast_path(...)`
6. any helper functions used to reconstruct query/key states from captured layer inputs and KV caches

### 5.4 Current results context

1. [ITERATION_RESULT.md](/home/fuxa/cascade_simultaneous/ITERATION_RESULT.md)
2. [PLAN_RESULT_IMPLEMENTATION.md](/home/fuxa/cascade_simultaneous/PLAN_RESULT_IMPLEMENTATION.md)
3. [PLAN_AUDIT_NOTE.md](/home/fuxa/cascade_simultaneous/PLAN_AUDIT_NOTE.md)

The notes may contain superseded conclusions, but they still provide context for why the code is structured as it is.


## 6. What You Are Trying To Build

Build the audio-side equivalent of:

- MT `qk_fast` AlignAtt under `sdpa`

Concretely, the desired end state is:

1. Gemma audio forced alignment can run in a fast path that does **not** require eager attention
2. it reconstructs the selected audio-attention rows from captured layer inputs and KV/cache information
3. it plugs into the existing calibrated head + offset + monotone aggregation pipeline
4. it is selectable explicitly, not silently mixed with the eager path

This should be implemented as a **new probe path**, not as a silent mutation of the current eager path.


## 7. Design Principle

Do not try to invent a brand-new alignment algorithm.

Reuse the current Gemma forced-alignment pipeline as much as possible:

1. same transcript-forcing contract
2. same audio-span detection
3. same calibrated head files
4. same aggregation into token positions
5. same conversion to word timings

Only replace the piece that currently depends on materialized `attn_weights`.

In other words:

- keep the alignment logic
- replace the attention extraction backend


## 8. Main Technical Hypothesis

The MT `qk_fast` path works because it does not need explicit attention matrices; it reconstructs selected attention rows from:

1. captured layer inputs
2. prompt KV snapshots
3. runtime suffix/past KV information
4. selected head indices

The audio alignment task is structurally similar enough that the same idea may work.

The core question is:

> can we reconstruct transcript-token attention into the audio-token span under `sdpa` without requiring eager attention?

This is what your implementation should aim to answer.


## 9. Likely Technical Differences vs MT

Do not assume the MT code can be copied unchanged.

You must think through these differences carefully.

### 9.1 Source definition differs

MT AlignAtt probes attention from generated target tokens into a **textual source span in the prompt**.

Audio forced alignment probes attention from transcript tokens into an **audio-token span** embedded in the multimodal prompt.

So the source mapping logic is different:

1. source positions are audio-token positions, not source-text token positions
2. the transcript tokens live in a teacher-forced assistant span, not in autoregressive draft output only

### 9.2 Prompt structure differs

The audio forced-alignment input is multimodal and includes:

1. text-before-audio ordering in the forced-alignment path
2. an assistant prefill transcript span

That means prompt/suffix boundaries may differ from the MT assumptions.

### 9.3 Probe target differs

MT probes target draft tokens.
Audio forced alignment probes the teacher-forced transcript tokens in the assistant span.

So you may need a dedicated audio-side `qk_fast` probe function rather than directly reusing the MT draft-token replay flow.

### 9.4 Audio tower interaction differs

The audio prompt involves audio features processed by Gemma's multimodal stack before the text LM sees the relevant sequence positions.

You need to reason carefully about which sequence positions correspond to the audio-token span available to the text model's self-attention.


## 10. Recommended Implementation Strategy

## Phase 1: Understand and isolate the reusable MT machinery

### Objective

Identify what part of the MT `qk_fast` path is generic and what part is MT-specific.

### Questions to answer

1. Which helper functions are already generic enough to reuse on the audio side?
2. Which assumptions are specific to source-text prompt layouts?
3. Which pieces should be moved or duplicated for audio use?

### Deliverable

A clear internal design decision in code structure:

1. reused helper(s)
2. audio-specific helper(s)
3. minimal duplication with maximal clarity

### Strong preference

Prefer extracting generic helper functions only if it actually improves clarity.
Do not force refactoring if it makes the code harder to follow.


## Phase 2: Add an audio-side layer-input fast-path probe

### Objective

Implement an audio equivalent of `extract_source_attention_rows_per_token_from_fast_path(...)`.

### Requirements

It should:

1. operate on captured layer inputs rather than full attentions
2. use selected alignment heads
3. reconstruct per-transcript-token rows over the audio-token positions
4. return the same kind of row tensors the current downstream code already expects

### Likely new function

Something like:

- `extract_audio_attention_rows_per_token_from_fast_path(...)`

Name can differ, but it should be explicit and not overloaded with MT semantics.

### Important

Keep the output contract aligned with the existing downstream utilities, so the rest of the pipeline can remain unchanged.


## Phase 3: Add a fast probe mode to the Gemma audio aligner

### Objective

Make the audio aligner able to choose between:

1. eager attention extraction
2. `qk_fast` extraction

### Recommended runtime config

Add a config flag similar in spirit to the MT side, for example:

- `gemma_audio_align_probe_mode = "eager" | "qk_fast"`

Use a clear name.
Do not piggyback on the MT config unless there is a very good reason.

### Requirements

1. default behavior should remain safe and explicit
2. the old eager path must remain available
3. the new path must be easy to select for later validation


## Phase 4: Keep the downstream alignment pipeline unchanged where possible

### Objective

Minimize conceptual surface area.

Once the new probe returns per-token source rows, the existing path should still handle:

1. head aggregation
2. argmax / source-position extraction
3. monotonicity enforcement
4. offset correction
5. token-to-word aggregation

If you find yourself rewriting all of that, you are probably going too far.


## Phase 5: Add diagnostics hooks for later runtime validation

### Objective

Since you cannot run anything now, leave the next agent a clean validation path.

### Add diagnostics such as

1. which probe backend was selected
2. whether fast-path reconstruction succeeded
3. any obvious unsupported conditions
4. if relevant, fallback reason metadata when `qk_fast` cannot be used

But do **not** add automatic silent fallback unless it is extremely well-justified.
Explicit is better here.


## 11. File-Touch Guidance

You will probably need to touch at least:

1. [cascade/alignment/gemma_transformers_asr_backend.py](/home/fuxa/cascade_simultaneous/cascade/alignment/gemma_transformers_asr_backend.py)
2. [qwen3asr_gemma_cascade_core.py](/home/fuxa/cascade_simultaneous/qwen3asr_gemma_cascade_core.py)

You may also need to touch:

1. [cascade/mt/base.py](/home/fuxa/cascade_simultaneous/cascade/mt/base.py)

But only if extracting genuinely generic helper code improves clarity.

Be careful here:

- MT is working
- do not destabilize it casually

If the cleanest option is to duplicate a small amount of helper logic into the audio path, that is acceptable.


## 12. What Success Looks Like in Code

Even without running anything, by the end of this pass the repo should have:

1. a new audio fast-path probe implementation
2. a clear runtime switch between eager and `qk_fast`
3. an unchanged or minimally changed downstream word-timing pipeline
4. explicit diagnostics / comments explaining the intended validation path
5. no accidental hidden dependence on eager in the new path


## 13. What Not To Do

1. Do not run any GPU code.
2. Do not start validating with real audio.
3. Do not train anything.
4. Do not integrate the current trained feature-aligner here.
5. Do not redesign the whole repo around this.
6. Do not silently remove the eager path.
7. Do not turn this into a giant cleanup unrelated to the audio `qk_fast` objective.


## 14. Suggested Deliverables

### Code deliverables

1. audio-side `qk_fast` attention-row extraction
2. Gemma audio aligner support for selecting `qk_fast` vs eager
3. any small supporting config changes
4. comments/docstrings that explain the new path

### Documentation deliverable

Add or update one short note explaining:

1. what was implemented
2. what still needs runtime validation
3. why the GPU was intentionally not used during this pass

This can be a small result note or a short implementation note.


## 15. Validation Plan For The Next Agent

Since you must not run anything now, leave the implementation in a state where the next runtime-enabled agent can do this:

1. run one offline forced-alignment comparison on `smoke18`
2. compare `eager` vs `qk_fast`
3. compare MAE / monotonicity / runtime
4. if promising, test one harder clip

Your code should make that validation straightforward.


## 16. Final Standard

At the end of this iteration, another researcher should be able to inspect the repo and understand:

1. why audio `qk_fast` is the right next experiment
2. how it relates to the existing MT `qk_fast` path
3. what code was added to support it
4. why no runtime claims are being made yet
5. how to validate it later once the GPU is free

That is the bar for this pass.
