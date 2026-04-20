# PLAN.md

Date: 2026-04-20

## Objective

Get **Gemma 4 ASR via vLLM + AlignAtt** to run correctly in long-form
SimulStreaming, with a paper-defensible implementation.

Target is not “15 safety heuristics that hide failures”.
Target is:

- clean long-form streaming
- pure/defensible AlignAtt-based commit behavior
- no prompt leakage into the transcript
- no reset-induced collapse after public commits
- short-form ASR ceiling around the already observed `~0.34-0.35 WER`
- long-form behavior that actually reaches the end of the audio and remains
  usable

This file is a handoff for the next agent.

## Scope and operating constraints

- Active runtime: [cascade/runtime.py](/home/fuxa/cascade_simultaneous/cascade/runtime.py)
- Active Gemma ASR backend:
  [cascade/alignment/gemma_vllm_asr_backend.py](/home/fuxa/cascade_simultaneous/cascade/alignment/gemma_vllm_asr_backend.py)
- Evaluation harness:
  [scripts/compare_asr_full_audio.py](/home/fuxa/cascade_simultaneous/scripts/compare_asr_full_audio.py)
- Use `.venv-inference`
- Stay on **one talk** until the mechanism is stable:
  - short debug: `dev-set/audio/ccpXHNfaoy_short60s.wav`
  - full long-form: `dev-set/audio/ccpXHNfaoy.wav`
- Avoid broad sweeps
- Runs are expensive because Gemma vLLM reloads/compiles often

## Current bottom line

We have already fixed several real implementation bugs.

What is fixed:

- Gemma long-form no longer depends on hacky text-windowing to stay under the
  30 s audio cap.
- Public commit no longer hard-resets the streaming state like a brand-new
  utterance.
- The `generate` path no longer double-counts audio.
- The `generate` path now actually distinguishes `with_prefix` vs `no_prefix`
  prompt regimes.

What is **not** fixed:

- The best `with_prefix` prompt is still unresolved.
- There is a strong trade-off between:
  - suppressing prompt leakage
  - preserving actual audio-grounded continuation
- We do **not** yet have a full-360 s Gemma vLLM AlignAtt ASR run that is both
  stable and trustworthy.

## Important code changes already made

### 1. Pure-AlignAtt-ish long-form runtime

Recent work concentrated commit decisions into AlignAtt-driven logic instead of
 piling on textual guardrails.

Relevant areas:

- [cascade/runtime.py](/home/fuxa/cascade_simultaneous/cascade/runtime.py)
- [cascade/simulstream_processor.py](/home/fuxa/cascade_simultaneous/cascade/simulstream_processor.py)
- [run_simulstream_batch.py](/home/fuxa/cascade_simultaneous/run_simulstream_batch.py)
- [run_simulstream_compare.py](/home/fuxa/cascade_simultaneous/run_simulstream_compare.py)

Notable surfaces:

- `asr_commit_mode`
- `asr_alignatt_frontier_margin_ms`
- `asr_alignatt_boundary_gap_ms`
- `asr_streaming_prefix_enabled`
- `asr_streaming_prefix_max_words`

### 2. Public commit no longer destroys ASR continuation state

This was a major bug.

Before the fix:

- AlignAtt committed a public boundary
- runtime reset the streaming state/backend
- Gemma restarted like a fresh utterance
- continuation collapsed immediately after a mid-sentence boundary

This was fixed by preserving the post-commit remainder as the next streaming
seed instead of resetting everything.

Primary file:

- [cascade/runtime.py](/home/fuxa/cascade_simultaneous/cascade/runtime.py)

### 3. vLLM `generate` path prompt bug

This is a key discovery from today.

Even after adding a special `with_prefix` prompt regime, the `generate` path
still built its user-side prompt as if `assistant_prefix=""`.

In practice, that meant:

- no-prefix instruction was always used in streaming `generate`
- the supposedly separate `with_prefix` prompt was not actually active
- many earlier conclusions about “with-prefix prompt behavior” were polluted by
  this bug

This is now fixed in:

- [cascade/alignment/gemma_vllm_asr_backend.py](/home/fuxa/cascade_simultaneous/cascade/alignment/gemma_vllm_asr_backend.py)

Look at:

- `_build_prompt_layout(...)`
- `_render_asr_instruction(...)`

## Prompt regimes currently in play

Defined in
[cascade/alignment/gemma_vllm_asr_backend.py](/home/fuxa/cascade_simultaneous/cascade/alignment/gemma_vllm_asr_backend.py):

- `verbatim_strict`
- `english_into_english`
- `original_language`
- `continue_from_prefix`

Current defaults:

- `no_prefix`: `verbatim_strict`
- `with_prefix`: `continue_from_prefix`

But this default is **not settled**. It is only the latest experimental state.

## Stable short-form ceiling

The short-form offline ceiling is still roughly:

- `WER ~0.34-0.35`

Useful artifact:

- [outputs/hf_best_practices_segmented_full_20260420/summary.json](/home/fuxa/cascade_simultaneous/outputs/hf_best_practices_segmented_full_20260420/summary.json)
- [outputs/plan_impl_validation_segmented_20260420/segment_eval.json](/home/fuxa/cascade_simultaneous/outputs/plan_impl_validation_segmented_20260420/segment_eval.json)

Interpretation:

- Gemma ASR short-form is not solved to the model-card level, but it is not
  the main blocker right now.
- Long-form streaming behavior is the real failure surface.

## Most important recent streaming results

### A. Full360 before state-carry fix

Artifact:

- [outputs/alignatt_gapcommit_strongestgap_full360_gap250_eosfix_direct_20260420/summary.json](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_strongestgap_full360_gap250_eosfix_direct_20260420/summary.json)

Key result:

- `WER=0.821`
- `predicted_boundary_count=33`

Main failure:

- post-commit reset killed continuation

### B. Full360 after state-carry fix

Artifact:

- [outputs/alignatt_gapcommit_statecarry_full360_gap250_20260420/summary.json](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_full360_gap250_20260420/summary.json)
- [streaming run JSON](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_full360_gap250_20260420/runs/gemma_vllm_qk_fast__sp-on__rb-0__uf-2__margin-500__gap-250__streaming_full.json)

Key result:

- `WER=0.6609`
- `predicted_boundary_count=15`
- better than the previous full run

Main failure:

- prompt contamination still entered the transcript
- run still did not behave like a fully trustworthy 360 s streaming decode

### C. Short60 with old accidental no-prefix prompt in streaming

Artifact:

- [outputs/alignatt_gapcommit_statecarry_short60_promptfix_20260420/summary.json](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_short60_promptfix_20260420/summary.json)

Key result:

- `WER=0.4369`
- `predicted_boundary_count=9`
- `meta_response_count=4`

Interpretation:

- good WER
- but polluted by prompt leakage
- this run was still affected by the prompt-layout bug

### D. Short60 with properly wired `continue_from_prefix`

Artifact:

- [outputs/alignatt_gapcommit_statecarry_short60_promptfix2_20260420/summary.json](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_short60_promptfix2_20260420/summary.json)

Key result:

- `WER=1.3981`
- `predicted_boundary_count=5`
- `meta_response_count=0`

Interpretation:

- leakage gone
- but the model drifts into plausible self-introduction / semantic completion
  that is **not audio-grounded**
- the prompt is too compact / under-constraining

### E. Short60 with properly wired `english_into_english`

Artifact:

- [outputs/alignatt_gapcommit_statecarry_short60_wpenglish_20260420/summary.json](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_short60_wpenglish_20260420/summary.json)

Key result:

- `WER=0.6117`
- `predicted_boundary_count=21`
- `meta_response_count=1`

Interpretation:

- better grounded than `continue_from_prefix`
- still too leak-prone / over-segmented
- likely too verbose / too easy to recycle into the transcript

## Current diagnosis

The core remaining problem is:

**find a `with_prefix` regime that preserves audio grounding without being
copied into the transcript.**

Observed failure modes:

1. `verbatim_strict`-like behavior:
   - prompt gets copied into the hypothesis
   - then recycled via prefix
   - long-form self-poisons

2. `continue_from_prefix` compact behavior:
   - prompt leak disappears
   - but the model starts making up plausible continuation text
   - audio grounding weakens too much

3. `english_into_english`:
   - better grounding
   - but still too verbose and too segmentation-happy

So the next step is **not** another runtime redesign first.
The next step is to settle the `with_prefix` continuation prompt.

## Hypothesis for the next agent

We probably need a **middle prompt**:

- shorter than `english_into_english`
- explicitly anchored to audio/transcription
- but much less copyable than the full formatting-heavy instruction

Candidate shape:

- “Continue transcribing the audio after the provided prefix. Output only the
  next words.”

This is not implemented yet.

## Exact next steps

### Step 1. Inspect the new traces before adding more code

Read these runs side by side:

- [outputs/alignatt_gapcommit_statecarry_short60_promptfix_20260420/runs/gemma_vllm_qk_fast__sp-on__rb-0__uf-2__margin-500__gap-250__streaming_full.json](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_short60_promptfix_20260420/runs/gemma_vllm_qk_fast__sp-on__rb-0__uf-2__margin-500__gap-250__streaming_full.json)
- [outputs/alignatt_gapcommit_statecarry_short60_promptfix2_20260420/runs/gemma_vllm_qk_fast__sp-on__rb-0__uf-2__margin-500__gap-250__streaming_full.json](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_short60_promptfix2_20260420/runs/gemma_vllm_qk_fast__sp-on__rb-0__uf-2__margin-500__gap-250__streaming_full.json)
- [outputs/alignatt_gapcommit_statecarry_short60_wpenglish_20260420/runs/gemma_vllm_qk_fast__wp-english_into_english__sp-on__rb-0__uf-2__margin-500__gap-250__streaming_full.json](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_short60_wpenglish_20260420/runs/gemma_vllm_qk_fast__wp-english_into_english__sp-on__rb-0__uf-2__margin-500__gap-250__streaming_full.json)

Confirm:

- where the first leak appears
- where the first non-audio-grounded semantic drift appears
- whether bad behavior starts before or after the first public commit

### Step 2. Implement exactly one new `with_prefix` prompt mode

Do **not** start a broad sweep.

Add one intermediate mode, something like:

- `continue_transcribing_audio`

Goal:

- retain explicit audio grounding
- remain short enough to avoid instruction leakage

Suggested location:

- [cascade/alignment/gemma_vllm_asr_backend.py](/home/fuxa/cascade_simultaneous/cascade/alignment/gemma_vllm_asr_backend.py)

Also expose it in:

- [scripts/compare_asr_full_audio.py](/home/fuxa/cascade_simultaneous/scripts/compare_asr_full_audio.py)

### Step 3. Run only one `short60` probe with that new mode

Use the same config as the recent pure-AlignAtt probes:

```bash
PYTHONPATH=. .venv-inference/bin/python scripts/compare_asr_full_audio.py run \
  --wav dev-set/audio/ccpXHNfaoy_short60s.wav \
  --eval-mode streaming_full \
  --backends gemma_vllm_qk_fast \
  --chunk-ms 800 \
  --min-start-seconds 2.0 \
  --asr-commit-mode auto \
  --asr-alignatt-frontier-margin-ms 500 \
  --asr-alignatt-boundary-gap-ms 250 \
  --asr-streaming-prefix-enabled \
  --asr-streaming-rollback-words 0 \
  --asr-streaming-unfixed-chunks 2 \
  --gemma-with-prefix-prompt-mode <NEW_MODE> \
  --skip-plot \
  --output-dir outputs/<new_short60_dir>
```

### Step 4. Compare against these three short60 anchors

- leakage-prone but good WER:
  [promptfix_20260420](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_short60_promptfix_20260420/summary.json)
- leak-free but ungrounded:
  [promptfix2_20260420](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_short60_promptfix2_20260420/summary.json)
- more grounded but verbose:
  [wpenglish_20260420](/home/fuxa/cascade_simultaneous/outputs/alignatt_gapcommit_statecarry_short60_wpenglish_20260420/summary.json)

### Step 5. Only then rerun full360

Do **not** rerun full360 until the new short60 candidate is clearly better.

A short60 candidate is worth promoting to full360 only if it improves on the
current trade-off in a meaningful way:

- `meta_response_count == 0` or very close
- WER materially better than `1.398`
- boundary count materially better than `21`
- no obvious self-introduction / assistant-style semantic drift

Ideal short60 target before full rerun:

- `WER <= 0.55`
- `predicted_boundary_count` closer to `7` than `21`
- no copied prompt text

If a candidate clears that bar, rerun:

```bash
PYTHONPATH=. .venv-inference/bin/python scripts/compare_asr_full_audio.py run \
  --wav dev-set/audio/ccpXHNfaoy.wav \
  --eval-mode streaming_full \
  --backends gemma_vllm_qk_fast \
  --chunk-ms 800 \
  --min-start-seconds 2.0 \
  --asr-commit-mode auto \
  --asr-alignatt-frontier-margin-ms 500 \
  --asr-alignatt-boundary-gap-ms 250 \
  --asr-streaming-prefix-enabled \
  --asr-streaming-rollback-words 0 \
  --asr-streaming-unfixed-chunks 2 \
  --gemma-with-prefix-prompt-mode <WINNER> \
  --skip-plot \
  --output-dir outputs/<new_full360_dir>
```

## Things to avoid

- No lexical post-processing
- No content-aware string repair
- No dataset-specific rewrites
- No new safety layers unless they are generic and paper-defensible
- No broad multi-audio sweeps before one candidate behaves correctly on
  `ccpXHNfaoy_short60s.wav`

## Evaluation note

`META_RESPONSE_PATTERNS` in
[scripts/compare_asr_full_audio.py](/home/fuxa/cascade_simultaneous/scripts/compare_asr_full_audio.py)
was expanded today so obvious prompt leakage is no longer silently scored as
clean.

This is evaluation-only; it does not change runtime behavior.

## Success condition

“Gemma vLLM AlignAtt ASR turns correctly” means:

- it reaches the end of the full audio
- it does not inject prompt instructions into the transcript
- it does not collapse into made-up semantic continuation
- it keeps a believable streaming segmentation pattern
- it stays within the current pure-AlignAtt runtime philosophy
