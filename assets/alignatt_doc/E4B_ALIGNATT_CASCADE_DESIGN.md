# E4B AlignAtt Cascade Design

## Goal

Redesign the current incremental ASR->MT cascade as if **alignment-aware emission**
had been a foundational assumption from the start.

The target system should:

- keep the MT model free to produce a good draft continuation
- decide emission **outside** the model, using attention-derived alignments
- preserve a single monotone target prefix as the only truth reused at the next step
- separate clearly:
  - what the model is currently drafting
  - what the system has accepted as stable
  - what is displayed or emitted downstream


## Core Principle

AlignAtt is not "make the LLM say where it is uncertain".

AlignAtt is:

1. generate a candidate continuation
2. inspect where each candidate target token aligns on the source side
3. stop emission before the first token that depends on source evidence that is too recent

This is the key insight from `assets/alignatt_doc/alignatt_markdown.md`.


## Terminology

### Source Prefix

`source_prefix_t`

- the full English ASR hypothesis for the current sentence at update `t`
- always starts at the beginning of the current sentence
- may still be incomplete

### Draft Target

`draft_target_t`

- the full German MT hypothesis from the beginning of the current sentence at update `t`
- can revise text beyond the already accepted prefix
- never becomes the source of truth by itself

### Accepted Target Prefix

`accepted_target_t`

- the longest German prefix from the beginning of the current sentence that the system accepts as stable
- this is the only text that:
  - may be emitted to the outside world
  - may be reused as assistant prefill at `t+1`
  - may anchor the sentence-final pass

### Draft Acceptance Signal

No dedicated control token is required.

The runtime must be able to accept or reject target material using only:

- alignment-derived source positions
- intra-draft rewind detection
- last-complete-word truncation


## What Is Cached

### 1. Prompt / KV Cache

This is purely a runtime optimization.

We cache the prompt prefix shared between successive MT requests:

- system prompt
- few-shot examples
- previous confirmed sentence-pair context
- the repeated beginning of the current user message
- the assistant prefill equal to `accepted_target_t`

This cache exists to reduce compute, not to define semantics.

### 2. Semantic State

This is the real incremental MT state and must be explicit:

```python
@dataclass
class PartialTranslationState:
    source_prefix: str = ""
    draft_target: str = ""
    draft_token_ids: tuple[int, ...] = ()
    accepted_target: str = ""
    accepted_token_ids: tuple[int, ...] = ()
    source_accessible_unit_count: int = 0
    source_total_unit_count: int = 0
    last_accept_audio_seconds: float = 0.0
    last_num_cached_tokens: int | None = None
    last_prompt_num_tokens: int | None = None
```

Important:

- `draft_target` and `accepted_target` must be different fields
- cache stats must never be confused with semantic commitment
- bias decay or latency bookkeeping must be tied to `accepted_target` growth, not draft growth


## What The Prefix Is

The prefix reused at `t+1` is:

`accepted_target_t`

It is **not**:

- the raw generation text
- the text before `eos`
- the whole previous assistant completion if that completion contains speculative tail material

The assistant prefill must always be exactly the accepted target prefix:

```text
user:  [instruction + context + source_prefix_t]
assistant: accepted_target_t
```

Then generation continues after `accepted_target_t`.


## Prompting Strategy

### Required Prompt Contract

At every update:

- the user message contains the full current source prefix from sentence start
- the assistant message is prefilled with the accepted target prefix from sentence start

This keeps the decoding contract simple:

- the model continues the same sentence
- the runtime decides how much of that continuation is actually accepted

### Few-Shot Role

Few-shot examples should teach only the shape of revision / insertion points.

They should not be the main stability controller.

Good examples:

- subordinate-clause verb-final behavior
- infinitival purpose clauses
- object insertion in the middle of the German clause

The examples are there to improve draft quality, not to replace AlignAtt.


## AlignAtt Strategy For This Cascade

### High-Level Rule

For each newly proposed target token in `draft_target_t`:

1. inspect alignment heads
2. find the source token position most attended by that target token
3. if that aligned source position lies beyond the currently accessible source frontier, stop before that target token

This must be implemented as an inline decoding policy layered on top of
Gemma generation, not as prompt-only control.

### Whisper-Like Aggregation

To stay aligned with the Whisper implementation in
`assets/alignatt_doc/alignatt_whipser.py`, the decoding policy should:

1. keep only the selected alignment heads
2. restrict each head to the current source span only
3. normalize attention over the generated target-token axis
4. median-filter on the source axis
5. average the selected heads
6. compute `argmax` on the resulting source-only matrix

The important detail is that the argmax must be taken on the **source
slice only**, not on the whole prompt context.

### Source Accessibility Frontier

Because this is a text-only cascade, we do not have encoder audio frames inside Gemma.

So the analogue of AlignAtt's last `f` frames must be:

`the suffix of source units whose end timestamps are newer than current_audio_ms - inaccessible_ms`

where `inaccessible_ms` is the latency control knob.

Examples:

- `inaccessible_ms = 0`: latency-first default for the current cascade after calibration
- `inaccessible_ms = 100-200`: softer safety buffer when quality pressure dominates
- `inaccessible_ms = 300+`: conservative, often too restrictive for CU in this setup

### Acceptance Rule

Let:

- `draft_target_t` be the full newly generated target draft
- `align_pos(i)` be the aligned source token index of target token `i`
- `a` be the last accessible source token position for the current update

Then:

- accept tokens from the start of the draft
- stop before the first target token whose `align_pos(i)` falls beyond `a`

This yields an alignment-truncated draft:

`alignatt_target_t`

### Rewind Guard

Whisper also contains a local guard against attention failure: if the
most-attended source position suddenly rewinds too far backwards during
the same partial generation step, the segment is omitted.

For the text MT cascade, the clean analogue is:

- scan the per-token source argmaxes from left to right
- if a token jumps backwards by more than `r` source tokens relative to
  the previous accepted target token alignment, reject the whole newly
  generated suffix for this update

This is an **intra-draft** safety check, not a temporal agreement rule
between `t-1` and `t`.

Operationally, the rewind threshold should stay permissive enough to
avoid suppressing legitimate German reordering. For the current E4B
text-only cascade, `rewind_threshold = 8` is the calibrated default.

### Last-Word Truncation

For partial updates, the decoding policy should emit only complete target
words.

This mirrors Whisper's strategy of emitting all but the last unfinished
word in non-final streaming steps.

So the partial acceptance pipeline is:

1. generate `draft_target_t`
2. apply AlignAtt source-frontier truncation
3. apply rewind guard
4. drop the final incomplete target word

### Monotonic Prefix Invariant

We do **not** want a strong temporal consistency constraint such as:

`common_prefix(draft_{t-1}, draft_t)`

That over-constrains the system and is not part of canonical AlignAtt.

Instead, the only core invariant should be:

- if the backend-accepted target at `t` extends the previously accepted
  target, keep the extension
- otherwise keep the previous accepted target unchanged

This is the cleanest foundational design for the cascade:

- backend AlignAtt decides source-side safety
- a minimal monotonic-prefix invariant protects downstream consumers
- no extra temporal agreement policy is needed in the core


## Recommended Acceptance Pipeline

At update `t`:

1. ASR provides `source_prefix_t`
2. MT prompt is built with:
   - user: full `source_prefix_t`
   - assistant prefill: `accepted_target_{t-1}`
3. Gemma generates a draft continuation
4. runtime reconstructs `draft_target_t`
5. runtime computes token alignments from selected alignment heads
6. runtime applies AlignAtt source-frontier cutoff
7. runtime applies rewind rejection if attention jumps backwards too far
8. runtime drops the final incomplete target word
9. runtime updates:
   - `draft_target_t`
   - `accepted_target_t`
10. only the new suffix of `accepted_target_t` is emitted


## Alignment Heads

The alignment heads must be detected with the paper's criterion:

- for each aligned target token
- take the attention argmax over the **full sequence**
- not only over source positions

This is important because otherwise BOS, prompt, or target-self heads can look like false translation heads.

For Gemma E4B, the expected artifact is:

- `assets/attention_heads/translation_heads_google_gemma-4-E4B-it_en-de.json`

These heads should then be reduced to a smaller serving set for runtime use, for example:

- top 4 heads
- top 8 heads
- or one validated middle-layer cluster if the scores are concentrated


## Empirical Calibration Notes

The design above is not only theoretical. It has already been exercised on
the current talk-level test file with full streaming runs at `chunk_ms = 800`.

All evaluations below were run with `.venv-evaluation`. In practice we used
`--skip-comet` for the hot-kernel experiments because keeping Qwen3-ASR and
Gemma E4B resident in GPU memory makes `XCOMETXL` prone to CUDA OOM.


## Short Smoke Sweep

Before running full-talk experiments, we used the short
`alignatt_smoke18_*` clips to calibrate the source frontier.

| Setting | Updates with AlignAtt | Unsafe `source_frontier` | Unsafe `rewind` | Avg accepted tokens | Avg candidate tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `500 ms / 3` | 14 | 7 | 1 | 2.786 | 4.571 |
| `200 ms / 6` | 14 | 7 | 0 | 4.071 | 6.071 |
| `100 ms / 6` | 14 | 7 | 0 | 4.000 | 5.929 |
| `0 ms / 6` | 14 | 5 | 1 | 4.429 | 6.143 |

Takeaway:

- `500 / 3` was clearly too conservative.
- Lowering `inaccessible_ms` materially increased accepted progress.
- The smoke sweep suggested that `200 ms` and `0 ms` were the right full-run candidates.


## Full 800 ms Runs

The most relevant complete runs so far are:

| Run | Architecture / calibration | BLEU | chrF | LongYAAL CU | LongYAAL CA |
| --- | --- | ---: | ---: | ---: | ---: |
| `foundational_nomarker_defaultsafe` | old safe baseline, pre-source-frontier | 37.0393 | 67.3858 | 4419.0688 | 4297.6604 |
| `source_frontier_hotreload_live` | source frontier `500 ms / 3` | 36.2580 | 67.0110 | 4269.0472 | 4220.3349 |
| `source_frontier_ms200_rw6` | source frontier `200 ms / 6` | 36.1303 | 66.9475 | 4068.3914 | 4093.7408 |
| `source_frontier_ms0_rw8` | source frontier `0 ms / 8` | 35.3816 | 66.6986 | 3851.1828 | 3956.4983 |

For context, the best historical non-AlignAtt `chunk800` run
(`prompt_only_partial_anchor_chunk800_live`) is still much faster:

| Run | BLEU | chrF | LongYAAL CU | LongYAAL CA |
| --- | ---: | ---: | ---: | ---: |
| `prompt_only_partial_anchor_chunk800_live` | 38.3956 | 67.8364 | 1997.3458 | 2581.6145 |

So the current AlignAtt-first architecture is cleaner and more principled,
but it is **not yet latency-competitive** with the best older cascade.


## Full-Run Behavioral Summary

The frontier calibration also changed the behavior of the inline policy:

| Run | Unsafe `source_frontier` / `source_tail` | Unsafe `rewind` | Word trims | Avg accepted tokens | Avg candidate tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `foundational_nomarker_defaultsafe` | 329 | 44 | 294 | 0.836 | 2.150 |
| `source_frontier_hotreload_live` | 98 | 126 | 247 | 1.973 | 3.118 |
| `source_frontier_ms200_rw6` | 107 | 89 | 284 | 2.847 | 4.193 |
| `source_frontier_ms0_rw8` | 110 | 66 | 307 | 3.536 | 4.976 |

Takeaway:

- Moving from `500 / 3` to `0 / 8` reduced `LongYAAL CU` by about `418 ms`
  (`4269.0472 -> 3851.1828`).
- The gain comes from allowing more accepted target growth per update:
  `avg accepted tokens` rises from `1.973` to `3.536`.
- A more permissive rewind threshold also matters: rewinds fall from `126`
  to `66`, which is important for German reordering freedom.
- Even with this calibration, the frontier-only approach still leaves a large
  latency gap relative to the best historical run.


## Current Default And Rationale

The current default is therefore:

- `translation_alignatt_inaccessible_ms = 0.0`
- `translation_alignatt_rewind_threshold = 8`

Why this default:

- it is the best `LongYAAL CU` setting found so far on the full-talk run
- it keeps the design faithful to AlignAtt: source accessibility first,
  no special token, no heavy temporal agreement rule
- it preserves a safety mechanism through the inline rewind guard and
  last-complete-word truncation

Why this is not the final answer:

- the current bottleneck is no longer “how to make the frontier less strict”
- the next likely wins will come from lowering observation/emission cost elsewhere:
  `chunk_ms`, word-boundary emission policy, and possibly tighter integration of
  the inline policy with decoding cost


## Runtime Head Aggregation

At runtime, the alignment signal should come from a fixed small set of heads.

Recommended aggregation:

1. collect attention maps only from selected translation heads
2. normalize per head
3. smooth lightly over target positions
4. average across selected heads
5. compute per-target-token source argmax from the averaged map

This mirrors the spirit of `alignatt_whipser.py` while adapting it to text-only MT.


## Why Control Tokens Were Removed

The last iteration showed that a dedicated control token adds prompt
fragility without solving the real problem.

The stable design is to let:

- Gemma draft freely
- AlignAtt decide source-side safety
- the inline policy enforce monotonic emission

This removes an entire axis of tuning and makes the system less prompt-dependent.


## What Must Change In The Current Code

### State

Current state mixes draft and accepted text.

We need a real separation in `qwen3asr_gemma_cascade_core.py`:

- `draft_target`
- `accepted_target`

### Finalization

The final sentence pass must anchor only on `accepted_target`, never on speculative draft text.

### Emission Policy

`cascade_emission.py` should become a replay / ablation utility, not the main mechanism that repairs instability after the fact.

### Head Access

For runtime AlignAtt, the MT backend must expose the selected Gemma attention heads.

This is straightforward in `transformers` and not cleanly available in the current public `vLLM` API.


## vLLM Note

For the current environment:

- `vLLM` is good for fast serving and prefix caching
- it is not the right backend for one-off alignment-head detection
- the public API does not expose per-layer attentions for the decoding path we need

So:

- head detection should be done with `transformers`
- serving does not need to be uniform across ASR and MT
- runtime AlignAtt with `vLLM` would require additional custom instrumentation or a different MT backend


## Recommended Backend Split

The cleanest operational split is:

- `Qwen3-ASR` on `vLLM`
- `Gemma E4B MT` on `transformers`

Why:

- ASR benefits strongly from the existing `vLLM` stack and is already integrated that way
- MT AlignAtt needs attention access, which is natural in `transformers`
- this avoids building the whole redesign around `vLLM` internals just to recover attention maps

So the proposed future runtime is a hybrid cascade:

- fast ASR serving
- attention-visible MT serving
- explicit accepted-prefix control above both


## Recommended Development Order

1. detect and validate E4B en-de alignment heads
2. freeze a small runtime head set
3. refactor MT state into `draft_target` vs `accepted_target`
4. implement text-token AlignAtt cutoff
5. add intra-draft rewind rejection
6. add explicit prompt/KV reuse for the `transformers` MT backend


## Success Criterion

The redesigned cascade is successful if:

- the reused prefix is always `accepted_target`
- emitted text is monotone by construction
- MT can still draft beyond the accepted prefix
- alignment gating, not prompt hacks, decides how far emission goes
- latency tuning becomes a clean source-tail parameter instead of a collection of bias heuristics
