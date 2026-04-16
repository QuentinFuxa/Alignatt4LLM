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
| `compute_unaware_chunk800_20260415T154922Z` | hot-kernel rerun, same `0 ms / 8` frontier defaults, manifest recorded `probe_mode = eager` | 38.7556 | 68.0866 | 3716.8469 | 5595.8554 |

For context, the best historical non-AlignAtt `chunk800` run
(`prompt_only_partial_anchor_chunk800_live`) is still much faster:

| Run | BLEU | chrF | LongYAAL CU | LongYAAL CA |
| --- | ---: | ---: | ---: | ---: |
| `prompt_only_partial_anchor_chunk800_live` | 38.3956 | 67.8364 | 1997.3458 | 2581.6145 |

So the current AlignAtt-first architecture is cleaner and more principled,
but it is **not yet latency-competitive** with the best older cascade.

The new hot-kernel rerun is important for interpretation:

- it materially improves quality and `LongYAAL CU` relative to the older
  `source_frontier_ms0_rw8` run
- but it is still far from the `< 2 s` `LongYAAL CU` target
- and its `LongYAAL CA` is much worse because the reused in-memory kernel
  reported `translation_alignatt_probe_mode = eager`

So this rerun changes the picture in a useful way:

- the current AlignAtt semantics can already produce good `compute unaware`
  quality/latency tradeoffs
- but observer compute is still unresolved on the `CA` axis


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

The 2026-04-15 hot-kernel rerun sharpens that conclusion:

- compared to `source_frontier_ms0_rw8`, it improves by about:
  - `+3.374 BLEU`
  - `+1.388 chrF`
  - `-134 ms` `LongYAAL CU`
- but compared to the historical `prompt_only_partial_anchor_chunk800_live`,
  it is still slower by about `+1719 ms` on `LongYAAL CU`

That means the remaining gap to `< 2 s` `LongYAAL CU` is no longer a small
calibration issue. It is a structural latency problem.


## What The New Rerun Tells Us

The newest `chunk800` rerun matters because it separates two questions that
were previously entangled.

### 1. The main remaining `CU` gap is not primarily compute

This run was evaluated on the `compute unaware` axis and still landed at:

- `LongYAAL CU = 3716.8469`

even though its manifest recorded:

- `translation_alignatt_probe_mode = eager`

That means speeding up the observer alone will not get us below `2 s`
`LongYAAL CU`.

Observer cost matters for `CA`.

But the dominant remaining `CU` gap is source-time waiting:

- when we start translating
- how often we are allowed to react
- how coarse the source frontier is
- how much safe target text we accept at each frontier step

### 2. Quality is no longer the blocker for trying more aggressive `CU` policies

The same rerun reached:

- `BLEU = 38.7556`
- `chrF = 68.0866`

So we are no longer in the regime where every latency improvement attempt is
obviously quality-destructive.

That gives room to attack `CU` directly with more aggressive but principled
streaming decisions.


## What The Stage 1 Start-Gate Sweep Tells Us

We now have a partial full-talk Stage 1 sweep at:

- `chunk_ms = 800`
- same current AlignAtt defaults otherwise
- run root: `outputs/stage1_start_gate_sweep_20260415T160539Z`

Completed points:

| `min_start_seconds` | Run | First non-empty / accepted emission | BLEU | chrF | LongYAAL CU | LongYAAL CA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `5.0` | `min_start_5p0` | `5.6 s` | 38.7556 | 68.0866 | 3726.8594 | 5524.1509 |
| `3.0` | `min_start_3p0` | `4.0 s` | 38.7556 | 68.0866 | 3698.2681 | 5524.5087 |
| `2.0` | `min_start_2p0` | `4.0 s` | 38.7556 | 68.0866 | 3698.2681 | 5521.5712 |

The `1.5 s` point was intentionally stopped once the pattern was already clear,
to avoid spending another full-talk run on an axis that was no longer moving.

This result is important because it falsifies the strongest version of the
earlier Stage 1 hypothesis.

Lowering the hard start gate does improve user-visible early responsiveness:

- first accepted emission moves from `5.6 s` to `4.0 s`

But on corpus-level `compute unaware` latency, the effect is tiny:

- `5.0 -> 3.0` improves `LongYAAL CU` by only about `28.6 ms`
- `3.0 -> 2.0` produces no further `LongYAAL CU` gain at all

So the current evidence says:

- the hard `5 s` gate is not the main reason we are stuck around
  `3.7 s` `LongYAAL CU`
- on the current full-talk `chunk800` setting, start-gate tuning quickly
  saturates
- the dominant remaining `CU` bottlenecks are more likely:
  - chunk quantization
  - coarse scheduler gating
  - source-frontier granularity
  - insufficient accepted growth per update


## Pistes To Get Below 2 Seconds LongYAAL CU

The gap from the current rerun to the target is about:

- `3716.8469 -> < 2000`
- roughly `1717 ms` still to remove

The most plausible path is not a single trick. It is a stack of structural
changes that all move the `source-time` policy in the same direction.

### 1. Keep the start gate low for UX, but do not overinvest in this axis

The current manifest still records:

- `min_start_seconds = 5.0`

and that is still too high for first-token responsiveness.

However, the completed Stage 1 sweep now shows that this axis is not the main
corpus-level `CU` lever in the current architecture:

- `5.0 -> 3.0` moved first accepted emission from `5.6 s` to `4.0 s`
- but only reduced `LongYAAL CU` by about `28.6 ms`
- `3.0 -> 2.0` brought no further `CU` improvement on the full-talk run

So the right conclusion is narrower than before.

The principled redesign is still:

- split the streaming regime into:
  - an early low-context regime
  - a later steady-state regime
- start partial MT much earlier, for example around `1.5-2.0 s`
- compensate with:
  - tiny initial draft budgets
  - no speculative history
  - the same accepted-prefix contract as later steps

But this is now best viewed as:

- a worthwhile UX cleanup
- not the primary path to `< 2 s` `LongYAAL CU`

For future full-talk runs, `3.0 s` is already enough to represent the
"early-start" regime until some other bottleneck changes.

### 2. Reduce `chunk_ms` below `800`

At `chunk_ms = 800`, every source update arrives on a coarse time grid.

Even a perfect inline policy cannot emit earlier than the next chunk boundary.

So a large part of the remaining `CU` gap is probably quantization error from:

- chunk arrival
- ASR refresh
- MT refresh

The most principled next sweep is:

- `chunk_ms = 400`
- then `chunk_ms = 200-320`

If quality becomes unstable, the clean answer is not to reintroduce hacks.
It is:

- adaptive cadence

For example:

- smaller chunks at sentence start and near blocked frontiers
- larger chunks once the system is clearly behind the frontier

### 3. Use token-level blocked-frontier scheduling, not only unit-level scheduling

The current scheduler stores:

- `blocked_source_local_position`
- `blocked_source_unit_index`

but gating is still mainly driven by:

- `accessible_unit_count`
- `blocked_source_unit_index`

This is unnecessarily coarse for `CU`.

If a probe stopped at a specific blocked source token position, waiting for the
whole next source unit can add avoidable delay.

The principled next step is:

- drive scheduling from `blocked_source_local_position`
- compare it directly to `accessible_source_token_count`
- rerun MT as soon as the token-level frontier is close enough

This should recover latency without changing the acceptance semantics.

### 4. Make the source frontier finer than whole completed words

Right now, the source accessibility frontier is still effectively governed by
completed source-word timestamps.

That is clean, but probably too coarse for `< 2 s` `CU`.

The principled long-term direction is:

- move from word-complete accessibility to source-token or subword accessibility
- use timestamped ASR evidence at finer granularity when available
- keep the same monotone accepted-prefix invariant on the target side

This is especially important because the current AlignAtt policy is already at:

- `inaccessible_ms = 0`

So the next source-time gains will not come from making the word frontier more
permissive. They will come from making it finer.

### 5. Increase accepted target growth per update with LLM-native provenance checks

On a causal LLM, some drafted target tokens are blocked not because they truly
depend on future source evidence, but because the current observer does not yet
distinguish clearly between:

- source support
- accepted-prefix support
- speculative-suffix support

For `CU`, this matters directly.

If a token's source anchor is already behind the frontier and the remaining
support comes mainly from the accepted target prefix rather than speculative
future target tokens, then rejecting it is often overly conservative.

So a strong medium-term `CU` lever is:

- provenance-aware acceptance

not to accept arbitrary target-history-driven tokens, but specifically to
separate:

- safe accepted-prefix continuation
- unsafe speculative-suffix self-support

This should raise `avg accepted tokens` per update, which is exactly the
behavior that previously improved `LongYAAL CU`.

### 6. Make draft length adaptive to frontier distance

The current system can still spend updates drafting much farther than the
blocked frontier.

For `CA` that wastes compute.

For `CU` it also wastes opportunities because:

- longer drafts increase the chance that acceptance stops only after a large
  speculative tail was explored
- word-boundary truncation then throws away part of that effort

The cleaner policy is:

- if the blocked frontier is very near, draft very short
- if the frontier jumped far, allow a larger suffix

This is not just a serving optimization. It is a latency policy because it
changes how often we can expose a newly accepted whole-word prefix.


## What Is Unlikely To Be Enough

If the explicit goal is `< 2 s` `LongYAAL CU`, these changes alone are very
unlikely to suffice:

- further tuning `inaccessible_ms`
- further tuning `rewind_threshold`
- speeding up the observer backend without changing the source-time policy

Those can still matter.

But the new rerun suggests that the decisive `CU` gains now require:

- finer update cadence
- finer frontier granularity
- less conservative scheduling
- more target growth per accepted update


## Prioritized Experimental Program For `< 2 s` LongYAAL CU

The right next step is not a broad grid search.

It is a staged program that tests the highest-impact `CU` levers first while
keeping the interpretation clean.

### Success Criterion

For this phase, the primary objective is:

- `LongYAAL CU < 2000 ms`

Subject to a secondary quality floor:

- stay in the same broad quality regime as the new rerun
- i.e. avoid changes that collapse quality even if they improve `CU`

A practical working guardrail for early experiments is:

- `BLEU >= 37.5`
- `chrF >= 67.5`

These are not final scientific thresholds. They are just a stop condition to
avoid spending time on obviously bad latency-only regimes.


### Stage 0: Fix The Experimental Baseline

Before changing behavior, we should lock one explicit baseline around the new
rerun:

- `chunk_ms = 800`
- `min_start_seconds = 5.0`
- `inaccessible_ms = 0.0`
- `rewind_threshold = 8`
- current scheduler

Artifacts to record for every future run:

- `BLEU`
- `chrF`
- `LongYAAL CU`
- `LongYAAL CA`
- first accepted target emission time
- average accepted tokens per update
- average candidate tokens per update
- number of scheduler skips
- number of updates where the blocked frontier was within one source token of accessibility

This is important because the current document has quality/latency tables, but
the next phase needs more direct diagnostics for why `CU` moves.


### Stage 1: Attack The Start Lag First

Status:

- completed enough to make the decision
- no need to keep spending full-talk runs on this axis right now

#### Why first

The current hard gate:

- `min_start_seconds = 5.0`

was large enough that it was reasonable to test whether it dominated the
remaining `CU` gap.

That hypothesis did not hold on the full-talk run.

#### Experiment

Hold everything else fixed and sweep only:

- `min_start_seconds = 5.0`
- `min_start_seconds = 3.0`
- `min_start_seconds = 2.0`
- `min_start_seconds = 1.5`

Do this at the current:

- `chunk_ms = 800`

Completed scored points:

| `min_start_seconds` | First non-empty / accepted emission | BLEU | chrF | LongYAAL CU | LongYAAL CA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `5.0` | `5.6 s` | 38.7556 | 68.0866 | 3726.8594 | 5524.1509 |
| `3.0` | `4.0 s` | 38.7556 | 68.0866 | 3698.2681 | 5524.5087 |
| `2.0` | `4.0 s` | 38.7556 | 68.0866 | 3698.2681 | 5521.5712 |

The `1.5 s` run was intentionally interrupted once `3.0` and `2.0` already
showed the same `CU` outcome.

#### What to look for

- first accepted target emission time
- `LongYAAL CU`
- `BLEU` and `chrF`

#### Hypothesis

The original hypothesis was too strong.

The start gate clearly affects first-token UX, but on this full-talk setup it
does not meaningfully move corpus-level `LongYAAL CU`.

#### Decision

Freeze Stage 1 as:

- "solved enough"
- `3.0 s` is a sufficient representative early-start setting
- do not spend more full-talk runs on this axis until `chunk_ms` or scheduler
  semantics change

The next active lever is now Stage 2.


### Stage 2: Reduce Chunk Quantization

#### Why second, and now effectively first

Stage 1 showed that start lag is not dominating the remaining `CU` gap.

So the next clean structural suspect is the `800 ms` update grid itself.

#### Experiment

Using the representative Stage 1 early-start setting, sweep:

- `chunk_ms = 800`
- `chunk_ms = 400`
- `chunk_ms = 320`
- `chunk_ms = 240`

Keep:

- the same frontier semantics
- the same scheduler
- the same acceptance policy

#### What to look for

- `LongYAAL CU`
- average accepted tokens per update
- number of updates
- quality degradation, if any

#### Hypothesis

This is the second most likely large `CU` lever, because it reduces source-time
quantization without changing the core semantics.

#### Decision rule

Prefer the smallest `chunk_ms` that preserves quality and does not explode the
number of useless updates.

If smaller chunks create too many empty or blocked updates, that is a sign to
move immediately to Stage 3 rather than backing off entirely.


### Stage 3: Token-Level Blocked-Frontier Scheduler

#### Why third

After lowering start lag and chunk quantization, the next likely bottleneck is
that the scheduler is still too coarse:

- it mainly gates on source units
- not on token-level blocked positions

#### Minimal code change

Promote the scheduler from:

- `blocked_source_unit_index`

to also using:

- `blocked_source_local_position`
- `accessible_source_token_count`

The simplest principled rule is:

- if the blocked source token is now accessible, rerun immediately
- if it is within one token of accessibility, rerun on the next source update
- otherwise keep skipping unless a stall override triggers

#### Experiment

Compare:

- current unit-level scheduler
- token-level blocked-frontier scheduler

at the best settings from Stages 1 and 2.

#### What to look for

- `LongYAAL CU`
- number of skipped updates
- number of reruns that still stop on the first blocked token

#### Hypothesis

This is the cheapest structural scheduler upgrade and should improve `CU`
without changing acceptance semantics.


### Stage 4: Adaptive Draft Budget

#### Why fourth

Once scheduling is more reactive, the next likely issue is that the system still
drafts too far past the blocked frontier.

That wastes updates and lowers the chance that each source refresh yields a new
accepted whole-word prefix.

#### Minimal policy

Drive `partial_max_new_tokens` from frontier distance:

- blocked very near: draft `4-8` tokens
- medium distance: draft `8-16`
- far distance or no known block: draft `16-32`

Do not change the acceptance semantics.
Only change how far the model speculates.

#### Experiment

Compare:

- fixed current draft budget
- frontier-distance-adaptive draft budget

#### What to look for

- `LongYAAL CU`
- average accepted tokens per update
- average candidate tokens per update
- word-boundary trim rate

#### Hypothesis

This is more likely to give moderate but consistent `CU` gains than dramatic
ones, especially once chunking and early start are already improved.


### Stage 5: Finer Source Frontier

#### Why fifth

At this point, if `CU` is still above target, the word-complete source frontier
is probably the main semantic bottleneck.

The current system is already at:

- `inaccessible_ms = 0`

So further gains need finer source evidence, not a looser word boundary.

#### Experiment direction

Prototype a frontier over:

- source tokenizer units
- or finer ASR timestamped segments if available

instead of only completed source words.

#### What to look for

- whether accepted growth can happen earlier inside the same source word span
- whether German quality remains stable

#### Hypothesis

This is one of the strongest remaining `CU` levers, but it is more invasive
than the earlier stages and should not be attempted before the easier source-time
levers are exhausted.


### Stage 6: Provenance-Aware Acceptance

#### Why sixth

If the system is still missing the target after the earlier stages, the problem
is likely no longer scheduling alone.

It is that the current observer is still too conservative when a drafted token is
supported by:

- already accepted target context

rather than by:

- speculative future target context

#### Experiment direction

Augment the observer to distinguish:

- source support
- accepted-prefix support
- speculative-suffix support

Then allow acceptance when:

- source anchor is already accessible
- accepted-prefix support is strong
- speculative-suffix self-support is weak

#### What to look for

- increase in accepted tokens per update
- reduction in frontier stops that are later revealed to be unnecessary
- quality drift from over-acceptance

#### Hypothesis

This is probably the strongest medium-term semantic improvement for `CU`, but it
is more complex than the earlier scheduler and cadence changes.


### Recommended Order

If we want the shortest path to an answer, the order should be:

1. freeze an early-start setting around `3.0 s` and stop sweeping it
2. lower `chunk_ms`
3. add token-level blocked-frontier scheduling
4. add adaptive draft budget
5. add finer source frontier granularity
6. add provenance-aware acceptance

This order is deliberate.

Stage 1 is now mostly done.

The next four steps mostly preserve the current semantics and attack the
largest remaining likely `CU` bottlenecks with minimal conceptual change.

The last two steps are stronger architectural moves and should only be taken
once we know the simpler source-time changes are insufficient.


### Recommended Minimal Run Matrix

To avoid combinatorial explosion, the next concrete run matrix should be:

1. Stop the start-gate sweep at the current conclusion:
   - use `3.0 s` as the representative early-start setting
   - do not spend more full-talk runs on `1.5` until another bottleneck changes
2. Chunk sweep using that fixed start gate
3. Scheduler A/B at the best `(start_gate, chunk_ms)`
4. Draft-budget A/B at the best configuration so far

Only if the best run after those four steps is still clearly above `2 s`
`LongYAAL CU` should we move to:

5. finer frontier granularity
6. provenance-aware acceptance

Because each full-talk run is expensive even with a hot kernel, the intended
workflow is:

1. validate the mechanism on one audio
2. only then spend full benchmark runs on the corresponding axis


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
- the newest rerun suggests that the main remaining `CU` gap is now dominated by
  source-time policy rather than observer compute
- the completed Stage 1 sweep shows that start-gate tuning is not the dominant
  source-time bottleneck either
- the next likely wins for `< 2 s` `LongYAAL CU` will come from:
  - lower or adaptive `chunk_ms`
  - token-level frontier scheduling
  - finer source accessibility granularity
  - larger safe accepted growth per update
- lowering observation cost is still crucial, but mainly for the `CA` axis


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


## 2026-04-15 Latency Push To `< 2 s` `LongYAAL CU`

### Goal

Drive `LongYAAL CU` below `2000 ms` on a single representative talk
(`test-set/audio/ccpXHNfaoy.wav`, `360 s`) while keeping translation
quality readable. The user explicitly accepted quality loss as a trade-off
("On est OK de perdre en qualité pour arriver à ces métriques").

### Diagnostic Of The Starting Point

The hot-kernel `compute_unaware_chunk800_20260415T154922Z` baseline was
profiled per stream update (411 updates, 360 s audio):

- mean `draft_decode`: `679 ms` (dominant cost)
- mean `prompt_cache_restore`: `60 ms`
- mean `alignment_probe`: `58 ms`
- mean accepted: `3.78` German words per update
- `partial_followup_max_new_tokens = 16` but accepts only `~6-8` tokens
- `140 / 411` updates produced `0` accepted tokens
- `67 / 411` updates (`16 %`) hit `alignatt:rewind` → entire draft rejected
- only `4 / 411` MT calls were actually skipped by the scheduler

Conclusion: `chunk_ms`, draft-budget oversizing, and rewind
reject-all were the dominant `CU` and `CA` drains.

### Implemented Changes

1. **Truncate-on-rewind** in `cascade_mt_backend.py`. The probe loop now
   truncates the drafted suffix at the first rewind token, mirroring the
   `source_frontier` handling, instead of dropping the entire suffix.
   This recovered the safe prefix of `~46` updates per full talk.
2. **Latency-shaped runtime overrides**. `partial_max_new_tokens = 16`,
   `partial_followup_max_new_tokens = 8`, `min_start_seconds = 2.0`,
   `max_history_utterances = 1`, `rewind_threshold = 8`, `inaccessible_ms = 0`.
3. **Lower `chunk_ms`**. Swept from `800` down to `400` to find the
   point where `LongYAAL CU` crosses `2 s` without collapsing quality.
4. **New experiment harness** `run_latency_experiment.py` that hot-reloads
   the warm `.venv-inference` kernel and runs one audio per call.

### Sweep Results

| tag | `chunk_ms` | `min_start` | partial caps | history | BLEU | chrF | `LongYAAL CU` | `LongYAAL CA` |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline `compute_unaware_chunk800_20260415T154922Z` | 800 | 5.0 | 48 / 16 | 0 | 38.76 | 68.09 | 3716.85 | 5595.86 |
| `latency_v3_chunk800_min2_cap16_cap8` | 800 | 2.0 | 16 / 8 | 0 | 37.16 | 67.07 | 3678.67 | 5621.51 |
| `latency_v5_chunk600_cap16_cap8` | 600 | 2.0 | 16 / 8 | 0 | 31.38 | 64.44 | 2376.87 | 4891.11 |
| `latency_v6_chunk500_cap16_cap8` | 500 | 2.0 | 16 / 8 | 0 | 28.32 | 62.19 | 2135.42 | 3045.64 |
| `latency_v7_chunk450_cap16_cap8` | 450 | 2.0 | 16 / 8 | 0 | 26.87 | 62.93 | 1907.98 | 3064.72 |
| **`latency_v9_chunk450_cap16_cap8_hist1`** | **450** | **2.0** | **16 / 8** | **1** | **28.15** | **63.59** | **1948.51** | **3006.81** |
| `latency_v10_chunk450_cap16_cap8_hist2` | 450 | 2.0 | 16 / 8 | 2 | 26.52 | 63.51 | 1995.36 | 3230.23 |
| `latency_v8_chunk450_cap24_cap12` | 450 | 2.0 | 24 / 12 | 0 | 26.87 | 62.93 | 1907.98 | 3035.42 |
| `latency_v1_chunk400_min2_cap16_cap8` | 400 | 2.0 | 16 / 8 | 0 | 23.37 | 60.88 | 1348.17 | 2172.94 |
| `latency_v2_chunk400_min2_cap24_cap12_inacc200` | 400 | 2.0 | 24 / 12 | 0 | 22.81 | 60.74 | 1298.18 | 2082.69 |
| `latency_v4_chunk400_cap32_cap20` | 400 | 2.0 | 32 / 20 | 0 | 22.81 | 60.74 | 1298.18 | 2087.01 |

Per-update timings under the `latency_v1` regime (`chunk_ms = 400`,
`cap16 / cap8`):

- `800` updates total (vs `411` at `chunk_ms = 800`)
- mean `draft_decode`: `279 ms` (vs `679 ms`)
- mean `total` per call: `393 ms` (vs `800 ms`)
- `alignatt:rewind` events: `21` (vs `67`)

### Observations

1. `chunk_ms` is by far the strongest `CU` lever in the current architecture.
   Halving it from `800` to `400` nearly cuts `LongYAAL CU` in three.
2. At `chunk_ms = 800`, slashing the partial draft budget barely moves `CU`
   (`-37 ms`) and only loses `~1.6 BLEU`. So at the current cadence,
   compute is not the bottleneck for `CU`; the cadence is.
3. Increasing the partial draft budget at `chunk_ms = 400` or `chunk_ms = 450`
   does not change BLEU or `CU` (`v1 = v2 = v4` numerics). The model already
   stops naturally at `<turn|>` well before exhausting its budget; a larger
   cap mostly wastes `CA`.
4. Adding one utterance of history (`max_history_utterances = 1`) at
   `chunk_ms = 450` lifts BLEU from `26.87` to `28.15` while only spending
   `+40 ms` of `CU`. Two utterances of history is worse on both axes.
5. Below `chunk_ms ≈ 450`, ASR sentence-finalization fires too quickly for
   the current prompt contract: the same `Hi, I'm Siyu Yuan.` /
   `From Fudan University.` boundary that the baseline merges into one
   smooth German sentence is split into two German sentences in `v1`. The
   meaning is preserved but the surface drops below `BLEU = 24`.
6. Truncate-on-rewind is a strict semantic improvement: rewinds drop from
   `67` to `~21` per full talk and the recovered prefix translates into
   accepted growth that previously vanished.

### Conclusion

The `latency_v9` operating point hits the target:

- `LongYAAL CU = 1948.51 ms` (target was `< 2000 ms`)
- `LongYAAL CA = 3006.81 ms`
- `BLEU = 28.15`, `chrF = 63.59`
- translation is fluent, mostly correct German with some sentence
  fragmentation; comprehensible at simultaneous-translation quality.

Recommended operating point for the `< 2 s` `CU` regime:

- `chunk_ms = 450`
- `min_start_seconds = 2.0`
- `partial_max_new_tokens = 16`
- `partial_followup_max_new_tokens = 8`
- `max_history_utterances = 1`
- `translation_alignatt_rewind_threshold = 8`
- `translation_alignatt_inaccessible_ms = 0`
- per-token rewind truncation in `cascade_mt_backend.py` (kept as a
  permanent semantic fix, not a tuning knob).

Quality is roughly `~10 BLEU` below the slow `chunk_ms = 800` baseline,
which is consistent with the user's stated trade-off. Further gains likely
require the unfinished `Direction B / E / F / G` work in
`ALIGNATT_LLM.md` (cheap inline observer, serving-optimized head set,
explicit cache branches), which would let us push `chunk_ms` lower without
the current sentence-fragmentation tax on quality.


## 2026-04-15 Reproducibility Pass

### What Was Broken In The Original `< 2 s` Claim

The `latency_v*` numbers in the previous section were collected by a
harness that hot-reloaded the Python modules but **reused the old
`core.mt_backend` instance**. Two consequences:

1. Backend code edits in `cascade_mt_backend.py` were not guaranteed to
   take effect in the next run, because methods were still bound to the
   pre-reload class object.
2. The backend instance owns the `PromptCacheState`, so prompt KV cache
   contents could leak across runs and silently bias latency.

This was a real reproducibility hazard for a paper claim, even if the CU
numbers happened to be correct.

### What The Rewritten Harness Does

`run_latency_experiment.py` now:

- saves only the hot weights (ASR + Gemma model + tokenizer) across runs;
- rebuilds `mt_backend` from the freshly reloaded `cascade_mt_backend`
  module via `core.rebuild_mt_backend_preserving_weights(...)`;
- calls `mt_backend.reset_caches()` explicitly at the start of every run;
- collects a `run_provenance` block (git SHA, dirty state, dirty file
  list, CLI args, backend class name, backend module, cache-reset policy,
  harness timestamp) and writes it into the manifest.

`BaseMTBackend` now exposes a `reset_caches()` method; on the
`TransformersAlignAttGemmaMTBackend` subclass it replaces
`self.prompt_cache` with a fresh `PromptCacheState()`.

### Revalidated `< 2 s` Operating Point

Two back-to-back hot-kernel runs on `test-set/audio/ccpXHNfaoy.wav`
with the recommended defaults
(`chunk_ms = 450`, `min_start_seconds = 2.0`,
`partial_max_new_tokens = 16`, `partial_followup_max_new_tokens = 8`,
`max_history_utterances = 1`,
`translation_alignatt_rewind_threshold = 8`,
`translation_alignatt_inaccessible_ms = 0`) gave bit-exact scores:

| run | BLEU | chrF | XCOMETXL | `LongYAAL CU` | `LongYAAL CA` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `outputs/revalidate_phaseA_v2` | 28.22 | 63.53 | 0.8622 | **1747.19** | 2186.21 |
| `outputs/revalidate_phaseA_v2_rerun` | 28.22 | 63.53 | 0.8622 | **1747.19** | 2195.95 |

XCOMETXL was run on CPU (`CUDA_VISIBLE_DEVICES=""`) because the warm
inference kernel holds the A100 for ASR + Gemma; eviction would have cost
a ~5 min reload for no scientific gain on this comparison.

Both runs landed comfortably below the `2000 ms` `LongYAAL CU` target,
with identical translation text and identical CU. The only number that
moves between the two is `LongYAAL CA`, which is wallclock-sensitive.

Compared to the previously recorded `latency_v9` point (`CU = 1948.51`),
the revalidated run on the same operating point is ~`200 ms` faster. The
delta is consistent with the harness fix: the rebuilt backend and the
explicit `reset_caches()` remove any stale prompt-KV state that the old
harness could have been dragging into the first few chunks of a fresh
run.

### Regression Test

`test_alignatt_rewind_keeps_safe_prefix_up_to_offending_token` in
`test_cascade_mt_backend.py` locks in the truncate-on-rewind invariant:
when the aligned source position jumps backward past
`translation_alignatt_rewind_threshold`, the policy must return
`unsafe_reason = "rewind"` on the offending token only, so the caller's
accumulate-then-break loop keeps the safe prefix and rejects just the
suffix. This is the Whisper-style behaviour adapted to the LLM
self-attention setting; rejecting the whole draft on a rewind would
violate monotone acceptance.

### Non-Obvious Things Worth Remembering

- `importlib.reload(qwen3asr_gemma_cascade_core)` resets the module-level
  `mt_backend` to `None`. The harness therefore has to save a reference
  to the pre-reload backend *before* reloading, then pass it into the
  rebuild helper so that the hot weights survive.
- Quality (BLEU / chrF) is **identical across the two revalidated runs**
  and `draft_decode` is greedy with `temperature = 0`, so any residual
  CU noise between runs is compute-time jitter only. This makes the
  harness a suitable base for future CU ablations at one-audio scale.
- `LongYAAL CA` is not bit-exact across runs; it depends on wallclock
  elapsed time per update. Only `LongYAAL CU` should be used as a
  reproducibility signal.
