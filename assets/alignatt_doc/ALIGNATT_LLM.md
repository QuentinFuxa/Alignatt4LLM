# AlignAtt on a Causal LLM

## Scope

This note consolidates the current analysis of AlignAtt in the English ASR -> German MT cascade built around:

- `Qwen3-ASR` on `vLLM`
- `Gemma 4 E4B` on `transformers`
- runtime AlignAtt derived from selected Gemma attention heads

The goal is to answer a precise design question:

How should we implement AlignAtt on a causal LLM in the most principled and performant way, given that canonical AlignAtt was originally designed for encoder-decoder cross-attention?

This note is based on:

- `assets/alignatt_doc/alignatt_markdown.md`
- `assets/alignatt_doc/alignatt_whipser.py`
- `assets/alignatt_doc/E4B_ALIGNATT_CASCADE_DESIGN.md`
- `qwen3asr_gemma_cascade_core.py`
- `cascade_mt_backend.py`


## Short Conclusion

The current cascade is conceptually moving in the right direction:

- `accepted_target` is treated as the only reusable truth
- AlignAtt decides emission outside the model
- the MT model remains free to draft beyond the accepted prefix

However, the implementation still pays the cost of treating a causal LLM too much like Whisper.

That is the main issue.

In Whisper:

- alignment comes from a dedicated decoder -> encoder cross-attention channel
- the source side is structurally separated from prompt and target history
- extracting alignment is relatively natural

In a causal LLM:

- alignment is inferred from decoder self-attention
- source tokens compete with prompt tokens and previously generated target tokens
- the attention signal is noisier and more ambiguous
- the runtime cost of observing it naively can be much higher

So the best future architecture is not:

`slow LLM decoding with attention capture at every token`

It is much closer to:

`fast LLM drafting + cheap alignment probe + explicit monotone acceptance state`


## Latest Empirical Read (2026-04-15)

The newest full-talk `compute unaware` rerun at `chunk_ms = 800` reached:

- `BLEU = 38.7556`
- `chrF = 68.0866`
- `LongYAAL CU = 3716.8469`

So the current AlignAtt-first LLM design is already in a good quality regime,
but it is still far from the `< 2 s` `LongYAAL CU` target.

### Fresh diagnostic on the 2026-04-15 run (411 updates, 360 s audio)

- Mean `draft_decode` per call: `679 ms` (dominant cost).
- Mean `prompt_cache_restore`: `60 ms`.
- Mean `alignment_probe`: `58 ms`.
- `partial_followup_max_new_tokens = 16`, but mean accepted is `3.78` target
  words ≈ `6-8` tokens.
- `140 / 411` updates produce `0` accepted tokens.
- `67 / 411` updates (`16%`) are full-draft rejections on the rewind guard.
- The scheduler skipped only `4 / 411` MT calls: almost every chunk triggers MT.

So the dominant structural `CU` problems today are:

1. the MT scheduler runs on virtually every chunk even when the accepted
   prefix barely grows
2. the draft budget is much larger than the number of tokens that survive
   acceptance
3. the rewind guard rejects the entire new suffix, throwing away the
   already-safe prefix of the same draft

We also now have a partial full-talk start-gate sweep with everything else held
fixed:

| `min_start_seconds` | First non-empty / accepted emission | BLEU | chrF | LongYAAL CU |
| ---: | ---: | ---: | ---: | ---: |
| `5.0` | `5.6 s` | 38.7556 | 68.0866 | 3726.8594 |
| `3.0` | `4.0 s` | 38.7556 | 68.0866 | 3698.2681 |
| `2.0` | `4.0 s` | 38.7556 | 68.0866 | 3698.2681 |

This is a useful correction to the earlier intuition.

Lowering the hard start gate does improve first-token responsiveness, but on
the current full-talk `chunk800` setting it barely changes corpus-level
`LongYAAL CU`.

So the main remaining `CU` problem is probably not:

- the observer compute alone
- nor the hard start gate alone

It is much more likely to be:

- coarse update cadence
- coarse scheduler gating
- coarse source accessibility granularity
- insufficient accepted growth per update


## Canonical AlignAtt in Whisper

Canonical AlignAtt uses cross-attention:

- for each target token, find the most attended source frame
- if that aligned source position falls into the last inaccessible source region, stop emission before that token

Whisper implementation details that matter:

- selected alignment heads only
- normalization over generated target-token axis
- median filtering on source axis
- averaging across selected heads
- argmax over source-side positions
- local rewind guard
- drop the last incomplete word on non-final updates

This is a very clean fit for an encoder-decoder model because the alignment channel is structurally dedicated.


## What Changes on a Causal LLM

This is the central design shift.

On Gemma, there is no encoder-decoder cross-attention for translation. AlignAtt is recovered from self-attention heads that have translation-alignment behavior.

That has several consequences.

### 1. The attention signal is intrinsically mixed

A target token can attend to:

- source tokens
- system prompt tokens
- history/context tokens
- assistant prefill tokens
- previously generated target tokens

So unlike Whisper, "best source position" is only meaningful if the head is truly source-oriented for that token.

This means a good LLM AlignAtt policy should not rely only on:

- source-restricted argmax

It should also consider:

- how much attention mass actually lands on the source
- whether the selected heads agree
- whether the best accessible source position beats the best inaccessible one with margin
- whether the head is mostly reading target history instead of source evidence

### 2. KV cache is more central

In Whisper, the alignment mechanism naturally lives inside a streaming seq2seq loop.

In a causal LLM, the performance story is dominated by:

- prompt reuse
- assistant prefill reuse
- branch reuse after the accepted prefix
- avoiding expensive replays in `eager` mode

So AlignAtt must be designed with KV cache as a first-class concern.

One concrete implication that only appears once we implement this on Gemma 4:

- some decoder layers use KV sharing (`is_kv_shared_layer`)
- some layers are full attention, others are sliding attention

That means a "fast q/k reconstruction" probe cannot assume that every layer owns its
own local `k_proj(hidden_states)` in the same way.

For an LLM-native AlignAtt probe, the clean rule is:

- always recover `Q` from the current layer input for the drafted suffix
- prefer reading `K` from the runtime KV cache produced by the fast forward itself
- for KV-shared layers, resolve keys through the shared source layer, not the local layer
- for sliding-window layers, be careful: the visible prompt prefix may already be truncated by the cache policy, so replay-free reconstruction must respect the current visible window rather than the original full prompt

In other words, the probe should be KV-cache-native, not just attention-math-native.

### 3. Reordering is legitimate and frequent

For English -> German, good partial drafts often need:

- verb-final subordinate clauses
- `um ... zu ...`
- delayed lexical verbs
- object insertions
- clause packaging that looks non-monotone locally

So a naive monotonicity or rewind heuristic can suppress exactly the reordering that makes German good.

This is why the current design is right to avoid strong temporal agreement constraints and to keep only a permissive intra-draft rewind guard.


## Current Cascade: What Is Good

Several foundational choices are correct.

### Explicit semantic state

`qwen3asr_gemma_cascade_core.py` separates:

- `draft_target`
- `accepted_target`

This is the right semantic split.

`accepted_target` is the only thing that should:

- be emitted
- be reused as assistant prefill
- anchor the final pass

That part matches the intended design in `E4B_ALIGNATT_CASCADE_DESIGN.md`.

### Prompt contract

The current structured prompt contract is also right:

- user contains the full current source prefix from sentence start
- assistant is prefilled with the accepted target prefix only

This is the cleanest contract for incremental causal decoding.

### Runtime acceptance outside the model

The system correctly keeps emission policy outside Gemma:

- Gemma drafts
- runtime inspects alignments
- runtime truncates acceptance

This is the right architecture in principle.

### Backend split

The current operational split is also sensible:

- ASR on `vLLM`
- MT on `transformers`

For the current environment, attention-visible MT serving matters more than uniform serving stacks.


## Current Cascade: Main Problems

The main problems are not conceptual purity. They are cost, observability design, and a few possible semantic mismatches.

### 1. The probe is still expensive even though decoding is already two-pass

The current code is better than a naive Whisper port.

In `cascade_mt_backend.py`, partial MT already does:

- a fast draft pass with `sdpa`
- then a second replay pass on a fast path when possible via selected-layer input capture
- and only falls back to `eager` when the fast q/k observer is unavailable
- then alignment-based truncation outside the model

So the main cost is no longer "capture attention on every decode step".

The real costs are:

- replaying the whole drafted suffix at least once
- cloning and restoring full prompt KV snapshots
- carrying full-layer KV state even though the observer only needs a tiny slice of it
- reconstructing or resolving key states for sliding-window and KV-shared layers
- occasionally materializing full selected-layer attentions on the `eager` fallback path

This is already a meaningful improvement over token-by-token live capture, but it is still too expensive for a causal LLM.

The practical corollary is that the first serious optimization target is not "less
Python around the probe". It is:

- stay on the fast attention backend for the replay
- read as much as possible from the existing KV/cache state
- only fall back to explicit `k_proj` reconstruction where cache semantics make that necessary (for example some sliding-window cases)

### 2. The current implementation is much closer to the right architecture, but not yet LLM-native enough

The code already moved toward the right high-level shape:

- fast draft
- separate alignment probe
- explicit acceptance state

That part is good.

One earlier concern is now largely addressed in code.

The current Gemma backend already uses:

- a batched suffix replay
- `compute_prefix_online_alignatt_source_argmaxes`
- `IncrementalAlignAttTracker`
- left-to-right runtime stopping

So future drafted tokens no longer rewrite earlier token decisions through one full-suffix-global normalization pass.

That is a real improvement and much closer to canonical AlignAtt semantics.

What is still not LLM-native enough is different:

- the observer is still a replayed suffix module rather than a first-class part of fast decoding
- the decision signal is still mostly "source-local argmax", not a richer provenance decomposition over the causal context
- accepted-prefix support and speculative-suffix support are not distinguished explicitly even though they should have very different safety implications

### 3. Source-only argmax is not enough on a causal LLM

At runtime, source attention rows are extracted only over the source positions, then normalized and argmaxed there.

That is useful, but insufficient as a confidence signal for a causal LLM.

A token can have:

- a plausible best source argmax
- while still relying mostly on non-source context

So the future policy should include source-vs-non-source diagnostics, not only source-local argmax.

### 4. Prompt/KV reuse is semantically clean but still operationally heavy

The current prompt cache machinery snapshots and restores KV state.

This is much better than recomputing from scratch, but it still looks heavier than the ideal design:

- cloning cache tensors
- restoring full dynamic cache objects
- snapshotting values that the AlignAtt observer never reads
- replaying prompt deltas frequently

For a causal LLM, cache branch management is the optimization problem, not a side detail.

### 5. The MT scheduler is still too eager

This section is partly outdated now.

The current scheduler already skips several low-value probes:

- unchanged source prefix
- unchanged accessibility frontier
- blocked frontier not yet reached
- stall-based override only after time has passed

That is directionally right.

The remaining problem is that the gating is still coarse:

- it is mostly unit-level, not token-level
- it does not estimate how many target tokens are plausibly acceptable before drafting
- it does not yet shrink the draft budget aggressively when the last blocked position is still very near

### 6. Prefix bookkeeping looks cleaner than feared, but exact online equivalence is still unproven

After reading the code more carefully, the earlier bookkeeping suspicion is weaker than it first looked.

Statically, `accepted_token_ids` are re-encoded from the full `acceptance_text`, so monotone-prefix checking appears to compare full semantic prefixes, not only the newly accepted suffix.

That is good.

The more important correctness question is different:

- does the batched suffix probe make the same accept / wait decision that an online Whisper-style prefix-growing loop would make?

That should still be tested explicitly, especially with:

- non-empty `assistant_prefill`
- accepted-prefix continuation
- German reorderings near the frontier

There is also a strong hint in the code that this exact question was already anticipated:

- `IncrementalAlignAttTracker` now exists and is used in the runtime probe

So the next correctness step is probably not another semantic rewrite.

It is:

- validating that the current batched-prefix-online probe matches the intended online behavior under realistic prefills and reorderings

### 7. Runtime head selection is optimized for quality signal, not serving cost

The current top heads are spread across several layers.

On paper that is fine.

At runtime, however, the real cost is driven heavily by:

- how many layers must expose attentions
- not only how many heads are used inside those layers

So serving head selection should optimize:

- translation signal
- calibration robustness
- layer concentration
- capture cost

not only raw translation-score rank.


## What the Best LLM AlignAtt Design Probably Looks Like

The strongest direction is to decouple:

- fast drafting
- alignment observation
- semantic acceptance

### Layer 1: Fast decoder

Use the fastest available causal decode path for Gemma:

- `sdpa` or flash-compatible path
- normal KV reuse
- no attention capture on the critical path

Its only job is to draft a short continuation.

### Layer 2: Alignment probe

After drafting a short suffix, run a cheaper alignment-observation step for that suffix only.

This probe should answer:

- which source positions the drafted tokens align to
- whether those positions are accessible
- whether the source evidence is strong enough
- whether there is suspicious rewind

This probe should be much cheaper than "full eager attention capture on every decode step".

### Layer 3: Acceptance state

The semantic state remains:

- `draft_target`
- `accepted_target`

with the invariant:

- only `accepted_target` survives across updates

This part of the current design should be preserved.


## Recommended Architectural Directions

### Direction A: Prefix-online batched probe

This direction is now effectively the current backend shape, and it was the right move.

The key idea:

1. decode a short suffix quickly with `sdpa`
2. replay that suffix once under an alignment-visible path
3. extract source rows for all drafted tokens in one batch
4. feed those rows through a prefix scan that mimics Whisper's online growing-prefix semantics
5. accept the longest safe prefix

The important detail is step 4.

Do not compute the final decision for token `i` from statistics that already include tokens `i+1...n`.

Instead, the probe should maintain running normalization and rewind state while scanning the drafted suffix from left to right.

That is exactly the kind of logic `IncrementalAlignAttTracker` is pointing toward.

Advantages:

- keeps the current two-pass structure
- preserves the latency win of batched replay
- gets much closer to canonical AlignAtt semantics
- avoids paying `N` eager replays for `N` drafted tokens

The remaining work here is no longer first implementation.

It is:

- validating equivalence more carefully
- instrumenting the remaining replay cost
- deciding when this batched observer should be replaced by a lighter inline observer

### Direction B: Cheap alignment probe from selected q/k states

This is more ambitious and likely the cleanest long-term LLM solution.

Instead of asking the full model to expose attention weights during decoding:

- capture only the query/key states needed for selected alignment heads
- compute alignment scores outside the main decode path

For a causal LLM, this should be made more concrete than "maybe capture q/k".

The LLM-native observer would ideally:

- keep fast decoding on `sdpa` / flash
- cache the source-side key slice for selected heads
- reuse accepted-prefix cache state without cloning the whole prompt branch
- compute scores only for:
  - drafted query rows
  - selected heads
  - source columns
- optionally keep a tiny summary of non-source competition:
  - best non-source logit
  - total non-source mass

In other words, the observer should avoid materializing the full:

- `heads x drafted_tokens x full_context`

attention tensor if the decision only needs:

- source alignment
- source-vs-non-source competition
- frontier safety

Conceptually this would turn AlignAtt into:

- a small observer attached to a fast decoder

rather than:

- a property of the decoder execution mode itself

This is also the cleanest place to exploit the fact that a causal LLM already revolves around KV cache.

The observer should piggyback on cached keys, not fight them.

The next LLM-native refinement should be even more explicit:

- during fast autoregressive drafting, capture the selected layer inputs for the just-materialized token on the same `sdpa` pass
- score that query immediately against cached observer keys
- accept online with at most a one-token lag, or pay one tiny extra micro-step to resolve the last drafted token

That would turn the current `two-pass` observer into a `one-and-a-half-pass` observer:

- fast draft remains unchanged
- the observer becomes nearly replay-free
- the only unavoidable cost is causal timing, not a full suffix replay

### Direction C: Smarter scheduler before MT

Do not run partial MT on every ASR perturbation.

Run MT only when at least one of the following is true:

- `accessible_unit_count` increased
- a new complete source word became available
- punctuation or a strong clause boundary appeared
- the source delta exceeded a minimum threshold
- the accepted target has been stalled for enough audio time to justify a probe

The current code already uses the previous AlignAtt result through `blocked_source_unit_index`.

That is a good first step, but the next smarter version should refine it.

If the last probe stopped because token `i` aligned to source position `p`, and the source frontier is still before `p`, then another MT call is usually wasted.

That gives a predictive scheduler:

- store the first blocked source position from the previous probe
- skip MT until the accessible frontier reaches or nears that position
- override only on strong textual changes or long stall

This is likely one of the cheapest wins.

The scheduler should also drive the draft budget itself:

- if the last blocked source position is only slightly ahead of the current frontier, draft very short
- if the frontier jumped far, allow a longer suffix
- if repeated probes stop on the first or second drafted token, shrink `max_new_tokens` aggressively instead of paying for long speculative tails

### Direction D: Confidence-aware LLM AlignAtt

A future LLM AlignAtt policy should combine:

- best source argmax position
- source attention mass
- margin between best accessible and best inaccessible source token
- agreement across heads
- optional target-history dominance penalty

This would make acceptance more LLM-native and more robust than source-only argmax.

For a causal LLM, the most important missing detail is provenance partitioning.

The observer should distinguish at least:

- source tokens
- accepted target prefix tokens
- speculative drafted suffix tokens
- all other prompt tokens

This matters because these are not equivalent competitors.

Strong reliance on the accepted prefix can be perfectly legitimate:

- it often reflects fluent target continuation anchored on already committed text

Strong reliance on the speculative suffix is much less safe:

- it means the token is being justified by other not-yet-accepted target material

So the best LLM AlignAtt signal is not merely:

- `source vs non-source`

It is closer to:

- `source vs accepted-prefix vs speculative-suffix vs other-prompt`

### Direction E: Serving-optimized head set

Freeze a runtime serving set chosen for:

- alignment quality
- stability across examples
- low number of distinct layers
- high frontier discriminability
- low dependence on awkward cache semantics

The serving set should not necessarily be the raw top-K TS list.

In practice, this likely means preferring heads that are:

- concentrated in a narrow middle-layer span
- stable under prefix growth
- less sensitive to sliding-window truncation
- less sensitive to KV-sharing reconstruction edge cases

### Direction F: Adaptive observer and draft budget

Not every update needs the same observer cost.

A strong serving pattern for LLM AlignAtt is:

1. cheap probe with a tiny head set concentrated in a small layer span
2. if the token is far from the source frontier and confidence is high, accept immediately
3. only if the decision is near-boundary or low-confidence, escalate to:
   - more heads
   - more layers
   - or the full observer

This is especially attractive on a causal LLM because the expensive part is often:

- how many layers must expose alignment evidence

not merely how many final heads are averaged.

The same principle should apply to draft length:

1. near the frontier, ask for a tiny suffix
2. far from the frontier, allow a larger suffix
3. after repeated early stops, cut the suffix budget before cutting quality elsewhere

### Direction G: Explicit three-branch cache design

The current cache code is semantically correct, but the optimal LLM structure is even more explicit.

Treat cache state as three branches:

- immutable base prompt branch:
  - system prompt
  - examples
  - committed history
- accepted-prefix branch:
  - base prompt plus `accepted_target`
- ephemeral draft branch:
  - accepted-prefix branch plus speculative suffix

Then design the observer around those same branches.

That makes it possible to:

- reuse the accepted branch across updates
- avoid cloning the whole prompt cache for every probe
- cache the source-side observer keys together with the accepted branch
- keep speculative draft state disposable

There should also be a distinct observer cache, not only a decode cache.

That observer cache can store:

- selected-head or selected-KV-head source keys
- partition boundaries for:
  - source
  - accepted prefix
  - speculative suffix
- cheap logsumexp summaries or other competition summaries if they prove useful

The critical implementation point is:

- the observer does not need full values
- the observer does not need every layer
- the observer often does not need every prompt token equally

So cloning full `K` and `V` for all layers should be treated as a transitional implementation cost, not as the target architecture

### Direction H: Frontier-aware head calibration

Offline alignment-head discovery and online simultaneous emission are not the same objective.

A head can be excellent for offline token alignment yet mediocre for online emission control if it:

- fires too late near the frontier
- tracks source tokens well offline but spends little probability mass on the source online
- depends heavily on sliding-window or shared-KV behavior that is awkward at serving time

So runtime head calibration should explicitly optimize:

- frontier discrimination
- prefix-growth stability
- source-mass robustness
- serving cost

not just offline translation-score rank


## Suggested Critical View of the Current Implementation

If we are strict:

- the current cascade has the right semantics
- the current cascade does not yet have the right LLM-native mechanics

That matters because the latest results suggest the main remaining `CU`
bottleneck is no longer:

- frontier conservatism

and also not simply:

- hard start delay

For `CU`, it is much more likely to be:

- coarse `chunk_ms` quantization
- over-coarse MT triggering and blocked-frontier gating
- insufficient accepted growth per update
- source granularity that is still too word-level

For `CA`, the major remaining issues are still:

- observation cost
- cache handling overhead
- replay probing, especially on the `eager` fallback

So if we keep tuning only:

- `inaccessible_ms`
- `rewind_threshold`

we will probably hit diminishing returns.

The next gains are much more likely to come from:

- for `CU`:
  - chunking
  - scheduling
  - acceptance policy
- for `CA`:
  - cheaper alignment probing
  - better cache structure


## Empirical Sweep To Cross Below 2 s (2026-04-15, `ccpXHNfaoy.wav`)

After implementing the principled fixes below, a focused single-audio sweep
shows the clear shape of the trade-off:

| tag | `chunk_ms` | `min_start` | partial caps | history | BLEU | chrF | `LongYAAL CU` | `LongYAAL CA` |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline `compute_unaware_chunk800_20260415T154922Z` | 800 | 5.0 | 48 / 16 | 0 | 38.76 | 68.09 | 3716.85 | 5595.86 |
| `latency_v3_chunk800_min2_cap16_cap8` | 800 | 2.0 | 16 / 8 | 0 | 37.16 | 67.07 | 3678.67 | 5621.51 |
| `latency_v5_chunk600_cap16_cap8` | 600 | 2.0 | 16 / 8 | 0 | 31.38 | 64.44 | 2376.87 | 4891.11 |
| `latency_v6_chunk500_cap16_cap8` | 500 | 2.0 | 16 / 8 | 0 | 28.32 | 62.19 | 2135.42 | 3045.64 |
| `latency_v7_chunk450_cap16_cap8` | 450 | 2.0 | 16 / 8 | 0 | 26.87 | 62.93 | 1907.98 | 3064.72 |
| `latency_v9_chunk450_cap16_cap8_hist1` | 450 | 2.0 | 16 / 8 | 1 | **28.15** | **63.59** | **1948.51** | **3006.81** |
| `latency_v10_chunk450_cap16_cap8_hist2` | 450 | 2.0 | 16 / 8 | 2 | 26.52 | 63.51 | 1995.36 | 3230.23 |
| `latency_v1_chunk400_min2_cap16_cap8` | 400 | 2.0 | 16 / 8 | 0 | 23.37 | 60.88 | 1348.17 | 2172.94 |

Relevant observations:

1. `chunk_ms` is by far the strongest `CU` lever on the current architecture.
   Dropping it from `800` to `450` nearly halves `LongYAAL CU` without any
   other change.
2. On `chunk_ms = 800`, aggressive draft-budget cuts barely move `CU`
   (`37 ms` improvement); quality drops by about `1.6 BLEU`. So at this
   cadence, compute is not the bottleneck.
3. The `latency_v9` point at `chunk_ms = 450` with one utterance of history
   is the sweet spot: `LongYAAL CU = 1948 ms`, `BLEU = 28.15`, `chrF = 63.59`.
   `CA` also collapses from `~5600 ms` to `~3000 ms` because the draft caps
   are sized to actual accepted growth.
4. Lower `chunk_ms` than `450` keeps improving `CU` but the ASR
   sentence-finalization cadence gets too fragmented for the current prompt
   contract, and BLEU collapses below `24`.
5. `truncate-on-rewind` (instead of `reject-all-on-rewind`) combined with
   smaller partial caps cuts rewind events from `67` to `21` per full talk
   and recovers acceptance growth that was previously discarded.

The principled recommendation for this phase is therefore to treat
`chunk_ms = 450`, `partial caps = 16 / 8`, `min_start_seconds = 2.0`,
`max_history_utterances = 1`, `rewind_threshold = 8` as the operating point
below `2 s` `LongYAAL CU`, with the per-token rewind truncation in
`cascade_mt_backend.py` as a standing semantic fix rather than a tuning knob.


## Focused Plan To Cross Below 2 s `LongYAAL CU`

The levers that move `CU` most directly, ranked by expected impact on a single
full talk:

1. **Chunk quantization**. Drop `chunk_ms` from `800` to `400`. Each accepted
   word is delayed on average by `chunk_ms / 2`; halving the chunk shaves
   roughly `200 ms` of pure CU quantization.
2. **Start gate**. Lower `min_start_seconds` to `2.0`. First-token UX matters
   for the early part of the stream even if corpus-level CU plateaued at `3.0`
   in the earlier sweep.
3. **Adaptive draft budget**. `partial_max_new_tokens = 48` and
   `partial_followup_max_new_tokens = 16` are wildly oversized for an average
   accept of `~4` words. Default the follow-up cap to `~8` tokens; keep an
   initial-partial cap of `~16` for the first probe after a sentence break.
   This has no CU downside because we do not accept those extra tokens anyway,
   and it unlocks the `chunk_ms = 400` regime on the CA axis.
4. **Truncate-on-rewind instead of reject-all**. Today `alignatt:rewind`
   wipes the entire drafted suffix, including its already-safe prefix. The
   principled change is to truncate before the offending token, the same way
   `alignatt:source_frontier` already does. That recovers `~16 %` of updates
   that currently produce zero growth.
5. **Skip MT when accepted prefix is unlikely to grow**. Extend the scheduler
   to skip MT when the accessibility frontier advanced by fewer than one
   source word since the last probe and no stall override has fired yet.
6. **Higher `translation_alignatt_rewind_threshold` for German reordering**.
   The current default is `8`. German often reorders far more than that, and
   rewinds are the single largest source of zero-growth updates.

Steps `1-3` are purely latency-shaping: they do not touch AlignAtt semantics.

Step `4` is the principled fix that makes the rewind guard align with the
source-frontier guard: both become "stop before the first unsafe token" rather
than "drop the whole suffix".

Steps `5` and `6` are orthogonal refinements that compound on top.


## Concrete Suggestions

### High Priority

- Keep the current `fast draft` plus prefix-online `alignment probe` split.
- Do not spend more full-talk benchmark runs on `min_start_seconds` alone until another bottleneck changes; `3.0` and `2.0` already plateau on `CU` in the current setup.
- Treat lower `chunk_ms` as the next highest-priority `CU` experiment.
- Refine the existing predictive scheduler from blocked unit indices to finer frontier-distance and draft-budget control.
- Build a serving head set optimized for both signal and layer concentration.
- Add a cheap first-tier observer and escalate only near the frontier or on low confidence.
- Add provenance-aware diagnostics that separate:
  - source
  - accepted-prefix
  - speculative-suffix
  - other-prompt

### Medium Priority

- Add confidence features beyond source-only argmax.
- Restructure cache handling into explicit base / accepted / draft branches plus an observer cache.
- Log per-update cost by phase:
  - prompt replay
  - draft decode
  - alignment probe
  - acceptance
- Add focused tests for prefix-online equivalence with non-empty `assistant_prefill`.
- Add a replay-free inline observer prototype on top of the existing selected-layer-input capture.

### Lower Priority but Valuable

- Explore external q/k-based alignment computation for selected heads.
- Compare replayed q/k probing vs inline one-and-a-half-pass probing.
- Evaluate whether a tiny dedicated alignment observer could approximate the same decisions more cheaply.


## Task List

### Phase 1: Correctness and Instrumentation

- [ ] Add a focused unit test that verifies the batched probe matches a prefix-growing online probe on the same drafted suffix.
- [ ] Add a focused unit test that verifies monotone acceptance across two consecutive partial updates with non-empty prefill.
- [ ] Add a focused unit test with German local reordering near the source frontier.
- [ ] Log per-update timings for:
  - prompt processing
  - prompt cache restore
  - decoding
  - attention capture
  - alignment filtering
- [ ] Log how many MT calls were made without any increase in `accessible_unit_count`.
- [ ] Log how often the same blocked source position caused repeated skipped or wasted probes.
- [ ] Log per-token provenance partitions:
  - source
  - accepted prefix
  - speculative suffix
  - other prompt

### Phase 2: Cheap Wins

- [ ] Refine the existing scheduler from blocked unit gating to token-level or near-frontier gating when possible.
- [ ] Add a scheduler gate that skips MT for tiny ASR perturbations that do not create a new complete source word.
- [ ] Make `max_new_tokens` adaptive to the last blocked position and frontier distance.
- [ ] Benchmark the latency impact of these gates on the existing talk-level run.

### Phase 3: Runtime AlignAtt Refactor

- [ ] Keep the existing two-pass partial decode mode:
  - fast draft of a short suffix
  - replay of that suffix for alignment inspection
- [ ] Keep the current prefix-online batched scan as the reference implementation.
- [ ] Validate the current `IncrementalAlignAttTracker` behavior against an explicit online oracle.
- [ ] Compare:
  - current replayed prefix-online probing
  - inline one-and-a-half-pass probing
  on quality, latency, and accepted-token growth.

### Phase 4: Better LLM-Native Decision Signal

- [ ] Add source-mass diagnostics for each drafted token.
- [ ] Add accessible-vs-inaccessible margin diagnostics.
- [ ] Add head-consensus diagnostics.
- [ ] Add accepted-prefix-vs-speculative-suffix provenance diagnostics.
- [ ] Evaluate whether these signals reduce false frontier stops or false rewinds.

### Phase 5: Serving Head Optimization

- [ ] Re-rank the current Gemma AlignAtt serving heads with a cost-aware criterion:
  - translation score
  - stability
  - number of distinct runtime layers touched
- [ ] Add frontier-discriminability and cache-friendliness to that ranking.
- [ ] Compare top-8 TS vs low-layer-span head clusters.
- [ ] Freeze a smaller serving set if it preserves decisions while reducing runtime cost.
- [ ] Prototype a tiered serving policy:
  - cheap head cluster first
  - full head set only on low-confidence or near-frontier cases

### Phase 6: Long-Term Architecture

- [ ] Prototype an external alignment probe based on selected q/k states rather than full attention capture.
- [ ] Make the observer cache-aware:
  - cache source-side keys with the accepted branch
  - score drafted queries against explicit provenance partitions
- [ ] Measure whether this can approximate current decisions with materially lower cost.
- [ ] Decide whether AlignAtt should remain part of decoding or become an external observer module.


## Recommended Design Principle Going Forward

The foundational principle should be:

AlignAtt is a decision policy over an explicit incremental MT state.

It should not dictate a slow decoding path if a cheaper observation mechanism can provide the same safety decision.

In other words:

- preserve the semantics
- redesign the mechanics

That is the right way to "break" AlignAtt onto a causal LLM.
