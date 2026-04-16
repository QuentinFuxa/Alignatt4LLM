# Feature Aligner Audit Note

## Bottom line

This is a **promising proof of concept**, not a convincing aligner result yet.

What is credible:

1. A very small transcript-conditioned aligner on frozen Gemma audio features can be trained quickly.
2. It can output monotone word timings.
3. It may eventually become a useful Qwen-independent inference-time aligner.

What is **not** credible yet:

1. the claim that the current result demonstrates robust generalization
2. the claim that it already beats the current Gemma eager aligner in a meaningful way
3. the claim that it is already ~100x faster in an end-to-end sense
4. the claim that the 3-clip evaluation is a real 3-clip held-out validation


## Main findings

### Finding 1 — There is effectively no held-out evaluation

The training clips are:

1. `smoke18`
2. `ccpXHNfaoy_30s_48s`

The reported evaluation clips are:

1. `smoke18`
2. `ccpXHNfaoy_18s`
3. `ccpXHNfaoy_30s_48s`

This means:

1. `smoke18` is evaluated in-sample
2. `ccpXHNfaoy_30s_48s` is evaluated in-sample
3. `ccpXHNfaoy_18s` is not an independent content check in any useful sense for this claim

So the current evaluation is basically:

- train-set evaluation
- plus one near-duplicate / same-intro evaluation

That is not enough to support a strong result claim.

### Finding 2 — `smoke18` and `ccpXHNfaoy_18s` are effectively duplicate evaluation items

The teacher files:

1. `tmp/alignment_research/frontier_smoke18_qwen_teacher.json`
2. `tmp/alignment_research/ccpXHNfaoy_18s_qwen_teacher.json`

contain the same transcript text.

The saved feature-aligner outputs for these two clips also have the same transcript and exactly the same aggregate metrics.

That strongly suggests the reported 3-clip summary is not really 3 independent checks.

At best, it is 2 unique transcript conditions.
At worst, it is 2 training clips plus a duplicate intro.

### Finding 3 — The model is trained to imitate Qwen teacher timestamps, then evaluated against Qwen teacher timestamps

That is acceptable as a first-stage distillation experiment.
It is **not** evidence of true alignment quality beyond the teacher.

So the right interpretation is:

- the small aligner can fit the teacher signal

The wrong interpretation is:

- the small aligner has now established superior real alignment quality

Those are not the same claim.

### Finding 4 — The speed claim is currently overstated

The reported `inference_time_s` in the evaluation script measures only the small aligner head after:

1. Gemma audio features have already been extracted
2. Gemma text embeddings have already been computed

So the reported `~0.005s` to `~0.05s` is **not end-to-end runtime**.

It is only the runtime of the lightweight prediction head on precomputed features.

That means the comparison:

- `feature aligner ~0.007s`
- `Gemma eager ~seconds`

is not apples-to-apples yet.

A fair runtime claim must include:

1. audio feature extraction time
2. text embedding extraction time
3. aligner forward time

Even so, this path may still end up much faster than full eager alignment.
But that has not been demonstrated yet.

### Finding 5 — The current result is still interesting

Despite the problems above, the result is not worthless.

The credible interesting signal is:

1. frozen Gemma audio features seem alignment-informative
2. a tiny transcript-conditioned head can fit timing targets on top of them
3. the representation is good enough that the idea deserves a real follow-up

That is a valid and useful result.
It is just much narrower than the current result note suggests.


## What should be claimed right now

A defensible claim would be:

> A small dedicated aligner on frozen Gemma audio features can fit Qwen teacher timestamps on a tiny proof-of-concept setup, producing monotone word timings extremely cheaply once features are available.

That is defensible.

A claim that is **not** defensible yet would be:

> The 1M-parameter aligner is already a validated replacement for the current Gemma eager aligner.

We do not have the evidence for that.


## What should happen next

### Priority 1 — Build a real held-out evaluation

Minimum requirement:

1. train on one set of clips
2. evaluate on genuinely different clips
3. include different transcripts
4. include different speakers if possible

At least one of those held-out clips should be:

- `tmp/rxrToXvRyM_first18.wav`

once teacher timestamps are available.

### Priority 2 — Measure end-to-end runtime honestly

Report two runtimes separately:

1. feature extraction runtime
2. aligner-head runtime

And then report:

3. total offline alignment runtime

That is the only fair comparison against eager Gemma forced alignment.

### Priority 3 — Evaluate against multiple targets

Keep Qwen-teacher MAE, but add at least one more view:

1. comparison to current Gemma eager alignment
2. if possible, comparison to any available manual or more trusted alignment reference

Otherwise the model is only being graded on how well it mimics its teacher.

### Priority 4 — Stop calling the current 3-clip summary a generalization result

It is not.
It is a tiny bootstrap result.

Use language like:

- preliminary
- proof-of-concept
- teacher-distillation pilot

Do not use language like:

- robust
- validated replacement
- clearly better


## Recommendation

Keep this line of work alive.
Do **not** throw it away.

But reset the claim level immediately:

1. interesting prototype: yes
2. validated new aligner: no
3. end-to-end speed win: unproven
4. real generalization: unproven

The next iteration should be an honest held-out evaluation pass, not more celebratory prose.
