# Held-Out Evaluation: Gemma Feature Aligner

## Verdict

The feature aligner **does not generalize** beyond its training clips.
The original result was a tiny-fit artifact.

This path should not be pursued further without a fundamental change
in approach (e.g., much more training data, different architecture,
or different training objective).

## What was tested

A small transcript-conditioned aligner (~1M parameters, 2-layer
TransformerDecoder with cross-attention) trained on frozen Gemma audio
tower features and frozen Gemma text embeddings.

Supervision: Qwen teacher timestamps (word-level midpoints mapped
to Gemma audio token positions via Gaussian soft targets).

## Split

| Split | Clips | Talks | Total audio |
|-------|-------|-------|-------------|
| Train | 15    | 410, 468 (2 speakers) | 112s |
| Val   | 4     | 567 (1 speaker) | 36s |
| Test  | 5     | 111 (1 speaker) | 41s |

Source: acl-speech gold segments (sentence-level, 16kHz mono).
Train and test differ by both talk and speaker.

## Results

### Train-fit (in-sample)

| Metric | Value |
|--------|-------|
| Mean MAE | ~0.15s |
| All monotone | Yes |

The model fits its training data well.

### Validation (talk 567, held out)

| Metric | Value |
|--------|-------|
| Mean MAE | 1.78s |
| Mean median error | 1.85s |
| Mean P90 | 3.06s |
| All monotone | Yes |

### Test (talk 111, held out)

| Metric | Value |
|--------|-------|
| Mean MAE | 1.63s |
| Mean median error | 1.62s |
| Mean P90 | 2.77s |
| All monotone | Yes |

## Interpretation

The held-out MAE is ~10x worse than the train-fit MAE.
For clips of 7-12 seconds, a 1.6s MAE means the model is placing
words in the wrong quarter of the audio on average.

The model preserves monotonicity (the attention structure still
produces ordered outputs), but the actual positions are useless.

## Why it fails

The most likely explanation: with only 15 training clips, the model
memorizes the specific audio-to-text timing patterns of those clips
rather than learning a general alignment function. The frozen Gemma
audio features may not encode enough speaker-invariant timing
information at this resolution (40ms/token) for a small head to
learn general alignment from so few examples.

## Runtime (for completeness)

| Component | Mean time |
|-----------|-----------|
| Feature extraction (Gemma audio tower) | ~0.09s |
| Aligner head inference | ~0.003s |
| Total alignment | ~0.09s |

The runtime is fast, but irrelevant given the quality collapse.

## What this means for the project

1. The feature aligner is **not** a viable replacement for eager
   Gemma alignment at this training scale.
2. The original 2-clip result was misleadingly good because it
   evaluated on (near-)duplicates of training data.
3. The frozen Gemma audio features may still be useful for alignment,
   but a ~1M parameter head trained on 15 clips is not enough.

## Possible next steps (if pursuing this path)

1. Scale training data significantly (100+ clips across many speakers)
2. Try a simpler model (e.g., learned linear projection + CTC-style
   loss instead of cross-attention)
3. Fine-tune the audio tower (partial unfreezing) rather than using
   it fully frozen
4. Use a different training objective that doesn't require per-word
   timestamps

## Supervision disclosure

The aligner was trained to imitate Qwen teacher timestamps and
evaluated against Qwen teacher timestamps. This is a distillation
evaluation — it tests whether the model can reproduce Qwen's
alignment output from Gemma features, not whether it produces
objectively correct alignments.

Qwen is not used at inference time in the feature-aligner path.
