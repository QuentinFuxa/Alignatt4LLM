# Training a Gemma Feature Aligner: Results and Conclusions

## Goal

Train a small dedicated alignment head on frozen Gemma audio tower features to replace the eager Gemma attention-based aligner or the Qwen forced aligner. The head takes frozen Gemma audio features (1536-dim, 40ms/token) and frozen Gemma text embeddings (2560-dim) as input, and predicts per-word timestamps.

Supervision: Qwen teacher timestamps (word-level).

## Data

Source: ACL-Speech gold segments (sentence-level, 16kHz mono WAV).

Five talks, five different speakers:

| Talk | Sentences | Speaker |
|------|-----------|---------|
| 410  | 100       | Allan (ByteDance) |
| 468  | 84        | Antoine (Maastricht) |
| 567  | 56        | VALSE presenter |
| 597  | 91        | Kamezawa (U Tokyo) |
| 111  | 85        | Asaf Harari |

Clips filtered to 3–18s duration. Qwen3-ASR teacher timestamps generated for all clips.

## Experiments

### v2: Small model, small data (proof of concept)

- **Architecture**: 2-layer TransformerDecoder, 128-dim, 4 heads (~1M params)
- **Loss**: Gaussian soft targets (σ=5) over audio positions
- **Train**: 15 clips from talks 410+468
- **Val**: 4 clips from talk 567
- **Test**: 5 clips from talk 111

| Split | MAE | P90 |
|-------|-----|-----|
| Train (in-sample) | 0.15s | — |
| Val (held out) | 1.78s | 3.06s |
| Test (held out) | 1.63s | 2.77s |

**Diagnosis**: Perfect train fit, complete generalization failure. The model memorized 15 clips but learned nothing transferable.

### v3a: Bigger model, full data, discrete classification

Inspired by Qwen3-ForcedAligner's approach of treating timestamps as discrete classification.

- **Architecture**: 4-layer TransformerDecoder, 256-dim, 8 heads (~5.3M params)
- **Loss**: Cross-entropy with label smoothing (0.1) on hard audio position targets
- **Train**: 288 clips from talks 410+468+567+597
- **Test**: 10 clips from talk 111

| Split | MAE |
|-------|-----|
| Train (in-sample) | 2.38s |
| Test (held out) | 2.25s |

**Diagnosis**: Can't even fit training data. Discrete classification with dot-product scoring is too hard to optimize.

### v3b: Bigger model, full data, Gaussian regression

Same as v3a but reverted to Gaussian soft targets (the loss that worked for fitting in v2).

- **Architecture**: 4-layer TransformerDecoder, 256-dim, 8 heads (~5.3M params)
- **Loss**: Gaussian soft targets (σ=5) over audio positions
- **Train**: 288 clips from talks 410+468+567+597
- **Test**: 10 clips from talk 111

| Split | MAE |
|-------|-----|
| Train (in-sample) | 1.60s |
| Test (held out) | 1.55s |

**Diagnosis**: Can't fit training data either. 20x more data and 5x more parameters didn't help. The model is underfitting.

## Summary table

| Experiment | Train clips | Params | Loss | Train MAE | Test MAE |
|------------|------------|--------|------|-----------|----------|
| v2         | 15         | 1.0M   | Gaussian | 0.15s | 1.63s |
| v3a        | 288        | 5.3M   | Discrete | 2.38s | 2.25s |
| v3b        | 288        | 5.3M   | Gaussian | 1.60s | 1.55s |

A usable aligner needs MAE well under 0.3s. None of these come close on held-out data.

## Conclusions

### 1. The original 2-clip result was pure memorization

The v2 model achieved 0.15s MAE on its 15 training clips but 1.63s on unseen clips. This was a tiny-fit artifact, not evidence that the approach works.

### 2. Frozen Gemma audio features don't support learned alignment

When given enough data to prevent memorization (288 clips), the model cannot learn the mapping — not even on training data. The frozen Gemma audio tower features at 40ms resolution do not encode sufficient fine-grained timing information for a small head to learn a general text-to-audio alignment function.

This is fundamentally different from Qwen3-ForcedAligner, which is a 0.6B-parameter model with its own audio encoder trained end-to-end specifically for alignment.

### 3. Architecture and loss function are secondary

We tried two loss functions (Gaussian regression, discrete classification) and two model sizes (1M, 5.3M params). Neither made a meaningful difference. The bottleneck is the input representation, not the head.

### 4. Possible next steps (if pursuing this path)

These would require a fundamentally different approach:

- **Partially unfreeze the audio tower** — let the last few conformer layers adapt during alignment training. This changes the problem from "learn alignment from frozen features" to "fine-tune features for alignment."
- **Use a much larger head** — closer to Qwen3-ForcedAligner's scale (0.6B). But this defeats the purpose of a lightweight alternative.
- **Use a different input representation** — e.g., raw mel spectrograms instead of Gemma's learned features, with a dedicated encoder.

All of these are substantially larger investments than the original idea.

## Files

- `tmp/feature_aligner/split_manifest.json` — 15-clip split (v2)
- `tmp/feature_aligner/split_manifest_full.json` — 288-clip split (v3)
- `tmp/feature_aligner/teachers/` — Qwen teacher timestamps (373 clips)
- `tmp/feature_aligner/aligner_v2.pt` — v2 checkpoint
- `tmp/feature_aligner/aligner_v3.pt` — v3b checkpoint
- `tmp/feature_aligner/training_summary_v2.json` — v2 training summary
- `tmp/feature_aligner/training_summary_v3.json` — v3b training summary
- `tmp/feature_aligner/heldout_eval_v2.json` — v2 held-out evaluation
