# PLAN.md
# Held-Out Evaluation for the Gemma Feature Aligner — COMPLETED

## Outcome: Rule C — held-out quality collapsed

The feature aligner does not generalize. The original 2-clip result was a tiny-fit artifact.

| Split | Clips | MAE | P90 |
|-------|-------|-----|-----|
| Train (in-sample) | 15 | ~0.15s | — |
| Validation (talk 567) | 4 | 1.78s | 3.06s |
| Test (talk 111) | 5 | 1.63s | 2.77s |

Held-out MAE is ~10x worse than train-fit. This path should not be integrated further without a fundamental change in approach.

## What was done

### Phase 1: Split ✓
- Built `tmp/feature_aligner/split_manifest.json`
- Train: 15 clips from talks 410+468 (2 speakers, 112s)
- Val: 4 clips from talk 567 (1 speaker, 36s)
- Test: 5 clips from talk 111 (1 speaker, 41s)
- Source: acl-speech gold segments
- Train/test differ by both talk and speaker

### Phase 2: Teacher targets ✓
- Generated Qwen teacher timestamps for all 24 clips
- Saved to `tmp/feature_aligner/teachers/`

### Phase 3: Retrained ✓
- Same architecture (2-layer TransformerDecoder, 128-dim, 1M params)
- 2000 epochs, lr=3e-4
- Trained on train split only
- Checkpoint: `tmp/feature_aligner/aligner_v2.pt`

### Phase 4: Honest evaluation ✓
- Evaluated on val and test separately
- Measured end-to-end runtime (feature extraction + aligner head)
- Results: `tmp/feature_aligner/heldout_eval_v2.json`

### Phase 5: Honest result note ✓
- `tmp/feature_aligner/RESULT_NOTE_v2.md`

## Decision

Per Rule C: "If held-out quality collapses, do not integrate further."
The current result should be treated as a tiny-fit artifact.

## Files created/updated

- `build_split_manifest.py` — split manifest builder
- `run_generate_split_teachers.py` — Qwen teacher generation for split
- `run_gemma_feature_aligner_train.py` — updated to use manifest, train-only
- `run_gemma_feature_aligner_eval.py` — updated for held-out evaluation
- `tmp/feature_aligner/split_manifest.json` — the split
- `tmp/feature_aligner/teachers/*.json` — 24 teacher artifacts
- `tmp/feature_aligner/aligner_v2.pt` — trained checkpoint
- `tmp/feature_aligner/training_summary_v2.json` — training summary
- `tmp/feature_aligner/heldout_eval_v2.json` — held-out evaluation
- `tmp/feature_aligner/RESULT_NOTE_v2.md` — honest result note
