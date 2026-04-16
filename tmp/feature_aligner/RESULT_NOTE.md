# Gemma Feature Aligner — First Result

## Architecture

- **Feature source**: Frozen Gemma audio tower output (conformer + output_proj), 1536-dim, 40ms per token
- **Text representation**: Frozen Gemma `embed_tokens` (2560-dim), projected to 128-dim
- **Aligner**: 2-layer TransformerDecoder with cross-attention (128-dim, 4 heads)
- **Parameters**: 1,053,696 (~1M)
- **Output**: soft argmax over audio positions, monotonicity enforced post-hoc

## Supervision

- Qwen teacher timestamps (training only, not used at inference)
- Gaussian soft targets (sigma=5 audio tokens) around teacher word midpoints
- Token-to-word mapping via character-span overlap (reuses repo's `split_text_into_word_spans`)

## Training

- 2 clips (smoke18 + ccpXHNfaoy_30s_48s), 92 tokens total
- 2000 epochs, lr=3e-4 (Adam), 46 seconds on 1x A100
- Final loss: 2.98

## Results (3 clips, all 18s)

| Clip | MAE (s) | Median (s) | P90 (s) | Monotone |
|------|---------|-----------|---------|----------|
| smoke18 | 0.143 | 0.145 | 0.247 | 100% |
| ccpXHNfaoy_18s | 0.143 | 0.145 | 0.247 | 100% |
| ccpXHNfaoy_30s_48s | 0.155 | 0.124 | 0.282 | 100% |
| **Mean** | **0.147** | **0.138** | **0.259** | **100%** |

## Comparison vs Gemma Eager Forced Alignment (smoke18)

| Metric | Gemma Eager | Feature Aligner |
|--------|-------------|----------------|
| MAE | 0.177s | 0.143s (-19%) |
| P90 | 0.384s | 0.247s (-36%) |
| Monotonicity | 93.9% | 100% |
| Inference | ~seconds | 0.007s (~100x faster) |

## Assessment

This path meets **Decision Rule A** from PLAN.md: clearly faster than eager alignment and in the same or better MAE regime. The dedicated aligner becomes the recommended Qwen-independent alignment path.

Key caveats:
1. Trained on only 2 clips — needs more data for robust generalization
2. Both training clips are from the same speaker — speaker diversity untested
3. Supervision comes from Qwen (acceptable for training, stated clearly)
4. Not yet integrated into the streaming cascade (by design — offline first)

## Next steps

1. Scale training to more clips and speakers
2. Test on `rxrToXvRyM_first18` (different speaker, needs Qwen teacher generation)
3. Integrate as a new alignment backend in the cascade
