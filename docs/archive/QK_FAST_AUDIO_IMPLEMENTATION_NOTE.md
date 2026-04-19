# QK-Fast Audio Implementation Note

This pass implemented the static code path for Gemma audio forced alignment under `qk_fast`, without running any model-backed validation.

## What changed

- `cascade/alignment/gemma_transformers_asr_backend.py`
  - added an explicit `gemma_audio_align_probe_mode` switch for forced alignment
  - kept `eager` as the default safe path
  - added a new `qk_fast` replay path that:
    - snapshots the multimodal prompt prefix KV cache
    - replays the teacher-forced transcript suffix under fast attention
    - reconstructs transcript-token rows into the audio-token span with the shared MT `qk_fast` helper
  - kept the downstream timing pipeline unchanged: head aggregation, argmax, monotonic projection, offset correction, and token-to-word grouping still run exactly as before
- `qwen3asr_gemma_cascade_core.py`
  - exposed `config.gemma_audio_align_probe_mode` with default `eager`
- `gemma_two_pass_frontend.py` and `hybrid_cascade/alignment/base.py`
  - now surface the selected Gemma probe backend in diagnostics
- `run_alignment_single_audio.py`
  - `gemma_forced_align` now accepts `--probe-mode eager|qk_fast`

## What was intentionally not done

- no GPU/model runs
- no eager-vs-qk_fast quality claims
- no head recalibration changes
- no trained feature-aligner integration

The machine guidance in `PLAN_qk_fast_audio.md` was followed exactly: this is a static implementation pass only.

## Next validation step

When the GPU is free, start with one clip only:

```bash
.venv-inference/bin/python run_alignment_single_audio.py gemma_forced_align \
  --wav tmp/alignatt_smoke18.wav \
  --teacher tmp/alignatt_smoke18_qwen.json \
  --heads-path assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json \
  --probe-mode eager \
  --output tmp/smoke18_gemma_forced_eager.json

.venv-inference/bin/python run_alignment_single_audio.py gemma_forced_align \
  --wav tmp/alignatt_smoke18.wav \
  --teacher tmp/alignatt_smoke18_qwen.json \
  --heads-path assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json \
  --probe-mode qk_fast \
  --output tmp/smoke18_gemma_forced_qk_fast.json
```

Then compare MAE, monotonicity, and runtime before trying a harder clip.
