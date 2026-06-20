# Data

The repository tracks only lightweight, redistributable inputs needed for code
inspection, policy tests, and reproducibility anchors.

## Tracked

- `data/devset/` — segmentation metadata and references.
- `data/alignatt_heads/` — calibrated attention-head payloads.
- `data/context_artifacts/` — extracted extra-context JSON artifacts for the
  context sub-track runtime.
- `data/smoke/` — small text fixtures for local smoke workflows.
- `data/alignment_corpora/` — tiny head-discovery support corpora.

## Not Tracked

- Dataset audio (`*.wav`) and local test-set audio.
- Model weights and Hugging Face caches.
- Generated outputs, logs, plots, and per-run diagnostics.

Place local dev-set audio under `data/devset/audio/` when using the default
paths. Audio files are ignored by Git.
