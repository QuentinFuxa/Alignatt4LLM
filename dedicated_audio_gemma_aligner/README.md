# Dedicated Audio Gemma Aligner

This folder is the self-contained workspace for the trained transcript-conditioned Gemma audio aligner.

Contents:
- `PLAN.md`: current subproject plan
- `training_gemma_aligner.md`: experiment write-up and conclusions
- `gemma_audio_features.py`: frozen Gemma audio feature extraction
- `gemma_feature_aligner.py`: learned transcript-conditioned alignment head
- `run_gemma_feature_aligner_train.py`: training entrypoint
- `run_gemma_feature_aligner_eval.py`: full held-out evaluation entrypoint
- `build_split_manifest.py`: small v2 split builder
- `build_split_manifest_full.py`: full ACL split builder
- `run_generate_split_teachers.py`: Qwen teacher generation for the split
- `artifacts/feature_aligner/`: checkpoints, manifests, teacher JSON, evaluations, and notes

Compatibility:
