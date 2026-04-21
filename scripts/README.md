# `scripts/`

Maintained utility scripts that sit next to the active runtime.

## ASR workflow

- `compare_asr_full_audio.py` — single-audio ASR evaluation harness
- `compare_asr_per_audio_batch.py` — hot-bundle ASR batch comparison over the tracked dev-set
- `compare_asr_word_end_bias.py` — Qwen-vs-Gemma word-end timing bias analysis for ASR AlignAtt
- `eval_asr_holdback_proxy.py` — evaluate the maintained offset-calibrated Gemma word-end proxy against the last-250 ms hold-back rule
- `eval_asr_per_audio_longyaal.py` — LongYAAL and aggregate scoring for those per-audio ASR runs
- `plot_asr_comparison.py` — plotting/report helper for the ASR comparison outputs
- `asr_trace_to_hypothesis_jsonl.py` — convert ASR traces to OmniSTEval-style hypothesis bundles

## Submission and additive calibration

- `run_testset_submission.sh` — sequential test-set submission wrapper derived from preset metadata
- `run_additive_chunk_sweep.py` — additive `chunk_ms` calibration over `data/devset/`
- `run_additive_full_pipeline.sh` — orchestrate additive inference + scoring + submission sync
- `score_additive_chunks.sh` — score additive outputs already present on disk

## Runtime resources

- `build_alignatt_head_set.py` — materialize shared-kernel or multilingual-union head sets from the tracked head payloads

All other dated one-off research scripts were removed from the maintained tree.
