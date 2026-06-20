# `scripts/`

Maintained utility scripts that sit next to the active runtime.

## ASR workflow

- `compare_asr_full_audio.py` — single-audio ASR evaluation harness
- `compare_asr_per_audio_batch.py` — hot-bundle ASR batch comparison over the tracked dev-set
- `compare_asr_word_end_bias.py` — Qwen-vs-Gemma word-end timing bias analysis for ASR AlignAtt
- `eval_asr_holdback_proxy.py` — evaluate the maintained offset-calibrated Gemma word-end proxy against the last-250 ms hold-back rule
- `qwen_local_agreement_capture.py` — capture Qwen chunk hypotheses with reinjected stable prefix and per-word forced-align timings
- `plot_asr_reference_tail_risk_curve.py` — paper-style smoothed live-tail ASR reference-error curve with audio bootstrap bands
- `eval_asr_per_audio_longyaal.py` — LongYAAL and aggregate scoring for those per-audio ASR runs
- `eval_voxtral_asr_longyaal.py` — convert retained Voxtral realtime traces to ASR hypotheses and score LongYAAL
- `gemma_e4b_asr_local_agreement_capture.py` — capture direct Gemma E4B ASR local-agreement traces without AlignAtt
- `eval_gemma_e4b_asr_longyaal.py` — convert Gemma ASR captures to OmniSTEval hypotheses and score LongYAAL
- `plot_asr_comparison.py` — plot the Qwen/Voxtral/Gemma ASR comparison figure and summaries
- `asr_trace_to_hypothesis_jsonl.py` — convert ASR traces to OmniSTEval-style hypothesis bundles

## Runtime sweeps and calibration

- `run_iwslt_testset_tracks.sh` — sequential local test-set wrapper derived from runtime preset metadata
- `run_additive_chunk_sweep.py` — additive `chunk_ms` calibration over `data/devset/`
- `run_additive_full_pipeline.sh` — orchestrate additive inference + scoring
- `score_additive_chunks.sh` — score additive outputs already present on disk
- `run_mt_cutoff_policy_sweep.py` — run AlignAtt vs fixed target-cutoff policy sweeps
- `report_mt_cutoff_policy_tradeoff.py` — render quality-latency reports for cutoff sweeps
- `report_mt_alignatt_cutoff_microscope.py` — summarize token-level cutoff microscope runs

## Runtime resources

- `build_alignatt_head_set.py` — materialize shared-kernel or multilingual-union head sets from the tracked head payloads

All other dated one-off research scripts were removed from the maintained tree.
