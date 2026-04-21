# Status

## Cleanup state

- duplicate submission trees were removed
- extra-context code now lives in `cascade/paper_context/`
- tracked evaluation assets now live under `data/`
- `dev-set/` was restored as a compatibility alias to `data/devset/`
- tracked research clutter was removed from `tmp/`
- paper entrypoint is `paper/main.tex`

## Active workflows

- End-to-end streaming: `run_simulstream_batch.py`
- Single-audio backend comparison: `run_simulstream_compare.py`
- ASR evaluation trio: `scripts/compare_asr_full_audio.py`, `scripts/compare_asr_per_audio_batch.py`, `scripts/eval_asr_per_audio_longyaal.py`
- Submission rendering/export: `submission/render_preset_yaml.py`, `submission/sync_artifacts.py`, `submission/export_standalone_bundle.py`

## Decisions

- `cascade/submission.py` is the sole submission preset source of truth
- The maintained submission surface is the two main presets defined in `cascade/submission.py`
- The main repo no longer vendors a second frozen copy of the runtime
- Historical one-off scripts and notebook-era compatibility layers were removed instead of preserved in place
