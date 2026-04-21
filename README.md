# cascade_simultaneous

Research repo for simultaneous speech translation with a streaming
ASR-to-MT cascade.

## Current runtime

- ASR default: `qwen_forced`
- MT default: `gemma_vllm_alignatt`
- Canonical runner: `run_simulstream_batch.py`
- Canonical single-audio A/B: `run_simulstream_compare.py`
- Submission presets: `main_low_latency` = `chunk_ms=850`, `main_high_latency` = `chunk_ms=1500`
- MT AlignAtt no longer uses an anti-rewind threshold; legitimate EN->ZH reorderings
  make that heuristic a bad fit for streaming MT.

## Canonical commands

```bash
.venv-inference/bin/python run_simulstream_batch.py \
  --inputs data/devset/audio/ccpXHNfaoy.wav \
  --output-dir outputs/my_run

.venv-evaluation/bin/python evaluate_cascade_outputs.py \
  --output-dir outputs/my_run
```

curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs


## Repo layout

- `cascade/` — active runtime package
- `data/devset/` — tracked development set and references
- `dev-set/` — compatibility alias to `data/devset/`
- `data/alignatt_heads/` — tracked AlignAtt head payloads used by runtime and paper tooling
- `data/smoke/` — tiny reproducible smoke fixtures
- `docs/` — current system, results, status, and submission docs
- `scripts/` — maintained utility scripts only
- `submission/` — single submission/export surface
- `paper/` — paper sources and retained generated TeX fragments

## Docs

- [`docs/system.md`](docs/system.md) — runtime architecture, supported backends, operational notes
- [`docs/results.md`](docs/results.md) — consolidated calibration and reference numbers
- [`docs/status.md`](docs/status.md) — current repo status and cleanup decisions
- [`docs/submission.md`](docs/submission.md) — submission workflow and bundle/export story
- [`submission/README.md`](submission/README.md) — concrete submission workspace usage
- [`docs/reference/README.md`](docs/reference/README.md) — retained upstream reading material
