# cascade_simultaneous

Research repo for simultaneous speech translation with a streaming
ASR-to-MT cascade.

## Current runtime

- ASR default: `qwen_forced`
- MT default: `gemma_vllm_alignatt`
- Canonical runner: `run_simulstream_batch.py`
- Canonical single-audio A/B: `run_simulstream_compare.py`
- Submission presets: `main_low_latency` = `chunk_ms=850`, `main_high_latency` = `chunk_ms=1500`
- Docker submission directions: EN->DE, EN->IT, EN->ZH only
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

## Docker submission

Build on an NVIDIA H100 host with a Hugging Face token available as a BuildKit
secret:

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,src="$HF_TOKEN_FILE" \
  -t "$DOCKERHUB_REPO:latest" .
```

Direct inference is the default container mode:

```bash
docker run --gpus all --rm \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  -v /host/wavs:/io/wavs:ro \
  -v /host/out:/io/out \
  "$DOCKERHUB_REPO:latest" \
  infer /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

The same image can expose a SimulStream HTTP speech processor:

```bash
docker run --gpus all --rm -p 8080:8080 \
  -e PRESET=main_low_latency \
  -e TGT_LANG_CODE=de \
  "$DOCKERHUB_REPO:latest" serve
```

## Repo layout

- `cascade/` — active runtime package
- `data/devset/` — tracked development set and references
- `dev-set/` — compatibility alias to `data/devset/`
- `data/alignatt_heads/` — tracked AlignAtt head payloads used by runtime and paper tooling
- `data/smoke/` — tiny reproducible smoke fixtures
- `docs/` — current system, results, status, and submission docs
- `scripts/` — maintained utility scripts only
- `submission/` — Docker submission surface
- `paper/` — paper sources and retained generated TeX fragments

## Docs

- [`docs/system.md`](docs/system.md) — runtime architecture, supported backends, operational notes
- [`docs/results.md`](docs/results.md) — consolidated calibration and reference numbers
- [`docs/status.md`](docs/status.md) — current repo status and cleanup decisions
- [`docs/submission.md`](docs/submission.md) — submission workflow and bundle/export story
- [`submission/README.md`](submission/README.md) — concrete submission workspace usage
- [`docs/reference/README.md`](docs/reference/README.md) — retained upstream reading material
