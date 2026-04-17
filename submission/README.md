# IWSLT 2026 Submission Notes

Point of contact: Quentin Fuxa  
Email: quentin.fuxa@gmail.com

## What this package contains

- `run_iwslt_submission.py`
  - `batch`: frozen offline generation for log-based submission
  - `server`: websocket server for Docker-based submission
- `Dockerfile`
  - CUDA 12.9 + vLLM/Qwen/Gemma inference environment
- `submission/docker-entrypoint.sh`
  - starts the frozen submission server inside the container

## Frozen presets

- `main_low_latency`
  - `qwen_forced + gemma_vllm_alignatt`
  - `chunk_ms=450`
- `main_high_latency`
  - `qwen_forced + gemma_vllm_alignatt`
  - `chunk_ms=700`
- `context_low_latency`
  - `title_abstract + min_source_mass=0.3`
  - `chunk_ms=450`
- `context_high_latency`
  - `title_abstract + min_source_mass=0.3`
  - `chunk_ms=700`

These presets freeze the validated runtime knobs from `docs/RESULTS.md` and
`docs/CONTEXT_INJECTION.md` so submission runs do not depend on a long list of
hand-entered flags.

## Log-based submission

Main track, low latency:

```bash
.venv-inference/bin/python run_iwslt_submission.py batch \
  --preset main_low_latency \
  --source en \
  --target de \
  --input-dir /path/to/dev_or_test_media \
  --output-dir outputs/iwslt26_main_low_ende
```

Extra-context track with one artifact per talk:

```bash
.venv-inference/bin/python run_iwslt_submission.py batch \
  --preset context_low_latency \
  --source en \
  --target de \
  --input-dir /path/to/dev_or_test_media \
  --paper-context-dir /path/to/paper_artifacts \
  --output-dir outputs/iwslt26_context_low_ende
```

`--paper-context-dir` matches `talk.mp4` or `talk.wav` to
`paper_artifacts/talk.json`.

The output directory contains:

- `manifest.json`
- `hypothesis.jsonl`
- `stream_updates.jsonl`

These are the files to keep for the log-based path, together with this README
and the exact preset name used.

## Docker submission

Build:

```bash
docker build -t cascade-simul-iwslt26 .
```

Run the default websocket server:

```bash
docker run --gpus all --rm -p 8765:8765 \
  -e CASCADE_SUBMISSION_PRESET=main_low_latency \
  -e CASCADE_SOURCE_LANG=en \
  -e CASCADE_TARGET_LANG=de \
  cascade-simul-iwslt26
```

The container entrypoint starts:

```bash
python run_iwslt_submission.py server \
  --preset <preset> \
  --host 0.0.0.0 \
  --port 8765
```

The websocket server expects the standard SimulStream PCM audio stream. For
extra-context runs, the client may send a metadata JSON message containing
`"paper_context_path": "/absolute/path/to/talk.json"` before audio chunks for
that stream.

## Model paths / offline cache

The runtime is offline by default. The container therefore needs local model
snapshots available under the Hugging Face cache, or explicit overrides via:

- `CASCADE_QWEN_ASR_SNAPSHOT`
- `CASCADE_QWEN_ALIGNER_SNAPSHOT`
- `CASCADE_GEMMA_SNAPSHOT`

If you keep the standard cache layout, mount or bake the cache under:

```text
/root/.cache/huggingface/hub
```

The runtime now also honors `HF_HOME` and `HF_HUB_CACHE`.

## Important practical note

The local batch runner accepts both `.wav` and `.mp4` inputs. This matters for
the IWSLT 2026 ACL-talk directions, where the official files are provided in
MP4 format.
