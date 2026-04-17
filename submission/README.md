# IWSLT 2026 Submission Notes

Point of contact: Quentin Fuxa  
Email: quentin.fuxa@gmail.com

## What this package contains

- `run_iwslt_submission.py`
  - `batch`: frozen offline generation for log-based submission
  - `server`: auxiliary websocket path kept for direct SimulStream serving
- `Dockerfile`
  - CUDA 12.9 + vLLM/Qwen/Gemma inference environment
- `submission/docker-entrypoint.sh`
  - runs the frozen log-based submission entrypoint inside the container

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

These presets freeze the current simplified runtime surface used by the Docker
entrypoint, so submission runs do not depend on a long list of hand-entered
flags. The exact historical manifests from earlier dev-set runs remain the
source of truth for those earlier experiments.

Current worktree note: the shipped runtime has a fixed ASR commit path
(`punctuation_lcp` + EOS flush). Historical `asr_commit_mode` ablations remain
in `docs/RESULTS.md` / `DECISIONS.md`, but they are not public preset knobs
any more.

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

Run the frozen log-based entrypoint:

```bash
docker run --gpus all --rm \
  -e PRESET=main_low_latency \
  -e SRC_LANG=English \
  -e TGT_LANG=German \
  -e SRC_LANG_CODE=en \
  -e TGT_LANG_CODE=de \
  -v /host/wavs:/io/wavs:ro \
  -v /host/out:/io/out \
  cascade-simul-iwslt26 \
  /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

The container entrypoint renders the frozen speech-processor YAML and then executes:

```bash
python -m simulstream.inference \
  --speech-processor-config /tmp/.../speech_processor.yaml \
  --wav-list-file <wavlist.txt> \
  --src-lang <SRC_LANG> \
  --tgt-lang <TGT_LANG> \
  --metrics-log-file <metrics.jsonl>
```

`wavlist.txt` must follow the SimulStream contract: paths are relative to the
wavlist file location. For extra-context runs, pass `PAPER_CONTEXT_PATH` to the
container when using a single artifact, or use the offline batch flow with
`--paper-context-dir` outside Docker when each talk has its own artifact.

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
