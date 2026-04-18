# Standard SimulStream Bundle

This directory is the non-Docker version of the submitted cascade. It keeps the
official `simulstream_inference` entry point and ships only the local runtime
files that the cascade needs on top of SimulStream itself.

The system is a two-stage cascade. `Qwen3-ASR-1.7B` plus the forced aligner
produces an incremental English prefix with word times. `Gemma-4-E4B-it` then
translates that prefix incrementally. `AlignAtt` is applied on the MT LLM so a
target token is emitted only when its attention support points to source speech
that is already accessible. The shipped probe mode is `qk_fast`, which rebuilds
the source-side accessibility signal from query/key states without materializing
the full attention matrix. Both the ASR side and the MT side use `vLLM` in the
runtime that was frozen for submission.

## What is in here

- `configs/main_low_latency/en-de.yaml`
- `configs/main_low_latency/en-it.yaml`
- `configs/main_low_latency/en-zh.yaml`
- `configs/main_high_latency/en-de.yaml`
- `configs/main_high_latency/en-it.yaml`
- `configs/main_high_latency/en-zh.yaml`
- `Dockerfile`
- `docker-entrypoint.sh`
- `bin/run_simulstream_inference.sh`
- `bin/render_submission_preset.sh`
- `cascade/` runtime package copied from the active repo
- `context_injection/`
- `assets/attention_heads/`

Supported directions in this bundle are `en->de`, `en->it`, and `en->zh`.

## Setup

Create the inference environment in place:

```bash
./setup_inference_qwen_asr_vllm.sh
```

If you want to reuse an existing environment, point the wrapper at it:

```bash
CASCADE_ENV_DIR=/path/to/.venv-inference ./setup_inference_qwen_asr_vllm.sh /path/to/.venv-inference
```

## Main-track run

```bash
./bin/run_simulstream_inference.sh \
  configs/main_low_latency/en-de.yaml \
  /path/to/wavlist.txt \
  /path/to/metrics.jsonl
```

`wavlist.txt` follows the normal SimulStream convention: audio paths are
resolved relative to the wavlist file.

Frozen main-track presets:

- `main_low_latency`: `chunk_ms=850`
- `main_high_latency`: `chunk_ms=1500`

## Docker rebuild from this bundle

If you want a container recipe tied directly to this standalone bundle, build it
from this directory:

```bash
docker build -t cascade-standalone-simulstream .
```

Main-track example:

```bash
docker run --gpus all --rm \
  -e PRESET=main_low_latency \
  -e SRC_LANG=English \
  -e TGT_LANG=German \
  -e SRC_LANG_CODE=en \
  -e TGT_LANG_CODE=de \
  -v /host/wavs:/io/wavs:ro \
  -v /host/out:/io/out \
  cascade-standalone-simulstream \
  /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

The Docker entrypoint renders the requested preset to a temporary
`speech_processor.yaml` and then invokes the canonical
`python -m simulstream.inference` path.

## Extra-context run

The extra-context presets need a paper artifact path, so they are rendered on
the fly:

```bash
./bin/render_submission_preset.sh \
  context_low_latency \
  en \
  de \
  configs/generated/context_low_latency.en-de.yaml \
  /path/to/paper_artifact.json
```

Then run the generated config with the same launcher:

```bash
./bin/run_simulstream_inference.sh \
  configs/generated/context_low_latency.en-de.yaml \
  /path/to/wavlist.txt \
  /path/to/metrics.jsonl
```

Frozen extra-context presets:

- `context_low_latency`: `chunk_ms=850`, `paper_context_mode=title_abstract`
- `context_high_latency`: `chunk_ms=1500`, `paper_context_mode=title_abstract`

## Runtime assumptions

- Model weights are not bundled here.
- The runtime expects the Hugging Face snapshots to already be available in the
  local cache, or to be provided through:
  - `CASCADE_QWEN_ASR_SNAPSHOT`
  - `CASCADE_QWEN_ALIGNER_SNAPSHOT`
  - `CASCADE_GEMMA_SNAPSHOT`
- Offline loading is the default path.
- The launcher disables vLLM DeepGEMM by default:
  - `VLLM_USE_DEEP_GEMM=0`
  - `VLLM_MOE_USE_DEEP_GEMM=0`
