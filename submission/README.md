# IWSLT 2026 Simultaneous Submission

Point of contact: Quentin Fuxa  
Email: quentin.fuxa@gmail.com

This folder now contains the development logs plus one runnable code bundle:

- `standard_simulstream/`

The prebuilt Docker archive was removed from this copy because it was too large
to keep in the transfer package. What remains is the standalone SimulStream
bundle, together with the Docker recipe that rebuilds the same frozen runtime
from that bundle.

The system itself is a cascade. The ASR side is `Qwen3-ASR-1.7B` plus the
forced aligner. The MT side is `Gemma-4-E4B-it`. `AlignAtt` is applied on the
translation LLM so target tokens are only emitted when their attention support
lands on source speech that has become accessible. The shipped probe mode is
`qk_fast`, which reconstructs that accessibility signal from query/key states
instead of materializing full attention matrices. Both ASR and MT use `vLLM` in
the submitted runtime.

## Frozen presets

Main track:

- `main_low_latency`: `chunk_ms=850`, `translation_alignatt_border_margin=1`
- `main_high_latency`: `chunk_ms=1500`, `translation_alignatt_border_margin=1`

Extra-context track:

- `context_low_latency`: `chunk_ms=850`, `paper_context_mode=title_abstract`
- `context_high_latency`: `chunk_ms=1500`, `paper_context_mode=title_abstract`

All presets use `qwen_forced` on the ASR side and `gemma_vllm_alignatt` on the
MT side.

## Included files

- `dev_logs/`: six dev-set SimulStream runs for the frozen main-track regimes
- `DEV_LOG_INDEX.json`: machine-readable index of those runs and scores
- `DOCKER_TEST.md`: notes about the Docker validation done on this machine
- `standard_simulstream/README.md`: setup and run instructions for the
  standalone bundle
- `standard_simulstream/SMOKE_TEST.md`: single-audio standalone validation note
- `standard_simulstream/Dockerfile`: container rebuild recipe for the
  standalone bundle
- `standard_simulstream/docker-entrypoint.sh`: preset-aware Docker entrypoint

The dev-set bundles are:

- `dev_logs/low/en-de/`
- `dev_logs/low/en-it/`
- `dev_logs/low/en-zh/`
- `dev_logs/high/en-de/`
- `dev_logs/high/en-it/`
- `dev_logs/high/en-zh/`

Each one contains `manifest.json`, `hypothesis.jsonl`, `stream_updates.jsonl`,
`evaluation.json`, `evaluation.report.txt`, and `scores.tsv`.

Summary of the frozen main-track runs:

| Regime | Direction | chunk_ms | BLEU | chrF | XCOMET-XL | LongYAAL CU | LongYAAL CA |
|---|---|---:|---:|---:|---:|---:|---:|
| low | en->de | 850 | 28.76 | 62.14 | 0.8752 | 1997.8 ms | 1628.9 ms |
| low | en->it | 850 | 40.10 | 68.02 | 0.8052 | 1983.7 ms | 1621.4 ms |
| low | en->zh | 850 | 36.01 | 34.97 | 0.7432 | 1947.0 ms | 1766.8 ms |
| high | en->de | 1500 | 32.63 | 64.21 | 0.9018 | 3528.2 ms | 3136.1 ms |
| high | en->it | 1500 | 44.46 | 70.06 | 0.8407 | 3484.3 ms | 3096.0 ms |
| high | en->zh | 1500 | 39.86 | 37.81 | 0.7781 | 3271.9 ms | 3085.3 ms |

## Docker path

There is no pre-exported `.tar` in this copy anymore. The intended Docker path
is now:

1. rebuild the image from `standard_simulstream/`
2. run that image with the frozen preset you want

Build from the bundled recipe:

```bash
cd standard_simulstream
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

Extra-context example:

```bash
docker run --gpus all --rm \
  -e PRESET=context_low_latency \
  -e SRC_LANG=English \
  -e TGT_LANG=German \
  -e SRC_LANG_CODE=en \
  -e TGT_LANG_CODE=de \
  -e PAPER_CONTEXT_PATH=/io/context/paper_artifact.json \
  -v /host/wavs:/io/wavs:ro \
  -v /host/context:/io/context:ro \
  -v /host/out:/io/out \
  cascade-standalone-simulstream \
  /io/wavs/wavlist.txt /io/out/metrics.jsonl
```

This rebuild path is meant as a practical replacement for the removed archive.
The code, presets, entrypoint logic and environment setup are all inside
`standard_simulstream/`.

`wavlist.txt` follows the standard SimulStream convention: paths are resolved
relative to the wavlist file location.

## Standalone use

The fallback path is the same runtime without Docker. It uses the canonical
`simulstream_inference` CLI directly, with the frozen YAML configs already
rendered for the main track.

Setup:

```bash
cd standard_simulstream
./setup_inference_qwen_asr_vllm.sh
```

Run:

```bash
./bin/run_simulstream_inference.sh \
  configs/main_low_latency/en-de.yaml \
  /path/to/wavlist.txt \
  /path/to/metrics.jsonl
```

Why keep this fallback:

- it is much smaller to transfer than an exported Docker archive
- it keeps the standard SimulStream execution path
- it is easier to inspect and rebuild if something needs to be changed quickly
- it was smoke-tested end to end from this bundle on one audio

The exact standalone validation command and its outcome are recorded in
`standard_simulstream/SMOKE_TEST.md`.

## Docker vs fallback

Use Docker when you want a sealed runtime and the target machine has a normal
NVIDIA Docker setup. Use the standalone bundle when moving a huge image archive
is inconvenient, when you want to inspect the runtime directly, or when you
need a fast rebuild from source files already in the submission folder.

In both cases the intended runtime is the same:

- same frozen presets
- same `qwen_forced` ASR path
- same `gemma_vllm_alignatt` MT path
- same AlignAtt head files and prompt/runtime modules
- same Hugging Face snapshot assumptions

## Runtime assumptions

- The runtime is offline by default for model loading.
- Model snapshots are not bundled in either form.
- Snapshot overrides can be provided through:
  - `CASCADE_QWEN_ASR_SNAPSHOT`
  - `CASCADE_QWEN_ALIGNER_SNAPSHOT`
  - `CASCADE_GEMMA_SNAPSHOT`
