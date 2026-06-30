# Reproducibility

## Environments

Use separate environments for inference and evaluation:

```bash
tools/bootstrap/setup_inference_qwen_asr_vllm.sh
uv venv .venv-evaluation --python 3.13
UV_PROJECT_ENVIRONMENT=.venv-evaluation uv sync --group evaluation
```

The inference bootstrap pins the vLLM/CUDA stack used by this project and
patches the Qwen ASR package for the validated Transformers version.

The cascade runs the ASR and MT engines side by side on one GPU. With the
pinned stack the defaults fit. On a newer vLLM (its CUDA-graph memory profiler
reserves extra memory) or a smaller card, the second engine can fail with
`No available memory for the cache blocks`. If that happens, set
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` and/or raise
`--mt-vllm-gpu-memory-utilization` (validated on a 40 GB A100 with vLLM 0.23 at
`--mt-vllm-gpu-memory-utilization 0.72`).

## Smoke Run

```bash
.venv-inference/bin/alignatt-compare --wav <local.wav>
```

## Batch Run

```bash
.venv-inference/bin/alignatt-batch \
  --inputs <local.wav> \
  --target zh \
  --mt-backend-name milmmt_vllm_alignatt \
  --output-dir outputs/milmmt_zh_smoke
```

## Scoring

```bash
.venv-evaluation/bin/alignatt-eval \
  --output-dir outputs/milmmt_zh_smoke
```

Claims should cite the output directory, `manifest.json`, `evaluation.json`,
and the exact command used to produce them.
