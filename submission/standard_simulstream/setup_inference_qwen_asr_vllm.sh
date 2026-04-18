#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${1:-${UV_PROJECT_ENVIRONMENT:-.venv-inference}}"
PYTHON_BIN="$ENV_DIR/bin/python"

cd "$ROOT_DIR"

if [ -d "$ENV_DIR" ]; then
  echo "Reusing existing inference environment at $ENV_DIR"
else
  echo "Creating inference environment at $ENV_DIR"
  uv venv "$ENV_DIR" --python 3.13
fi

echo "Syncing inference dependencies from pyproject.toml"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync --group inference

# qwen_asr's published [vllm] extra is pinned to an older torch/vllm stack.
# We install qwen_asr via the inference group, then layer a validated vLLM
# build used by the ASR path on top for this repo's runtime.
echo "Upgrading to a vLLM build compatible with the CUDA 12.9 stack"
uv pip install --python "$PYTHON_BIN" -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

echo "Installing the validated transformers 5.x version"
uv pip install --python "$PYTHON_BIN" transformers==5.5.0

echo "Patching qwen_asr for transformers 5.x compatibility"
"$PYTHON_BIN" patch_qwen_asr_for_transformers5.py

echo "Verifying installation"
"$PYTHON_BIN" -c "import qwen_asr, transformers; print('qwen_asr:', qwen_asr.__file__); print('transformers:', transformers.__version__)"
