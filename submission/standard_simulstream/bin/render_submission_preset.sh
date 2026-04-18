#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${CASCADE_ENV_DIR:-$ROOT_DIR/.venv-inference}"
PYTHON_BIN="$ENV_DIR/bin/python"

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 <preset> <source_lang_code> <target_lang_code> <output.yaml> [paper_context.json]" >&2
  exit 1
fi

PRESET="$1"
SOURCE_LANG_CODE="$2"
TARGET_LANG_CODE="$3"
OUTPUT_PATH="$4"
PAPER_CONTEXT_PATH="${5:-}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing $PYTHON_BIN" >&2
  echo "Run ./setup_inference_qwen_asr_vllm.sh first, or set CASCADE_ENV_DIR." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/render_preset_yaml.py"
  --preset "$PRESET"
  --source-lang-code "$SOURCE_LANG_CODE"
  --target-lang-code "$TARGET_LANG_CODE"
  --output "$OUTPUT_PATH"
)

if [ -n "$PAPER_CONTEXT_PATH" ]; then
  CMD+=(--paper-context-path "$PAPER_CONTEXT_PATH")
fi

"${CMD[@]}"
