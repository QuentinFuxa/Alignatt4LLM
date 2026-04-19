#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${CASCADE_ENV_DIR:-$ROOT_DIR/.venv-inference}"
SIMULSTREAM_BIN="$ENV_DIR/bin/simulstream_inference"

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <speech_processor.yaml> <wavlist.txt> <metrics.jsonl>" >&2
  exit 1
fi

CONFIG_PATH="$1"
WAVLIST_PATH="$2"
METRICS_LOG_PATH="$3"

if [ ! -x "$SIMULSTREAM_BIN" ]; then
  echo "Missing $SIMULSTREAM_BIN" >&2
  echo "Run ./setup_inference_qwen_asr_vllm.sh first, or set CASCADE_ENV_DIR." >&2
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [ ! -f "$WAVLIST_PATH" ]; then
  echo "Wav list not found: $WAVLIST_PATH" >&2
  exit 1
fi

SOURCE_LANG_CODE="$(awk -F': *' '$1 == "source_lang_code" {print $2; exit}' "$CONFIG_PATH")"
TARGET_LANG_CODE="$(awk -F': *' '$1 == "target_lang_code" {print $2; exit}' "$CONFIG_PATH")"

if [ -z "$SOURCE_LANG_CODE" ] || [ -z "$TARGET_LANG_CODE" ]; then
  echo "Could not read source_lang_code / target_lang_code from $CONFIG_PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

exec "$SIMULSTREAM_BIN" \
  --speech-processor-config "$CONFIG_PATH" \
  --wav-list-file "$WAVLIST_PATH" \
  --src-lang "$SOURCE_LANG_CODE" \
  --tgt-lang "$TARGET_LANG_CODE" \
  --metrics-log-file "$METRICS_LOG_PATH"
