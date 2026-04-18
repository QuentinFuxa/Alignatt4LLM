#!/usr/bin/env bash
set -euo pipefail

PRESET="${PRESET:-${CASCADE_SUBMISSION_PRESET:-main_low_latency}}"
SRC_LANG="${SRC_LANG:-${CASCADE_SOURCE_LANG:-English}}"
TGT_LANG="${TGT_LANG:-${CASCADE_TARGET_LANG:-German}}"
SRC_LANG_CODE="${SRC_LANG_CODE:-en}"
TGT_LANG_CODE="${TGT_LANG_CODE:-de}"
PAPER_CONTEXT_PATH="${PAPER_CONTEXT_PATH:-}"

WAV_LIST_FILE="${1:-${WAV_LIST_FILE:-}}"
METRICS_LOG_FILE="${2:-${METRICS_LOG_FILE:-/io/out/metrics.jsonl}}"

if [ -z "$WAV_LIST_FILE" ]; then
  echo "Usage: docker run ... <image> <wavlist.txt> [<metrics.jsonl>]" >&2
  echo "  or set env var WAV_LIST_FILE" >&2
  exit 2
fi

export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

CONFIG_DIR="$(mktemp -d)"
CONFIG_FILE="${CONFIG_DIR}/speech_processor.yaml"

python /app/render_preset_yaml.py \
  --preset "$PRESET" \
  --source-lang-code "$SRC_LANG_CODE" \
  --target-lang-code "$TGT_LANG_CODE" \
  ${PAPER_CONTEXT_PATH:+--paper-context-path "$PAPER_CONTEXT_PATH"} \
  --output "$CONFIG_FILE"

mkdir -p "$(dirname "$METRICS_LOG_FILE")"

exec python -m simulstream.inference \
  --speech-processor-config "$CONFIG_FILE" \
  --wav-list-file "$WAV_LIST_FILE" \
  --src-lang "$SRC_LANG" \
  --tgt-lang "$TGT_LANG" \
  --metrics-log-file "$METRICS_LOG_FILE"
