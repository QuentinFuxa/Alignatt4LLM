#!/usr/bin/env bash
set -euo pipefail

language_name() {
  case "$1" in
    en) printf '%s\n' "English" ;;
    de) printf '%s\n' "German" ;;
    it) printf '%s\n' "Italian" ;;
    zh) printf '%s\n' "Simplified Chinese" ;;
    *)
      echo "Unsupported language code: $1" >&2
      exit 2
      ;;
  esac
}

MODE="${MODE:-}"
if [ "$#" -gt 0 ] && { [ "$1" = "infer" ] || [ "$1" = "serve" ]; }; then
  MODE="$1"
  shift
fi
MODE="${MODE:-infer}"

PRESET="${PRESET:-${CASCADE_SUBMISSION_PRESET:-main_low_latency}}"
SRC_LANG_CODE="${SRC_LANG_CODE:-en}"
TGT_LANG_CODE="${TGT_LANG_CODE:-de}"
SRC_LANG="${SRC_LANG:-${CASCADE_SOURCE_LANG:-$(language_name "$SRC_LANG_CODE")}}"
TGT_LANG="${TGT_LANG:-${CASCADE_TARGET_LANG:-$(language_name "$TGT_LANG_CODE")}}"

export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"

CONFIG_DIR="$(mktemp -d)"
CONFIG_FILE="${CONFIG_DIR}/speech_processor.yaml"

python /app/submission/render_preset_yaml.py \
  --preset "$PRESET" \
  --source-lang-code "$SRC_LANG_CODE" \
  --target-lang-code "$TGT_LANG_CODE" \
  --output "$CONFIG_FILE"

if [ "$MODE" = "infer" ]; then
  WAV_LIST_FILE="${1:-${WAV_LIST_FILE:-}}"
  METRICS_LOG_FILE="${2:-${METRICS_LOG_FILE:-/io/out/metrics.jsonl}}"

  if [ -z "$WAV_LIST_FILE" ]; then
    echo "Usage: docker run ... cascade-simul-iwslt26 [infer] <wavlist.txt> [<metrics.jsonl>]" >&2
    echo "  or set WAV_LIST_FILE and optionally METRICS_LOG_FILE" >&2
    exit 2
  fi

  mkdir -p "$(dirname "$METRICS_LOG_FILE")"

  exec simulstream_inference \
    --speech-processor-config "$CONFIG_FILE" \
    --wav-list-file "$WAV_LIST_FILE" \
    --src-lang "$SRC_LANG" \
    --tgt-lang "$TGT_LANG" \
    --metrics-log-file "$METRICS_LOG_FILE"
fi

if [ "$MODE" = "serve" ]; then
  HOST="${HOST:-0.0.0.0}"
  PORT="${PORT:-8080}"
  POOL_SIZE="${POOL_SIZE:-1}"
  TTL="${TTL:-3600}"
  SERVER_CONFIG="${CONFIG_DIR}/http_server.yaml"

  cat >"$SERVER_CONFIG" <<EOF
pool_size: ${POOL_SIZE}
hostname: ${HOST}
port: ${PORT}
ttl: ${TTL}
EOF

  exec python -m simulstream.server.speech_processors.remote.http_speech_processor_server \
    --server-config "$SERVER_CONFIG" \
    --speech-processor-config "$CONFIG_FILE"
fi

echo "Unknown MODE: $MODE" >&2
echo "Expected 'infer' or 'serve'." >&2
exit 2
