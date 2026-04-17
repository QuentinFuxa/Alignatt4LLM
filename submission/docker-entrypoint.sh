#!/usr/bin/env bash
set -euo pipefail

PRESET="${CASCADE_SUBMISSION_PRESET:-main_low_latency}"
HOST="${CASCADE_HOST:-0.0.0.0}"
PORT="${CASCADE_PORT:-8765}"
POOL_SIZE="${CASCADE_POOL_SIZE:-1}"
ACQUIRE_TIMEOUT="${CASCADE_ACQUIRE_TIMEOUT:-600}"
SOURCE_LANG="${CASCADE_SOURCE_LANG:-en}"
TARGET_LANG="${CASCADE_TARGET_LANG:-de}"
METRICS_LOG_FILE="${CASCADE_METRICS_LOG_FILE:-metrics.jsonl}"

exec python /app/run_iwslt_submission.py server \
  --preset "$PRESET" \
  --host "$HOST" \
  --port "$PORT" \
  --pool-size "$POOL_SIZE" \
  --acquire-timeout "$ACQUIRE_TIMEOUT" \
  --source "$SOURCE_LANG" \
  --target "$TARGET_LANG" \
  --metrics-log-file "$METRICS_LOG_FILE" \
  "$@"
