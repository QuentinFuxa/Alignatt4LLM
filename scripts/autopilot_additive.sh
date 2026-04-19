#!/usr/bin/env bash
# Autopilot: wait for the in-flight chunk_ms=850 sweep to finish, then run
# the downstream pipeline (chunk_ms=1900 inference + score both + sync).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date -Iseconds)] waiting for chunk850 pid=$WAIT_PID to exit"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date -Iseconds)] chunk850 pid=$WAIT_PID has exited"
fi

exec bash scripts/run_additive_full_pipeline.sh
