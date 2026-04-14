#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STATE_DIR="${HOME}/.local/share/ralph-loop"
LOG_DIR="${HOME}/.local/state/ralph-loop/logs"
FOLLOW=0
FORMAT_MODE="compact"

usage() {
  cat <<'EOF'
Usage: ralph_watch.sh [options]

Options:
  --state-dir PATH   Ralph state directory
  --log-dir PATH     Ralph log directory
  --format MODE      raw, pretty, or compact recent-event rendering
  --follow           Follow the current iteration in real time
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-dir) STATE_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --format) FORMAT_MODE="$2"; shift 2 ;;
    --follow) FOLLOW=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$FORMAT_MODE" in
  raw|pretty|compact) ;;
  *)
    echo "Unsupported --format: $FORMAT_MODE" >&2
    exit 1
    ;;
esac

show_snapshot() {
  local status_file="${STATE_DIR}/status.txt"
  local current_link="${LOG_DIR}/current"
  local current_dir=""
  local events_file=""
  local stderr_file=""
  local final_file=""

  if [[ -L "$current_link" || -d "$current_link" ]]; then
    current_dir="$(cd "$current_link" && pwd)"
    events_file="${current_dir}/codex.events.jsonl"
    stderr_file="${current_dir}/codex.stderr.log"
    final_file="${current_dir}/codex.final.txt"
  fi

  echo "== Ralph Status =="
  if [[ -f "$status_file" ]]; then
    cat "$status_file"
  else
    echo "status file not found: $status_file"
  fi

  echo
  echo "== Current Iteration =="
  if [[ -n "$current_dir" ]]; then
    echo "$current_dir"
  else
    echo "no current iteration"
  fi

  echo
  echo "== Final Message =="
  if [[ -f "$final_file" ]]; then
    cat "$final_file"
  else
    echo "not available yet"
  fi

  echo
  echo "== Recent Events =="
  if [[ -f "$events_file" ]]; then
    tail -n 20 "$events_file" | bash "${SCRIPT_DIR}/ralph_format_codex_jsonl.sh" --mode "$FORMAT_MODE"
  else
    echo "events file not available yet"
  fi

  echo
  echo "== Recent Stderr =="
  if [[ -f "$stderr_file" ]]; then
    tail -n 20 "$stderr_file"
  else
    echo "stderr file not available yet"
  fi
}

follow_current() {
  local current_link="${LOG_DIR}/current"
  local events_file="${current_link}/codex.events.jsonl"
  local stderr_file="${current_link}/codex.stderr.log"

  echo "Watching ${STATE_DIR}/status.txt"
  echo "Watching ${events_file}"
  echo "Watching ${stderr_file}"
  echo

  while true; do
    clear || true
    show_snapshot
    sleep 2
  done
}

if [[ "$FOLLOW" -eq 1 ]]; then
  follow_current
else
  show_snapshot
fi
