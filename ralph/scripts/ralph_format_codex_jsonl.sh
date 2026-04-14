#!/usr/bin/env bash
set -euo pipefail

MODE="pretty"

usage() {
  cat <<'EOF'
Usage: ralph_format_codex_jsonl.sh [options]

Options:
  --mode MODE   One of: raw, pretty, compact
  -h, --help    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "raw" && "$MODE" != "pretty" && "$MODE" != "compact" ]]; then
  echo "Unsupported mode: $MODE" >&2
  exit 1
fi

if [[ "$MODE" == "raw" ]]; then
  cat
  exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  cat
  exit 0
fi

pretty_print_line() {
  local line="$1"
  if printf '%s\n' "$line" | jq -e . >/dev/null 2>&1; then
    printf '%s\n' "$line" | jq -M .
  else
    printf '%s\n' "$line"
  fi
}

compact_print_line() {
  local line="$1"
  if ! printf '%s\n' "$line" | jq -e . >/dev/null 2>&1; then
    printf '%s\n' "$line"
    return
  fi

  printf '%s\n' "$line" | jq -r '
    def s:
      if . == null then ""
      elif type == "string" then .
      elif type == "number" or type == "boolean" then tostring
      else ""
      end;
    def squash:
      gsub("[[:space:]]+"; " ")
      | sub("^ "; "")
      | sub(" $"; "");
    def first_nonempty($items):
      $items
      | map(select(. != null and . != ""))
      | .[0];
    . as $event
    | (first_nonempty([
        $event.type,
        $event.event,
        "event"
      ]) // "event") as $type
    | (first_nonempty([
        $event.timestamp,
        $event.time,
        $event.created_at
      ]) // "") as $timestamp
    | (first_nonempty([
        ($event.thread_id | s),
        ($event.thread.id | s),
        ($event.turn_id | s),
        ($event.turn.id | s),
        ($event.item.id | s),
        ($event.id | s)
      ]) // "") as $id
    | (first_nonempty([
        ($event.item.type | s),
        ($event.item.kind | s),
        ($event.kind | s),
        ($event.role | s),
        ($event.status | s)
      ]) // "") as $kind
    | ((first_nonempty([
        ($event.message | s),
        ($event.text | s),
        ($event.summary | s),
        ($event.detail | s),
        ($event.error.message | s),
        ($event.error | s),
        ($event.item.command | s),
        ($event.item.text | s),
        ($event.item.summary | s),
        ($event.delta | s),
        ($event.output_text | s)
      ]) // "") | squash) as $message
    | (
        "[" + $type + "]"
        + (if $timestamp != "" then " " + $timestamp else "" end)
        + (if $id != "" then " id=" + $id else "" end)
        + (if $kind != "" then " kind=" + $kind else "" end)
      )
    , (if ($message // "") != "" then "  " + $message else empty end)
  '
}

while IFS= read -r line || [[ -n "$line" ]]; do
  case "$MODE" in
    pretty) pretty_print_line "$line" ;;
    compact) compact_print_line "$line" ;;
  esac
done
