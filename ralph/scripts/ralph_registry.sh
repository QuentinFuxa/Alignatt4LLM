#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ralph_registry.sh <command> [args]

Commands:
  validate-json JSON_PATH
  render-state JSON_PATH
  summarize JSON_PATH
  validate-state JSON_PATH STATE_PATH
  validate-plan JSON_PATH PLAN_PATH
  validate-history-tail HISTORY_PATH
  validate-stop JSON_PATH FINAL_MESSAGE
  validate-budget BEFORE_JSON_PATH AFTER_JSON_PATH
EOF
}

die() {
  echo "$1" >&2
  exit 1
}

ensure_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "Missing required command: $1"
  fi
}

normalize_lines() {
  sed '/^[[:space:]]*$/d' | LC_ALL=C sort -u
}

assert_same_set() {
  local label="$1"
  local expected_path="$2"
  local actual_path="$3"
  if ! diff -u <(normalize_lines < "$expected_path") <(normalize_lines < "$actual_path") >/dev/null; then
    echo "Mismatched ${label}." >&2
    diff -u <(normalize_lines < "$expected_path") <(normalize_lines < "$actual_path") >&2 || true
    exit 1
  fi
}

extract_markdown_section() {
  local heading="$1"
  local path="$2"
  awk -v heading="## ${heading}" '
    $0 == heading {capture=1; next}
    capture && /^## / {exit}
    capture {print}
  ' "$path"
}

section_ids() {
  sed -nE 's/^[[:space:]]*-[[:space:]]*`?([A-Za-z0-9._-]+)`?.*/\1/p'
}

extract_latest_history_entry() {
  local path="$1"
  awk '
    /^## / {entry=$0 ORS; next}
    {entry=entry $0 ORS}
    END {printf "%s", entry}
  ' "$path"
}

validate_json() {
  local json_path="$1"
  jq -e '
    def string_array:
      type == "array" and all(.[]?; type == "string");
    def non_empty_string:
      type == "string" and length > 0;
    def iteration_value:
      (type == "string" and length > 0) or type == "number";
    def allowed_kind:
      . == "runtime_candidate"
      or . == "paper_claim"
      or . == "analysis_aux";
    def allowed_status:
      . == "open"
      or . == "active"
      or . == "replay_candidate"
      or . == "supported_local"
      or . == "falsified"
      or . == "frozen_annex"
      or . == "promoted_runtime"
      or . == "blocked_human"
      or . == "blocked_external"
      or . == "superseded";
    def allowed_priority:
      . == "primary"
      or . == "secondary"
      or . == "annex";
    def allowed_reopen_rule:
      . == "new_talk"
      or . == "new_delay"
      or . == "new_runtime_artifact"
      or . == "human_override";
    def allowed_global_status:
      . == "active"
      or . == "terminal_no_bounded_next_step"
      or . == "terminal_human_decision_required"
      or . == "terminal_blocked_external_dependency"
      or . == "terminal_unsafe_repo_state";
    . as $root
    | type == "object"
    and (.version | non_empty_string)
    and (.global_goal | string_array and length > 0)
    and (.global_status | allowed_global_status)
    and (.active_focus_ids | string_array)
    and (.reopen_policy | type == "object")
    and (.reopen_policy.default_rules | string_array)
    and (all(.reopen_policy.default_rules[]?; allowed_reopen_rule))
    and (.reopen_policy.note | non_empty_string)
    and (.hypotheses | type == "array")
    and (.iteration_ledger | type == "array")
    and (
      [ .hypotheses[].id ] as $ids
      | ($ids | length) == ($ids | unique | length)
      and all(.active_focus_ids[]?; $ids | index(.) != null)
    )
    and all(.hypotheses[]?;
      type == "object"
      and (.id | non_empty_string)
      and (.family | non_empty_string)
      and (.title | non_empty_string)
      and (.question | non_empty_string)
      and (.kind | allowed_kind)
      and (.status | allowed_status)
      and (.priority | allowed_priority)
      and (.scope | type == "object")
      and (.scope.talks | string_array)
      and (.scope.delays | string_array)
      and (.scope.slice | non_empty_string)
      and (.scope.artifacts | string_array)
      and (.success_signal | non_empty_string)
      and (.failure_signal | non_empty_string)
      and (.reopen_only_if | string_array)
      and all(.reopen_only_if[]?; allowed_reopen_rule)
      and (.budget | type == "object")
      and (.budget.max_local_iterations | type == "number" and . >= 0)
      and (.budget.used | type == "number" and . >= 0)
      and (.last_touched_commit | non_empty_string)
      and (.last_touched_iteration | iteration_value)
      and (.evidence | type == "array")
      and all(.evidence[]?;
        type == "object"
        and (.kind | non_empty_string)
        and (.ref | non_empty_string)
        and (.summary | non_empty_string)
      )
    )
    and all(.iteration_ledger[]?;
      type == "object"
      and (.iteration | iteration_value)
      and (.commit | non_empty_string)
      and (.hypothesis_ids_touched | string_array)
      and all(.hypothesis_ids_touched[]?; $root.hypotheses | map(.id) | index(.) != null)
      and (.status_transitions | string_array)
      and (.bounded_outcome | non_empty_string)
    )
  ' "$json_path" >/dev/null || die "Invalid Ralph hypotheses JSON: $json_path"
}

render_state() {
  local json_path="$1"
  local rendered_from
  if [[ "$json_path" == */* ]]; then
    rendered_from="$(basename "$(dirname "$json_path")")/$(basename "$json_path")"
  else
    rendered_from="${json_path#./}"
  fi
  jq -r --arg rendered_from "$rendered_from" '
    . as $root
    |
    def join_or_none:
      if length == 0 then "none" else join(", ") end;
    def scope_summary($scope):
      "talks=" + (($scope.talks // []) | join_or_none)
      + "; delays=" + (($scope.delays // []) | join_or_none)
      + "; slice=" + ($scope.slice // "none");
    def bullet_lines($items):
      if ($items | length) == 0 then
        ["- none"]
      else
        $items
      end;
    def hypothesis_line:
      "- `\(.id)` [status=`\(.status)` kind=`\(.kind)` priority=`\(.priority)`] \(.title) | \((scope_summary(.scope)))";
    [
      "# Ralph State",
      "",
      "Derived from `\($rendered_from)`. Do not edit by hand.",
      "",
      "## Goal"
    ]
    + (.global_goal | map("- " + .))
    + [
      "",
      "## Active Focus"
    ]
    + bullet_lines(
      [ $root.hypotheses[]
        | select(.id as $id | $root.active_focus_ids | index($id) != null)
        | hypothesis_line
      ]
    )
    + [
      "",
      "## Open Runtime Candidates"
    ]
    + bullet_lines(
      [ $root.hypotheses[]
        | select(.kind == "runtime_candidate")
        | select(.status == "open" or .status == "active" or .status == "replay_candidate" or .status == "supported_local")
        | hypothesis_line
      ]
    )
    + [
      "",
      "## Frozen Or Falsified Branches"
    ]
    + bullet_lines(
      [ $root.hypotheses[]
        | select(.status == "frozen_annex" or .status == "falsified" or .status == "superseded")
        | hypothesis_line
      ]
    )
    + [
      "",
      "## Paper-Only Claims"
    ]
    + bullet_lines(
      [ $root.hypotheses[]
        | select(.kind == "paper_claim")
        | hypothesis_line
      ]
    )
    + [
      "",
      "## Reopen Conditions"
    ]
    + bullet_lines(
      [ $root.hypotheses[]
        | select(.status == "frozen_annex" or .status == "falsified" or .status == "superseded")
        | "- `\(.id)` -> " + ((.reopen_only_if // []) | join_or_none)
      ]
    )
    + [
      "",
      "## Recent Iterations"
    ]
    + bullet_lines(
      [ .iteration_ledger
        | reverse
        | .[:5][]
        | "- `\(.iteration)` `\(.commit)` | ids=" + ((.hypothesis_ids_touched // []) | join_or_none) + " | " + .bounded_outcome
      ]
    )
    | join("\n")
  ' "$json_path"
}

summarize_registry() {
  local json_path="$1"
  jq -r '
    . as $root
    |
    def join_or_none:
      if length == 0 then "none" else join(", ") end;
    def scope_summary($scope):
      "talks=" + (($scope.talks // []) | join_or_none)
      + "; delays=" + (($scope.delays // []) | join_or_none)
      + "; slice=" + ($scope.slice // "none");
    [
      "- global_status: `\(.global_status)`",
      "- active_focus_ids: " + ((.active_focus_ids | map("`" + . + "`")) | join_or_none),
      (
        $root.hypotheses[]
        | select(.id as $id | $root.active_focus_ids | index($id) != null)
        | "- active_focus_detail: `\(.id)` | \(.title) | " + (scope_summary(.scope))
      ),
      "- frozen_branch_ids: " + (
        [
          $root.hypotheses[]
          | select(.status == "frozen_annex" or .status == "falsified" or .status == "superseded")
          | "`" + .id + "`"
        ] | join_or_none
      ),
      (
        $root.iteration_ledger[-3:][]?
        | "- recent_outcome: `\(.iteration)` -> " + .bounded_outcome
      )
    ] | join("\n")
  ' "$json_path"
}

validate_state() {
  local json_path="$1"
  local state_path="$2"
  local tmp
  tmp="$(mktemp)"
  render_state "$json_path" > "$tmp"
  if ! diff -u "$state_path" "$tmp" >/dev/null; then
    echo "RALPH_STATE.md is out of sync with RALPH_HYPOTHESES.json." >&2
    diff -u "$state_path" "$tmp" >&2 || true
    rm -f "$tmp"
    exit 1
  fi
  rm -f "$tmp"
}

validate_plan() {
  local json_path="$1"
  local plan_path="$2"
  local focus_tmp blocked_tmp expected_tmp
  focus_tmp="$(mktemp)"
  blocked_tmp="$(mktemp)"
  expected_tmp="$(mktemp)"

  grep -qx '## Focus Hypothesis IDs' "$plan_path" || die "PLAN.md missing required section: Focus Hypothesis IDs"
  grep -qx '## Blocked Or Frozen IDs' "$plan_path" || die "PLAN.md missing required section: Blocked Or Frozen IDs"
  grep -qx '## Why This Is Not A Revisit' "$plan_path" || die "PLAN.md missing required section: Why This Is Not A Revisit"

  extract_markdown_section "Focus Hypothesis IDs" "$plan_path" | section_ids > "$focus_tmp"
  extract_markdown_section "Blocked Or Frozen IDs" "$plan_path" | section_ids > "$blocked_tmp"
  jq -r '.active_focus_ids[]?' "$json_path" > "$expected_tmp"

  [[ -s "$focus_tmp" ]] || die "PLAN.md Focus Hypothesis IDs must list at least one hypothesis id."
  assert_same_set "focus hypothesis ids" "$expected_tmp" "$focus_tmp"

  while IFS= read -r blocked_id; do
    [[ -z "$blocked_id" ]] && continue
    jq -e --arg blocked_id "$blocked_id" '
      .hypotheses[]
      | select(.id == $blocked_id)
      | .status == "frozen_annex" or .status == "falsified" or .status == "superseded" or .status == "blocked_human" or .status == "blocked_external"
    ' "$json_path" >/dev/null || die "Blocked Or Frozen id is not currently blocked/frozen: $blocked_id"
  done < "$blocked_tmp"

  local why_section why_tokens
  why_section="$(extract_markdown_section "Why This Is Not A Revisit" "$plan_path")"
  [[ -n "$why_section" ]] || die "PLAN.md Why This Is Not A Revisit must not be empty."
  why_tokens="$(printf '%s\n' "$why_section" | grep -Eo 'new_talk|new_delay|new_runtime_artifact|human_override' | LC_ALL=C sort -u || true)"

  while IFS= read -r focus_id; do
    [[ -z "$focus_id" ]] && continue
    local focus_status allowed_reopen
    focus_status="$(jq -r --arg focus_id "$focus_id" '.hypotheses[] | select(.id == $focus_id) | .status' "$json_path")"
    if [[ "$focus_status" == "frozen_annex" || "$focus_status" == "falsified" || "$focus_status" == "superseded" ]]; then
      allowed_reopen="$(jq -r --arg focus_id "$focus_id" '.hypotheses[] | select(.id == $focus_id) | .reopen_only_if[]?' "$json_path" | LC_ALL=C sort -u)"
      [[ -n "$why_tokens" ]] || die "PLAN.md focuses blocked hypothesis $focus_id without an allowed reopen token in Why This Is Not A Revisit."
      if ! comm -12 <(printf '%s\n' "$why_tokens") <(printf '%s\n' "$allowed_reopen") | grep -q .; then
        die "PLAN.md focuses blocked hypothesis $focus_id without satisfying reopen_only_if."
      fi
    fi
  done < "$focus_tmp"

  rm -f "$focus_tmp" "$blocked_tmp" "$expected_tmp"
}

validate_history_tail() {
  local history_path="$1"
  local latest_entry
  latest_entry="$(extract_latest_history_entry "$history_path")"
  [[ -n "$latest_entry" ]] || die "RALPH_HISTORY.md is empty."
  grep -q '^### Hypothesis IDs touched$' <<< "$latest_entry" || die "Latest RALPH_HISTORY entry is missing ### Hypothesis IDs touched."
  grep -q '^### Status transitions$' <<< "$latest_entry" || die "Latest RALPH_HISTORY entry is missing ### Status transitions."
  grep -A80 '^### Hypothesis IDs touched$' <<< "$latest_entry" | section_ids | grep -q . || die "Latest RALPH_HISTORY entry has no hypothesis ids."
  grep -A80 '^### Status transitions$' <<< "$latest_entry" | grep -E '^- ' -q || die "Latest RALPH_HISTORY entry has no status transitions."
}

validate_stop() {
  local json_path="$1"
  local final_message="$2"
  [[ "$final_message" =~ ^RALPH_STOP:\ ([a-z_]+):\ .+ ]] || die "Invalid stop message format."
  local category expected_status open_primary_count
  category="${BASH_REMATCH[1]}"
  case "$category" in
    no_bounded_next_step) expected_status="terminal_no_bounded_next_step" ;;
    human_decision_required) expected_status="terminal_human_decision_required" ;;
    blocked_on_external_dependency) expected_status="terminal_blocked_external_dependency" ;;
    unsafe_repo_state) expected_status="terminal_unsafe_repo_state" ;;
    *) die "Unsupported stop category: $category" ;;
  esac

  local global_status
  global_status="$(jq -r '.global_status' "$json_path")"
  [[ "$global_status" == "$expected_status" ]] || die "Stop category ${category} does not match global_status ${global_status}."

  open_primary_count="$(jq '[.hypotheses[] | select(.priority == "primary") | select(.status == "open" or .status == "active" or .status == "replay_candidate")] | length' "$json_path")"
  [[ "$open_primary_count" == "0" ]] || die "Cannot stop Ralph while primary hypotheses remain open."
}

validate_budget() {
  local before_json_path="$1"
  local after_json_path="$2"
  local tmp
  tmp="$(mktemp)"
  jq -n --slurpfile before "$before_json_path" --slurpfile after "$after_json_path" '
    ($before[0]) as $before
    | ($after[0]) as $after
    |
    def scope_widened($prev; $curr):
      (($curr.scope.talks // []) - ($prev.scope.talks // []) | length > 0)
      or (($curr.scope.delays // []) - ($prev.scope.delays // []) | length > 0)
      or (($curr.scope.artifacts // []) - ($prev.scope.artifacts // []) | length > 0);
    [
      $after.active_focus_ids[] as $focus_id
      | ($after.hypotheses[] | select(.id == $focus_id)) as $curr
      | select($curr.budget.used > $curr.budget.max_local_iterations)
      | ($before.hypotheses[]? | select(.id == $focus_id)) as $prev
      | select($prev != null)
      | select(($curr.status == $prev.status) and (scope_widened($prev; $curr) | not))
      | {
          id: $focus_id,
          used: $curr.budget.used,
          max: $curr.budget.max_local_iterations,
          status: $curr.status
        }
    ]
  ' > "$tmp"

  if jq -e 'length > 0' "$tmp" >/dev/null; then
    echo "Active focus exceeded its local budget without status change or scope widening." >&2
    jq -r '.[] | "- \(.id): used=\(.used) max=\(.max) status=\(.status)"' "$tmp" >&2
    rm -f "$tmp"
    exit 1
  fi
  rm -f "$tmp"
}

main() {
  ensure_cmd jq
  local command="${1:-}"
  if [[ -z "$command" ]]; then
    usage >&2
    exit 1
  fi
  shift

  case "$command" in
    validate-json)
      [[ $# -eq 1 ]] || die "validate-json expects JSON_PATH"
      validate_json "$1"
      ;;
    render-state)
      [[ $# -eq 1 ]] || die "render-state expects JSON_PATH"
      render_state "$1"
      ;;
    summarize)
      [[ $# -eq 1 ]] || die "summarize expects JSON_PATH"
      summarize_registry "$1"
      ;;
    validate-state)
      [[ $# -eq 2 ]] || die "validate-state expects JSON_PATH STATE_PATH"
      validate_state "$1" "$2"
      ;;
    validate-plan)
      [[ $# -eq 2 ]] || die "validate-plan expects JSON_PATH PLAN_PATH"
      validate_plan "$1" "$2"
      ;;
    validate-history-tail)
      [[ $# -eq 1 ]] || die "validate-history-tail expects HISTORY_PATH"
      validate_history_tail "$1"
      ;;
    validate-stop)
      [[ $# -eq 2 ]] || die "validate-stop expects JSON_PATH FINAL_MESSAGE"
      validate_stop "$1" "$2"
      ;;
    validate-budget)
      [[ $# -eq 2 ]] || die "validate-budget expects BEFORE_JSON_PATH AFTER_JSON_PATH"
      validate_budget "$1" "$2"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      die "Unknown command: $command"
      ;;
  esac
}

main "$@"
