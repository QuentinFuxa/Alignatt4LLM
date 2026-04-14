#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ralph_loop.sh [options]

Options:
  --repo-root PATH                 Git repository root
  --plan-path PATH                 Relative path to PLAN.md
  --history-path PATH              Relative path to RALPH_HISTORY.md
  --hypotheses-path PATH           Relative path to RALPH_HYPOTHESES.json
  --state-path PATH                Relative path to RALPH_STATE.md
  --agent-instructions-path PATH   Relative path to the stable worker prompt
  --state-dir PATH                 Directory for loop state
  --log-dir PATH                   Directory for per-iteration logs
  --branch NAME                    Require this git branch
  --codex-bin PATH                 Codex executable
  --model NAME                     Codex model
  --reasoning-effort LEVEL         Codex reasoning effort
  --timeout-seconds N              Hard timeout per codex exec
  --sleep-seconds N                Sleep between iterations / retries
  --max-consecutive-failures N     Stop after N failed iterations
  --max-iterations N               0 means run forever
  --stop-on-unchanged-plan N       Stop after N unchanged PLAN.md hashes
  --require-gpu                    Require nvidia-smi before each iteration
  --no-require-gpu                 Skip GPU preflight
  --require-env NAME               Require an environment variable; repeatable
  --remote NAME                    Git remote for push
  --push                           Push after each successful commit
  --terminal-event-format MODE     raw, pretty, or compact terminal output
  --allow-dirty-start              Skip clean-worktree preflight
  --dry-run                        Render prompt and stop
  -h, --help                       Show this help
EOF
}

REPO_ROOT="."
PLAN_PATH="ralph_mission/PLAN.md"
HISTORY_PATH="ralph_mission/RALPH_HISTORY.md"
HYPOTHESES_PATH="ralph_mission/RALPH_HYPOTHESES.json"
STATE_PATH="ralph_mission/RALPH_STATE.md"
AGENT_INSTRUCTIONS_PATH="ralph_mission/RALPH_AGENT.md"
STATE_DIR="${HOME}/.local/share/ralph-loop"
LOG_DIR="${HOME}/.local/state/ralph-loop/logs"
BRANCH=""
CODEX_BIN="codex"
MODEL="gpt-5.4"
REASONING_EFFORT="xhigh"
TIMEOUT_SECONDS=21600
SLEEP_SECONDS=30
MAX_CONSECUTIVE_FAILURES=3
MAX_ITERATIONS=0
STOP_ON_UNCHANGED_PLAN=3
REQUIRE_GPU=1
PUSH=0
REMOTE="origin"
TERMINAL_EVENT_FORMAT="pretty"
ALLOW_DIRTY_START=0
DRY_RUN=0
REQUIRE_ENV_VARS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --plan-path) PLAN_PATH="$2"; shift 2 ;;
    --history-path) HISTORY_PATH="$2"; shift 2 ;;
    --hypotheses-path) HYPOTHESES_PATH="$2"; shift 2 ;;
    --state-path) STATE_PATH="$2"; shift 2 ;;
    --agent-instructions-path) AGENT_INSTRUCTIONS_PATH="$2"; shift 2 ;;
    --state-dir) STATE_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --codex-bin) CODEX_BIN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --sleep-seconds) SLEEP_SECONDS="$2"; shift 2 ;;
    --max-consecutive-failures) MAX_CONSECUTIVE_FAILURES="$2"; shift 2 ;;
    --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
    --stop-on-unchanged-plan) STOP_ON_UNCHANGED_PLAN="$2"; shift 2 ;;
    --require-gpu) REQUIRE_GPU=1; shift ;;
    --no-require-gpu) REQUIRE_GPU=0; shift ;;
    --require-env) REQUIRE_ENV_VARS+=("$2"); shift 2 ;;
    --remote) REMOTE="$2"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --terminal-event-format) TERMINAL_EVENT_FORMAT="$2"; shift 2 ;;
    --allow-dirty-start) ALLOW_DIRTY_START=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

state_get() {
  local name="$1"
  local path="${STATE_DIR}/${name}"
  if [[ -f "$path" ]]; then
    cat "$path"
  fi
}

state_set() {
  local name="$1"
  local value="$2"
  mkdir -p "$STATE_DIR"
  printf '%s' "$value" > "${STATE_DIR}/${name}"
}

state_del() {
  local name="$1"
  rm -f "${STATE_DIR}/${name}"
}

write_status() {
  local phase="$1"
  local iteration="${2:-}"
  local iter_dir="${3:-}"
  local extra="${4:-}"
  mkdir -p "$STATE_DIR"
  cat > "${STATE_DIR}/status.txt" <<EOF
timestamp=$(timestamp_utc)
phase=$phase
iteration=$iteration
repo_root=$REPO_ROOT
branch=${BRANCH:-$(git_branch 2>/dev/null || true)}
state_dir=$STATE_DIR
log_dir=$LOG_DIR
current_log_dir=$iter_dir
extra=$extra
EOF
}

set_current_log_dir() {
  local iter_dir="$1"
  mkdir -p "$LOG_DIR"
  ln -sfn "$iter_dir" "${LOG_DIR}/current"
}

ensure_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

hash_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    shasum -a 256 "$path" | awk '{print $1}'
  fi
}

git_branch() {
  git -C "$REPO_ROOT" branch --show-current
}

git_head() {
  git -C "$REPO_ROOT" rev-parse HEAD
}

git_clean_or_die() {
  local status
  status="$(git -C "$REPO_ROOT" status --porcelain)"
  if [[ -n "$status" ]]; then
    echo "Worktree must be clean before iteration start." >&2
    echo "$status" >&2
    exit 1
  fi
}

git_clean_check() {
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]]
}

stop_category_from_message() {
  local message="$1"
  if [[ "$message" =~ ^RALPH_STOP:\ ([a-z_]+):\ .+ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
}

require_envs() {
  local name
  for name in "${REQUIRE_ENV_VARS[@]-}"; do
    [[ -z "$name" ]] && continue
    if [[ -z "${!name:-}" ]]; then
      echo "Missing required environment variable: $name" >&2
      exit 1
    fi
  done
}

preflight() {
  mkdir -p "$STATE_DIR" "$LOG_DIR"
  ensure_cmd git
  ensure_cmd jq
  ensure_cmd flock
  ensure_cmd "$CODEX_BIN"
  if [[ "$REQUIRE_GPU" -eq 1 ]]; then
    ensure_cmd nvidia-smi
  fi

  if [[ ! -d "$REPO_ROOT/.git" ]]; then
    echo "Not a git repository: $REPO_ROOT" >&2
    exit 1
  fi

  if [[ ! -f "$REPO_ROOT/$PLAN_PATH" ]]; then
    echo "Missing plan file: $REPO_ROOT/$PLAN_PATH" >&2
    exit 1
  fi

  if [[ ! -f "$REPO_ROOT/$AGENT_INSTRUCTIONS_PATH" ]]; then
    echo "Missing agent instructions: $REPO_ROOT/$AGENT_INSTRUCTIONS_PATH" >&2
    exit 1
  fi

  if [[ ! -f "$REPO_ROOT/$HYPOTHESES_PATH" ]]; then
    echo "Missing hypotheses file: $REPO_ROOT/$HYPOTHESES_PATH" >&2
    exit 1
  fi

  if [[ ! -f "$REPO_ROOT/$STATE_PATH" ]]; then
    echo "Missing derived state file: $REPO_ROOT/$STATE_PATH" >&2
    exit 1
  fi

  if [[ ! -f "${SCRIPT_DIR}/ralph_format_codex_jsonl.sh" ]]; then
    echo "Missing formatter script: ${SCRIPT_DIR}/ralph_format_codex_jsonl.sh" >&2
    exit 1
  fi

  if [[ ! -f "${SCRIPT_DIR}/ralph_registry.sh" ]]; then
    echo "Missing registry helper: ${SCRIPT_DIR}/ralph_registry.sh" >&2
    exit 1
  fi

  case "$TERMINAL_EVENT_FORMAT" in
    raw|pretty|compact) ;;
    *)
      echo "Unsupported --terminal-event-format: $TERMINAL_EVENT_FORMAT" >&2
      exit 1
      ;;
  esac

  if [[ -n "$BRANCH" ]]; then
    local current_branch
    current_branch="$(git_branch)"
    if [[ "$current_branch" != "$BRANCH" ]]; then
      echo "Expected branch $BRANCH, found $current_branch" >&2
      exit 1
    fi
  fi

  if [[ "$ALLOW_DIRTY_START" -ne 1 ]]; then
    git_clean_or_die
  fi

  if [[ -z "$(git -C "$REPO_ROOT" config --get user.name || true)" ]]; then
    echo "git user.name is not configured" >&2
    exit 1
  fi

  if [[ -z "$(git -C "$REPO_ROOT" config --get user.email || true)" ]]; then
    echo "git user.email is not configured" >&2
    exit 1
  fi

  require_envs
  bash "${SCRIPT_DIR}/ralph_registry.sh" validate-json "$REPO_ROOT/$HYPOTHESES_PATH"
  bash "${SCRIPT_DIR}/ralph_registry.sh" validate-state "$REPO_ROOT/$HYPOTHESES_PATH" "$REPO_ROOT/$STATE_PATH"
  bash "${SCRIPT_DIR}/ralph_registry.sh" validate-plan "$REPO_ROOT/$HYPOTHESES_PATH" "$REPO_ROOT/$PLAN_PATH"
  bash "${SCRIPT_DIR}/ralph_registry.sh" validate-history-tail "$REPO_ROOT/$HISTORY_PATH"
  write_status "idle" "" "" "preflight_ok"
}

build_prompt() {
  local prompt_path="$1"
  local iteration="$2"
  local branch="$3"
  local head_before="$4"
  local gpu_summary="$5"
  local history_tail
  local registry_summary
  history_tail="$(tail -n 80 "$REPO_ROOT/$HISTORY_PATH" 2>/dev/null || true)"
  if [[ -z "$history_tail" ]]; then
    history_tail="(history file empty)"
  fi
  registry_summary="$(bash "${SCRIPT_DIR}/ralph_registry.sh" summarize "$REPO_ROOT/$HYPOTHESES_PATH")"

  cat > "$prompt_path" <<EOF
$(cat "$REPO_ROOT/$AGENT_INSTRUCTIONS_PATH")

Automation metadata:
- Iteration: $iteration
- Branch: $branch
- Head before: $head_before
- Repo root: $REPO_ROOT
- PLAN.md path: $PLAN_PATH
- RALPH_HISTORY.md path: $HISTORY_PATH
- RALPH_HYPOTHESES.json path: $HYPOTHESES_PATH
- RALPH_STATE.md path: $STATE_PATH
- GPU preflight:
$gpu_summary

Current Ralph registry summary:
\`\`\`markdown
$registry_summary
\`\`\`

Recent Ralph history tail:
\`\`\`markdown
$history_tail
\`\`\`

Current PLAN.md contents:
\`\`\`markdown
$(cat "$REPO_ROOT/$PLAN_PATH")
\`\`\`
EOF
}

run_codex_iteration() {
  local iteration="$1"
  local branch="$2"
  local head_before="$3"
  local iter_dir="$4"
  local prompt_path="${iter_dir}/prompt.md"
  local terminal_path="${iter_dir}/codex.terminal.log"
  local events_path="${iter_dir}/codex.events.jsonl"
  local stderr_path="${iter_dir}/codex.stderr.log"
  local final_path="${iter_dir}/codex.final.txt"
  local hypotheses_before_path="${iter_dir}/ralph_hypotheses.before.json"
  local gpu_summary="- not requested"

  if [[ "$REQUIRE_GPU" -eq 1 ]]; then
    gpu_summary="$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader)"
  fi

  cp "$REPO_ROOT/$HYPOTHESES_PATH" "$hypotheses_before_path"
  build_prompt "$prompt_path" "$iteration" "$branch" "$head_before" "$gpu_summary"
  set_current_log_dir "$iter_dir"
  write_status "running_codex" "$iteration" "$iter_dir" "head_before=$head_before terminal_event_format=$TERMINAL_EVENT_FORMAT"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run completed. Prompt written to $prompt_path" >&2
    write_status "dry_run_complete" "$iteration" "$iter_dir" "prompt=$prompt_path"
    return 10
  fi

  local -a cmd
  cmd=(
    "$CODEX_BIN" exec
    -C "$REPO_ROOT"
    -m "$MODEL"
    -c "model_reasoning_effort=\"$REASONING_EFFORT\""
    -s danger-full-access
    --json
    -o "$final_path"
    -
  )

  if command -v timeout >/dev/null 2>&1; then
    if ! timeout --preserve-status "$TIMEOUT_SECONDS" "${cmd[@]}" < "$prompt_path" > >(tee "$events_path" | bash "${SCRIPT_DIR}/ralph_format_codex_jsonl.sh" --mode "$TERMINAL_EVENT_FORMAT" | tee "$terminal_path") 2> >(tee "$stderr_path" >&2); then
      write_status "codex_failed" "$iteration" "$iter_dir" "see=$stderr_path"
      return 1
    fi
  else
    if ! "${cmd[@]}" < "$prompt_path" > >(tee "$events_path" | bash "${SCRIPT_DIR}/ralph_format_codex_jsonl.sh" --mode "$TERMINAL_EVENT_FORMAT" | tee "$terminal_path") 2> >(tee "$stderr_path" >&2); then
      write_status "codex_failed" "$iteration" "$iter_dir" "see=$stderr_path"
      return 1
    fi
  fi

  if [[ ! -f "$final_path" ]]; then
    echo "Codex did not produce $final_path" >&2
    write_status "codex_failed" "$iteration" "$iter_dir" "missing_final=$final_path"
    return 1
  fi

  write_status "codex_completed" "$iteration" "$iter_dir" "final=$final_path"
  return 0
}

main() {
  REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
  preflight

  exec 9> "${STATE_DIR}/loop.lock"
  if ! flock -n 9; then
    echo "Another Ralph loop already holds ${STATE_DIR}/loop.lock" >&2
    exit 0
  fi

  local pending_push_head pending_push_error
  pending_push_head="$(state_get pending_push_head || true)"
  pending_push_error="$(state_get pending_push_error || true)"
  if [[ -n "$pending_push_head" ]]; then
    echo "Pending push failure for commit $pending_push_head" >&2
    if [[ -n "$pending_push_error" ]]; then
      echo "$pending_push_error" >&2
    fi
    exit 0
  fi

  local consecutive_failures=0
  local iteration_count
  iteration_count="$(state_get iteration_count || true)"
  iteration_count="${iteration_count:-0}"

  local unchanged_count
  unchanged_count="$(state_get consecutive_unchanged_plan || true)"
  unchanged_count="${unchanged_count:-0}"

  while true; do
    local branch current_branch head_before plan_hash_before started_at iter_dir
    branch="$(git_branch)"
    if [[ -z "$branch" ]]; then
      echo "Repository is in detached HEAD state." >&2
      exit 0
    fi
    current_branch="$branch"
    if [[ -n "$BRANCH" && "$current_branch" != "$BRANCH" ]]; then
      echo "Expected branch $BRANCH, found $current_branch" >&2
      exit 0
    fi

    if [[ "$ALLOW_DIRTY_START" -ne 1 ]]; then
      git_clean_or_die
    fi

    iteration_count=$((iteration_count + 1))
    started_at="$(timestamp_utc)"
    iter_dir="${LOG_DIR}/iter_$(printf '%04d' "$iteration_count")_${started_at//:/-}"
    mkdir -p "$iter_dir"
    set_current_log_dir "$iter_dir"
    head_before="$(git_head)"
    plan_hash_before="$(hash_file "$REPO_ROOT/$PLAN_PATH")"
    write_status "iteration_start" "$iteration_count" "$iter_dir" "branch=$branch head_before=$head_before"

    if run_codex_iteration "$iteration_count" "$branch" "$head_before" "$iter_dir"; then
      :
    else
      local rc=$?
      if [[ "$rc" -eq 10 ]]; then
        exit 0
      fi
      consecutive_failures=$((consecutive_failures + 1))
      echo "Codex iteration failed ($consecutive_failures/$MAX_CONSECUTIVE_FAILURES)." >&2
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "failures=$consecutive_failures"
      if [[ -f "${iter_dir}/codex.stderr.log" ]]; then
        tail -n 50 "${iter_dir}/codex.stderr.log" >&2 || true
      fi
      if (( consecutive_failures >= MAX_CONSECUTIVE_FAILURES )); then
        echo "Stopping after $MAX_CONSECUTIVE_FAILURES consecutive failures." >&2
        exit 0
      fi
      sleep "$SLEEP_SECONDS"
      continue
    fi

    consecutive_failures=0

    local final_message head_after plan_hash_after changed_paths
    final_message="$(tr -d '\r' < "${iter_dir}/codex.final.txt" | sed '/^[[:space:]]*$/d')"

    if [[ -z "$final_message" ]]; then
      echo "Codex final message is empty." >&2
      exit 0
    fi

    head_after="$(git_head)"
    if [[ "$head_after" == "$head_before" ]]; then
      echo "Codex did not create a new commit." >&2
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "no_new_commit"
      exit 0
    fi

    if ! git_clean_check; then
      echo "Worktree is dirty after Codex iteration. Expected a clean repo after commit." >&2
      git -C "$REPO_ROOT" status --short >&2 || true
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "dirty_after_commit"
      exit 0
    fi

    changed_paths="$(git -C "$REPO_ROOT" diff-tree --no-commit-id --name-only -r HEAD)"
    if ! grep -qx "$PLAN_PATH" <<< "$changed_paths"; then
      echo "Latest commit does not modify $PLAN_PATH" >&2
      echo "$changed_paths" >&2
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "plan_not_changed"
      exit 0
    fi

    if ! grep -qx "$HISTORY_PATH" <<< "$changed_paths"; then
      echo "Latest commit does not modify $HISTORY_PATH" >&2
      echo "$changed_paths" >&2
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "history_not_changed"
      exit 0
    fi

    if ! grep -qx "$HYPOTHESES_PATH" <<< "$changed_paths"; then
      echo "Latest commit does not modify $HYPOTHESES_PATH" >&2
      echo "$changed_paths" >&2
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "hypotheses_not_changed"
      exit 0
    fi

    if ! grep -qx "$STATE_PATH" <<< "$changed_paths"; then
      echo "Latest commit does not modify $STATE_PATH" >&2
      echo "$changed_paths" >&2
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "state_not_changed"
      exit 0
    fi

    if ! bash "${SCRIPT_DIR}/ralph_registry.sh" validate-json "$REPO_ROOT/$HYPOTHESES_PATH"; then
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "invalid_hypotheses_json"
      exit 0
    fi

    if ! bash "${SCRIPT_DIR}/ralph_registry.sh" validate-state "$REPO_ROOT/$HYPOTHESES_PATH" "$REPO_ROOT/$STATE_PATH"; then
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "state_out_of_sync"
      exit 0
    fi

    if ! bash "${SCRIPT_DIR}/ralph_registry.sh" validate-plan "$REPO_ROOT/$HYPOTHESES_PATH" "$REPO_ROOT/$PLAN_PATH"; then
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "invalid_plan_focus"
      exit 0
    fi

    if ! bash "${SCRIPT_DIR}/ralph_registry.sh" validate-history-tail "$REPO_ROOT/$HISTORY_PATH"; then
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "invalid_history_tail"
      exit 0
    fi

    if ! bash "${SCRIPT_DIR}/ralph_registry.sh" validate-budget "${iter_dir}/ralph_hypotheses.before.json" "$REPO_ROOT/$HYPOTHESES_PATH"; then
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "budget_exceeded_without_pivot"
      exit 0
    fi

    if [[ "$PUSH" -eq 1 ]]; then
      local target_branch push_error
      target_branch="$branch"
      if [[ -n "$BRANCH" ]]; then
        target_branch="$BRANCH"
      fi
      write_status "pushing" "$iteration_count" "$iter_dir" "target=${REMOTE}/${target_branch}"
      if ! git -C "$REPO_ROOT" push "$REMOTE" "HEAD:${target_branch}" > "${iter_dir}/git.push.stdout.log" 2> "${iter_dir}/git.push.stderr.log"; then
        push_error="$(tail -n 50 "${iter_dir}/git.push.stderr.log" 2>/dev/null || true)"
        state_set pending_push_head "$head_after"
        state_set pending_push_error "$push_error"
        echo "Push failed for $head_after" >&2
        echo "$push_error" >&2
        write_status "push_failed" "$iteration_count" "$iter_dir" "head=$head_after"
        exit 0
      fi
      state_del pending_push_head
      state_del pending_push_error
    fi

    plan_hash_after="$(hash_file "$REPO_ROOT/$PLAN_PATH")"
    if [[ "$plan_hash_after" == "$plan_hash_before" ]]; then
      unchanged_count=$((unchanged_count + 1))
    else
      unchanged_count=0
    fi

    state_set iteration_count "$iteration_count"
    state_set consecutive_unchanged_plan "$unchanged_count"
    state_set last_head "$head_after"
    state_set last_plan_hash "$plan_hash_after"
    printf '%s\n' "$changed_paths" > "${iter_dir}/changed_paths.txt"
    git -C "$REPO_ROOT" show --stat --summary --no-patch HEAD > "${iter_dir}/git.commit.txt"
    write_status "iteration_committed" "$iteration_count" "$iter_dir" "head_after=$head_after"

    if [[ "$final_message" == "RALPH_CONTINUE" ]]; then
      :
    elif [[ "$final_message" == RALPH_STOP:* ]]; then
      local stop_category
      stop_category="$(stop_category_from_message "$final_message" || true)"
      case "$stop_category" in
        no_bounded_next_step|human_decision_required|blocked_on_external_dependency|unsafe_repo_state)
          ;;
        *)
          echo "Unexpected stop category from Codex:" >&2
          echo "$final_message" >&2
          write_status "iteration_failed" "$iteration_count" "$iter_dir" "invalid_stop_category"
          exit 0
          ;;
      esac
      if ! bash "${SCRIPT_DIR}/ralph_registry.sh" validate-stop "$REPO_ROOT/$HYPOTHESES_PATH" "$final_message"; then
        write_status "iteration_failed" "$iteration_count" "$iter_dir" "invalid_stop_state"
        exit 0
      fi
      echo "$final_message" >&2
      write_status "stopped_by_agent" "$iteration_count" "$iter_dir" "$final_message"
      exit 0
    else
      echo "Unexpected final message from Codex:" >&2
      echo "$final_message" >&2
      write_status "iteration_failed" "$iteration_count" "$iter_dir" "unexpected_final_message"
      exit 0
    fi

    if (( MAX_ITERATIONS > 0 && iteration_count >= MAX_ITERATIONS )); then
      echo "Reached max iterations: $MAX_ITERATIONS" >&2
      write_status "stopped_max_iterations" "$iteration_count" "$iter_dir" "max=$MAX_ITERATIONS"
      exit 0
    fi

    if (( STOP_ON_UNCHANGED_PLAN > 0 && unchanged_count >= STOP_ON_UNCHANGED_PLAN )); then
      echo "Stopping because PLAN.md stayed unchanged for $unchanged_count iterations." >&2
      write_status "stopped_unchanged_plan" "$iteration_count" "$iter_dir" "unchanged_count=$unchanged_count"
      exit 0
    fi

    write_status "sleeping" "$iteration_count" "$iter_dir" "sleep_seconds=$SLEEP_SECONDS"
    sleep "$SLEEP_SECONDS"
  done
}

main "$@"
