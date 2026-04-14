# README_RALPH

This branch contains a clean Ralph bootstrap: the loop scripts, the registry
guardrails, and one blank mission directory under `ralph/ralph_mission/`.

## Files

### Loop and helpers

- `scripts/ralph_loop.sh`
  Main supervisor. It runs one fresh `codex exec` session per iteration,
  validates the mission registry before and after the run, and requires the
  worker to leave a clean commit behind.
- `scripts/ralph_registry.sh`
  Validation and rendering helper for `RALPH_HYPOTHESES.json`,
  `RALPH_STATE.md`, `PLAN.md`, and the tail of `RALPH_HISTORY.md`.
- `scripts/ralph_format_codex_jsonl.sh`
  Pretty-printer for `codex exec --json` event logs.
- `scripts/ralph_watch.sh`
  Local watcher for `status.txt`, current logs, and recent events.

### Mission files

- `ralph_mission/EXISTING.md`
  Human context: what already exists, what matters, what is out of scope.
- `ralph_mission/RALPH_AGENT.md`
  Stable worker prompt. This is the contract Ralph follows every iteration.
- `ralph_mission/RALPH_HYPOTHESES.json`
  Machine-readable source of truth for open, frozen, blocked, and supported
  hypotheses.
- `ralph_mission/RALPH_STATE.md`
  Human-readable state derived from `RALPH_HYPOTHESES.json`.
- `ralph_mission/PLAN.md`
  The current bounded iteration plan. Ralph must replace it every iteration.
- `ralph_mission/RALPH_HISTORY.md`
  Iteration-by-iteration decision log. Ralph appends one entry per iteration.

## Prerequisites

Required:

- `git`
- `jq`
- `flock`
- `codex`

Optional but common:

- `nvidia-smi` if you want `--require-gpu`
- `timeout`

## Initialize a blank mission

The branch already contains one blank mission in `ralph_mission/`.

Edit these files before the first real run:

1. `ralph_mission/EXISTING.md`
   Replace placeholders with the actual mission, code surface, constraints,
   datasets, evaluation rules, and known blockers.
2. `ralph_mission/RALPH_AGENT.md`
   Keep the loop contract, but adapt mission-specific goals and constraints.
3. `ralph_mission/RALPH_HYPOTHESES.json`
   Replace the bootstrap hypothesis with the first real bounded hypothesis for
   your mission.
4. `ralph_mission/PLAN.md`
   Make sure the active focus IDs match `active_focus_ids` from the JSON and
   that the plan names one bounded next step.

Then regenerate and validate the derived files:

```bash
bash scripts/ralph_registry.sh render-state ralph_mission/RALPH_HYPOTHESES.json > ralph_mission/RALPH_STATE.md

bash scripts/ralph_registry.sh validate-json ralph_mission/RALPH_HYPOTHESES.json
bash scripts/ralph_registry.sh validate-state ralph_mission/RALPH_HYPOTHESES.json ralph_mission/RALPH_STATE.md
bash scripts/ralph_registry.sh validate-plan ralph_mission/RALPH_HYPOTHESES.json ralph_mission/PLAN.md
bash scripts/ralph_registry.sh validate-history-tail ralph_mission/RALPH_HISTORY.md
```

## Launch Ralph

Defaults now point to `ralph_mission/`, so the minimal command is:

```bash
bash scripts/ralph_loop.sh \
  --repo-root "$(pwd)" \
  --branch "$(git branch --show-current)" \
  --no-require-gpu
```

Typical Linux GPU run:

```bash
bash scripts/ralph_loop.sh \
  --repo-root /srv/project \
  --branch ralph/mission \
  --state-dir ~/.local/share/ralph-loop \
  --log-dir ~/.local/state/ralph-loop/logs \
  --push \
  --require-gpu \
  --require-env HF_TOKEN
```

To inspect a running loop:

```bash
bash scripts/ralph_watch.sh --follow --format compact
```

## Notes

- Ralph always works in the local checkout it is launched from.
- The runtime state and logs stay out of Git under `~/.local/share/ralph-loop`
  and `~/.local/state/ralph-loop/logs` unless you override them.
- If you want multiple missions in the same repo, copy `ralph_mission/` to a
  new directory and pass `--plan-path`, `--history-path`, `--hypotheses-path`,
  `--state-path`, and `--agent-instructions-path` to `scripts/ralph_loop.sh`.
