#!/usr/bin/env bash
# Run the frozen main_low_latency submission preset on the blind IWSLT test-set
# for all three main-track directions, back to back on a single GPU.
#
# Each direction re-initialises the cascade so that only one ASR + MT pair is
# resident at any time, matching the AGENTS.md operating rules.
set -euo pipefail

cd "$(dirname "$0")/.."

PRESET="${PRESET:-main_low_latency}"
INPUT_DIR="${INPUT_DIR:-data/testset/audio}"
PYTHON_BIN="${PYTHON_BIN:-.venv-inference/bin/python}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

OUTPUT_PREFIX="${OUTPUT_PREFIX:-$(PRESET_NAME="$PRESET" "$PYTHON_BIN" - <<'PY'
from cascade.submission import get_submission_preset
import os

preset = get_submission_preset(os.environ["PRESET_NAME"])
print(f"outputs/iwslt26_testset_{preset.name}")
PY
)}"

for TGT in de it zh; do
  OUT="${OUTPUT_PREFIX}_en${TGT}"
  LOG="${LOG_DIR}/testset_${PRESET}_en${TGT}.log"
  echo "[$(date -Iseconds)] en->$TGT preset=$PRESET -> $OUT"
  "$PYTHON_BIN" run_iwslt_submission.py batch \
    --preset "$PRESET" \
    --source en \
    --target "$TGT" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUT" \
    >"$LOG" 2>&1
  echo "[$(date -Iseconds)] en->$TGT done (log: $LOG)"
done
