#!/usr/bin/env bash
# Full dev-set validation of the chunk_ms=1100 / border_margin=1 HIGH preset
# per PLAN Phase 1 after the 2-clip pilot selected chunk_ms=1100 as the
# COMET-optimal candidate. Sequential en->{de,it,zh} on a single GPU.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv-inference/bin/python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-.venv-evaluation/bin/python}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

for TGT in de it zh; do
  OUT="outputs/iwslt26_devset_chunk1100_borderp1_en${TGT}"
  LOG="${LOG_DIR}/devset_chunk1100_borderp1_en${TGT}.log"
  echo "[$(date -Iseconds)] devset en->$TGT chunk_ms=1100 -> $OUT"
  "$PYTHON_BIN" run_simulstream_batch.py \
    --input-dir dev-set/audio \
    --output-dir "$OUT" \
    --chunk-ms 1100 \
    --source en --target "$TGT" \
    --translation-alignatt-border-margin 1 \
    >"$LOG" 2>&1
  echo "[$(date -Iseconds)] devset en->$TGT inference done (log: $LOG)"

  SCORE_LOG="${LOG_DIR}/score_chunk1100_borderp1_en${TGT}.log"
  echo "[$(date -Iseconds)] devset en->$TGT scoring"
  CUDA_VISIBLE_DEVICES=0 "$EVAL_PYTHON_BIN" evaluate_cascade_outputs.py \
    --output-dir "$OUT" \
    >"$SCORE_LOG" 2>&1
  echo "[$(date -Iseconds)] devset en->$TGT scoring done (log: $SCORE_LOG)"
done
