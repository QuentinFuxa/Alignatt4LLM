#!/usr/bin/env bash
# Orchestrate the remaining additive-chunk pipeline:
#   1. chunk_ms=1900 dev-set inference (all 3 directions)
#   2. score every additive dev-set output (chunk850 + chunk1900)
#   3. re-materialize submission/ with additive bundles appended
#
# chunk_ms=850 inference is expected to have already finished before this
# script starts; it re-runs idempotently for anything that is missing.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv-inference/bin/python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-.venv-evaluation/bin/python}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

needs_infer() {
  local chunk="$1"
  for TGT in de it zh; do
    if [[ ! -f "outputs/iwslt26_devset_chunk${chunk}_borderp1_en${TGT}/hypothesis.jsonl" ]]; then
      return 0
    fi
  done
  return 1
}

CHUNKS="${CHUNKS:-850 1500 1900}"
for CHUNK in $CHUNKS; do
  if needs_infer "$CHUNK"; then
    LOG="${LOG_DIR}/additive_chunk${CHUNK}.log"
    echo "[$(date -Iseconds)] starting chunk_ms=${CHUNK} dev-set sweep -> ${LOG}"
    "${PYTHON_BIN}" scripts/run_additive_chunk_sweep.py \
      --chunk-ms "${CHUNK}" \
      --output-tag "chunk${CHUNK}_borderp1" \
      >"${LOG}" 2>&1
    echo "[$(date -Iseconds)] chunk_ms=${CHUNK} inference done"
  else
    echo "[$(date -Iseconds)] skip chunk_ms=${CHUNK} (all dev-set hypotheses already present)"
  fi
done

echo "[$(date -Iseconds)] scoring additive dev-set outputs"
bash scripts/score_additive_chunks.sh

echo "[$(date -Iseconds)] syncing submission bundle"
"${PYTHON_BIN}" submission/sync_artifacts.py

echo "[$(date -Iseconds)] additive pipeline done"
