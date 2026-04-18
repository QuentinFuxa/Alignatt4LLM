#!/usr/bin/env bash
# Score every additive calibration dev-set run that has already produced
# hypothesis.jsonl + manifest.json but does not yet have evaluation.json.
# Evaluation runs sequentially on a single GPU from `.venv-evaluation` and
# is safe to run concurrently with inference on .venv-inference because
# COMET fits comfortably within the remaining GPU budget after the cascade
# releases its models.
set -euo pipefail

cd "$(dirname "$0")/.."

EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-.venv-evaluation/bin/python}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

CHUNKS="${CHUNKS:-850 1500 1900}"
for CHUNK in $CHUNKS; do
  for TGT in de it zh; do
    OUT="outputs/iwslt26_devset_chunk${CHUNK}_borderp1_en${TGT}"
    LOG="${LOG_DIR}/score_additive_chunk${CHUNK}_en${TGT}.log"

    if [[ ! -f "${OUT}/hypothesis.jsonl" ]]; then
      echo "[$(date -Iseconds)] skip ${OUT} (hypothesis.jsonl missing)"
      continue
    fi
    if [[ -f "${OUT}/evaluation.json" ]]; then
      echo "[$(date -Iseconds)] skip ${OUT} (already scored)"
      continue
    fi
    echo "[$(date -Iseconds)] scoring chunk_ms=${CHUNK} en->${TGT} from ${OUT}"
    CUDA_VISIBLE_DEVICES=0 "${EVAL_PYTHON_BIN}" evaluate_cascade_outputs.py \
      --output-dir "${OUT}" \
      >"${LOG}" 2>&1
    echo "[$(date -Iseconds)] chunk_ms=${CHUNK} en->${TGT} done (log: ${LOG})"
  done
done
