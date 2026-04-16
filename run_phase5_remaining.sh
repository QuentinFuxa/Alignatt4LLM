#!/usr/bin/env bash
# Phase 5: Run remaining full-set batch evaluations for delivery slots.
# Prerequisites: en->de <2s batch already completed separately.
# This script runs the remaining 5 slots sequentially, keeping models hot
# across same-language runs.

set -euo pipefail

PYTHON=".venv-inference/bin/python"
EVAL_PYTHON=".venv-evaluation/bin/python"
COMMON_ARGS="--wav-dir test-set/audio/ --min-start-seconds 2.0 --max-history-utterances 1 --partial-max-new-tokens 16 --partial-followup-max-new-tokens 8"

run_slot() {
    local name="$1" chunk_ms="$2" target="$3"
    local outdir="outputs/phase5_fullset_${name}"

    echo "=========================================="
    echo "Running: $name (chunk=${chunk_ms}ms, target=${target})"
    echo "=========================================="

    $PYTHON run_simulstream_batch.py \
        $COMMON_ARGS \
        --output-dir "$outdir" \
        --chunk-ms "$chunk_ms" \
        --source en \
        --target "$target"

    echo "Evaluating: $name"
    $EVAL_PYTHON evaluate_cascade_outputs.py \
        --output-dir "$outdir" \
        --skip-comet

    echo "Results for $name:"
    grep -A 15 "^Scores" "$outdir/evaluation.report.txt" || true
    echo ""
}

# <2s family (chunk_ms=450)
run_slot "enit_2s" 450 "it"
run_slot "enzh_2s" 450 "zh"

# <4s family (chunk_ms=800)
run_slot "ende_4s" 800 "de"
run_slot "enit_4s" 800 "it"
run_slot "enzh_4s" 800 "zh"

echo "=========================================="
echo "Phase 5 complete. All 5 remaining slots done."
echo "=========================================="
