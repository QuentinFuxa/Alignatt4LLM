#!/usr/bin/env bash
# Phase B τ sweep runner. Usage: bash tmp/run_phase_b_sweep.sh <direction> <tau1> [<tau2> ...]
# e.g. bash tmp/run_phase_b_sweep.sh ende 0.05 0.10 0.15
# Runs batch inference + evaluation for each τ, sequentially.
set -eu
cd /home/cascade_simultaneous

direction="$1"; shift
case "$direction" in
    ende) target=de; inputs="dev-set/audio/ccpXHNfaoy.wav dev-set/audio/OiqEWDVtWk.wav" ;;
    enit) target=it; inputs="dev-set/audio/ccpXHNfaoy.wav dev-set/audio/OiqEWDVtWk.wav" ;;
    enzh) target=zh; inputs="dev-set/audio/ccpXHNfaoy.wav dev-set/audio/OiqEWDVtWk.wav" ;;
    *) echo "unknown direction: $direction" >&2; exit 1 ;;
esac

for tau in "$@"; do
    tag=$(python3 -c "print(f'{int(round(float(\"$tau\")*100)):03d}')")
    out="outputs/phase_b_${direction}_tau${tag}"
    mkdir -p "$out"
    echo ">>> [$(date -u +%H:%M:%S)] direction=${direction} tau=${tau} -> ${out}"
    VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
        .venv-inference/bin/python run_simulstream_batch.py \
        --alignment-backend-name qwen_forced \
        --mt-backend-name gemma_vllm_alignatt \
        --inputs $inputs \
        --source en --target "$target" \
        --output-dir "$out" \
        --min-start-seconds 2.0 \
        --max-history-utterances 1 \
        --partial-max-new-tokens 24 \
        --partial-followup-max-new-tokens 16 \
        --translation-scheduler-stall-seconds 0.8 \
        --translation-alignatt-argmax-mass-threshold "$tau" \
        --translation-alignatt-rewind-threshold 8 \
        --translation-alignatt-border-margin 0 \
        --translation-alignatt-min-source-mass 0.0 \
        --chunk-ms 450 &> "$out/run.log"
    echo ">>> [$(date -u +%H:%M:%S)] inference done, evaluating..."
    .venv-evaluation/bin/python evaluate_cascade_outputs.py \
        --output-dir "$out" &>> "$out/run.log"
    echo ">>> [$(date -u +%H:%M:%S)] eval done: $out"
    grep -E "BLEU|chrF|LongYAAL|COMET|Total Instances" "$out/evaluation.report.txt" || true
done
echo ">>> [$(date -u +%H:%M:%S)] sweep complete: $direction"
