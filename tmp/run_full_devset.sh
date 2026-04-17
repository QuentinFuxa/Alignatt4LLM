#!/usr/bin/env bash
# Full dev-set run + eval. Usage: bash tmp/run_full_devset.sh <direction> <tau> <output-tag>
# e.g. bash tmp/run_full_devset.sh ende 0.05 best_candidate_v1
set -eu
cd /home/cascade_simultaneous

direction="$1"
tau="$2"
tag="$3"
case "$direction" in
    ende) target=de; ref=dev-set/ref/de.txt ;;
    enit) target=it; ref=dev-set/ref/it.txt ;;
    enzh) target=zh; ref=dev-set/ref/zh.txt ;;
    *) echo "unknown direction: $direction" >&2; exit 1 ;;
esac

out="outputs/phase_b_fulldev_${direction}_${tag}"
mkdir -p "$out"
echo ">>> [$(date -u +%H:%M:%S)] FULL dev-set ${direction} tau=${tau} -> ${out}"

devset_inputs=$(ls dev-set/audio/*.wav | grep -v short60s)
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
    .venv-inference/bin/python run_simulstream_batch.py \
    --alignment-backend-name qwen_forced \
    --mt-backend-name gemma_vllm_alignatt \
    --inputs $devset_inputs \
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

.venv-evaluation/bin/python evaluate_cascade_outputs.py --output-dir "$out" &>> "$out/run.log"
echo ">>> [$(date -u +%H:%M:%S)] FULL dev-set eval done: $out"
grep -E "BLEU|chrF|LongYAAL|COMET|Empty Predictions|Total Instances" "$out/evaluation.report.txt" || true
