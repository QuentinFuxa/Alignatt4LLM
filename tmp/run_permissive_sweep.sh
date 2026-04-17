#!/usr/bin/env bash
# Permissive AlignAtt sweep at chunk_ms=800, τ=0.
# Usage: bash tmp/run_permissive_sweep.sh <direction>
set -eu
cd /home/cascade_simultaneous

direction="${1:-ende}"
case "$direction" in
    ende) target=de; inputs="dev-set/audio/ccpXHNfaoy.wav dev-set/audio/OiqEWDVtWk.wav" ;;
    enit) target=it; inputs="dev-set/audio/ccpXHNfaoy.wav dev-set/audio/OiqEWDVtWk.wav" ;;
    enzh) target=zh; inputs="dev-set/audio/ccpXHNfaoy.wav dev-set/audio/OiqEWDVtWk.wav" ;;
    *) echo "unknown direction: $direction" >&2; exit 1 ;;
esac

# Configs: label  partial_max  partial_followup  rewind  border  stall
cat > /tmp/perm_configs.tsv <<EOF
permA	48	32	16	2	0.6
permB	64	48	24	4	0.5
permC	96	64	32	8	0.4
EOF

while IFS=$'\t' read -r label pmax pfol rew brd stl; do
    out="outputs/phase_b_${direction}_chunk800_${label}"
    mkdir -p "$out"
    echo ">>> [$(date -u +%H:%M:%S)] ${direction} ${label}: chunk=800 partial=${pmax}/${pfol} rewind=${rew} border=${brd} stall=${stl} -> ${out}"
    VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
        .venv-inference/bin/python run_simulstream_batch.py \
        --alignment-backend-name qwen_forced \
        --mt-backend-name gemma_vllm_alignatt \
        --inputs $inputs \
        --source en --target "$target" \
        --output-dir "$out" \
        --min-start-seconds 2.0 \
        --max-history-utterances 1 \
        --partial-max-new-tokens "$pmax" \
        --partial-followup-max-new-tokens "$pfol" \
        --translation-scheduler-stall-seconds "$stl" \
        --translation-alignatt-argmax-mass-threshold 0.0 \
        --translation-alignatt-rewind-threshold "$rew" \
        --translation-alignatt-border-margin "$brd" \
        --translation-alignatt-min-source-mass 0.0 \
        --chunk-ms 800 &> "$out/run.log"
    echo ">>> [$(date -u +%H:%M:%S)] ${label} inference done, evaluating..."
    .venv-evaluation/bin/python evaluate_cascade_outputs.py \
        --output-dir "$out" &>> "$out/run.log"
    echo ">>> [$(date -u +%H:%M:%S)] ${label} eval done: $out"
    grep -E "BLEU|chrF|LongYAAL|COMET|Total Instances" "$out/evaluation.report.txt" || true
done < /tmp/perm_configs.tsv

echo ">>> [$(date -u +%H:%M:%S)] permissive sweep complete: $direction"
