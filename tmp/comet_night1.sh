#!/bin/bash
# Add COMET (XCOMET-XL) scores to the canonical overnight outputs.
# Each invocation loads XCOMET-XL (~30 s) and scores quickly thereafter.
set -eu
cd /home/cascade_simultaneous
VENV=.venv-evaluation/bin/python

SOURCE=test-set/ref/en.txt

for dir in outputs/reanchor_chunk700 \
           outputs/night1_ende_stable_k3_chunk450 \
           outputs/night1_ende_stable_k4_chunk450 \
           outputs/night1_step6_ms10_punct \
           outputs/night1_step6_ms20_punct \
           outputs/night1_step6_ms00_freeze; do
  echo "=== $dir ==="
  $VENV evaluate_cascade_outputs.py --output-dir "$dir" \
    --source-reference "$SOURCE" 2>&1 | tail -5
  echo
done

echo "=== outputs/night1_enit_punct_chunk450 ==="
$VENV evaluate_cascade_outputs.py --output-dir outputs/night1_enit_punct_chunk450 \
  --target-lang-code it --target-reference test-set/ref/it.txt \
  --source-reference "$SOURCE" 2>&1 | tail -5
echo

echo "=== outputs/night1_enzh_punct_chunk450 ==="
$VENV evaluate_cascade_outputs.py --output-dir outputs/night1_enzh_punct_chunk450 \
  --target-lang-code zh --target-reference test-set/ref/zh.txt \
  --source-reference "$SOURCE" 2>&1 | tail -5
echo

echo
echo "=== SUMMARY ==="
for dir in outputs/reanchor_chunk450 outputs/reanchor_chunk700 \
           outputs/night1_ende_stable_k3_chunk450 \
           outputs/night1_ende_stable_k4_chunk450 \
           outputs/night1_enit_punct_chunk450 \
           outputs/night1_enzh_punct_chunk450 \
           outputs/night1_step6_ms10_punct \
           outputs/night1_step6_ms20_punct \
           outputs/night1_step6_ms00_freeze; do
  name=$(basename "$dir")
  scores=$(grep -E "BLEU|chrF|LongYAAL \(C[AU]\)|COMET" "$dir/evaluation.report.txt" 2>/dev/null | tr '\n' ' ')
  echo "$name  $scores"
done
