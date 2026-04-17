#!/bin/bash
# Evaluate all 4 night-1 experiments. Uses --skip-comet for speed;
# COMET can be re-added with a second pass if needed.
set -eu
cd /home/cascade_simultaneous
VENV=.venv-evaluation/bin/python

for dir in outputs/night1_ende_stable_k3_chunk450 \
           outputs/night1_ende_stable_k4_chunk450; do
  echo "=== $dir ==="
  $VENV evaluate_cascade_outputs.py --output-dir "$dir" --skip-comet 2>&1 | tail -3
done

# en→it
echo "=== outputs/night1_enit_punct_chunk450 ==="
$VENV evaluate_cascade_outputs.py --output-dir outputs/night1_enit_punct_chunk450 \
  --target-lang-code it --target-reference test-set/ref/it.txt --skip-comet 2>&1 | tail -3

# en→zh
echo "=== outputs/night1_enzh_punct_chunk450 ==="
$VENV evaluate_cascade_outputs.py --output-dir outputs/night1_enzh_punct_chunk450 \
  --target-lang-code zh --target-reference test-set/ref/zh.txt --skip-comet 2>&1 | tail -3

echo ""
echo "=== SUMMARY ==="
for dir in outputs/night1_*; do
  name=$(basename "$dir")
  scores=$(grep -E "BLEU|chrF|LongYAAL \(C[AU]\)" "$dir/evaluation.report.txt" 2>/dev/null | tr '\n' ' ')
  echo "$name  $scores"
done
