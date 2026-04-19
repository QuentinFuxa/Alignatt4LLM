#!/usr/bin/env bash
# Launch en→it and en→zh τ sweeps back-to-back after en→de is done.
set -eu
cd /home/cascade_simultaneous
bash tmp/run_phase_b_sweep.sh enit 0.00 0.05 0.10 0.15 &>> tmp/phase_b_enit_sweep.log
bash tmp/run_phase_b_sweep.sh enzh 0.00 0.05 0.10 0.15 &>> tmp/phase_b_enzh_sweep.log
echo ">>> [$(date -u +%H:%M:%S)] all sweeps complete"
