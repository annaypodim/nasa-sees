#!/usr/bin/env bash
# DIFFERENT question from the regularization sweep: is the pure faithful model's failure
# to beat IDW on Fresno an UNDER-CONVERGENCE / early-stop problem? The 4000-step +
# patience-10 recipe halts high-variance seeds early. Test longer training and a faster
# LR on fresno_med_max (30 nodes / 20 train, IDW 3.59). Pure faithful, 8 seeds each.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "run_graphy_faithful.sh" >/dev/null 2>&1; do sleep 20; done
echo "[sweep] train-length sweep on fresno_med_max $(date +%H:%M:%S)"

export CITY=fresno_med_max EPA=1 WIND=hrrr SEEDS="0 1 2 3 4 5 6 7"
run () { echo "[sweep] === $1 ==="; env $2 ./run_graphy_faithful.sh small "$3" 8; }

# 1) LONG training + generous patience: does it just need more steps to converge?
run "long 15k steps, patience 30, val-every 500"  "PATIENCE=30 VALEVERY=500"  15000
# 2) FASTER lr, moderate length: does it need faster convergence within budget?
run "lr 5e-4, 8k steps, patience 20"              "LR=5e-4 PATIENCE=20 VALEVERY=400"  8000

echo "[sweep] DONE $(date +%H:%M:%S)"
