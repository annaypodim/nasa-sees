#!/usr/bin/env bash
# SLC+WIND ablation, ROUND 2 (convergence fix). Round-1 pure runs hit the 15k step
# CAP, not a val plateau (best@9.5k-13k, high seed variance, R2<0 on one seed), so the
# pure/terrain comparison was under-converged noise. Fix: give the PURE configs a 40k
# cap + patience 20 (10k no-improve) so each seed early-stops at a real plateau. The
# HYBRID configs are IDW-anchored and already stable at 15k, so keep them short to save
# night-hours. Same recipe otherwise: lr 5e-4, val-every 500, 4 seeds, 4-way.
#   B - A = pure-model terrain novelty (converged).  D - C = hybrid terrain novelty.
set -uo pipefail
cd "$(dirname "$0")/../.."
while pgrep -f "run_graphy_faithful.sh" >/dev/null 2>&1; do sleep 20; done
echo "[slc2] START $(date +%H:%M:%S)"

export CITY=slc WIND=hrrr LR=5e-4 PATIENCE=20 VALEVERY=500 SEEDS="0 1 2 3"
run () { echo "[slc2] === $1 ($(date +%H:%M:%S)) ==="; env $2 scripts/run/run_graphy_faithful.sh small "$3" 4; }

# pure ablation at 40k (needs convergence), then hybrid ablation at 15k (stable)
run "A pure  | terrain OFF (control)"  ""                              40000
run "B pure  | terrain ON  (novelty)"  "TERRAIN=1"                     40000
run "C hybrid| terrain OFF (control)"  "IDWPRIOR=1"                    15000
run "D hybrid| terrain ON  (novelty)"  "IDWPRIOR=1 IDWELEV=1 TERRAIN=1" 15000

echo "[slc2] DONE $(date +%H:%M:%S)"
