#!/usr/bin/env bash
# OVERNIGHT SLC+WIND ablation. SLC now has real HRRR wind (scripts/fetch_wind_hrrr.py
# --city slc), so the convection module + our directional drainage gate finally have
# signal. 2x2, all WIND ON + long/high-LR to actually converge (pure model needs it):
#     terrain OFF (control)  vs  terrain ON (our novelty)
#     pure faithful          vs  hybrid (idw-prior)
# Delta terrain-on - terrain-off = OUR NOVELTY. pure-vs-hybrid answers "does learned
# physics beat interpolation-anchoring when wind advection is present."
# Converged recipe: lr 5e-4 (converges ~5x faster than 1e-4), 15k steps, patience 20.
# 4 seeds each, 4-way (efficient; 8-way oversubscribes 8 cores -> thrash). SLC = raw
# PurpleAir (no EPA), so no --epa-correct.
set -uo pipefail
cd "$(dirname "$0")/../.."
while pgrep -f "run_graphy_faithful.sh" >/dev/null 2>&1; do sleep 20; done
echo "[slc-sweep] START $(date +%H:%M:%S)"

export CITY=slc WIND=hrrr LR=5e-4 PATIENCE=20 VALEVERY=500 SEEDS="0 1 2 3"
run () { echo "[slc-sweep] === $1 ($(date +%H:%M:%S)) ==="; env $2 scripts/run/run_graphy_faithful.sh small 15000 4; }

# pure ablation first (the core novelty pair), then hybrid ablation
run "A pure  | terrain OFF (control)"  ""
run "B pure  | terrain ON  (novelty)"  "TERRAIN=1"
run "C hybrid| terrain OFF (control)"  "IDWPRIOR=1"
run "D hybrid| terrain ON  (novelty)"  "IDWPRIOR=1 IDWELEV=1 TERRAIN=1"

echo "[slc-sweep] DONE $(date +%H:%M:%S)"
