#!/usr/bin/env bash
# Base-model convergence sweep on the MAXIMAL-density Fresno net (fresno_med_max:
# 30 nodes / 20 train sensors, IDW 3.59 -- the closest analog to GraPhy's own setup
# of 28 sensors / IDW 3.07). Goal: get the PURE faithful GraPhy (no gate, no hybrid)
# to BEAT this net's IDW, reproducing GraPhy's headline (model beats its own IDW ~22%).
# Each config = 8 seeds via run_graphy_faithful.sh (8-way), sequential.
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "run_graphy_faithful.sh" >/dev/null 2>&1; do sleep 20; done
echo "[sweep] cores free -> base-convergence sweep on fresno_med_max $(date +%H:%M:%S)"

export CITY=fresno_med_max EPA=1 WIND=hrrr SEEDS="0 1 2 3 4 5 6 7"
run () { echo "[sweep] === $1 ==="; env $2 ./run_graphy_faithful.sh small 4000 8; }

run "small baseline (128/3)"        ""
run "small + wd1e-3"                "WD=1e-3"
run "small + wd1e-3 + dropout0.1"   "WD=1e-3 DROPOUT=0.1"
run "tiny 64/2 + wd1e-3"            "WD=1e-3 HID=64 LAY=2"

echo "[sweep] DONE $(date +%H:%M:%S)"
