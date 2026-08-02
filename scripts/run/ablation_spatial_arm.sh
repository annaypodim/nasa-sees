#!/usr/bin/env bash
# Ablation arm A: memoryless SPATIAL RK-elev kriging (no temporal memory, no GNN
# benefit from history). Same 5-fold partition + cities/gaps as confirm_kfold.sh.
# The "ST-kriging baseline" line here == the spatial-only prior (no persistence),
# because without --temporal the prior() returns spatial_rk alone.
set -uo pipefail
cd "$(dirname "$0")/../.."
K=5
for CITY in fresno slc pittsburgh; do
  for G in 24 48; do
    echo "########## $CITY  gap ${G}h  (${K}-fold, SPATIAL-ONLY arm) ##########"
    PYTHONPATH=. .venv/bin/python scripts/eval_temporal.py --city "$CITY" --wind hrrr \
      --rk-spatial --gap-len "$G" --kfold "$K" 2>&1 \
      | grep -iE "ST-kriging baseline|Persistence floor"
  done
done
echo ABLATION_SPATIAL_DONE
