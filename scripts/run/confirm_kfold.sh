#!/usr/bin/env bash
# Proper K-fold CV confirmation of the combined win (RK-elev prior + temporal
# memory + GNN) vs the ST-kriging baseline. Every fixed-length gap window is held
# out exactly once (no random test overlap). Fresno = GraPhy's home turf: the
# temporal-gap axis is precisely what memoryless GraPhy leaves open.
set -uo pipefail
cd "$(dirname "$0")/../.."
K=5
for CITY in fresno slc pittsburgh; do
  for G in 24 48; do
    echo "########## $CITY  gap ${G}h  (${K}-fold CV) ##########"
    PYTHONPATH=. .venv/bin/python scripts/eval_temporal.py --city "$CITY" --wind hrrr \
      --temporal --rk-spatial --gap-len "$G" --kfold "$K" 2>&1 \
      | grep -iE "OURS \(|ST-kriging|Persistence floor|vs ST-kriging|prior blend|fold [0-9]"
  done
done
echo ALL_KFOLD_DONE
