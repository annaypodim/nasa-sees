#!/usr/bin/env bash
# 8-seed confirmation of the combined win (RK-elev prior + temporal memory + GNN)
# vs the ST-kriging baseline, on TWO cities (SLC terrain + Pittsburgh flatter/denser).
set -uo pipefail
cd "$(dirname "$0")/../.."
for CITY in slc pittsburgh; do
  for G in 24 48; do
    echo "########## $CITY  gap ${G}h  (8 seeds) ##########"
    PYTHONPATH=. .venv/bin/python scripts/eval_temporal.py --city "$CITY" --wind hrrr \
      --temporal --rk-spatial --gap-len "$G" --n-gaps 6 --seeds 0,1,2,3,4,5,6,7 2>&1 \
      | grep -iE "OURS \(|ST-kriging|Persistence floor|vs ST-kriging|prior blend"
  done
done
echo ALL_CONFIRM_DONE
