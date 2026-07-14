#!/usr/bin/env bash
# GRIN learned-baseline on OUR data + OUR 5-fold masks, directly comparable to the
# OURS / ST-kriging numbers from confirm_kfold.sh. Ordered fresno -> slc ->
# pittsburgh so the headline city lands first (safe to Ctrl-C after fresno).
# ETA @80 epochs CPU: fresno ~3.3h, +slc ~6.5h, +pittsburgh ~13h.
# Override cities:  CITIES="fresno" bash scripts/run/confirm_grin.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
EPOCHS="${EPOCHS:-80}"
K="${K:-5}"
CITIES="${CITIES:-fresno slc pittsburgh}"
for CITY in $CITIES; do
  for G in 24 48; do
    echo "########## GRIN  $CITY  gap ${G}h  (${K}-fold, ${EPOCHS}ep) ##########"
    PYTHONPATH=. .venv/bin/python scripts/eval_grin.py --city "$CITY" --wind hrrr \
      --gap-len "$G" --kfold "$K" --epochs "$EPOCHS" 2>&1 \
      | grep -iE "GRIN MAE|fold [0-9]"
  done
done
echo ALL_GRIN_DONE
