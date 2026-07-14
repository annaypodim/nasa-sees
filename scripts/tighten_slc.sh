#!/usr/bin/env bash
# SLC tightening: does adding long-context lags (lag48 + weekly lag168) close the
# thin SLC margin on long gaps, where carry-forward persistence goes stale?
# 5-fold CV, SLC, both gaps, WITH vs WITHOUT --long-lags. Baseline (no long-lags)
# numbers already logged in faithful_logs/kfold_confirm.log; we re-print for A/B.
set -uo pipefail
cd "$(dirname "$0")/.."
K=5
for G in 24 48; do
  for MODE in base longlags; do
    EXTRA=""; [ "$MODE" = longlags ] && EXTRA="--long-lags"
    echo "########## slc  gap ${G}h  (${K}-fold, ${MODE}) ##########"
    PYTHONPATH=. .venv/bin/python scripts/eval_temporal.py --city slc --wind hrrr \
      --temporal --rk-spatial $EXTRA --gap-len "$G" --kfold "$K" 2>&1 \
      | grep -iE "OURS \(|ST-kriging baseline|vs ST-kriging|prior blend"
  done
done
echo TIGHTEN_SLC_DONE
