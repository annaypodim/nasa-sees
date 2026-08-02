#!/usr/bin/env bash
# Phase-2: per-node wind-SPEED feature (--wind-feature) on all 3 cities, gap 24/48,
# paired in-process against the wind=hrrr base. Bounded parallelism.
set -u
ROOT="/Users/annaypodimatopoulou/Code/side_quests/nasa-sees/.claude/worktrees/agent-ac26f922cb5129545"
PY="/Users/annaypodimatopoulou/Code/side_quests/nasa-sees/.venv/bin/python"
LOG="$ROOT/experiments/logs/temporal_covariates"
cd "$ROOT"
export PYTHONPATH=.
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

JOBS=(
  "pgh_windfeat_g24|pittsburgh|hrrr|24|base,windfeat"
  "pgh_windfeat_g48|pittsburgh|hrrr|48|base,windfeat"
  "fresno_windfeat_g24|fresno|hrrr|24|base,windfeat"
  "fresno_windfeat_g48|fresno|hrrr|48|base,windfeat"
  "slc_windfeat_g24|slc|hrrr|24|base,windfeat"
  "slc_windfeat_g48|slc|hrrr|48|base,windfeat"
)

run_one() {
  local spec="$1"
  IFS='|' read -r tag city wind gap variants <<< "$spec"
  "$PY" scripts/run_temporal_covariates.py --city "$city" --wind "$wind" \
      --gap-len "$gap" --kfold 5 --epochs 120 --variants "$variants" \
      > "$LOG/$tag.log" 2>&1
  echo "FINISHED $tag (exit $?)"
}
export -f run_one
export PY LOG ROOT

printf '%s\n' "${JOBS[@]}" | xargs -P 3 -I {} bash -c 'run_one "$@"' _ {}
echo "ALL_DONE_P2"
