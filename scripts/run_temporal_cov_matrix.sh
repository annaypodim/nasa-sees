#!/usr/bin/env bash
# Phase-1 covariate matrix on the temporal gap-fill task. Bounded parallelism.
set -u
ROOT="/Users/annaypodimatopoulou/Code/side_quests/nasa-sees/.claude/worktrees/agent-ac26f922cb5129545"
PY="/Users/annaypodimatopoulou/Code/side_quests/nasa-sees/.venv/bin/python"
LOG="$ROOT/experiments/logs/temporal_covariates"
cd "$ROOT"
export PYTHONPATH=.
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

# tag|city|wind|gap|variants
JOBS=(
  "pgh_hrrr_g24|pittsburgh|hrrr|24|base,temp,aod"
  "pgh_hrrr_g48|pittsburgh|hrrr|48|base,temp,aod"
  "pgh_zero_g24|pittsburgh|zero|24|base"
  "pgh_zero_g48|pittsburgh|zero|48|base"
  "fresno_hrrr_g24|fresno|hrrr|24|base"
  "fresno_hrrr_g48|fresno|hrrr|48|base"
  "fresno_zero_g24|fresno|zero|24|base"
  "fresno_zero_g48|fresno|zero|48|base"
  "slc_hrrr_g24|slc|hrrr|24|base"
  "slc_hrrr_g48|slc|hrrr|48|base"
  "slc_zero_g24|slc|zero|24|base"
  "slc_zero_g48|slc|zero|48|base"
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
echo "ALL_DONE"
