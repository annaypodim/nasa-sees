#!/usr/bin/env bash
# DENSITY CROSSOVER SWEEP: trace MAE(ours) and MAE(IDW) vs network density (# sensors)
# to find where the learned correction flips from >= IDW (dense) to < IDW (sparse).
# For each density N and each random subset (subsample-seed), run the inductive eval
# with 8 split-seeds; append a CSV row (config,N,subsample_seed,ours_mae,ours_std,
# idw_mae,idw_std). Averaging over subsample-seeds removes "unlucky subset" variance.
#
# Two OURS configs so we can see which (if any) beats the fixed IDW baseline when sparse:
#   krig  = kriging-prior (BLUP, range 5000) + corr-reg 0.6  -- worklog's SPARSE-best
#   idw   = 1/d IDW-prior + idw-feature + corr-reg 0.3        -- loosened residual on IDW
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT="${OUT:-experiments/logs/density_sweep}"
mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "config,N,subsample_seed,ours_mae,ours_std,idw_mae,idw_std" > "$CSV"

CITY="${CITY:-fresno_dense_abc}"
DENSITIES="${DENSITIES:-6 8 10 12 14 16 18 20 22}"
SUBSETS="${SUBSETS:-0 1 2}"
SEEDS="0,1,2,3,4,5,6,7"

COMMON=(--city "$CITY" --wind zero --epochs 120 --seeds "$SEEDS"
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --no-convection --loss huber --despike --spatial-qa)

cfg_flags () {  # bash 3.2 (macOS) has no associative arrays -> plain case
  case "$1" in
    krig) echo "--kriging-prior --kriging-range 5000 --corr-reg 0.6" ;;
    idw)  echo "--idw-prior --idw-feature --corr-reg 0.3" ;;
  esac
}

# NOTE: the summary line has MULTIPLE spaces before MAE=; match loosely. (Source of truth
# for aggregation is scripts/extract_density_results.py, which re-parses all logs robustly.)
parse () { local ln; ln=$(grep -F "$1" "$2" | grep -m1 "MAE="); \
  local m; m=$(printf '%s' "$ln" | grep -oE "MAE=[0-9.]+" | head -1 | cut -d= -f2); \
  printf '%s %s' "${m:-NA}" "NA"; }

for cfg in krig idw; do
  read -r -a EXTRA <<< "$(cfg_flags "$cfg")"
  for N in $DENSITIES; do
    for ss in $SUBSETS; do
      log="$OUT/${cfg}_N${N}_ss${ss}.log"
      "$PY" scripts/eval_inductive.py "${COMMON[@]}" --max-nodes "$N" --subsample-seed "$ss" \
        "${EXTRA[@]}" > "$log" 2>&1
      om=$(parse "OURS \(GraPhyNet\)" "$log"); im=$(parse "IDW baseline" "$log")
      echo "${cfg},${N},${ss},${om% *},${om#* },${im% *},${im#* }" >> "$CSV"
      echo "[done] cfg=$cfg N=$N ss=$ss  OURS=$om  IDW=$im"
    done
  done
done
echo "=== SWEEP COMPLETE -> $CSV ==="
column -t -s, "$CSV"
