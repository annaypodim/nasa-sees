#!/usr/bin/env bash
# Tier-1 + data-cleaning ablation sweep on density-matched Fresno (inductive, 8 seeds).
# Compares the current best config (A) against data QA (despike) and the three
# Tier-1 model levers: IDW-geometry features, kriging prior, L1+recalibration.
# Runs all 6 configs in parallel; logs to $OUT. Re-run any single line standalone.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
PY=.venv/bin/python
OUT="${OUT:-experiments/logs/tier1_sweep}"
mkdir -p "$OUT"

# shared config = the validated strong base (see WORKLOG "CURRENT BEST"):
# sqrt/huber-space, clip1, 4x16, drop convection, corr-reg 0.6, EPA-corrected data.
COMMON=(--city fresno --wind zero --epochs 120 --seeds 0,1,2,3,4,5,6,7
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --no-convection --corr-reg 0.6)

run () {  # run <tag> <extra flags...>
  local tag="$1"; shift
  echo "launching $tag -> $OUT/run_$tag.log"
  "$PY" scripts/eval_inductive.py "${COMMON[@]}" "$@" > "$OUT/run_$tag.log" 2>&1 &
}

run A --loss huber --idw-prior --idw-feature                                           # control (~4.30, no despike)
run B --despike --loss huber --idw-prior --idw-feature                                 # + data QA (despike)
run C --despike --loss huber --idw-prior --idw-feature --idw-geom-features             # + Tier1 #1 geom feats
run D --despike --loss huber --kriging-prior --idw-feature                             # + Tier1 #2 kriging prior
run E --despike --loss l1     --idw-prior --idw-feature --recal                        # + Tier1 #3 L1 + recal
run F --despike --loss l1     --kriging-prior --idw-feature --idw-geom-features --recal # + all together

wait
echo "=== SWEEP DONE — MAE summary (OURS vs IDW) ==="
for t in A B C D E F; do
  printf '%s  ' "$t"
  grep -E "OURS \(GraPhyNet\)|IDW baseline" "$OUT/run_$t.log" | tr '\n' ' '; echo
done
