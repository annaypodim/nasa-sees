#!/usr/bin/env bash
# Fix the model on the DENSE net (OURS 3.823 LOSES to IDW 3.391). Diagnose WHY:
#   is the KRIGING prior (r5000) over-smoothing on dense (worse than the 1/d IDW the
#   baseline uses), or is the CORRECTION adding noise? Isolate prior vs correction.
# fresno_dense, 27 usable, --wind zero, despike/huber base, 8 seeds.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT="${OUT:-experiments/logs/dense_eval}"
mkdir -p "$OUT"

COMMON=(--city fresno_dense --wind zero --epochs 120 --seeds 0,1,2,3,4,5,6,7
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --no-convection --despike --loss huber --idw-feature)

run () {  # run <tag> <extra...>
  local tag="$1"; shift
  echo "launching $tag -> $OUT/model_$tag.log"
  "$PY" scripts/eval_inductive.py "${COMMON[@]}" "$@" > "$OUT/model_$tag.log" 2>&1 &
}

# PRIOR test: 1/d IDW prior (matches the strong baseline) vs kriging prior, same corr.
run idwprior   --idw-prior                     --corr-reg 0.6           # 1/d prior + corr
run krigprior  --kriging-prior --kriging-range 5000 --corr-reg 0.6      # control = 3.823
# CORRECTION test: does suppressing the correction help (correction = noise)?
run idw_cr09   --idw-prior                     --corr-reg 0.9           # tiny correction
run idw_cr03   --idw-prior                     --corr-reg 0.3           # bigger correction
# CAPACITY: more layers/hidden now that there's more data (27 nodes, denser signal).
run idw_cap    --idw-prior --layers 6 --hidden 32 --corr-reg 0.6

wait
echo "=== DENSE MODEL DIAGNOSTIC DONE (IDW baseline = 3.391, kriging control = 3.823) ==="
for t in idwprior krigprior idw_cr09 idw_cr03 idw_cap; do
  printf '%-10s ' "$t"
  grep -E "OURS \(GraPhyNet\)|IDW baseline" "$OUT/model_$t.log" | tr '\n' ' '; echo
done
