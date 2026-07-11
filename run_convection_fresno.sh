#!/usr/bin/env bash
# Make CONVECTION contribute. On data/fresno (18 usable) we have REAL HRRR wind.
# Convection lives inside the corr-reg-suppressed residual + competes via softmax with
# local/diffusion -> it never spoke. It encodes DIRECTIONAL wind transport that the
# isotropic IDW/kriging prior CANNOT represent, so it's the one residual part that could
# add orthogonal signal. Test: does convection help, and does loosening the leash let it?
# Prior fixed to 1/d IDW to isolate convection. sparse fresno, --wind hrrr, 8 seeds.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT="${OUT:-convection_logs}"
mkdir -p "$OUT"

COMMON=(--city fresno --wind hrrr --epochs 120 --seeds 0,1,2,3,4,5,6,7
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --despike --loss huber --idw-prior --idw-feature)

run () {  # run <tag> <extra...>
  local tag="$1"; shift
  echo "launching $tag -> $OUT/conv_$tag.log"
  "$PY" scripts/eval_inductive.py "${COMMON[@]}" "$@" > "$OUT/conv_$tag.log" 2>&1 &
}

run off_06 --no-convection --corr-reg 0.6      # control: no convection
run on_06                  --corr-reg 0.6      # convection on, tight leash
run on_03                  --corr-reg 0.3      # convection on, looser leash
run on_01                  --corr-reg 0.1      # convection on, loose leash
run off_03 --no-convection --corr-reg 0.3      # isolate: looser reg WITHOUT convection

wait
echo "=== CONVECTION SWEEP DONE (data/fresno, HRRR wind, 8 seeds) ==="
for t in off_06 on_06 on_03 on_01 off_03; do
  printf '%-7s ' "$t"
  grep -E "OURS \(GraPhyNet\)|IDW baseline" "$OUT/conv_$t.log" | tr '\n' ' '; echo
done