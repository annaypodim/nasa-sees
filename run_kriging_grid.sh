#!/usr/bin/env bash
# Kriging range/nugget grid on the winning config D (despike + kriging + idw-feature).
# D baseline = range 3000, nugget 0.1 -> MAE 4.045. Vary range (nugget 0.1), then
# vary nugget at the incumbent range. All inductive Fresno, --wind zero, 8 seeds.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT="${OUT:-kriging_grid_logs}"
mkdir -p "$OUT"

# config D minus the kriging range/nugget (added per-run):
COMMON=(--city fresno --wind zero --epochs 120 --seeds 0,1,2,3,4,5,6,7
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --no-convection --corr-reg 0.6
        --despike --loss huber --kriging-prior --idw-feature)

run () {  # run <tag> <extra flags...>
  local tag="$1"; shift
  echo "launching $tag -> $OUT/run_$tag.log"
  "$PY" scripts/eval_inductive.py "${COMMON[@]}" "$@" > "$OUT/run_$tag.log" 2>&1 &
}

# range grid @ nugget 0.1  (r3000 = incumbent D)
run r1000 --kriging-range 1000 --kriging-nugget 0.1
run r2000 --kriging-range 2000 --kriging-nugget 0.1
run r3000 --kriging-range 3000 --kriging-nugget 0.1
run r5000 --kriging-range 5000 --kriging-nugget 0.1
run r8000 --kriging-range 8000 --kriging-nugget 0.1
# nugget check @ incumbent range 3000
run n0.05 --kriging-range 3000 --kriging-nugget 0.05
run n0.20 --kriging-range 3000 --kriging-nugget 0.20

wait
echo "=== KRIGING GRID DONE — MAE summary (OURS vs IDW) ==="
for t in r1000 r2000 r3000 r5000 r8000 n0.05 n0.20; do
  printf '%-7s ' "$t"
  grep -E "OURS \(GraPhyNet\)|IDW baseline" "$OUT/run_$t.log" | tr '\n' ' '; echo
done
