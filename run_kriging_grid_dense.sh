#!/usr/bin/env bash
# Re-tune kriging range on the DENSE net (43 fetched -> 27 usable). Sensors sit ~40%
# closer than the 18-node set, so the sparse-tuned range 5000 over-smooths and OURS
# (3.82) lost to IDW (3.39). Sweep shorter ranges. Tuned base, --wind zero, 8 seeds.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT="${OUT:-dense_eval_logs}"
mkdir -p "$OUT"

COMMON=(--city fresno_dense --wind zero --epochs 120 --seeds 0,1,2,3,4,5,6,7
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --no-convection --corr-reg 0.6
        --despike --loss huber --kriging-prior --idw-feature --kriging-nugget 0.1)

run () {  # run <tag> <range>
  local tag="$1"; local rng="$2"
  echo "launching $tag (range=$rng) -> $OUT/dense_$tag.log"
  "$PY" scripts/eval_inductive.py "${COMMON[@]}" --kriging-range "$rng" \
      > "$OUT/dense_$tag.log" 2>&1 &
}

run d800  800
run d1200 1200
run d1800 1800
run d2500 2500
run d3500 3500

wait
echo "=== DENSE KRIGING GRID DONE (IDW dense baseline = 3.391) ==="
for t in d800 d1200 d1800 d2500 d3500; do
  printf '%-6s ' "$t"
  grep -E "OURS \(GraPhyNet\)|IDW baseline" "$OUT/dense_$t.log" | tr '\n' ' '; echo
done
