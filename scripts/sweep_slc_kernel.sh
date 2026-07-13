#!/usr/bin/env bash
# Inversion-aware IDW-prior kernel A/B on the DENSE SLC net (36 usable nodes).
# The kernel modulates the IDW PRIOR only (wind-independent) -> run --wind zero to
# isolate it from convection. Control = the current symmetric terrain kernel
# (--idw-prior-elev). Novelty = signed vertical decay (steep uphill h_up, gentle
# downhill h_down: pollutant pools down) + an inversion CAP z* that cuts pairs
# straddling the inversion. z* swept across the valley band (1284-1708 m).
#   CTRL  - symmetric exp(-|dz|/200)                  (beat this)
#   SIGN  - signed h_up=100 / h_down=400
#   CAP14 - SIGN + cap z*=1400
#   CAP15 - SIGN + cap z*=1500
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=faithful_logs/slc_dense_kernel_ab
mkdir -p "$OUT"
SEEDS="0 1 2 3"
COMMON="--city slc --wind zero --despike --spatial-qa --config small \
  --steps 15000 --val-every 500 --val-hours 300 --patience 20 --idw-prior --idw-prior-elev"

run_cfg () {  # name  extra-args
  local name="$1"; shift
  for s in $SEEDS; do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. $PY scripts/eval_graphy_faithful.py \
      $COMMON "$@" --seeds "$s" > "$OUT/${name}_seed${s}.log" 2>&1
    echo "[kernel-ab] $name seed $s done ($(date +%H:%M:%S))"
  done
}

echo "[kernel-ab] START $(date +%H:%M:%S)  -> $OUT"
run_cfg CTRL
run_cfg SIGN  --idw-h-up 100 --idw-h-down 400
run_cfg CAP14 --idw-h-up 100 --idw-h-down 400 --idw-cap 1400 --idw-cap-w 50
run_cfg CAP15 --idw-h-up 100 --idw-h-down 400 --idw-cap 1500 --idw-cap-w 50
echo "[kernel-ab] DONE $(date +%H:%M:%S)"

# aggregate OURS mae per config
echo "=== SLC dense kernel A/B (OURS = hybrid MAE, IDW ref in parens) ==="
for name in CTRL SIGN CAP14 CAP15; do
  grep -hoE "OURS mae=[0-9.]+" "$OUT/${name}"_seed*.log | grep -oE "[0-9.]+" | \
    awk -v n="$name" '{s+=$1; ss+=$1*$1; c++} END{m=s/c; printf "%-6s MAE=%.3f +- %.3f  (n=%d)\n", n, m, sqrt(ss/c-m*m), c}'
done
