#!/usr/bin/env bash
# Definitive SLC A/B: does the neural corrector (over the RK-elev floor) and SPIN's
# AOD spatial-gradient loss beat the strong RK-elev baseline (3.65)?
#   A  rk-prior            = corrector over the elevation-kriging floor
#   B  rk-prior + aod 0.5  = A + masked AOD spatial-gradient training constraint
# Baselines (IDW 4.06, RK-elev 3.65) print inside every run for reference.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/logs/faithful/slc_rk_aod_ab
mkdir -p "$OUT"
SEEDS="0 1 2 3"
COMMON="--city slc --wind zero --despike --spatial-qa --config small \
  --steps 8000 --val-every 500 --val-hours 300 --patience 15 --rk-prior"

run_arm () {  # name  extra
  local name="$1"; shift
  for s in $SEEDS; do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. $PY scripts/eval_graphy_faithful.py \
      $COMMON "$@" --seeds "$s" > "$OUT/${name}_seed${s}.log" 2>&1
    echo "[rk-aod] $name seed $s done ($(date +%H:%M:%S))"
  done
}

echo "[rk-aod] START $(date +%H:%M:%S) -> $OUT"
run_arm A_rk
run_arm B_rk_aod --aod-weight 0.5
echo "[rk-aod] DONE $(date +%H:%M:%S)"

echo "=== SLC RK-prior A/B (OURS mean; IDW 4.06, RK-elev 3.65 refs) ==="
for name in A_rk B_rk_aod; do
  grep -hoE "OURS mae=[0-9.]+" "$OUT/${name}"_seed*.log | grep -oE "[0-9.]+" | \
    awk -v n="$name" '{s+=$1;ss+=$1*$1;c++} END{m=s/c; printf "%-9s OURS=%.3f +- %.3f (n=%d)\n", n, m, sqrt(ss/c-m*m), c}'
done
