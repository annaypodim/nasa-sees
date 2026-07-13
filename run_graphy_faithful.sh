#!/usr/bin/env bash
# Run the FAITHFUL GraPhy rebuild over 8 seeds, one process/seed, in WAVES of at
# most MAXPAR concurrent seeds, then aggregate.
# Usage:  ./run_graphy_faithful.sh <large|small> <steps> [maxpar]
# MEMORY NOTE: each hidden-512 (large) process holds ~1.5-2 GB. Running all 8 at
# once exhausted 16 GB RAM -> 10 GB swap -> thrash -> stall. Cap large at 4-way.
set -euo pipefail
CONFIG="${1:-large}"
STEPS="${2:-4000}"
MAXPAR="${3:-4}"
CITY="${CITY:-fresno_dense_abc}"
EPA="${EPA:-0}"                       # EPA=1 -> add --epa-correct
GATE="${GATE:-0}"                     # GATE=1 -> add --elev-gate (old 2-scalar gate)
TERRAIN="${TERRAIN:-0}"               # TERRAIN=1 -> add --terrain-gate (learned gates)
IDWPRIOR="${IDWPRIOR:-0}"             # IDWPRIOR=1 -> add --idw-prior (hybrid residual)
IDWELEV="${IDWELEV:-0}"               # IDWELEV=1 -> add --idw-prior-elev (terrain kernel)
WIND="${WIND:-hrrr}"                  # hrrr | zero | era5
WD="${WD:-0}"                         # Adam weight decay (regularizer)
DROPOUT="${DROPOUT:-0}"               # module-MLP dropout
HID="${HID:-}"                        # override hidden width (empty = config default)
LAY="${LAY:-}"                        # override layer count
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
EXTRA=""; TAG="$CITY"
if [ "$WD" != "0" ]; then EXTRA="$EXTRA --weight-decay $WD"; TAG="${TAG}_wd${WD}"; fi
if [ "$DROPOUT" != "0" ]; then EXTRA="$EXTRA --dropout $DROPOUT"; TAG="${TAG}_do${DROPOUT}"; fi
if [ -n "$HID" ]; then EXTRA="$EXTRA --hidden $HID"; TAG="${TAG}_h${HID}"; fi
if [ -n "$LAY" ]; then EXTRA="$EXTRA --layers $LAY"; TAG="${TAG}_l${LAY}"; fi
if [ "$EPA" = "1" ]; then EXTRA="$EXTRA --epa-correct"; TAG="${TAG}_epa"; fi
if [ "$GATE" = "1" ]; then EXTRA="$EXTRA --elev-gate"; TAG="${TAG}_gate"; fi
if [ "$TERRAIN" = "1" ]; then EXTRA="$EXTRA --terrain-gate"; TAG="${TAG}_terrain"; fi
if [ "$IDWPRIOR" = "1" ]; then EXTRA="$EXTRA --idw-prior"; TAG="${TAG}_idw"; fi
if [ "$IDWELEV" = "1" ]; then EXTRA="$EXTRA --idw-prior-elev"; TAG="${TAG}_idwelev"; fi
PATIENCE="${PATIENCE:-10}"            # early-stop patience (val checks)
VALEVERY="${VALEVERY:-250}"           # steps between val checks
LR="${LR:-1e-4}"                      # Adam lr
if [ "$LR" != "1e-4" ]; then EXTRA="$EXTRA --lr $LR"; TAG="${TAG}_lr${LR}"; fi
if [ "$PATIENCE" != "10" ]; then TAG="${TAG}_pat${PATIENCE}"; fi
PY=/Users/annaypodimatopoulou/Code/side_quests/nasa-sees/.venv/bin/python
OUT="faithful_logs/${CONFIG}_${STEPS}_${TAG}"
mkdir -p "$OUT"
echo "[run] config=$CONFIG steps=$STEPS maxpar=$MAXPAR city=$CITY epa=$EPA seeds=($SEEDS) -> $OUT"
# portable wave scheduler (macOS bash 3.2 has no `wait -n`): launch seeds in
# groups of MAXPAR and `wait` for the whole group before the next one.
group=()
for s in $SEEDS; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. "$PY" scripts/eval_graphy_faithful.py \
    --city "$CITY" --wind "$WIND" --despike --spatial-qa $EXTRA \
    --config "$CONFIG" --steps "$STEPS" --seeds "$s" \
    --val-every "$VALEVERY" --val-hours 300 --patience "$PATIENCE" \
    > "$OUT/seed_$s.log" 2>&1 &
  echo "[run] launched seed $s (pid $!)"
  group+=($!)
  if [ "${#group[@]}" -ge "$MAXPAR" ]; then
    for p in "${group[@]}"; do wait "$p" || true; done
    group=()
  fi
done
if [ "${#group[@]}" -gt 0 ]; then for p in "${group[@]}"; do wait "$p" || true; done; fi
echo "[run] all seeds finished; aggregating"
"$PY" scripts/aggregate_faithful.py "$OUT"/seed_*.log | tee "$OUT/summary.txt"
