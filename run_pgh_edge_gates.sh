#!/usr/bin/env bash
# TEMP-EDGE + AOD-EDGE gate ablations on PITTSBURGH (the only city with temp + AOD on
# disk). Each is the sensor-pair analog of the elevation edge gate: a per-edge gate from
# the signed Δcovariate(dst-src) that dampens both transports. Ablate each INDEPENDENTLY
# against an identical strong base (idw-prior + suppressed residual), inductive, 8 seeds.
# Question: does the EDGE form help where the NODE temp gate / AOD feature were null?
set -u
cd "$(dirname "$0")"
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT="${OUT:-pgh_edge_logs}"
mkdir -p "$OUT"

COMMON=(--city pittsburgh --wind zero --epochs 120 --seeds 0,1,2,3,4,5,6,7
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --no-convection --corr-reg 0.6 --loss huber --idw-prior --idw-feature
        --despike --spatial-qa)

run () { local tag="$1"; shift; echo "launching $tag -> $OUT/$tag.log"; \
  "$PY" scripts/eval_inductive.py "${COMMON[@]}" "$@" > "$OUT/$tag.log" 2>&1 & }

run base                                # strong base, no edge gate
run temp_edge --temp-edge-gate          # + temp edge gate
run aod_edge  --aod-edge-gate           # + AOD edge gate

wait
echo "=== PITTSBURGH EDGE-GATE ABLATION (8 seeds; base=no gate) ==="
for t in base temp_edge aod_edge; do
  printf '%-10s ' "$t"
  grep -E "OURS \(GraPhyNet\)|IDW baseline" "$OUT/$t.log" | tr '\n' '  '; echo
done
