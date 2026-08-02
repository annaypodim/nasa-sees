#!/usr/bin/env bash
# Does A/B-channel QA lower MAE? Compare the density set (fresno_dense) to the A/B-cleaned
# strict (fresno_dense_ab) and conservative (fresno_dense_abc) sets, all under the SAME
# full QA stack (despike+spatial-qa) + dense-best model, 8 seeds. Tension: A/B removes
# noisy sensors/cells (helps) but also loses density (hurts) -> let MAE decide.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT="${OUT:-experiments/logs/ab}"
mkdir -p "$OUT"

COMMON=(--wind zero --epochs 120 --seeds 0,1,2,3,4,5,6,7
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --no-convection --corr-reg 0.9 --loss huber --idw-prior --idw-feature
        --despike --spatial-qa)

run () { local city="$1"; echo "launching $city"; \
  "$PY" scripts/eval_inductive.py --city "$city" "${COMMON[@]}" > "$OUT/$city.log" 2>&1 & }

run fresno_dense       # baseline (no A/B QA)
run fresno_dense_ab    # A/B strict
run fresno_dense_abc   # A/B conservative

wait
echo "=== A/B QA EVAL (8 seeds; baseline fresno_dense OURS 3.368 / IDW 3.339) ==="
for c in fresno_dense fresno_dense_ab fresno_dense_abc; do
  printf '%-18s ' "$c"
  grep -E "nodes=" "$OUT/$c.log" | head -1 | grep -oE "nodes=[0-9]+" | tr '\n' ' '
  grep -E "OURS \(GraPhyNet\)|IDW baseline" "$OUT/$c.log" | tr '\n' '  '; echo
done
