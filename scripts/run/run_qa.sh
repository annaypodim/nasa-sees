#!/usr/bin/env bash
# QA ablation on fresno_dense (dense-best model: 1/d prior, corr-reg 0.9). Does the
# new flatline + spatial-outlier QA help beyond despike? 8 seeds.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT="${OUT:-experiments/logs/qa}"
mkdir -p "$OUT"

COMMON=(--city fresno_dense --wind zero --epochs 120 --seeds 0,1,2,3,4,5,6,7
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --no-convection --corr-reg 0.9 --loss huber --idw-prior --idw-feature)

run () { local tag="$1"; shift; echo "launching $tag"; \
  "$PY" scripts/eval_inductive.py "${COMMON[@]}" "$@" > "$OUT/$tag.log" 2>&1 & }

run qa_base --despike
run qa_flat --despike --flatline
run qa_spat --despike --spatial-qa
run qa_all  --despike --flatline --spatial-qa

wait
echo "=== QA ABLATION (fresno_dense, 8 seeds) ==="
for t in qa_base qa_flat qa_spat qa_all; do
  printf '%-9s ' "$t"
  grep -E "OURS \(GraPhyNet\)|IDW baseline" "$OUT/$t.log" | tr '\n' '  '; echo
done
