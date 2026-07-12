#!/usr/bin/env bash
# IGNNK-style implicit augmentation: randomize the per-step mask fraction so the GNN
# trains across reconstruction difficulties (predict-many-from-few) -> better general-
# ization to sparse held-out sensors. Question: does it let OURS finally BEAT dense IDW
# (3.339 with spatial-qa)? Also test with a looser correction leash (augmentation should
# make a bigger correction safe). fresno_dense, despike+spatial-qa, 1/d prior, 8 seeds.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT="${OUT:-ignnk_logs}"
mkdir -p "$OUT"

COMMON=(--city fresno_dense --wind zero --epochs 120 --seeds 0,1,2,3,4,5,6,7
        --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16
        --no-convection --loss huber --idw-prior --idw-feature
        --despike --spatial-qa)

run () { local tag="$1"; shift; echo "launching $tag"; \
  "$PY" scripts/eval_inductive.py "${COMMON[@]}" "$@" > "$OUT/$tag.log" 2>&1 & }

run base       --corr-reg 0.9 --mask-frac 0.25                     # current dense-best
run aug_wide   --corr-reg 0.9 --mask-frac 0.2 --mask-frac-hi 0.6   # randomized masking
run aug_hi     --corr-reg 0.9 --mask-frac 0.3 --mask-frac-hi 0.7   # harder masking
run aug_cr06   --corr-reg 0.6 --mask-frac 0.2 --mask-frac-hi 0.6   # aug + looser leash

wait
echo "=== IGNNK MASKING ABLATION (fresno_dense, dense IDW = 3.339) ==="
for t in base aug_wide aug_hi aug_cr06; do
  printf '%-9s ' "$t"
  grep -E "OURS \(GraPhyNet\)|IDW baseline" "$OUT/$t.log" | tr '\n' '  '; echo
done
