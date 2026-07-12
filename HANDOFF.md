# Handoff — GraPhy PM2.5 inductive-kriging reproduction (Fresno)

_Written 2026-07-11. Read this first, then `WORKLOG.txt` (bottom-up) for the blow-by-blow.
Persistent state also lives in the memory dir (`graphy-reproduction-fresno.md` finding #s
1–12). Everything described here is committed._

## The goal
Reproduce/beat **GraPhy**'s reported **MAE 2.38 µg/m³** on Fresno PurpleAir (inductive
sensor-holdout kriging: train on a disjoint sensor set, predict PM2.5 at held-out sensors
over every hour, score linear MAE in µg/m³). An **IDW baseline** is the model-free floor
we must beat — it's also GraPhy's headline claim (their GNN beats IDW).

## Current best result
**MAE: OURS 3.054 / IDW 3.021 → 1.27× off GraPhy 2.38** (was 1.8× at the start).
Reproduce it:
```bash
PYTHONPATH=. .venv/bin/python scripts/eval_inductive.py \
  --city fresno_dense_abc --wind zero --epochs 120 --seeds 0,1,2,3,4,5,6,7 \
  --epa-correct --transform sqrt --clip 1 --layers 4 --hidden 16 \
  --no-convection --corr-reg 0.9 --loss huber --idw-prior --idw-feature \
  --despike --spatial-qa
```
(22 usable sensors. Sparse 18-node variant: `--city fresno` uses `--kriging-prior
--kriging-range 5000 --corr-reg 0.6` instead → MAE 3.819.)

## The ONE thing to understand
**The gap to GraPhy is DATA QUALITY, not the model.** Every *data* lever moved MAE
(despike, then A/B-channel QA); every *model* lever hit a wall. On a dense network IDW is
near-optimal, and our architecture (`pred = IDW_prior + small learned residual`) can at
best *tie* IDW — the residual has no signal to add once the field is well-sampled. This is
confirmed **four independent ways** (dense prior-diagnostic, convection, IGNNK masking,
capacity). Do not re-litigate it; build on it.

## What WORKS (levers that moved MAE) — keep these
- **despike** (temporal MAD spike removal) — big early win.
- **A/B-channel QA** (`fresno_dense_abc`) — **−9.5%, biggest since despike.** Dropped 5
  sensors whose two laser channels are decorrelated (corr < 0.9) + masked scattered bad
  cells. Improved MAE *despite* fewer sensors → those sensors were corrupting.
- **spatial-qa** (per-hour cross-sensor outlier mask) — small win (−0.08).
- **Density** — going 18→27 usable sensors dropped IDW 4.16→3.34 (absolute-error lever).
- **Prior choice:** `--kriging-prior --kriging-range 5000` on SPARSE nets; plain 1/d
  `--idw-prior` on DENSE (kriging over-smooths dense geometry).
- **Suppressed correction** (`--corr-reg` 0.6 sparse / 0.9 dense) + zero-init head — keeps
  the GNN a small residual so it can't drag below the IDW floor.

## What's DEAD (ruled out — don't retry without a NEW reason)
- **convection** — HRRR wind is common-mode across a city (measured: ~17% cross-sensor
  variation, ~18° direction spread; Pittsburgh even more uniform). HRRR's 3 km grid can't
  resolve intra-city wind, so there's no differential advection to learn. Convection HURTS.
- **temperature gate, AOD** — common-mode, redundant with PM's own spatial structure.
- **L1 loss, isotonic recal** — recal unstable (a seed blew up); both regress.
- **IDW-geometry features** — noise.
- **Extra capacity** (more layers/hidden, deeper module MLPs) — OVERFITS at N=18–27.
- **IGNNK random-mask augmentation** (`--mask-frac-hi`) — no MAE gain; doesn't unlock the
  residual.
- **flatline QA** — inert (no stuck runs survive the [0,500] clip).
- **STRICT A/B drop** (also dropping high-corr sensors on cell-fraction) — over-drops good
  sensors, worse than conservative.

## Open levers (ranked) — where to go next
1. **More data QA toward GraPhy's ~2.7 IDW** (data path, the proven-productive direction):
   we've done despike + spatial + A/B. Remaining ideas: humidity/RH-based flagging,
   Barkjohn-correction edge cases, cross-sensor drift/bias detection over time. Diminishing
   but this is where the gains have been.
2. **Re-fetch the 12 empty original sensors** (their old fetches failed; the PurpleAir key
   works now). Would lift usable ~22→~30+, truly matching GraPhy's ~41. Costs credits.
   Helps IDW more than OURS. See WORKLOG "MATCH GRAPHY DENSITY" + the supplement workflow.
3. **Architectural rebuild** (research bet, LOW expected payoff given the ceiling finding):
   abandon residual-on-IDW for a learned-anisotropic-weight interpolator (GraPhy's actual
   learned diffusion; or GAT/KCN-style attention over neighbor values). Only worth it if
   you accept IDW is beatable on dense data — our evidence says barely.

## Key files
- `scripts/eval_inductive.py` — the whole eval + training loop + all model-lever flags
  (`--idw-prior/--kriging-prior/--kriging-range`, `--despike/--spatial-qa/--flatline`,
  `--corr-reg`, `--mask-frac-hi`, `--no-convection`, etc.). Prints OURS vs IDW vs GraPhy.
- `src/graph/preprocessing.py` — QA filters (mask_implausible, despike, mask_flatline,
  mask_spatial_outliers). Globals flipped on by eval flags.
- `scripts/apply_ab_qa.py` — A/B-channel QA → builds `data/fresno_dense_abc` (conservative,
  the best) and `fresno_dense_ab` (strict). `--frac-max 1.0` = conservative.
- `scripts/fetch_purpleair.py` — PurpleAir fetch. Flags: `--bbox`, `--extra-fields`
  (cf_1+humidity for EPA), `--ab-channels` (raw a/b), `--exclude-existing DIR` (supplement
  mode: fetch only NEW sensors, don't re-pull owned ones).
- `scripts/merge_fresno_dense.py` — unions data/fresno + data/fresno_extra → fresno_dense.
- `src/graph/build_graph2.py` — `CITY_CONFIG` (fresno, fresno_dense, fresno_dense_ab/abc).
- `src/model/{model,diffusion,convection,local,fusion,elevation,temperature}.py` — the GNN.
  Note diffusion uses a FIXED 1/d Laplacian (= IDW's kernel); convection has the MLPs.
- Run scripts: `run_qa.sh`, `run_ignnk.sh`, `run_ab_eval.sh`, `run_kriging_grid*.sh`,
  `run_convection_fresno.sh`, `run_dense_model.sh`, `run_tier1_sweep.sh`.

## Data on disk (data/)
- `fresno/` (33 sensors, ~18–21 usable, HRRR wind) — sparse set.
- `fresno_dense/` (43 sensors, 27 usable, zero wind) — density-matched (33 + 10 supplement).
- `fresno_ab/` (43 sensors, raw a/b channels, no cf_1/humidity).
- `fresno_dense_abc/` (22 usable) — **A/B-cleaned, the current best set.** `_ab` = strict.
- `.env.local` holds `PURPLEAIR_API_KEY` (gitignored; key works, has credits).

## How to run things
- venv python: `.venv/bin/python`, always `PYTHONPATH=.`.
- Sweeps: the `run_*.sh` use a bash ARRAY for args (`"${COMMON[@]}"`). Do NOT pass a
  space-joined `$VAR` of flags to a backgrounded python — it word-splits wrong and argparse
  rejects it (bit us twice). Use the array pattern or fully explicit args.
- Logs block-buffer to files; a 4×8-seed dense sweep takes ~8–12 min. Count progress with
  `grep -c 'seed [0-9]:' <log>`; final line has `OURS (GraPhyNet)` + `IDW baseline`.
- User controls commits — don't commit unless asked. Document results in WORKLOG.txt +
  the memory dir (the user relies on both).
```
