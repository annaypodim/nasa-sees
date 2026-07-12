# Handoff — Faithful GraPhy reproduction (next session starts here)

**Branch:** `graphy` (this is the working branch; the faithful-GraPhy commit is
`8f52824 "slc city elev with first faithful graphy"`).
**Next goal:** recreate GraPhy faithfully and **meet or beat their MAE 2.38** on
SLC or Fresno. We will fetch the rest of the Fresno data (paid PurpleAir API).

Full narrative + every number is in `WORKLOG.txt` (search "FAITHFUL GraPhy").
Long-term facts are in Claude memory `graphy-faithful-repro-task.md`.

---

## What already exists (built this session — DO NOT rebuild)
- `src/model/graphy_faithful.py` — the **published GraPhy architecture, standalone**:
  3 physics GNN modules (Diffusion Laplacian high-pass, Convection wind-message,
  Local source/sink) + **dynamic softmax fusion**. NO IDW-residual/kriging/gate
  scaffold. Optional `use_elev_gate` reuses `src/model/elevation.py` (per-layer,
  gates diffusion + convection; ungated path is byte-identical to base).
- `scripts/eval_graphy_faithful.py` — trains + inductively evaluates it under the
  SAME kriging protocol as `eval_inductive.py`. One-sensor masking, **plain MSE in
  raw µg/m³** (no log/sqrt/huber/clip — proven stable, 0 divergences), Adam 1e-4,
  batch 32, **val-based early stopping** (crucial — without it the model overfits
  the train sensors: test MAE 9.4 → 4.8). Block-diagonal batching (verified
  numerically identical to a per-example loop, 1.5e-8). Flags: `--config small|large`,
  `--epa-correct`, `--elev-gate`, `--wind hrrr|zero`, `--despike --spatial-qa`.
- `run_graphy_faithful.sh` — 8-seed (or N-seed) parallel runner in memory-safe waves.
  Env vars: `CITY=`, `EPA=0|1`, `GATE=0|1`, `WIND=`, `SEEDS=`. Usage:
  `CITY=slc WIND=zero GATE=1 SEEDS="0 1 2 3 4" ./run_graphy_faithful.sh small 4000 5`
- `scripts/aggregate_faithful.py` — aggregates per-seed logs → mean±std.
- Results: `faithful_logs/`.

## Results so far (inductive kriging, MAE µg/m³, held-out sensors)
| Setup | Faithful GraPhy | IDW (same splits) | note |
|---|---|---|---|
| Fresno abc, small, 8 seeds | 7.53 ± 2.56 | 3.99 | loses to IDW ~1.9x |
| Fresno abc, large(512/5), 4 seeds | 6.51 ± 0.73 | 3.71 | large>small, robust |
| Fresno abc, **+EPA**, small | **4.74 ± 1.12** | **3.04** | EPA = −42%; best Fresno |
| Fresno dense (+7 noisy sensors) +EPA | 5.64 | 3.42 | extra sensors are QA-rejects → worse |
| **SLC**, gate OFF, small | 5.49 ± 1.15 | 4.71 | 1.16x IDW |
| **SLC**, gate ON, small | **4.96 ± 0.94** | 4.71 | 1.05x IDW; gate −9.7% |
GraPhy paper: **2.38** (Fresno, 41 sensors, Oct'23–Jan'24).

## Why we're not at 2.38 yet — decomposed (all evidence in WORKLOG)
1. **Calibration (biggest lever):** raw PurpleAir over-reads ~40%. EPA/Barkjohn
   correction (`--epa-correct`, needs `pm2.5_cf_1`+`humidity` cols) cut MAE −42%
   (8.17→4.74) and even IDW −20% (→3.04). GraPhy used corrected data.
2. **Density of GOOD sensors:** we have ~22–29 usable Fresno sensors vs GraPhy's 41.
   IDW itself is 3.0–4.9 here (a no-model floor), so 2.38 is unreachable by ANY
   interpolator on this network. `fresno_dense` (43) does NOT help — its extra
   sensors are exactly the ones A/B QA drops (noisy) → makes IDW *and* the model worse.
3. Architecture is SOUND: stable, learns real signal (corr .85–.93), large>small
   like the paper, and on SLC terrain it's ~parity with IDW and the elevation gate
   helps (−9.7%). The Fresno gap is data, not the model.

## THE BLOCKER on matching 41 sensors (why we "don't have enough")
PurpleAir history API is **paid**. The 43-sensor Fresno fetch died at sensor 9/43
with **HTTP 402 PaymentRequiredError** (key out of points). 12 of 33 original CSVs
came back empty. `scripts/fetch_purpleair.py` supports `--bbox`, `--extra-fields`
(for EPA cols), `--exclude-existing`. Bbox that listed 43 usable:
`36.90,-119.92,36.60,-119.58`, window Oct'23–Jan'24.

## Recommended plan to MEET/BEAT 2.38
**Path A — Fresno (once credits are topped up):**
1. `scripts/fetch_purpleair.py --bbox 36.90,-119.92,36.60,-119.58 --extra-fields`
   for Oct'23–Jan'24 → ~41–43 sensors WITH cf_1+humidity. Land in a new
   `data/fresno_full/`, add a `CITY_CONFIG` entry (copy `fresno_dense_abc`, add
   `wind_hrrr_dir=data/fresno/wind_hrrr`).
2. Apply A/B-channel QA (`scripts/apply_ab_qa.py`) to keep only clean sensors.
3. Run: `CITY=fresno_full EPA=1 WIND=hrrr ./run_graphy_faithful.sh large 4000 4`
   (large, EPA on, 8 seeds if RAM allows — NOTE 8-way large thrashes 16GB → use
   waves of 4). Compare to IDW and 2.38. With 41 EPA'd sensors the faithful model
   should finally have enough signal to beat IDW (GraPhy's headline claim).

**Path B — SLC (no new data needed):**
- The gate already gets to 1.05x IDW. To BEAT: try `--config large` on SLC,
  and consider a terrain-aware element on the diffusion adjacency (WORKLOG's
  biggest SLC lever for the OLD model was a terrain-aware IDW *kernel*, −13.5%;
  the faithful model has no IDW prior, but the elevation gate is the analog —
  push it: larger capacity + more seeds, maybe per-edge learned anisotropy).
- SLC has no wind data (uses `--wind zero`) and no EPA cols (raw only).

## Gotchas (cost time this session)
- **Worktree/branch:** this session's worktree was based on an OLD commit missing
  all Fresno data/harness; had to reset onto `graphy`. Work directly on `graphy`.
- **Memory:** 8 parallel hidden-512 (large) models exhaust 16GB RAM → swap → stall.
  Use ≤4-way for large (`run_graphy_faithful.sh` maxpar arg = 3rd positional).
- **macOS bash 3.2:** no `wait -n`; `set -e` aborts on `false`/`$()`-returns-1 and
  `set -u` on empty `"${arr[@]}"`. The runner is already fixed for these.
- **Output buffering:** Python block-buffers stdout to redirected files; per-seed
  logs only flush at process exit. Judge progress by CPU time, not log contents.
