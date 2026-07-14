# nasa-sees

GNN-based PM2.5 spatial imputation for air-quality sensor networks (NASA SEES project).
Learns to fill gaps in PurpleAir networks using graph structure, terrain, and temporal memory,
benchmarked against IDW and (regression-)kriging baselines across several cities.

## Layout

```
src/                  Library code
  graph/              Graph construction from raw sensor data
    build_graph2.py     current builder + per-city CONFIG (coords, PM2.5, wind, terrain)
    build_graph.py      original builder (legacy)
    preprocessing.py    despike / QA / feature prep
  model/              Model + training
    train.py            training loop, priors (IDW / kriging), losses
    graphy_faithful.py  faithful GraPhy re-implementation
    model.py            GraPhyNet (our modular variant)
    diffusion.py convection.py fusion.py elevation.py temperature.py local.py
  viz/                Data / topography visualizations

scripts/              Entrypoints (flat Python package — modules import each other as scripts.*)
  fetch_purpleair.py fetch_aod.py fetch_wind_hrrr.py    data fetching
  eval_inductive.py eval_temporal.py eval_graphy_faithful.py eval_grin.py   benchmarks
  apply_ab_qa.py build_processed.py merge_fresno_dense.py aod_loss.py       data prep / losses
  plot_*.py aggregate_faithful.py extract_density_results.py               analysis
  run/                All shell runners (run_*.sh, sweep_*.sh, confirm_*.sh)

data/                 Datasets
  boulder/ slc/ pittsburgh/          per-city raw data (coords, pm25, wind, terrain)
  fresno_variants/                   the many fresno_* network variants (density/QA/channel sets)

docs/                 writeups/ ; the running journal is WORKLOG.txt at the repo root
poster-graphs/        Self-contained figure-generation code + rendered PNGs

experiments/          Run logs — LOCAL ONLY, git-ignored (regenerable)
outputs/              Generated matrices/viz — LOCAL ONLY, git-ignored (regenerable)
```

## Running

Scripts assume the repo root as CWD and use the project venv. Runners in `scripts/run/`
`cd` to the repo root themselves, so they can be invoked from anywhere:

```bash
# a benchmark directly
PYTHONPATH=. .venv/bin/python scripts/eval_inductive.py --city fresno_dense --seeds 0,1,2,3

# or via a runner (writes logs under experiments/logs/<name>/)
bash scripts/run/run_qa.sh
```

City names passed via `--city` (e.g. `fresno_dense`, `slc`, `pittsburgh`) are logical keys
resolved by the CONFIG dict in `src/graph/build_graph2.py`, which maps each to its data paths.

## Notes

- `experiments/` and `outputs/` are git-ignored local artifacts. Delete freely; re-runs recreate them.
- This is a research repo run in parallel on a laptop and a remote GPU box (astrapi). After any
  reorg, `git pull` on both before running so paths stay in sync.
- Requirements in `requirements.txt`; project venv at `.venv/`.
