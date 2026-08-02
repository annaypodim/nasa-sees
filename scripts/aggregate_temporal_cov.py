"""Aggregate RESULT: json lines from the temporal-covariate matrix logs into the
final results.csv. Also derives the WIND-ISOLATION row (wind hrrr vs zero), which is
paired per-fold across the two separate --wind processes (folds depend only on the
observed mask, identical across wind sources; run_seed(seed=k) is deterministic given
the graph, so per-fold pairing by fold index is exact -- only the wind input differs).
"""
import csv
import glob
import json
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "logs",
                       "temporal_covariates")
OUT = os.path.join(LOG_DIR, "results.csv")

COLS = ["city", "gap_len", "variant", "ours_mae", "ours_std", "stk_mae",
        "baseline_ours_mae", "delta_vs_baseline_pct", "folds_improved_over_baseline"]


def load():
    recs = {}
    for path in glob.glob(os.path.join(LOG_DIR, "*.log")):
        with open(path) as f:
            for line in f:
                if not line.startswith("RESULT: "):
                    continue
                r = json.loads(line[len("RESULT: "):])
                recs[(r["city"], r["gap_len"], r["variant"])] = r
    return recs


def main():
    recs = load()
    rows = []
    cities = sorted({k[0] for k in recs})
    gaps = sorted({k[1] for k in recs})

    for city in cities:
        for gap in gaps:
            base_h = recs.get((city, gap, "base_wind_hrrr"))
            base_z = recs.get((city, gap, "base_wind_zero"))
            if not base_h:
                continue
            # 1) baseline reference row (the strong model, no extra covariate, wind=hrrr)
            rows.append(dict(city=city, gap_len=gap, variant="base",
                             ours_mae=base_h["ours_mae"], ours_std=base_h["ours_std"],
                             stk_mae=base_h["stk_mae"],
                             baseline_ours_mae=base_h["ours_mae"],
                             delta_vs_baseline_pct=0.0,
                             folds_improved_over_baseline="n/a"))
            # 2) temp / aod / windfeat variants (paired in-process vs base_wind_hrrr)
            for v in ["temp", "aod", "windfeat"]:
                r = recs.get((city, gap, v))
                if not r:
                    continue
                rows.append(dict(city=city, gap_len=gap, variant=v,
                                 ours_mae=r["ours_mae"], ours_std=r.get("ours_std"),
                                 stk_mae=r["stk_mae"],
                                 baseline_ours_mae=r["baseline_ours_mae"],
                                 delta_vs_baseline_pct=r["delta_vs_baseline_pct"],
                                 folds_improved_over_baseline=r["folds_improved_over_baseline"]))
            # 3) WIND ISOLATION: variant = wind (hrrr) vs baseline = wind zero.
            #    negative delta => real wind lowers MAE vs zero (wind helps).
            if base_z and "ours_per_fold" in base_h and "ours_per_fold" in base_z:
                ph, pz = base_h["ours_per_fold"], base_z["ours_per_fold"]
                n = min(len(ph), len(pz))
                improved = sum(1 for i in range(n) if ph[i] < pz[i])  # folds wind helps
                delta = (base_h["ours_mae"] / base_z["ours_mae"] - 1.0) * 100.0
                rows.append(dict(city=city, gap_len=gap, variant="wind_hrrr_vs_zero",
                                 ours_mae=base_h["ours_mae"], ours_std=base_h["ours_std"],
                                 stk_mae=base_h["stk_mae"],
                                 baseline_ours_mae=base_z["ours_mae"],
                                 delta_vs_baseline_pct=round(delta, 3),
                                 folds_improved_over_baseline=improved))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    print(f"wrote {len(rows)} rows -> {OUT}")
    # echo a readable table
    for r in rows:
        print(f"{r['city']:11s} g{r['gap_len']:<3d} {r['variant']:20s} "
              f"ours={r['ours_mae']}  stk={r['stk_mae']}  base={r['baseline_ours_mae']}  "
              f"d={r['delta_vs_baseline_pct']}%  folds+={r['folds_improved_over_baseline']}")


if __name__ == "__main__":
    main()
