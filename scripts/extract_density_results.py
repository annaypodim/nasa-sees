#!/usr/bin/env python3
"""Rebuild density_sweep_logs/results.csv from the per-run logs (robust to the inline
parse bug: the summary line has multiple spaces before MAE=). One row per {config}_N{N}_ss{ss}
log that finished. Safe to re-run anytime; overwrites the CSV from whatever logs exist."""
import re
from pathlib import Path

LOGDIR = Path(__file__).resolve().parents[1] / "density_sweep_logs"
OUT = LOGDIR / "results.csv"
FN = re.compile(r"(?P<cfg>[a-z]+)_N(?P<N>\d+)_ss(?P<ss>\d+)\.log$")
OURS = re.compile(r"OURS \(GraPhyNet\)\s+MAE=([0-9.]+)(?:±([0-9.]+))?")
IDW = re.compile(r"IDW baseline\s+MAE=([0-9.]+)(?:±([0-9.]+))?")


def main():
    rows = []
    for f in sorted(LOGDIR.glob("*_N*_ss*.log")):
        m = FN.search(f.name)
        if not m:
            continue
        txt = f.read_text(errors="replace")
        o, i = OURS.search(txt), IDW.search(txt)
        if not (o and i):
            continue  # unfinished run
        rows.append((m["cfg"], int(m["N"]), int(m["ss"]),
                     o.group(1), o.group(2) or "", i.group(1), i.group(2) or ""))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    with OUT.open("w") as fh:
        fh.write("config,N,subsample_seed,ours_mae,ours_std,idw_mae,idw_std\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"[extract] {len(rows)} finished runs -> {OUT}")
    # quick aggregated preview (mean over subsets per N); baseline keyed as _IDW to avoid
    # colliding with the OURS config literally named "idw".
    import collections
    ours = collections.defaultdict(lambda: collections.defaultdict(list))  # N -> cfg -> [ours]
    base = collections.defaultdict(list)                                   # N -> [idw]
    for cfg, N, ss, om, os_, im, is_ in rows:
        ours[N][cfg].append(float(om)); base[N].append(float(im))
    def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
    print(f"\n{'N':>3} {'IDW':>8} {'OURS-krig':>10} {'OURS-idw':>9}   (mean over subsets; * = OURS<IDW)")
    for N in sorted(base):
        idw = mean(base[N]); k = mean(ours[N].get("krig", [])); i = mean(ours[N].get("idw", []))
        flag = " *" if (k < idw or i < idw) else ""
        print(f"{N:>3} {idw:8.3f} {k:10.3f} {i:9.3f}{flag}")


if __name__ == "__main__":
    main()
