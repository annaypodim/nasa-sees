#!/usr/bin/env python3
"""A/B channel-agreement QA -> a cleaned canonical Fresno set (data/fresno_dense_ab).

PurpleAir sensors have TWO laser channels (pm2.5_atm_a / pm2.5_atm_b). On a healthy
sensor they track almost perfectly (calibrated on 42327: corr 0.999, median |A-B|
0.7 ug/m3, p95 1.7). Gross disagreement = a malfunctioning channel -> the pm2.5_atm
average is untrustworthy. This is the QA GraPhy-grade pipelines apply that we couldn't
before (we only had the averaged atm).

Inputs (same box/window/API, so time_stamps align):
  data/fresno_dense/pm25/urban/*.csv  -> canonical (pm2.5_atm, pm2.5_cf_1, humidity)
  data/fresno_ab/pm25/urban/*.csv     -> raw channels (pm2.5_atm_a, pm2.5_atm_b)

Output: data/fresno_dense_ab/ = fresno_dense with
  - per-CELL: pm2.5_atm AND pm2.5_cf_1 blanked where the two channels grossly disagree
    (|A-B| > ABS and rel > REL), or one channel drops out (min~0 while max large) ->
    that hour becomes NOT observed downstream.
  - per-SENSOR: sensors whose channels are chronically inconsistent (corr < CORR_MIN or
    bad-cell fraction > FRAC_MAX) are DROPPED entirely (a bad node, not a bad hour).
  - coords rewritten for the surviving sensors.
Then: eval --city fresno_dense_ab and compare to fresno_dense (OURS 3.368 / IDW 3.339).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
GROUP = "urban"
SRC = DATA / "fresno_dense"
AB = DATA / "fresno_ab"
OUT = DATA / "fresno_dense_ab"

# per-CELL disagreement: BOTH an absolute and a relative gate must trip (healthy p95 =
# 1.7 ug/m3 abs, 0.22 rel -> these are well clear of normal sensor noise).
CELL_ABS = 5.0     # ug/m3
CELL_REL = 0.5     # |A-B| / mean(A,B)
DROPOUT_MIN = 0.1  # one channel <= this while...
DROPOUT_MAX = 5.0  # ...the other exceeds this = a dead channel that hour
# per-SENSOR drop: chronic channel inconsistency
CORR_MIN = 0.90    # A/B correlation over the record
FRAC_MAX = 0.10    # fraction of joint cells flagged bad

COORD_RE = re.compile(r"\[\s*(\d+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]")


def parse_coords(path: Path) -> dict[int, tuple]:
    out = {}
    for m in COORD_RE.finditer(path.read_text()):
        out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def write_coords(entries: dict[int, tuple], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Node Values (Sensor Index, Latitude, Longitude, Altitude)", "", "Urban", "["]
    for sid in sorted(entries):
        lat, lon, alt = entries[sid]
        lines.append(f"  [{sid},{lat:.5f},{lon:.5f},{alt:.0f}],")
    lines.append("]")
    path.write_text("\n".join(lines) + "\n")


def sid_of(name: str):
    head = name.split(" ", 1)[0]
    return int(head) if head.isdigit() else None


def cell_bad_mask(a: pd.Series, b: pd.Series) -> pd.Series:
    """True where the two channels grossly disagree (both present)."""
    both = a.notna() & b.notna()
    absdiff = (a - b).abs()
    mean = (a + b) / 2.0
    rel = absdiff / mean.clip(lower=1e-6)
    disagree = (absdiff > CELL_ABS) & (rel > CELL_REL)
    dropout = (np.minimum(a, b) <= DROPOUT_MIN) & (np.maximum(a, b) > DROPOUT_MAX)
    return both & (disagree | dropout)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="A/B channel QA -> cleaned Fresno set")
    ap.add_argument("--out-name", default="fresno_dense_ab",
                    help="output city dir under data/ (default fresno_dense_ab)")
    ap.add_argument("--frac-max", type=float, default=FRAC_MAX,
                    help="drop a sensor if this fraction of joint cells disagree "
                         "(set 1.0 to disable -> drop only on low A/B correlation)")
    ap.add_argument("--corr-min", type=float, default=CORR_MIN,
                    help="drop a sensor whose A/B correlation is below this")
    args = ap.parse_args()
    out_dir = DATA / args.out_name
    src_pm = SRC / "pm25" / GROUP
    ab_pm = AB / "pm25" / GROUP
    out_pm = out_dir / "pm25" / GROUP
    if not src_pm.exists():
        raise SystemExit(f"missing {src_pm}")
    if not ab_pm.exists():
        raise SystemExit(f"missing {ab_pm} (run fetch --ab-channels first)")
    if out_pm.exists():
        shutil.rmtree(out_pm)
    out_pm.mkdir(parents=True)

    ab_by_id = {sid_of(f.name): f for f in ab_pm.glob("*.csv") if sid_of(f.name)}
    kept_ids, dropped, total_cells, total_bad = [], [], 0, 0

    for f in sorted(src_pm.glob("*.csv")):
        sid = sid_of(f.name)
        if sid is None:
            continue
        df = pd.read_csv(f)
        abf = ab_by_id.get(sid)
        if abf is None:
            # no channel data (e.g. an empty sensor) -> keep as-is, can't A/B-check it
            df.to_csv(out_pm / f.name, index=False)
            kept_ids.append(sid)
            continue
        ab = pd.read_csv(abf)[["time_stamp", "pm2.5_atm_a", "pm2.5_atm_b"]]
        m = df.merge(ab, on="time_stamp", how="left")
        a, b = m["pm2.5_atm_a"], m["pm2.5_atm_b"]
        both = a.notna() & b.notna()
        n_both = int(both.sum())
        bad = cell_bad_mask(a, b)
        n_bad = int(bad.sum())

        # per-sensor health verdict
        corr = np.nan
        if n_both >= 50:
            corr = np.corrcoef(a[both], b[both])[0, 1]
        frac_bad = (n_bad / n_both) if n_both else 0.0
        unhealthy = (n_both >= 50 and (corr < args.corr_min or frac_bad > args.frac_max))
        if unhealthy:
            dropped.append((sid, round(float(corr), 3), round(frac_bad, 3)))
            continue

        # blank the disagreeing cells in the trusted columns -> not observed downstream
        for col in ("pm2.5_atm", "pm2.5_cf_1"):
            if col in m.columns:
                m.loc[bad, col] = np.nan
        m.drop(columns=["pm2.5_atm_a", "pm2.5_atm_b"]).to_csv(out_pm / f.name, index=False)
        kept_ids.append(sid)
        total_cells += n_both
        total_bad += n_bad

    coords = parse_coords(SRC / "coords" / "sensor_lat_long_alt")
    coords = {sid: coords[sid] for sid in kept_ids if sid in coords}
    write_coords(coords, out_dir / "coords" / "sensor_lat_long_alt")

    print(f"[ab-qa] sensors: {len(kept_ids)} kept, {len(dropped)} dropped for channel "
          f"inconsistency (corr<{args.corr_min} or frac_bad>{args.frac_max}): {dropped}")
    print(f"[ab-qa] per-cell: masked {total_bad}/{total_cells} joint cells "
          f"({100*total_bad/max(total_cells,1):.2f}%) as A/B-disagreement")
    print(f"[ab-qa] -> {out_dir}")


if __name__ == "__main__":
    main()
