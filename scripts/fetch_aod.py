"""
fetch_aod.py  -  pull MODIS/MAIAC MCD19A2 1 km AOD sampled at each PurpleAir
sensor location, via Google Earth Engine.

WHY THIS EXISTS
    Every reanalysis covariate we tried (temperature, wind, humidity from ERA5)
    is COMMON-MODE: near-identical across a small city at any hour, so it can't
    tell one location from another and is useless for interpolating PM2.5 to an
    unmonitored point. AOD is different: MAIAC retrieves it at 1 km, and our
    sensor networks span 12-40 km (~12-40 pixels), so column aerosol genuinely
    VARIES in space. It is also an independent physical observation of the
    aerosol column available at EVERY location -- exactly the thing missing at
    an empty query node. See memory: aod-fusion-direction.

    (We first tried NASA AppEEARS point-sampling -- it has NO aerosol product at
    all, 190 products and zero AOD. GEE carries MCD19A2 and samples at points
    server-side, handling the sinusoidal projection and multiple daily orbits.)

WHAT IT WRITES  (default data/<city>/aod/aod_points_<START>_<END>.csv)
    columns:
        id                 our sensor index
        lat, lon
        datetime           overpass time, ISO-8601 UTC. MCD19A2 has ~1-3 orbits
                           per day and drops out under cloud/snow, so this is
                           SUB-DAILY + gappy, not hourly. The loader aggregates
                           to a daily value and carries a have_aod mask.
        aod_055, aod_047   AOD @ 550 / 470 nm (raw DN * 0.001), NaN where missing.
        aod_qa             MCD19A2 AOD_QA bitmask (unscaled int).

AUTH  (you run this once; creds never touch the repo)
    Make/verify a Google Earth Engine account + a cloud project, then:
        .venv/bin/earthengine authenticate
    and pass your project id:
        export EE_PROJECT=your-ee-project-id
    (or --project your-ee-project-id)

USAGE
    export EE_PROJECT=...
    .venv/bin/python scripts/fetch_aod.py --city pittsburgh \
        --start 2023-01-01 --end 2023-12-31
    # SLC winter stress test (worst case: snow + inversion decoupling):
    .venv/bin/python scripts/fetch_aod.py --city slc \
        --start 2023-12-01 --end 2024-02-29

WIRING INTO THE PIPELINE
    Follow-up: load_aod() in build_graph2.py (mirrors load_temperature; aggregate
    to daily, forward-fill, carry have_aod mask) + an aod node-feature channel in
    train.py, never masked, behind USE_AOD_FEATURE / --no-aod-feature.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# MCD19A2.061 granule-level collection: Terra+Aqua MAIAC Land AOD, 1 km, sub-daily.
COLLECTION = "MODIS/061/MCD19A2_GRANULES"
AOD_SCALE = 0.001                       # raw DN -> unitless AOD
SCALE_M = 1000                          # sampling scale (native 1 km)

ROOT = Path(__file__).resolve().parents[1]
COORDS = {
    "pittsburgh": ROOT / "data/pittsburgh/coords/pittsburgh_loc_elev.txt",
    "slc":        ROOT / "data/slc/coords/sensor_lat_long_alt",
    "boulder":    ROOT / "data/boulder/coords/sensor_lat_long_alt",
}

_ROW = re.compile(
    r"\[\s*(\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+)\s*\]")


def parse_coords(path: Path) -> list[dict]:
    rows = _ROW.findall(path.read_text())
    if not rows:
        raise SystemExit(f"No sensor rows parsed from {path}")
    return [{"id": int(i), "lat": float(la), "lon": float(lo)}
            for i, la, lo, _alt in rows]


def _iso_utc(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch MCD19A2 AOD at sensor "
                                             "points via Google Earth Engine.")
    ap.add_argument("--city", default="pittsburgh", choices=list(COORDS))
    ap.add_argument("--start", default="2023-01-01", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", default="2023-12-31", help="YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--out", default=None, help="output CSV (default data/<city>/aod/...)")
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT", ""),
                    help="Earth Engine cloud project id (or set EE_PROJECT)")
    a = ap.parse_args()

    import ee
    if not a.project:
        sys.exit("No EE project. Run `earthengine authenticate` and set "
                 "EE_PROJECT (or pass --project).")
    ee.Initialize(project=a.project)

    sensors = parse_coords(COORDS[a.city])
    print(f"[coords] {len(sensors)} sensors for {a.city}")

    # end is inclusive -> GEE filterDate end is exclusive, so bump one day.
    end_excl = (datetime.strptime(a.end, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp() + 86400)
    end_excl_s = datetime.fromtimestamp(end_excl, tz=timezone.utc).strftime("%Y-%m-%d")

    col = (ee.ImageCollection(COLLECTION)
           .filterDate(a.start, end_excl_s)
           .select(["Optical_Depth_055", "Optical_Depth_047", "AOD_QA"]))

    out_path = Path(a.out) if a.out else (
        ROOT / "data" / a.city / "aod" / f"aod_points_{a.start}_{a.end}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # getRegion per point: returns [header, *rows] where each row is
    # [image_id, lon, lat, time_ms, Optical_Depth_055, Optical_Depth_047, AOD_QA].
    n_rows = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lat", "lon", "datetime", "aod_055", "aod_047", "aod_qa"])
        for n, s in enumerate(sensors, 1):
            pt = ee.Geometry.Point([s["lon"], s["lat"]])
            for attempt in range(3):
                try:
                    data = col.getRegion(pt, SCALE_M).getInfo()
                    break
                except Exception as e:              # transient EE/network errors
                    if attempt == 2:
                        raise
                    print(f"    retry {s['id']} ({e})")
                    time.sleep(3)
            header = {name: i for i, name in enumerate(data[0])}
            ti = header["time"]
            i55, i47 = header["Optical_Depth_055"], header["Optical_Depth_047"]
            iqa = header["AOD_QA"]
            got = 0
            for row in data[1:]:
                v55, v47 = row[i55], row[i47]
                if v55 is None and v47 is None:
                    continue                        # no retrieval this orbit
                w.writerow([
                    s["id"], f"{s['lat']:.5f}", f"{s['lon']:.5f}",
                    _iso_utc(row[ti]),
                    "" if v55 is None else f"{v55 * AOD_SCALE:.4f}",
                    "" if v47 is None else f"{v47 * AOD_SCALE:.4f}",
                    "" if row[iqa] is None else int(row[iqa]),
                ])
                got += 1
            n_rows += got
            print(f"  [{n}/{len(sensors)}] sensor {s['id']:>10}  {got:>5} AOD rows")
    print(f"[done] {n_rows} AOD rows -> {out_path}")


if __name__ == "__main__":
    main()
