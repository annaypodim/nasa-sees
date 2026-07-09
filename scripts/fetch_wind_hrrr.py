"""
fetch_wind_hrrr.py  -  pull HRRR (3 km) 10 m wind at each sensor point, via the
Open-Meteo historical-forecast archive (which stores past HRRR runs as JSON).

WHY THIS EXISTS
    All our wind so far is ERA5 (~30 km): over a 13 km metro that is ONE vector
    for the whole city, so "wind is common-mode" was CONFOUNDED by resolution --
    we never had wind that could vary at city scale. HRRR is 3 km and resolves
    terrain-channeled flow (river valleys, canyon drainage). This fetch lets us
    run the honest diagnostic: does fine-scale wind actually VARY across the
    sensor network (convergence, channeling), the one thing a single ERA5 vector
    can't provide and the cosine-projection can't fake. See memory:
    aod-fusion-direction (same diagnose-first discipline).

    Open-Meteo is used instead of raw HRRR GRIB from AWS because it needs only
    `requests` (no eccodes/cfgrib), and because it is itself gridded-model output
    available at ANY lat/lon regardless of ground-sensor density -- exactly the
    property we need for interpolating to unmonitored/rural locations.

WHAT IT WRITES  (default data/pittsburgh/wind_hrrr/sensor_<id>.csv)
    columns: time,u10,v10   (UTC, m/s) -- same schema load_wind() reads, so we can
    point CITY_CONFIG wind_dir at wind_hrrr/ later without touching the pipeline.
    u/v from meteorological speed+direction (dir = where wind comes FROM):
        u10 = -speed * sin(dir),  v10 = -speed * cos(dir).

USAGE
    .venv/bin/python -u scripts/fetch_wind_hrrr.py --city pittsburgh \
        --start 2023-01-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path

import requests

API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODEL = "gfs_hrrr"                 # HRRR 3 km (CONUS) in Open-Meteo's archive

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
            for i, la, lo, _ in rows]


def fetch_point(lat: float, lon: float, start: str, end: str) -> list[dict]:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms", "timezone": "UTC",
        "models": MODEL,
    }
    for attempt in range(4):
        r = requests.get(API, params=params, timeout=90)
        if r.status_code == 200:
            h = r.json().get("hourly", {})
            times = h.get("time", [])
            spd = h.get("wind_speed_10m", [])
            drc = h.get("wind_direction_10m", [])
            out = []
            for t, s, d in zip(times, spd, drc):
                if s is None or d is None:
                    continue
                rad = math.radians(d)
                out.append({"time": t,
                            "u10": -s * math.sin(rad),
                            "v10": -s * math.cos(rad)})
            return out
        if r.status_code in (429, 500, 502, 503):      # rate-limit / transient
            time.sleep(5 * (attempt + 1))
            continue
        raise SystemExit(f"[http {r.status_code}] {r.text[:300]}")
    raise SystemExit("[http] repeated rate-limit/5xx from Open-Meteo")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch HRRR 10 m wind at sensor "
                                             "points via Open-Meteo.")
    ap.add_argument("--city", default="pittsburgh", choices=list(COORDS))
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sensors = parse_coords(COORDS[a.city])
    out_dir = Path(a.out) if a.out else ROOT / "data" / a.city / "wind_hrrr"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[coords] {len(sensors)} sensors for {a.city}; model={MODEL} "
          f"{a.start}..{a.end} -> {out_dir}", flush=True)

    total = 0
    for n, s in enumerate(sensors, 1):
        rows = fetch_point(s["lat"], s["lon"], a.start, a.end)
        with open(out_dir / f"sensor_{s['id']}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "u10", "v10"])
            for r in rows:
                w.writerow([r["time"], f"{r['u10']:.3f}", f"{r['v10']:.3f}"])
        total += len(rows)
        print(f"  [{n}/{len(sensors)}] sensor {s['id']:>10}  {len(rows):>5} hours",
              flush=True)
        time.sleep(0.5)                                # be polite to the API
    print(f"[done] {total} wind rows -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
