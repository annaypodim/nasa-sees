"""
fetch_purpleair.py  -  pull a whole bounding box of PurpleAir sensors into the
exact on-disk layout build_graph2.py already reads.

WHY THIS EXISTS
    The manual workflow (click each sensor on the PurpleAir web map, download its
    60-min CSV, hand-type its lat/lon/altitude into `sensor_lat_long_alt`) does
    not scale to a dense valley like Salt Lake City. The PurpleAir API does the
    same job in two calls per run:
      1. GET /v1/sensors            -> the sensor list + lat/lon/altitude for the
                                       whole box  == your coords file, for free
      2. GET /v1/sensors/{id}/history/csv  -> the per-sensor 60-min PM2.5 series

WHAT IT WRITES  (default OUT_DIR = data/slc/)
    coords/sensor_lat_long_alt
        the custom text format parse_sensor_coords() expects:
            Node Values (Sensor Index, Latitude, Longitude, Altitude)

            Urban
            [
              [<id>, <lat>, <lon>, <altitude_ft>],
              ...
            ]
        altitude is written in FEET (PurpleAir reports feet; build_graph2 does
        the feet->m conversion), matching the Boulder file exactly.
    pm25/<GROUP>/<id> <START> <END> 60-Minute Average.csv
        columns: time_stamp,pm2.5_atm   (time_stamp as ISO-8601 UTC)

USAGE
    export PURPLEAIR_API_KEY=xxxxxxxx-xxxx-...        # your read key
    .venv/bin/python scripts/fetch_purpleair.py       # uses the config below
    # or override the window / box on the CLI:
    .venv/bin/python scripts/fetch_purpleair.py --start 2023-12-01 --end 2023-02-28

    Target WINTER (Dec-Feb) for SLC: the elevation gate only has signal during
    persistent cold-air-pool inversions. An annual pull averages the vertical
    gradient away and gives another Boulder-style null.

WIRING INTO THE PIPELINE
    build_graph2.py currently hard-codes Boulder paths + a GROUP_CONFIG. After a
    fetch you'll add an SLC entry there (or point COORDS_FILE / purple_air_dir at
    data/slc/...) and set SENSOR_SET to this GROUP. That's a separate ~5-line edit.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ===========================================================================
# CONFIG  -  edit these, or override the common ones on the CLI
# ===========================================================================
# Salt Lake Valley: valley floor (west/airport) up through the benches into the
# Wasatch canyon mouths -> ~700-800 m vertical spread within one air basin.
BBOX = dict(nwlat=40.85, nwlng=-112.10, selat=40.48, selng=-111.76)

START = "2023-01-01"   # inclusive (UTC). For SLC prefer a winter window.
END   = "2023-12-31"   # inclusive (UTC).

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "slc"
GROUP   = "urban"      # sub-folder under pm25/ AND the header name in the coords
                       # file. parse_sensor_coords maps any name containing
                       # "urban" -> group "urban", else -> "rural".

# Keep only outdoor sensors with recent data and a real altitude reading.
DROP_INDOOR      = True          # location_type == 1 is indoor
MAX_DAYS_STALE   = 365 * 3       # skip sensors not seen in this many days
MIN_ALTITUDE_FT  = -1000         # guard against null/garbage altitudes

# Density control. PurpleAir over-samples the valley floor; 197 raw sensors is
# far denser than the graph needs and costs points to download. Greedy spacing:
# keep a sensor only if it's >= MIN_SPACING_KM from every sensor already kept, so
# coverage stays even across the valley (and the floor->bench altitude range)
# instead of clumping downtown. 0 = keep all. --min-spacing-km overrides.
MIN_SPACING_KM   = 1.5

# API mechanics
BASE = "https://api.purpleair.com/v1"
HISTORY_AVERAGE  = 60            # minutes; matches "60-Minute Average" files
CHUNK_DAYS       = 14            # 60-min history is capped per request; chunk it
SLEEP_BETWEEN    = 1.0           # seconds between history calls (be polite)
TIMEOUT          = 60


# ===========================================================================
# tiny HTTP helper (stdlib only)
# ===========================================================================
def _get(path: str, params: dict, api_key: str) -> bytes:
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise SystemExit(f"[http {e.code}] {url}\n{body}")


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def thin_by_spacing(sensors: list[dict], min_km: float) -> list[dict]:
    """Greedily drop sensors closer than min_km to an already-kept one.

    Kept order is by descending altitude first, so the sparse high-elevation
    (bench/canyon) sensors -- the ones the gate actually needs -- always survive
    the thinning; the dense valley-floor cluster is what gets decimated.
    """
    if min_km <= 0:
        return sensors
    kept: list[dict] = []
    for s in sorted(sensors, key=lambda x: -x["alt_ft"]):
        if all(_haversine_km(s["lat"], s["lon"], k["lat"], k["lon"]) >= min_km
               for k in kept):
            kept.append(s)
    kept.sort(key=lambda x: x["id"])
    return kept


def _to_iso_utc(ts) -> str:
    """PurpleAir history time_stamp -> ISO-8601 UTC string.

    The CSV endpoint returns unix-epoch seconds; normalise so
    pd.to_datetime(..., utc=True) in build_graph2 parses it cleanly.
    """
    s = str(ts).strip()
    try:
        return datetime.fromtimestamp(int(float(s)), tz=timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return s  # already a date string; leave it


# ===========================================================================
# 1. sensor list for the box  ->  coords file
# ===========================================================================
def fetch_sensor_list(api_key: str) -> list[dict]:
    fields = ["latitude", "longitude", "altitude", "name",
              "location_type", "last_seen"]
    params = {"fields": ",".join(fields), **BBOX}
    payload = json.loads(_get("/sensors", params, api_key))
    cols = payload["fields"]
    idx = {c: i for i, c in enumerate(cols)}
    id_i = idx["sensor_index"]

    now = time.time()
    out = []
    for row in payload["data"]:
        lat, lon = row[idx["latitude"]], row[idx["longitude"]]
        alt = row[idx["altitude"]]
        if lat is None or lon is None or alt is None:
            continue
        if alt < MIN_ALTITUDE_FT:
            continue
        if DROP_INDOOR and row[idx["location_type"]] == 1:
            continue
        last_seen = row[idx["last_seen"]]
        if last_seen and (now - last_seen) > MAX_DAYS_STALE * 86400:
            continue
        out.append({
            "id": int(row[id_i]),
            "lat": float(lat),
            "lon": float(lon),
            "alt_ft": float(alt),   # PurpleAir altitude is already in feet
        })
    out.sort(key=lambda s: s["id"])
    return out


def write_coords_file(sensors: list[dict], path: Path, group: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_name = "Urban" if "urban" in group.lower() else "Rural"
    lines = ["Node Values (Sensor Index, Latitude, Longitude, Altitude)", "",
             header_name, "["]
    for s in sensors:
        lines.append(f"  [{s['id']},{s['lat']:.5f},{s['lon']:.5f},{s['alt_ft']:.0f}],")
    lines.append("]")
    path.write_text("\n".join(lines) + "\n")
    print(f"[coords] {len(sensors)} sensors -> {path}")


# ===========================================================================
# 2. per-sensor history  ->  "<id> <start> <end> 60-Minute Average.csv"
# ===========================================================================
def _chunks(start: datetime, end: datetime, days: int):
    step = days * 86400
    t = start.timestamp()
    end_ts = end.timestamp()
    while t < end_ts:
        lo = t
        hi = min(t + step, end_ts)
        yield int(lo), int(hi)
        t = hi


def fetch_history(sensor_id: int, start: datetime, end: datetime,
                  api_key: str) -> list[tuple[str, str]]:
    """Return [(time_stamp_iso, pm25), ...] across all chunks, time-sorted."""
    rows: dict[str, str] = {}
    for lo, hi in _chunks(start, end, CHUNK_DAYS):
        params = {
            "fields": "pm2.5_atm",
            "average": HISTORY_AVERAGE,
            "start_timestamp": lo,
            "end_timestamp": hi,
        }
        raw = _get(f"/sensors/{sensor_id}/history/csv", params, api_key)
        reader = csv.reader(io.StringIO(raw.decode("utf-8", "replace")))
        header = next(reader, None)
        if not header:
            continue
        h = {name.strip(): k for k, name in enumerate(header)}
        if "time_stamp" not in h or "pm2.5_atm" not in h:
            continue
        ti, pi = h["time_stamp"], h["pm2.5_atm"]
        for r in reader:
            if len(r) <= max(ti, pi) or not r[ti]:
                continue
            rows[_to_iso_utc(r[ti])] = r[pi]
        time.sleep(SLEEP_BETWEEN)
    return sorted(rows.items())


def write_history_csv(sensor_id: int, rows, start_s: str, end_s: str,
                      out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sensor_id} {start_s} {end_s} 60-Minute Average.csv"
    with open(out_dir / fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_stamp", "pm2.5_atm"])
        for ts, pm in rows:
            w.writerow([ts, pm])
    return len(rows)


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a PurpleAir bbox into the "
                                             "nasa-sees data layout.")
    ap.add_argument("--start", default=START, help="YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--end", default=END, help="YYYY-MM-DD (UTC, inclusive)")
    ap.add_argument("--out", default=str(OUT_DIR), help="output city dir")
    ap.add_argument("--group", default=GROUP, help="pm25 subfolder / coords header")
    ap.add_argument("--api-key", default=os.environ.get("PURPLEAIR_API_KEY", ""),
                    help="or set PURPLEAIR_API_KEY")
    ap.add_argument("--min-spacing-km", type=float, default=MIN_SPACING_KM,
                    help="drop sensors within this many km of a kept one "
                         "(0 = keep all). Thins the dense valley floor.")
    ap.add_argument("--list-only", action="store_true",
                    help="write the coords file only; skip history downloads")
    a = ap.parse_args()
    group = a.group

    if not a.api_key:
        sys.exit("No API key. Set PURPLEAIR_API_KEY or pass --api-key.")

    out_dir = Path(a.out)
    start_dt = datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # make END inclusive by covering through the end of that day
    end_dt = datetime.strptime(a.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = end_dt.replace(hour=23, minute=59, second=59)

    print(f"[box] nwlat={BBOX['nwlat']} nwlng={BBOX['nwlng']} "
          f"selat={BBOX['selat']} selng={BBOX['selng']}")
    sensors = fetch_sensor_list(a.api_key)
    if not sensors:
        sys.exit("No sensors returned for that box/filters.")
    n_raw = len(sensors)
    sensors = thin_by_spacing(sensors, a.min_spacing_km)
    if a.min_spacing_km > 0:
        print(f"[thin] {n_raw} -> {len(sensors)} sensors "
              f"(>= {a.min_spacing_km} km apart)")
    alts = [s["alt_ft"] for s in sensors]
    print(f"[box] {len(sensors)} usable sensors, "
          f"altitude {min(alts):.0f}-{max(alts):.0f} ft "
          f"(spread {max(alts) - min(alts):.0f} ft "
          f"= {(max(alts) - min(alts)) * 0.3048:.0f} m)")

    write_coords_file(sensors, out_dir / "coords" / "sensor_lat_long_alt", group)
    if a.list_only:
        print("[done] --list-only: coords written, history skipped.")
        return

    pm_dir = out_dir / "pm25" / group
    total_rows = 0
    for n, s in enumerate(sensors, 1):
        rows = fetch_history(s["id"], start_dt, end_dt, a.api_key)
        got = write_history_csv(s["id"], rows, a.start, a.end, pm_dir)
        total_rows += got
        print(f"  [{n}/{len(sensors)}] sensor {s['id']:>10}  "
              f"{got:>5} hourly rows")
    print(f"[done] {len(sensors)} sensors, {total_rows} rows -> {pm_dir}")


if __name__ == "__main__":
    main()
