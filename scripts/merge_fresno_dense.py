#!/usr/bin/env python3
"""Merge the existing Fresno set with newly-fetched supplement sensors into
data/fresno_variants/fresno_dense (the density-matched set the winning eval runs on).

SUPPLEMENT philosophy: we never re-download the 33 sensors we already own; we
only fetch NEW ones (scripts/fetch_purpleair.py --exclude-existing data/fresno_variants/fresno
-> data/fresno_variants/fresno_extra). This script unions the two by sensor id:

  pm25   = data/fresno_variants/fresno/pm25/<grp>/*.csv  U  data/fresno_variants/fresno_extra/pm25/<grp>/*.csv
  coords = union of both coords files, keyed by id, keeping ONLY ids that have a
           CSV present in the merged pm25 dir (guarantees coords<->pm25 consistency
           and never drops an existing sensor just because it went stale in the
           fresh sensor-list call).

Idempotent: re-running rebuilds data/fresno_variants/fresno_dense from the two source dirs.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
GROUP = "urban"

BASE = DATA / "fresno_variants" / "fresno"          # the 33 we already own
EXTRA = DATA / "fresno_variants" / "fresno_extra"   # newly-fetched supplement sensors
DENSE = DATA / "fresno_variants" / "fresno_dense"   # merged output

COORD_RE = re.compile(r"\[\s*(\d+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]")


def parse_coords(path: Path) -> dict[int, tuple[float, float, float]]:
    """id -> (lat, lon, alt) from a 'sensor_lat_long_alt' file."""
    out: dict[int, tuple[float, float, float]] = {}
    if not path.exists():
        return out
    for m in COORD_RE.finditer(path.read_text()):
        sid = int(m.group(1))
        out[sid] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def write_coords(entries: dict[int, tuple[float, float, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Node Values (Sensor Index, Latitude, Longitude, Altitude)", "",
             "Urban", "["]
    for sid in sorted(entries):
        lat, lon, alt = entries[sid]
        lines.append(f"  [{sid},{lat:.5f},{lon:.5f},{alt:.0f}],")
    lines.append("]")
    path.write_text("\n".join(lines) + "\n")


def csv_id(name: str) -> int | None:
    head = name.split(" ", 1)[0]
    return int(head) if head.isdigit() else None


def main() -> None:
    base_pm = BASE / "pm25" / GROUP
    extra_pm = EXTRA / "pm25" / GROUP
    dense_pm = DENSE / "pm25" / GROUP
    if not base_pm.exists():
        sys.exit(f"missing base pm25 dir: {base_pm}")

    # clean-rebuild the merged pm25 dir
    if dense_pm.exists():
        shutil.rmtree(dense_pm)
    dense_pm.mkdir(parents=True)

    have_ids: set[int] = set()
    n_base = n_extra = 0
    for src, tag in [(base_pm, "base"), (extra_pm, "extra")]:
        if not src.exists():
            print(f"[warn] no {tag} pm25 dir ({src}) -- skipping")
            continue
        for f in sorted(src.glob("*.csv")):
            sid = csv_id(f.name)
            if sid is None:
                continue
            if sid in have_ids:      # base wins on collision (shouldn't happen)
                continue
            shutil.copy2(f, dense_pm / f.name)
            have_ids.add(sid)
            if tag == "base":
                n_base += 1
            else:
                n_extra += 1

    # union coords, keep only ids that actually have a CSV
    coords = {**parse_coords(EXTRA / "coords" / "sensor_lat_long_alt"),
              **parse_coords(BASE / "coords" / "sensor_lat_long_alt")}
    coords = {sid: v for sid, v in coords.items() if sid in have_ids}
    missing = have_ids - set(coords)
    if missing:
        print(f"[warn] {len(missing)} CSVs have no coords entry, dropped: "
              f"{sorted(missing)[:10]}")
        for sid in missing:
            (dense_pm / next(p.name for p in dense_pm.glob(f"{sid} *.csv"))).unlink()
    write_coords(coords, DENSE / "coords" / "sensor_lat_long_alt")

    print(f"[merge] base={n_base}  extra(new)={n_extra}  "
          f"-> {len(coords)} sensors in {DENSE}")
    print(f"[merge] coords -> {DENSE / 'coords' / 'sensor_lat_long_alt'}")


if __name__ == "__main__":
    main()
