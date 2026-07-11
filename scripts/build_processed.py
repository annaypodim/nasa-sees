"""
Build the processed-data cache for every city-group.
=============================================================================
Runs the preprocessing pipeline (implausible-value masking, dud-sensor drop,
observed mask, input fill) on the raw PurpleAir CSVs and writes one traceable
artifact per city-group to data/<city>/processed/<group>/:

    pm_filled.csv   filled PM2.5 inputs   [timestamp x station_id]
    observed.csv    boolean observed mask [timestamp x station_id]
    meta.json       settings + kept/dropped ids that produced the tables

train.py loads these instead of recomputing, unless --rebuild is passed there.

run:  PYTHONPATH=. .venv/bin/python scripts/build_processed.py
"""
from src.graph import build_graph2 as bg
from src.graph import preprocessing as pp

# (city, sensor_set) pairs to build. Mirrors CITY_CONFIG's groups.
TARGETS = [
    ("boulder", "urban"),
    ("boulder", "rural"),
    ("slc", "urban"),
    ("pittsburgh", "urban"),
]


def build_one(city: str, sensor_set: str):
    bg.use_city(city)
    bg.SENSOR_SET = sensor_set
    cfg = bg.GROUP_CONFIG[sensor_set]

    coords = bg.parse_sensor_coords(bg.COORDS_FILE)
    coords = coords[coords["group"] == sensor_set]
    long_pm = bg.load_air_quality(cfg["purple_air_dir"], coords["station_id"].tolist())
    station_ids = sorted(set(coords["station_id"]) & set(long_pm["station_id"]))
    pm_raw = (
        long_pm.pivot_table(index="timestamp", columns="station_id", values="pm25")
        .reindex(columns=station_ids)
        .sort_index()
    )

    filled, observed, kept_ids, dropped = pp.preprocess(pm_raw)
    out = pp.save_processed(
        filled, observed, bg.DATA_DIR, city, sensor_set,
        dropped_ids=dropped,
        extra={"raw_sensors": int(pm_raw.shape[1])},
    )
    print(f"[saved] {city}-{sensor_set}: {filled.shape[0]} hours x "
          f"{filled.shape[1]} nodes -> {out}\n")


if __name__ == "__main__":
    for city, sset in TARGETS:
        print(f"=== {city}-{sset} ===")
        try:
            build_one(city, sset)
        except Exception as e:  # a missing city shouldn't abort the rest
            print(f"[skip] {city}-{sset}: {e}\n")
