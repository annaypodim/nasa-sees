"""Driver: run src/viz/topo_correlation for Fresno and SLC without editing
shared source. Uses bg.use_city() and patches the per-city DEM cache path
(the script otherwise hardcodes Boulder's cache, which would collide)."""
import shutil
from pathlib import Path

from src.graph import build_graph2 as bg
from src.viz import data_visualizations as dv
from src.viz import topo_correlation as tc

POSTER_DIR = Path(__file__).resolve().parent

for city in ["fresno", "slc"]:
    print(f"\n===== {city.upper()} =====")
    bg.use_city(city)
    bg.SENSOR_SET = "urban"
    # per-city DEM cache so Fresno/SLC don't clobber each other or Boulder
    tc.DEM_CACHE = bg.DATA_DIR / city / "dem" / f"dem_{city}.npz"
    tc.DEM_CACHE.parent.mkdir(parents=True, exist_ok=True)

    pm_wide, coords, ids = dv.load(apply_preprocess=True)
    tc.fig_topo_correlation(pm_wide, coords, ids, tc.DEFAULT_ZOOM)

    # copy the just-saved figure into poster-graphs/
    src = dv.VIZ_DIR / f"{dv._prefix()}_8_topo_correlation.png"
    dst = POSTER_DIR / f"topo_corr_{city}.png"
    shutil.copy(src, dst)
    print(f"copied -> {dst}")
