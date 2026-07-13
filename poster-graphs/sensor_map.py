"""Poster figure: Fresno & SLC sensor networks colored by the documented
28:4:9 inductive train/val/test split (seed 0), reproduced from
scripts/eval_inductive.py:split_nodes."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.lines import Line2D
from _style import apply, FS
apply()

# --- EDIT TITLE HERE ---
TITLE = "Sensor Networks & Inductive Split (28:4:9, seed 0)"

# --- real coordinates: [sensor_id, lat, lon, alt] from data/<city>/coords ---
FRESNO = [
    [473,36.72448,-119.70164],[18625,36.81895,-119.71606],[22751,36.80738,-119.79611],
    [27111,36.85110,-119.89152],[42327,36.74932,-119.70302],[46571,36.73728,-119.70873],
    [62137,36.84532,-119.73624],[64483,36.85043,-119.72064],[67603,36.82034,-119.76121],
    [85495,36.77714,-119.82079],[85615,36.83982,-119.83463],[87251,36.81605,-119.85010],
    [104230,36.72651,-119.73324],[111391,36.85807,-119.72782],[114721,36.79879,-119.82519],
    [118085,36.85515,-119.74519],[123359,36.83295,-119.81529],[139384,36.79923,-119.87532],
    [155829,36.75627,-119.80107],[163097,36.81932,-119.85510],[167071,36.75335,-119.79232],
    [197623,36.76304,-119.79864],[198161,36.76305,-119.79858],[198871,36.77409,-119.81482],
    [233743,36.81451,-119.75814],[236357,36.76763,-119.80016],[237167,36.72937,-119.68072],
    [240179,36.76045,-119.80259],[246355,36.74271,-119.80241],[246395,36.71642,-119.80082],
    [253369,36.80926,-119.74568],[296839,36.74136,-119.79464],[306814,36.76821,-119.79112],
]
SLC = [
    [5758,40.48866,-111.85717],[6288,40.70122,-111.96849],[6352,40.65652,-111.84560],
    [10808,40.50732,-111.89919],[18021,40.63794,-111.89346],[18115,40.61586,-111.85935],
    [18237,40.67275,-111.88601],[18469,40.73594,-111.80772],[20897,40.78279,-111.94669],
    [22647,40.57964,-111.80601],[39993,40.53756,-111.92394],[40713,40.84942,-111.87097],
    [42079,40.67282,-111.78331],[44157,40.57408,-111.95880],[46773,40.67895,-112.03383],
    [46863,40.63352,-112.04344],[47171,40.68980,-112.08269],[49155,40.73513,-111.88229],
    [89717,40.58134,-111.84770],[97391,40.76670,-111.90304],[105030,40.79087,-111.86294],
    [204009,40.59371,-111.89525],[205861,40.69498,-111.81860],[206185,40.49757,-112.03069],
    [235753,40.54298,-112.04023],[240129,40.55298,-111.99324],[243975,40.72264,-111.93582],
    [251023,40.75256,-111.84682],[262151,40.60068,-112.04675],[280142,40.61333,-111.81431],
    [283230,40.60684,-111.93956],[285111,40.66545,-111.98411],[298359,40.57040,-111.76149],
    [302286,40.61195,-112.00414],[303336,40.54382,-111.82624],
]

TRAIN, VAL, TEST = "#2a78d6", "#eb6834", "#0ca30c"


def split_nodes(N, seed=0):
    """Faithful copy of eval_inductive.py:split_nodes with the 28:4:9 scaling."""
    n_test = max(1, round(N * 9 / 41))
    n_val = max(1, round(N * 4 / 41))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    test = perm[:n_test]
    val = perm[n_test:n_test + n_val]
    train = perm[n_test + n_val:]
    lbl = np.empty(N, dtype=object)
    lbl[train] = "train"; lbl[val] = "val"; lbl[test] = "test"
    return lbl


def scale_bar(ax, lat0, lon0, km, y_frac=0.06):
    """Horizontal scale bar of `km` at lower-left, using local deg->km."""
    km_per_deg_lon = 111.320 * np.cos(np.radians(lat0))
    dlon = km / km_per_deg_lon
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    bx = x0 + 0.06 * (x1 - x0)
    by = y0 + y_frac * (y1 - y0)
    ax.plot([bx, bx + dlon], [by, by], color="#0b0b0b", lw=4, solid_capstyle="butt")
    ax.text(bx + dlon / 2, by + 0.015 * (y1 - y0), f"{km} km",
            ha="center", va="bottom", fontsize=FS)


def north_arrow(ax):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax_x = x1 - 0.08 * (x1 - x0)
    ax_y0 = y1 - 0.20 * (y1 - y0)
    ax.add_patch(FancyArrow(ax_x, ax_y0, 0, 0.11 * (y1 - y0), width=0.0,
                            head_width=0.02 * (x1 - x0), head_length=0.035 * (y1 - y0),
                            length_includes_head=True, color="#0b0b0b"))
    ax.text(ax_x, ax_y0 + 0.14 * (y1 - y0), "N", ha="center", va="bottom",
            fontsize=FS)


fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

for ax, data, name in [(axes[0], FRESNO, "Fresno, CA"),
                       (axes[1], SLC, "Salt Lake City, UT")]:
    arr = np.array(data, dtype=float)
    ids, lat, lon = arr[:, 0], arr[:, 1], arr[:, 2]
    lbl = split_nodes(len(arr), seed=0)
    for split, color in [("train", TRAIN), ("val", VAL), ("test", TEST)]:
        m = lbl == split
        ax.scatter(lon[m], lat[m], s=90, color=color, edgecolor="white",
                   linewidth=1.5, zorder=4)
    # keep geographic aspect
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xlabel("Longitude", fontsize=FS)
    ax.set_ylabel("Latitude", fontsize=FS)
    ax.set_title(f"{name}   (N={len(arr)})", fontsize=FS, pad=8)
    ax.tick_params(axis="both", labelsize=FS)
    ax.grid(True, color="#e1e0d9", lw=1)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    # pad limits a touch so bars/arrows don't clip
    dx = (lon.max() - lon.min()) * 0.10
    dy = (lat.max() - lat.min()) * 0.10
    ax.set_xlim(lon.min() - dx, lon.max() + dx)
    ax.set_ylim(lat.min() - dy, lat.max() + dy)
    scale_bar(ax, lat.mean(), lon.min(), km=5)
    north_arrow(ax)

# shared legend across both panels
handles = [Line2D([0], [0], marker="o", ls="", ms=9, mfc=TRAIN, mec="white", label="Train"),
           Line2D([0], [0], marker="o", ls="", ms=9, mfc=VAL, mec="white", label="Validation"),
           Line2D([0], [0], marker="o", ls="", ms=9, mfc=TEST, mec="white", label="Test")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FS,
           frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(TITLE, fontsize=FS, y=1.0)

fig.tight_layout(rect=[0, 0.05, 1, 0.97])
out = "sensor_map.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
