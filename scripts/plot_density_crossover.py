#!/usr/bin/env python3
"""Plot MAE(ours) vs MAE(IDW) against network density from the density sweep.

Reads experiments/logs/density_sweep/results.csv (config,N,subsample_seed,ours_mae,ours_std,
idw_mae,idw_std), averages over subsample-seeds per (config,N), and draws one MAE-vs-N
curve per OURS config plus the shared IDW baseline. Marks the crossover density where an
OURS config drops below IDW (the publishable "GNN beats kriging in the sparse regime").
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "experiments/logs/density_sweep" / "results.csv"
OUT = REPO / "experiments/logs/density_sweep" / "density_crossover.png"

# CVD-safe: blue vs orange (max separation), IDW = neutral gray reference floor.
C_KRIG, C_IDW_CFG, C_BASE = "#2a78d6", "#eb6834", "#8a8a86"
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
LABEL = {"krig": "OURS  (kriging-prior + corr)", "idw": "OURS  (IDW-prior + corr)"}
COLOR = {"krig": C_KRIG, "idw": C_IDW_CFG}


def agg(df, col):
    # area (mi^2) of the central-Fresno box for a density axis label; N is the primary x.
    g = df.groupby("N")[col]
    return g.mean(), g.std().fillna(0.0)


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV} -- run run_density_sweep.sh first")
    df = pd.read_csv(CSV)
    if df.empty:
        sys.exit("results.csv has no rows yet")

    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=150)
    ax.set_facecolor(SURF); fig.patch.set_facecolor(SURF)

    # shared IDW baseline (identical subsets across configs -> average all IDW rows per N)
    idw_m, idw_s = agg(df, "idw_mae")
    xs = idw_m.index.to_numpy()
    ax.fill_between(xs, idw_m - idw_s, idw_m + idw_s, color=C_BASE, alpha=0.12, lw=0)
    ax.plot(xs, idw_m.to_numpy(), color=C_BASE, lw=2.4, marker="o", ms=6, zorder=3,
            label="IDW baseline (literature floor)")

    for cfg in ("krig", "idw"):
        sub = df[df.config == cfg]
        if sub.empty:
            continue
        m, s = agg(sub, "ours_mae")
        x = m.index.to_numpy()
        ax.fill_between(x, m - s, m + s, color=COLOR[cfg], alpha=0.13, lw=0)
        ax.plot(x, m.to_numpy(), color=COLOR[cfg], lw=2.0, marker="s", ms=5.5, zorder=4,
                label=LABEL[cfg])
        # mark crossover: densities where OURS < IDW (beats the floor)
        base = idw_m.reindex(m.index)
        wins = m[m < base]
        if len(wins):
            ax.scatter(wins.index, wins.to_numpy(), s=150, facecolors="none",
                       edgecolors=COLOR[cfg], linewidths=2.2, zorder=5)

    ax.axhline(2.38, color="#199e70", lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.text(xs.max(), 2.38, "  GraPhy 2.38", color="#199e70", va="center",
            ha="left", fontsize=9, fontweight="bold")

    ax.set_xlabel("network density  (# sensors in central-Fresno box)", color=INK, fontsize=11)
    ax.set_ylabel("inductive MAE  (µg/m³, lower better)", color=INK, fontsize=11)
    ax.set_title("Where does the learned correction beat IDW?  MAE vs density",
                 color=INK, fontsize=12.5, fontweight="bold", pad=12)
    ax.tick_params(colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d9d8d4")
    ax.grid(axis="y", color="#ececea", lw=1, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    ax.annotate("circled = OURS < IDW (beats the floor)", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURF)
    print(f"[plot] -> {OUT}")
    # also dump the aggregated table
    print("\nN   IDW    OURS-krig  OURS-idw")
    for n in xs:
        k = df[(df.config == "krig") & (df.N == n)]["ours_mae"].mean()
        i = df[(df.config == "idw") & (df.N == n)]["ours_mae"].mean()
        print(f"{n:<3} {idw_m[n]:.3f}  {k:8.3f}  {i:8.3f}")


if __name__ == "__main__":
    main()
