"""Shared drawing helpers for the architecture flowcharts (box + arrow diagrams,
matplotlib-only so they render as clean vector-ish PNGs at poster DPI).

Palette mirrors GraPhy's figure so our diagram reads as a sibling of theirs:
    diffusion = orange, convection = blue, local = purple, prior = green.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FS = 11  # single uniform font size, matches poster-graphs/_style.py

# module / stream colours  (edge, fill)
ORANGE = ("#D98A1F", "#FbEcD4")   # diffusion
BLUE   = ("#3F6FB0", "#DBE6F4")   # convection
PURPLE = ("#7A66AE", "#E7E1F1")   # local
GREEN  = ("#3F7E4E", "#DCEBDF")   # physics prior / fusion band
GRAY   = ("#7A7A7A", "#F1F1F1")   # neutral sub-box
INK    = ("#333333", "#FFFFFF")   # input / plain box
GOLD   = ("#B8892A", "#F6EFDD")   # gates / modulators


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": FS,
        "axes.titleweight": "normal",
        # nicer inline equations: proper italic vars, subscripts, Greek.
        # "stixsans" keeps the sans look of the boxes; use "cm" for a
        # classic Computer-Modern / LaTeX feel.
        "mathtext.fontset": "stixsans",
        "mathtext.default": "regular",
    })


def box(ax, x, y, w, h, text, color=INK, fs=FS, lw=1.4, round=0.02,
        weight="normal", style="normal", ha="center", z=2, text_color="#111"):
    """Draw a rounded rectangle centred at (x, y) with wrapped text inside."""
    edge, fill = color
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0.0,rounding_size={round}",
                       linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z)
    ax.add_patch(p)
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, zorder=z + 1,
            weight=weight, style=style, color=text_color, wrap=True)
    return x, y, w, h


def titled_box(ax, x, y, w, h, title, body, color=INK, fs=FS, title_fs=None,
               lw=1.4, round=0.02, z=2, text_color="#111", gap=0.30):
    """Box with a BOLD concept title stacked above a lighter equation/detail body.

    `gap` is the fraction of the box height between the title baseline and the
    body block (title sits in the upper band, body in the lower).
    """
    edge, fill = color
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0.0,rounding_size={round}",
                       linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z)
    ax.add_patch(p)
    ax.text(x, y + h * gap, title, ha="center", va="center",
            fontsize=(title_fs or fs) + 1.2, weight="bold", color=text_color,
            zorder=z + 1)
    ax.text(x, y - h * (gap * 0.62), body, ha="center", va="center",
            fontsize=fs, weight="normal", color="#5a5a5a", zorder=z + 1)
    return x, y, w, h


def label(ax, x, y, text, fs=FS, color="#111", weight="normal", ha="center",
          style="normal", z=5):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            weight=weight, style=style, zorder=z)


def arrow(ax, p0, p1, color="#444", lw=1.6, style="-|>", rad=0.0, z=1, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                        lw=lw, color=color, zorder=z,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)


def band(ax, x, y, w, h, color=GREEN, lw=1.4, round=0.03, z=0, alpha=1.0):
    """A light grouping panel drawn behind boxes."""
    edge, fill = color
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0.0,rounding_size={round}",
                       linewidth=lw, edgecolor=edge, facecolor=fill,
                       zorder=z, alpha=alpha)
    ax.add_patch(p)
