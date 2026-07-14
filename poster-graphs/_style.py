"""Shared plot style for the poster graphs: one small uniform font size,
plain weight -- matches the look of src/viz/topo_correlation.py."""
import matplotlib.pyplot as plt

FS = 11  # single font size used for EVERYTHING (title, labels, ticks, text)


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "figure.titlesize": FS,
        "axes.titleweight": "normal",
        "figure.titleweight": "normal",
    })
