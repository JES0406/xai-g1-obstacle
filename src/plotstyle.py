"""Shared matplotlib style for the report figures (light, print).

Categorical colours in fixed order, checked for colour-blind separation;
identity is never colour alone: every series also gets a marker and a direct label.
"""

import matplotlib as mpl

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def use():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })
