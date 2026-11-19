"""Shared matplotlib style for the report figures (light, print).

Categorical colours in fixed order, checked for colour-blind separation;
identity is never colour alone: every series also gets a marker and a direct label.
"""

import matplotlib as mpl

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def _maps():
    from matplotlib.colors import LinearSegmentedColormap
    positive = LinearSegmentedColormap.from_list("pos", [(1, 1, 1, 0), (0.92, 0.41, 0.20, 1)])   # transparent -> orange
    signed = LinearSegmentedColormap.from_list("signed", [SERIES[0], "#f0efec", SERIES[1]])        # blue - grey - orange
    return positive, signed


def overlay(ax, image, m, mask, signed):
    """Attribution map over the grey image, box outline in ink. Positive evidence
    is orange in every map; signed maps use symmetric limits around grey."""
    import numpy as np

    positive, diverging = _maps()
    ax.imshow(image.mean(-1), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    if signed:
        lim = np.abs(m).max() or 1
        ax.imshow(m, cmap=diverging, vmin=-lim, vmax=lim, alpha=0.6, interpolation="nearest")
    else:
        ax.imshow(m, cmap=positive, vmin=0, vmax=m.max() or 1, interpolation="bilinear")
    if (mask >= 0).any():
        ax.contour(mask >= 0, levels=[0.5], colors=[INK], linewidths=0.6)
    ax.set_xticks([]), ax.set_yticks([])


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
