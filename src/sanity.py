"""Checks on the explanations: localisation against the box segmentation (q_RRA),
and later parameter randomisation, ROAD, agreement between methods and
stability (q_LLE / q_AS).

Localisation, per frame with at least one box pixel:
    s        box pixels (union of visible boxes, logged mask)
    q_RRA    |top-|s| pixels of the map ∩ s| / |s|  (relevance rank accuracy,
             = q_Top-K with K = |s|); ranks the signed map, so only positive
             evidence counts
    random   expected q_RRA of a random map: |s| / d
    inside   share of the positive attribution that falls inside s
"""

import numpy as np
import pandas as pd


def localisation(maps, masks):
    """Per-frame q_RRA, its random reference and the positive mass inside the
    boxes. maps (N, H, W) float, masks (N, H, W) int (box index, -1 none).
    Frames without box pixels get NaN."""
    n = len(maps)
    flat = maps.reshape(n, -1)
    inside_box = (masks >= 0).reshape(n, -1)
    k = inside_box.sum(1)
    d = flat.shape[1]
    q = np.full(n, np.nan)
    positive_inside = np.full(n, np.nan)
    for i in range(n):
        if k[i] == 0:
            continue
        top = np.argpartition(-flat[i], k[i] - 1)[:k[i]]
        q[i] = inside_box[i, top].mean()
        pos = np.clip(flat[i], 0, None)
        total = pos.sum()
        positive_inside[i] = pos[inside_box[i]].sum() / total if total > 0 else np.nan
    return pd.DataFrame({"q_rra": q, "random": np.where(k > 0, k / d, np.nan),
                         "positive_inside": positive_inside, "box_pixels": k})
