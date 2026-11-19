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


# -- Parameter randomisation (Adebayo et al., 2018), cascading -----------------------

def _cascade(policy):
    """(stage, module) from the output towards the input: actor head, actor MLP,
    then the CNN's linear layer and convolutions 3, 2, 1."""
    image = policy.features_extractor.extractors["image"]
    return [("action_head", policy.action_net), ("actor_mlp", policy.mlp_extractor.policy_net),
            ("cnn_linear", image.linear), ("conv3", image.cnn[4]), ("conv2", image.cnn[2]), ("conv1", image.cnn[0])]


def _reinit(module, seed):
    import torch

    torch.manual_seed(seed)
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def map_similarity(a, b):
    """Spearman rank correlation and SSIM between two maps, on absolute values
    (SSIM after scaling each map to [0, 1])."""
    from scipy.stats import spearmanr
    from skimage.metrics import structural_similarity

    a, b = np.abs(a), np.abs(b)
    if a.max() == 0 or b.max() == 0:
        return np.nan, np.nan
    rho = spearmanr(a.ravel(), b.ravel()).statistic
    return float(rho), float(structural_similarity(a / a.max(), b / b.max(), data_range=1.0))


def randomisation_cascade(net, image, state, scalars, methods, steps=256, fill=None, seed=0):
    """Maps of each method after randomising the network in cascade, compared
    with the original maps. A method that depends on the learned weights must
    lose similarity as more layers are randomised. Returns one row per stage,
    method, scalar and frame (stage 'original' = 1 by construction, omitted)."""
    import copy

    from src.attribution import local_maps
    from src.load_policy import SteeringNet

    original = {(s, m): local_maps(net, image, state, s, m, steps=steps, fill=fill) for s in scalars for m in methods}
    randomised = SteeringNet(copy.deepcopy(net.policy)).eval()
    rows = []
    for k, (stage, module) in enumerate(_cascade(randomised.policy)):
        _reinit(module, seed + k)
        for s in scalars:
            for m in methods:
                maps = local_maps(randomised, image, state, s, m, steps=steps, fill=fill)
                for i in range(len(maps)):
                    rho, ssim = map_similarity(original[(s, m)][i], maps[i])
                    rows.append({"stage": stage, "depth": k + 1, "method": m, "scalar": s, "frame": i,
                                 "spearman": rho, "ssim": ssim})
    return pd.DataFrame(rows)


# -- ROAD (Rong et al., 2022): remove the top-p % pixels with noisy linear imputation --

_NEIGHBOURS = [((1, 0), 1 / 6), ((-1, 0), 1 / 6), ((0, 1), 1 / 6), ((0, -1), 1 / 6),
               ((1, 1), 1 / 12), ((1, -1), 1 / 12), ((-1, 1), 1 / 12), ((-1, -1), 1 / 12)]


def noisy_linear_impute(image, removed, rng, noise=0.01):
    """ROAD's noisy linear imputation. image (H, W, C) float in [0, 1]; removed
    (H, W) bool. Each removed pixel becomes the weighted mean of its 8
    neighbours (direct 1/6, diagonal 1/12), solved jointly as a sparse linear
    system, plus Gaussian noise (std `noise`)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    H, W, C = image.shape
    ys, xs = np.nonzero(removed)
    n = len(ys)
    if n == 0:
        return image.copy()
    index = -np.ones((H, W), dtype=np.int64)
    index[ys, xs] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], []
    diag = np.zeros(n)
    rhs = np.zeros((n, C))
    for (dy, dx), w in _NEIGHBOURS:
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        diag[ok] += w
        known = ok.copy()
        known[ok] = index[ny[ok], nx[ok]] < 0
        rhs[known] += w * image[ny[known], nx[known]]
        unknown = ok & ~known
        rows.append(np.flatnonzero(unknown))
        cols.append(index[ny[unknown], nx[unknown]])
        vals.append(np.full(unknown.sum(), -w))
    vals = [diag] + vals
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
    solved = spsolve(A, rhs)
    out = image.copy()
    out[ys, xs] = np.clip(solved.reshape(n, C) + rng.normal(0, noise, size=(n, C)), 0, 1)
    return out


ROAD_LEVELS = tuple(range(0, 55, 5))     # % of pixels removed


def road_curves(net, image, state, scalar, maps, std, levels=ROAD_LEVELS, seed=0, sign=None):
    """|Δs| / std(s) after removing the top-p % pixels by each map's ranking
    (most important first), for every p in `levels`. maps: {name: (N, H, W)};
    pass a 'random' entry for the control order. Returns rows (name, frame, p, delta)."""
    import torch

    from src.attribution import oriented, to_tensors

    f = oriented(net, scalar, sign)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(image)):
        x = image[i].astype(np.float64) / 255
        with torch.no_grad():
            base = float(f(*to_tensors(image[i:i + 1].astype(np.float32), state[i:i + 1]))[0])
        for name, m in maps.items():
            order = np.argsort(-(m[i].ravel() + rng.uniform(0, 1e-9, m[i].size)))   # random tie-break
            batch = []
            for p in levels:
                removed = np.zeros(x.shape[0] * x.shape[1], bool)
                removed[order[:int(round(p / 100 * removed.size))]] = True
                batch.append(noisy_linear_impute(x, removed.reshape(x.shape[:2]), rng) * 255)
            with torch.no_grad():
                s = f(*to_tensors(np.array(batch, dtype=np.float32), np.repeat(state[i:i + 1], len(levels), 0))).numpy()
            for p, v in zip(levels, s):
                rows.append({"order": name, "frame": i, "p": p, "delta": abs(v - base) / std})
    return pd.DataFrame(rows)


def road_auc(curves):
    """Area under the mean |Δs|/std curve per order, p in [0, 0.5] (trapezoid)."""
    out = {}
    for name, g in curves.groupby("order"):
        m = g.groupby("p")["delta"].mean()
        out[name] = float(np.trapezoid(m.to_numpy(), m.index.to_numpy() / 100))
    return out


# -- Agreement between methods --------------------------------------------------------

def agreement(maps, top=0.1):
    """Per frame and pair of methods: Spearman correlation of the signed maps and
    overlap of their top-`top` pixels. maps: {method: (N, H, W)}."""
    from itertools import combinations

    from scipy.stats import spearmanr

    names = list(maps)
    n = len(next(iter(maps.values())))
    d = next(iter(maps.values()))[0].size
    k = int(round(top * d))
    rows = []
    for a, b in combinations(names, 2):
        for i in range(n):
            x, y = maps[a][i].ravel(), maps[b][i].ravel()
            rho = spearmanr(x, y).statistic if x.std() > 0 and y.std() > 0 else np.nan
            overlap = len(np.intersect1d(np.argpartition(-x, k - 1)[:k], np.argpartition(-y, k - 1)[:k])) / k
            rows.append({"a": a, "b": b, "frame": i, "spearman": rho, "top_overlap": overlap})
    return pd.DataFrame(rows)


# -- Stability: q_LLE and q_AS ---------------------------------------------------------

def perturbations(image, rng, M=20, sigmas=(1, 2), brightness=(0.95, 1.05)):
    """M perturbed copies of one uint8 image (float, 0-255): half Gaussian noise
    with sigma in `sigmas` (/255), half brightness scaled by U(brightness)."""
    x = image.astype(np.float32)
    out = []
    for j in range(M):
        if j % 2 == 0:
            out.append(np.clip(x + rng.normal(0, rng.choice(sigmas), x.shape), 0, 255))
        else:
            out.append(np.clip(x * rng.uniform(*brightness), 0, 255))
    return np.array(out, dtype=np.float32)


def stability(net, image, state, scalar, methods, std, M=20, tolerance=0.1, steps=256, fill=None, seed=0):
    """q_LLE and q_AS per frame and method (EE XAI 7 definitions), x in [0, 1]:
        q_LLE = max ||Φ(x) - Φ(x')||_2 / ||x - x'||_2
        q_AS  = mean ||Φ(x) - Φ(x')||_2 / ||x||_2
    over the kept perturbations x' (those that move the explained output by less
    than tolerance * std). Also scale-free versions with ||Φ(x)|| in the
    numerator's denominator, to compare methods whose maps differ in scale."""
    from src.attribution import local_maps, scalar_outputs, SCALARS

    rng = np.random.default_rng(seed)
    j = SCALARS.index(scalar)
    rows = []
    base_maps = {m: local_maps(net, image, state, scalar, m, steps=steps, fill=fill) for m in methods}
    for i in range(len(image)):
        xs = perturbations(image[i], rng, M)
        st = np.repeat(state[i:i + 1], M, 0)
        base_out = scalar_outputs(net, image[i:i + 1].astype(np.float32), state[i:i + 1])[0, j]
        keep = np.abs(scalar_outputs(net, xs, st)[:, j] - base_out) < tolerance * std
        if not keep.any():
            continue
        xs, st = xs[keep], st[keep]
        x0 = image[i].astype(np.float32) / 255
        dx = np.sqrt(((xs / 255 - x0) ** 2).reshape(len(xs), -1).sum(1))
        for m in methods:
            phi = base_maps[m][i]
            maps = local_maps(net, xs, st, scalar, m, steps=steps, fill=fill)
            dphi = np.sqrt(((maps - phi) ** 2).reshape(len(xs), -1).sum(1))
            scale = np.linalg.norm(phi) or np.nan
            rows.append({"frame": i, "method": m, "scalar": scalar, "kept": int(keep.sum()),
                         "q_lle": float(np.max(dphi / dx)), "q_as": float(np.mean(dphi) / np.linalg.norm(x0)),
                         "q_lle_rel": float(np.max(dphi / scale / dx)), "q_as_rel": float(np.mean(dphi) / scale / np.linalg.norm(x0)),
                         "map_change_rel": float(np.mean(dphi) / scale)})
    return pd.DataFrame(rows)


def shift(image, dy, dx):
    """Translate (H, W, ...) by (dy, dx) pixels, repeating the edge."""
    pad = [(max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0))] + [(0, 0)] * (image.ndim - 2)
    p = np.pad(image, pad, mode="edge")
    H, W = image.shape[:2]
    y0, x0 = max(-dy, 0), max(-dx, 0)
    return p[y0:y0 + H, x0:x0 + W]


def translation_sensitivity(net, image, state, scalar, methods, std, steps=256, fill=None):
    """1-px translations (4 directions), reported apart from q_LLE: change of the
    explained output (in stds) and relative change of the map after shifting
    it back into register (border row/column excluded)."""
    from src.attribution import local_maps, scalar_outputs, SCALARS

    j = SCALARS.index(scalar)
    rows = []
    base_out = scalar_outputs(net, image, state)[:, j]
    base_maps = {m: local_maps(net, image, state, scalar, m, steps=steps, fill=fill) for m in methods}
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        moved = np.array([shift(im, dy, dx) for im in image])
        out = scalar_outputs(net, moved, state)[:, j]
        for m in methods:
            maps = local_maps(net, moved, state, scalar, m, steps=steps, fill=fill)
            for i in range(len(image)):
                back = shift(maps[i], -dy, -dx)[1:-1, 1:-1]
                phi = base_maps[m][i][1:-1, 1:-1]
                scale = np.linalg.norm(phi) or np.nan
                rows.append({"frame": i, "method": m, "scalar": scalar, "shift": f"{dy},{dx}",
                             "output_change_std": float(abs(out[i] - base_out[i]) / std),
                             "map_change_rel": float(np.linalg.norm(back - phi) / scale)})
    return pd.DataFrame(rows)
