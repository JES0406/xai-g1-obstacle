"""Attribution for the steering policy: global (Group PFI, IG aggregated by input
group) and, later, local maps (Grad-CAM, IG, occlusion).

Outputs explained, always the raw mean before the clip and the EMA:
    vx, vy (the manoeuvre), yaw (control only) and V (the critic shares the CNN).

Input groups: the image, and the three parts of the 6-number state:
    speed        the user's commanded speed (constant within an episode)
    heading      sin and cos of the heading
    prev_action  the previous action, exactly clip(mean_{t-1})

Only episodes saved with image are used (the image and mask the policy saw).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCALARS = ("vx", "vy", "yaw", "V")
STATE_GROUPS = {"speed": [0], "heading": [1, 2], "prev_action": [3, 4, 5]}
GROUPS = ("image", *STATE_GROUPS)


@dataclass
class Frames:
    image: np.ndarray       # (N, 64, 64, 3) uint8
    state: np.ndarray       # (N, 6)
    mean: np.ndarray        # (N, 3) logged raw mean
    value: np.ndarray       # (N,)
    clearance: np.ndarray   # (N,) m to the nearest box face (inf: no boxes)
    mask: np.ndarray        # (N, 64, 64) int8 box index, -1 none
    episode: np.ndarray     # (N,) int, index into `episodes`
    step: np.ndarray        # (N,)
    set: np.ndarray         # (N,) str
    boxes: np.ndarray       # (N,)
    episodes: list          # meta dicts

    def __len__(self):
        return len(self.state)

    def subset(self, idx):
        idx = np.asarray(idx)
        fields = {k: getattr(self, k)[idx] for k in
                  ("image", "state", "mean", "value", "clearance", "mask", "episode", "step", "set", "boxes")}
        return Frames(**fields, episodes=self.episodes)


def load_frames(root, sets=None, policy_only=True):
    """All frames of the episodes saved with image, optionally only some sets.
    policy_only: skip baselines (their image is not the camera's)."""
    parts, episodes = [], []
    for path in sorted(Path(root).glob("*/b*_s*.npz")):
        if path.name.endswith(".tmp.npz") or (sets and path.parent.name not in sets):
            continue
        with np.load(path) as f:
            if "image" not in f.files:
                continue
            meta = json.loads(str(f["meta"]))
            if policy_only and meta["baseline"]:
                continue
            T = meta["steps"]
            e = len(episodes)
            episodes.append({**meta, "path": str(path)})
            parts.append({
                "image": f["image"], "state": f["state"], "mean": f["mean"], "value": f["value"],
                "clearance": f["clearance"], "mask": f["mask"], "episode": np.full(T, e),
                "step": np.arange(T), "set": np.full(T, meta["set"], dtype=object), "boxes": np.full(T, meta["boxes"]),
            })
    if not parts:
        raise FileNotFoundError(f"no episodes with image under {root} (sets={sets})")
    joined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return Frames(**joined, episodes=episodes)


# -- Forward ---------------------------------------------------------------------


def to_tensors(image, state):
    """uint8 HWC images and states -> float CHW (0-255) and float state tensors."""
    x = torch.as_tensor(np.ascontiguousarray(image)).permute(0, 3, 1, 2).float()
    return x, torch.as_tensor(np.asarray(state), dtype=torch.float32)


def scalar_outputs(net, image, state, batch=512):
    """(N, 4) array of [vx, vy, yaw, V] for uint8 images and states."""
    out = []
    with torch.no_grad():
        for i in range(0, len(state), batch):
            x, s = to_tensors(image[i:i + batch], state[i:i + batch])
            mean, value = net(x, s)
            out.append(torch.cat([mean, value], dim=1).numpy())
    return np.concatenate(out)


# -- Group permutation feature importance -------------------------------------------


def permutation_from_other_episodes(episode, rng, tries=100):
    """Index array: frame i takes its group from frame idx[i] of another episode.
    A random permutation, with the (few) same-episode draws re-drawn at random."""
    idx = rng.permutation(len(episode))
    for _ in range(tries):
        bad = np.flatnonzero(episode[idx] == episode)
        if len(bad) == 0:
            return idx
        idx[bad] = rng.integers(0, len(episode), size=len(bad))
    raise RuntimeError("could not avoid same-episode draws: too few episodes")


def group_pfi(net, frames, repeats=10, seed=0, batch=512):
    """Mean |output change| when one input group is replaced by that of a frame
    from another episode. Per group and scalar: the mean over `repeats`
    permutations with a 95 % t-interval, and the same relative to the
    output's standard deviation over the frames."""
    from scipy import stats

    rng = np.random.default_rng(seed)
    base = scalar_outputs(net, frames.image, frames.state, batch)
    spread = base.std(axis=0)
    deltas = {g: [] for g in GROUPS}
    for _ in range(repeats):
        for g in GROUPS:
            idx = permutation_from_other_episodes(frames.episode, rng)
            image, state = frames.image, frames.state
            if g == "image":
                image = frames.image[idx]
            else:
                state = frames.state.copy()
                cols = STATE_GROUPS[g]
                state[:, cols] = frames.state[idx][:, cols]
            out = scalar_outputs(net, image, state, batch)
            deltas[g].append(np.abs(out - base).mean(axis=0))
    rows = []
    t = stats.t.ppf(0.975, repeats - 1) if repeats > 1 else np.nan
    for g in GROUPS:
        d = np.array(deltas[g])                          # (repeats, 4)
        m, half = d.mean(axis=0), t * d.std(axis=0, ddof=1) / np.sqrt(repeats)
        for j, s in enumerate(SCALARS):
            rows.append({"group": g, "scalar": s, "delta": m[j], "lo": m[j] - half[j], "hi": m[j] + half[j],
                         "relative": m[j] / spread[j], "output_std": spread[j],
                         "frames": len(frames), "episodes": len(np.unique(frames.episode)), "repeats": repeats})
    return pd.DataFrame(rows)


# -- Integrated Gradients aggregated by input group ---------------------------------

CLEARANCE_BINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf)


def stratified_by_clearance(frames, per_bin, seed=0, no_box_frames=None):
    """Indices: up to `per_bin` frames per clearance bin (frames with boxes),
    plus up to `no_box_frames` (default per_bin) from episodes without boxes."""
    rng = np.random.default_rng(seed)
    chosen = []
    with_boxes = frames.boxes > 0
    for lo, hi in zip(CLEARANCE_BINS[:-1], CLEARANCE_BINS[1:]):
        pool = np.flatnonzero(with_boxes & (frames.clearance >= lo) & (frames.clearance < hi))
        chosen.append(rng.choice(pool, size=min(per_bin, len(pool)), replace=False))
    pool = np.flatnonzero(~with_boxes)
    k = per_bin if no_box_frames is None else no_box_frames
    chosen.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.sort(np.concatenate(chosen))


def image_baseline(kind, image):
    """IG baseline image (float CHW batch, 0-255) for float CHW images."""
    if kind == "black":
        return torch.zeros_like(image)
    if kind == "blur":
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(image.numpy(), sigma=(0, 0, 8, 8), mode="nearest")   # strong blur, per channel
        return torch.from_numpy(blurred)
    raise ValueError(kind)


def ig_by_group(net, frames, idx, scalar, baseline="black", state_baseline=None, steps=256,
                chunk=8, internal_batch=1024):
    """Integrated Gradients of one scalar for the frames `idx`, summed per input
    group. Returns a DataFrame, one row per frame: signed and absolute
    attribution per group, image share of the absolute total, and the
    completeness error |sum(attr) - (f(x) - f(baseline))| / |f(x) - f(baseline)|."""
    from captum.attr import IntegratedGradients

    j = SCALARS.index(scalar)

    def f(image, state):
        mean, value = net(image, state)
        return torch.cat([mean, value], dim=1)[:, j]

    ig = IntegratedGradients(f)
    state_ref = torch.as_tensor(state_baseline if state_baseline is not None else frames.state.mean(axis=0),
                                dtype=torch.float32)
    rows = []
    for start in range(0, len(idx), chunk):
        part = np.asarray(idx[start:start + chunk])
        x, s = to_tensors(frames.image[part], frames.state[part])
        bx, bs = image_baseline(baseline, x), state_ref.expand_as(s).clone()
        (a_img, a_st), delta = ig.attribute((x, s), baselines=(bx, bs), n_steps=steps,
                                            internal_batch_size=internal_batch, return_convergence_delta=True)
        with torch.no_grad():
            gap = (f(x, s) - f(bx, bs)).numpy()
        a_img = a_img.detach().numpy().reshape(len(part), -1)
        a_st = a_st.detach().numpy()
        signed = {"image": a_img.sum(1), **{g: a_st[:, c].sum(1) for g, c in STATE_GROUPS.items()}}
        absolute = {"image": np.abs(a_img).sum(1), **{g: np.abs(a_st[:, c]).sum(1) for g, c in STATE_GROUPS.items()}}
        total = sum(absolute.values())
        for k, i in enumerate(part):
            row = {"frame": int(i), "set": frames.set[i], "episode": int(frames.episode[i]), "step": int(frames.step[i]),
                   "boxes": int(frames.boxes[i]), "clearance": float(frames.clearance[i]), "scalar": scalar,
                   "baseline": baseline, "output": float(frames.mean[i, j]) if j < 3 else float(frames.value[i]),
                   "gap": float(gap[k]), "completeness_error": float(abs(delta[k]) / max(abs(gap[k]), 1e-8)),
                   "image_share": float(absolute["image"][k] / total[k]) if total[k] > 0 else np.nan}
            for g in GROUPS:
                row[f"abs_{g}"] = float(absolute[g][k])
                row[f"signed_{g}"] = float(signed[g][k])
            rows.append(row)
    return pd.DataFrame(rows)
