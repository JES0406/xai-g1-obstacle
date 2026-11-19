"""Evaluation metrics of the rollouts, with 95 % confidence intervals.

Episode outcomes (success, box hit, fall) are proportions: Wilson interval.
Continuous per-episode quantities (minimum clearance, progress, speed ratio)
are averaged over episodes: percentile bootstrap.

    from src.metrics import episode_frame, table
    df = episode_frame("data/raw")          # one row per episode
    print(table(df, by=["set", "boxes"]))
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

Z = 1.96


def wilson(k, n):
    """Proportion k/n with its 95 % Wilson interval: (p, low, high)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean(x, resamples=2000, seed=0):
    """Mean of x with a 95 % percentile-bootstrap interval: (mean, low, high)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(resamples, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def episode_frame(root):
    """One row per episode: the job, its outcome and per-episode summaries of
    the step logs (images are not loaded)."""
    rows = []
    for path in sorted(Path(root).glob("*/b*_s*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            mean, clearance = f["mean"], f["clearance"]
            qvel, state, command = f["qvel"], f["state"], f["command"]
        speed = meta["speed"]
        heading = np.arctan2(state[:, 1], state[:, 2])
        forward = qvel[:, 0] * np.cos(heading) + qvel[:, 1] * np.sin(heading)   # base speed along its heading
        policy = meta["baseline"] in ("", "black", "noise")
        rows.append({
            **{k: meta[k] for k in ("set", "variant", "baseline", "boxes", "seed", "speed", "steps",
                                    "success", "box_hit", "box_hit_part", "fall", "progress", "lateral", "end_y")},
            "min_clearance": float(np.min(clearance)) if np.isfinite(clearance).any() else np.nan,
            "speed_ratio": float(forward[10:].mean() / speed) if len(forward) > 10 else np.nan,  # after 1 s of start-up
            "correction_vx": float((command[:, 0] - speed).mean()),
            "correction_vy": float(command[:, 1].mean()),
            "correction_yaw": float(command[:, 2].mean()),
            "saturated_vx": float((np.abs(mean[:, 0]) > 1).mean()) if policy else np.nan,
            "saturated_vy": float((np.abs(mean[:, 1]) > 1).mean()) if policy else np.nan,
            "frames": len(clearance),
        })
    return pd.DataFrame(rows)


RATES = ("success", "box_hit", "fall")
MEANS = ("min_clearance", "progress", "speed_ratio", "end_y", "saturated_vx", "saturated_vy")


def table(df, by=("set", "boxes"), rates=RATES, means=MEANS):
    """Per group: n, each rate with its Wilson CI, each mean with its bootstrap CI.
    Columns <name>, <name>_lo, <name>_hi."""
    out = []
    for key, g in df.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key), n=len(g))
        for r in rates:
            row[r], row[f"{r}_lo"], row[f"{r}_hi"] = wilson(int(g[r].sum()), len(g))
        for m in means:
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = bootstrap_mean(g[m].to_numpy())
        out.append(row)
    return pd.DataFrame(out)


def command_deviation(df):
    """Test-ID-0 (no boxes): how far the policy moves the robot from the user's
    command when there is nothing to avoid."""
    g = df[df["set"] == "test_id_0"]
    rows = []
    for m in ("speed_ratio", "progress", "end_y", "correction_vx", "correction_vy", "correction_yaw"):
        mean, lo, hi = bootstrap_mean(g[m].to_numpy())
        rows.append({"quantity": m, "mean": mean, "lo": lo, "hi": hi, "n": len(g)})
    return pd.DataFrame(rows)


def fmt(p, lo, hi, pct=True):
    """'84 % [76, 90]' or '0.79 [0.74, 0.84]'."""
    if pct:
        return f"{p:.0%} [{lo * 100:.0f}, {hi * 100:.0f}]"
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"
