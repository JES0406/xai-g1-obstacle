"""Global explanation 2: Integrated Gradients aggregated by input group, against
the distance to the nearest box (temporal attribution over the population).

    python scripts/ig_groups.py                          # ~2000 Test-ID/Test-ID-0 frames, vx vy V, black + blur
    python scripts/ig_groups.py --per-bin 20 --steps 64  # quick check

Frames are stratified by clearance (bins in src/attribution.py) plus frames
from episodes without boxes. State baseline: the mean state of the frames
(state = 0 is out of distribution). Writes results/ig_groups.csv and
results/figures/ig_image_share.png, and prints the completeness errors.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src import plotstyle  # noqa: E402
from src.attribution import CLEARANCE_BINS, ig_by_group, load_frames, stratified_by_clearance  # noqa: E402
from src.load_policy import SteeringNet, load_ppo  # noqa: E402
from src.metrics import bootstrap_mean  # noqa: E402

NAMES = {"vx": "vx", "vy": "vy", "V": "V"}


def bin_labels():
    labels = []
    for lo, hi in zip(CLEARANCE_BINS[:-1], CLEARANCE_BINS[1:]):
        labels.append(f"{lo:g}–{hi:g}" if np.isfinite(hi) else f"≥{lo:g}")
    return labels + ["no boxes"]


def assign_bin(df):
    b = np.digitize(df["clearance"], CLEARANCE_BINS[1:-1])
    b[df["boxes"].to_numpy() == 0] = len(CLEARANCE_BINS) - 1
    return b


def figure(df, path):
    plotstyle.use()
    labels = bin_labels()
    baselines = list(dict.fromkeys(df["baseline"]))
    fig, axes = plt.subplots(1, len(baselines), figsize=(3.6 * len(baselines), 2.6), sharey=True, squeeze=False)
    for ax, base in zip(axes[0], baselines):
        for k, scalar in enumerate(NAMES):
            d = df[(df.baseline == base) & (df.scalar == scalar)]
            if d.empty:
                continue
            stats = [bootstrap_mean(d.loc[d.bin == b, "image_share"]) for b in range(len(labels))]
            m, lo, hi = (np.array(v) for v in zip(*stats))
            x = np.arange(len(labels))
            colour = plotstyle.SERIES[k]
            ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)
            ax.plot(x, m, color=colour, marker=plotstyle.MARKERS[k], label=NAMES[scalar])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        ax.set_xlabel("distance to nearest box (m)")
        ax.set_title(f"IG baseline: {base} image, mean state")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("image share of |attribution|")
    axes[0][0].legend(loc="lower left", ncols=3, title="output explained", title_fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--sets", nargs="+", default=["test_id", "test_id_0"])
    parser.add_argument("--per-bin", type=int, default=250, help="frames per clearance bin (8 bins: ~2000 frames)")
    parser.add_argument("--scalars", nargs="+", default=["vx", "vy", "V"])
    parser.add_argument("--baselines", nargs="+", default=["black", "blur"])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--out", default=str(REPO / "results"))
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    frames = load_frames(args.data, sets=args.sets)
    idx = stratified_by_clearance(frames, args.per_bin)
    print(f"{len(idx)} frames of {len(frames)} ({', '.join(args.sets)}), {args.steps} steps")
    net = SteeringNet(load_ppo().policy).eval()
    state_mean = frames.state.mean(axis=0)
    parts = []
    for base in args.baselines:
        for scalar in args.scalars:
            start = time.time()
            parts.append(ig_by_group(net, frames, idx, scalar, baseline=base, state_baseline=state_mean, steps=args.steps))
            err = parts[-1]["completeness_error"]
            print(f"  {base:>5} {scalar:>3}: {time.time() - start:6.0f} s   completeness error median {err.median():.1%}, p90 {err.quantile(0.9):.1%}")
    df = pd.concat(parts, ignore_index=True)
    df["bin"] = assign_bin(df)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ig_groups.csv", index=False)
    figure(df, out / "figures" / "ig_image_share.png")
    summary = df.groupby(["baseline", "scalar", "bin"])["image_share"].mean().unstack("bin")
    summary.columns = [bin_labels()[c] for c in summary.columns]
    print(summary.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"saved {out / 'ig_groups.csv'} and {out / 'figures' / 'ig_image_share.png'}")


if __name__ == "__main__":
    main()
