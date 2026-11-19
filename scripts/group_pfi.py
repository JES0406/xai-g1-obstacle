"""Global explanation 1: group permutation feature importance (offline).

    python scripts/group_pfi.py                     # Test-ID + Test-ID-0 frames, B = 10

For every input group (image, speed, heading, previous action), replace it by
the same group of a frame from another episode and measure the mean |change|
of vx, vy, yaw and V. Writes results/group_pfi.csv and
results/figures/group_pfi.png.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src import plotstyle  # noqa: E402
from src.attribution import GROUPS, group_pfi, load_frames  # noqa: E402
from src.load_policy import SteeringNet, load_ppo  # noqa: E402

LABELS = {"image": "image", "speed": "user speed", "heading": "heading", "prev_action": "previous action"}
TITLES = {"vx": "μ vx", "vy": "μ vy", "V": "V (critic)"}


def figure(df, path):
    plotstyle.use()
    scalars = ["vx", "vy", "V"]
    fig, axes = plt.subplots(1, len(scalars), figsize=(7.2, 2.4), sharey=True, layout="constrained")
    y = np.arange(len(GROUPS))[::-1]
    for ax, s in zip(axes, scalars):
        d = df[df.scalar == s].set_index("group").loc[list(GROUPS)]
        ax.barh(y, d["relative"], height=0.55, color=plotstyle.SERIES[0])
        err = np.vstack([d["relative"] - d["lo"] / d["output_std"], d["hi"] / d["output_std"] - d["relative"]])
        ax.errorbar(d["relative"], y, xerr=err, fmt="none", ecolor=plotstyle.INK_2, elinewidth=1, capsize=2)
        for yi, v in zip(y, d["relative"]):
            ax.text(v, yi, f"  {v:.2f}", va="center", fontsize=8, color=plotstyle.INK_2)
        ax.set_title(TITLES[s])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(1.0, d["hi"].div(d["output_std"]).max() * 1.25))
    axes[0].set_yticks(y, [LABELS[g] for g in GROUPS])
    fig.supxlabel("mean |Δ output| when the group is permuted, relative to the output's std", fontsize=9,
                  color=plotstyle.INK_2)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--sets", nargs="+", default=["test_id", "test_id_0"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", default=str(REPO / "results"))
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    frames = load_frames(args.data, sets=args.sets)
    print(f"{len(frames)} frames from {len(np.unique(frames.episode))} episodes ({', '.join(args.sets)})")
    net = SteeringNet(load_ppo().policy).eval()
    df = group_pfi(net, frames, repeats=args.repeats)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "group_pfi.csv", index=False)
    figure(df, out / "figures" / "group_pfi.png")
    print(df[df.scalar != "yaw"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"saved {out / 'group_pfi.csv'} and {out / 'figures' / 'group_pfi.png'}")


if __name__ == "__main__":
    main()
