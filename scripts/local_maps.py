"""Local explanations: Grad-CAM (conv3, conv2), IG (black, blur), occlusion, on
automatically selected frames, and their localisation (q_RRA) over a
population of frames with a box in view.

    python scripts/local_maps.py                               # lab: figures + q_RRA on 500 frames per set
    python scripts/local_maps.py --frames 50 --steps 64        # quick check

Maps explain the oriented scalars s = -mean vx (evidence for braking) and
s = -mean vy (evidence for reducing the push to the left). Writes
results/local_maps.npz, results/figures/local_maps_<scalar>.png,
results/q_rra.csv (per set, method, scalar; all frames and non-saturated) and
results/q_rra_frames.csv.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src import plotstyle  # noqa: E402
from src.plotstyle import overlay  # noqa: E402
from src.attribution import METHODS, load_frames, local_maps, select_frames  # noqa: E402
from src.load_policy import SteeringNet, load_ppo  # noqa: E402
from src.metrics import bootstrap_mean  # noqa: E402
from src.sanity import localisation  # noqa: E402

TITLES = {"gradcam_conv3": "Grad-CAM conv3", "gradcam_conv2": "Grad-CAM conv2", "ig_black": "IG (black)",
          "ig_blur": "IG (blur)", "occlusion": "occlusion"}
KINDS = {"onset": "before braking", "avoiding": "mid-avoidance", "no_box": "no box in view", "collision": "before a hit"}
SETS = ["test_id", "test_ood_box_colour", "test_ood_floor_tint", "test_ood_lighting"]


def example_figure(frames, picks, maps, scalar, path):
    plotstyle.use()
    rows = [k for k in KINDS if len(picks.get(k, []))]
    cols = ["image", *METHODS]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.25 * len(cols), 1.3 * len(rows) + 0.3), squeeze=False)
    for r, kind in enumerate(rows):
        i = picks[kind][0]
        k = int(np.flatnonzero(maps["index"] == i)[0])
        ax = axes[r][0]
        ax.imshow(frames.image[i], interpolation="nearest")
        if (frames.mask[i] >= 0).any():
            ax.contour(frames.mask[i] >= 0, levels=[0.5], colors=["white"], linewidths=0.6)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(f"{KINDS[kind]}\nμvx {frames.mean[i, 0]:.2f}", fontsize=7)
        for c, method in enumerate(METHODS, start=1):
            overlay(axes[r][c], frames.image[i], maps[f"{scalar}/{method}"][k], frames.mask[i],
                    signed=not method.startswith("gradcam"))
    for c, name in enumerate(cols):
        axes[0][c].set_title(TITLES.get(name, name), fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--sets", nargs="+", default=SETS)
    parser.add_argument("--frames", type=int, default=500, help="box-visible frames per set for q_RRA")
    parser.add_argument("--scalars", nargs="+", default=["vx", "vy"])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--out", default=str(REPO / "results"))
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    net = SteeringNet(load_ppo().policy).eval()

    # 1. examples on the in-distribution frames
    frames = load_frames(args.data, sets=["test_id", "test_id_0"])
    fill = frames.image.reshape(-1, 3).mean(0)        # occlusion patch colour: mean pixel of ID frames
    picks = select_frames(frames, per_kind=10)
    print("selected:", {k: len(v) for k, v in picks.items()})
    chosen = np.unique(np.concatenate([v for v in picks.values() if len(v)]))
    saved = {"index": chosen, "fill": fill}
    for kind, v in picks.items():
        saved[f"kind/{kind}"] = v
    for scalar in args.scalars:
        for method in METHODS:
            saved[f"{scalar}/{method}"] = local_maps(net, frames.image[chosen], frames.state[chosen], scalar, method,
                                                     steps=args.steps, fill=fill)
        example_figure(frames, picks, saved, scalar, out / "figures" / f"local_maps_{scalar}.png")
    np.savez_compressed(out / "local_maps.npz", **saved)

    # 2. localisation over a population of box-visible frames, per set
    rows, per_frame = [], []
    for name in args.sets:
        fs = load_frames(args.data, sets=[name])
        visible = np.flatnonzero((fs.mask >= 0).reshape(len(fs), -1).any(1))
        idx = np.sort(np.random.default_rng(0).choice(visible, min(args.frames, len(visible)), replace=False))
        for scalar in args.scalars:
            j = ["vx", "vy"].index(scalar)
            saturated = np.abs(fs.mean[idx, j]) > 1
            for method in METHODS:
                maps = local_maps(net, fs.image[idx], fs.state[idx], scalar, method, steps=args.steps, fill=fill)
                loc = localisation(maps, fs.mask[idx])
                loc = loc.assign(set=name, scalar=scalar, method=method, frame=idx, saturated=saturated)
                per_frame.append(loc)
                for subset, keep in (("all", np.ones(len(idx), bool)), ("not_saturated", ~saturated)):
                    q, lo, hi = bootstrap_mean(loc.q_rra[keep])
                    r = float(np.nanmean(loc.random[keep])) if keep.any() else np.nan
                    p, plo, phi = bootstrap_mean(loc.positive_inside[keep])
                    rows.append({"set": name, "scalar": scalar, "method": method, "frames": subset, "n": int(keep.sum()),
                                 "q_rra": q, "q_rra_lo": lo, "q_rra_hi": hi, "random": r, "ratio": q / r if r else np.nan,
                                 "positive_inside": p, "positive_inside_lo": plo, "positive_inside_hi": phi})
                print(f"  {name:<22} {scalar} {method:>14}: q_RRA {rows[-2]['q_rra']:.3f} (random {rows[-2]['random']:.3f})")
    table = pd.DataFrame(rows)
    table.to_csv(out / "q_rra.csv", index=False)
    pd.concat(per_frame, ignore_index=True).to_csv(out / "q_rra_frames.csv", index=False)
    view = table[table.frames == "all"].pivot_table(index=["scalar", "method"], columns="set", values="ratio")
    print("\nq_RRA / random reference (all frames):")
    print(view.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"saved {out / 'q_rra.csv'}, {out / 'local_maps.npz'} and figures/local_maps_*.png")


if __name__ == "__main__":
    main()
