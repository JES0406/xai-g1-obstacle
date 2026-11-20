"""Sanity checks of the explanations, offline on the logged Test-ID frames.

    python scripts/sanity_checks.py                        # lab: 200 frames, 256 IG steps, M = 20
    python scripts/sanity_checks.py --frames 12 --steps 32 --M 6   # quick check

1. Cascading parameter randomisation (Adebayo et al., 2018): Spearman and SSIM
   between original and randomised maps, layer by layer from the output.
2. ROAD faithfulness: |Δs| / std(s) after removing the top-p % pixels with
   noisy linear imputation, p = 0, 5, ..., 50 %; AUC per method vs random order.
3. Agreement between methods: Spearman and top-10 % overlap per frame.
4. Stability: q_LLE and q_AS (noise sigma 1-2/255 and brightness x[0.95, 1.05],
   only perturbations moving the output < 0.1 std), and 1-px translations apart.

Writes results/sanity_*.csv and results/figures/sanity_*.png.
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
from src import plotstyle, sanity  # noqa: E402
from src.attribution import METHODS, load_frames, local_maps  # noqa: E402
from src.load_policy import SteeringNet, load_ppo  # noqa: E402
from src.metrics import bootstrap_mean  # noqa: E402
from src.plotstyle import overlay  # noqa: E402

LABELS = {"gradcam_conv3": "Grad-CAM conv3", "gradcam_conv2": "Grad-CAM conv2", "ig_black": "IG (black)",
          "ig_blur": "IG (blur)", "occlusion": "occlusion", "random": "random order"}


def mean_ci(values):
    m, lo, hi = bootstrap_mean(values)
    return m, lo, hi


def fig_randomisation(df, path):
    plotstyle.use()
    stages = list(dict.fromkeys(df.stage))
    scalars = list(dict.fromkeys(df.scalar))
    fig, axes = plt.subplots(1, len(scalars), figsize=(3.6 * len(scalars), 3.1), sharey=True, squeeze=False, layout="constrained")
    for ax, s in zip(axes[0], scalars):
        x = np.arange(len(stages) + 1)
        for k, m in enumerate(METHODS):
            g = df[(df.method == m) & (df.scalar == s)]
            stats = [(1.0, 1.0, 1.0)] + [mean_ci(g.loc[g.stage == st, "spearman"]) for st in stages]
            mu, lo, hi = (np.array(v) for v in zip(*stats))
            ax.fill_between(x, lo, hi, color=plotstyle.SERIES[k], alpha=0.12, linewidth=0)
            ax.plot(x, mu, color=plotstyle.SERIES[k], marker=plotstyle.MARKERS[k], label=LABELS[m])
        ax.axhline(0, color=plotstyle.MUTED, linewidth=0.8)
        ax.set_xticks(x, ["original", *[st.replace("_", " ") for st in stages]], rotation=35, ha="right")
        ax.set_title(f"μ {s}: randomised from the output down")
    axes[0][0].set_ylabel("Spearman with original map (|map|)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(METHODS), fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def fig_road(curves, aucs, scalar, path):
    plotstyle.use()
    fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
    for k, name in enumerate([*METHODS, "random"]):
        g = curves[curves.order == name]
        stats = [mean_ci(g.loc[g.p == p, "delta"]) for p in sorted(g.p.unique())]
        mu, lo, hi = (np.array(v) for v in zip(*stats))
        p = np.array(sorted(g.p.unique()))
        colour = plotstyle.MUTED if name == "random" else plotstyle.SERIES[k]
        ax.fill_between(p, lo, hi, color=colour, alpha=0.12, linewidth=0)
        ax.plot(p, mu, color=colour, marker=None if name == "random" else plotstyle.MARKERS[k],
                linestyle="--" if name == "random" else "-", label=f"{LABELS[name]} (AUC {aucs[name]:.3f})")
    ax.set_xlabel("pixels removed, most attributed first (%)")
    ax.set_ylabel(f"|Δ s| / std(s),  s = -μ {scalar}")
    ax.set_title("ROAD (noisy linear imputation): higher AUC = more faithful", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def fig_agreement_examples(image, mask, maps, scores, path):
    """The frame where the methods agree most and the one where they agree least."""
    plotstyle.use()
    picks = [("agree", int(scores.idxmax())), ("disagree", int(scores.idxmin()))]
    fig, axes = plt.subplots(2, 1 + len(METHODS), figsize=(1.3 * (1 + len(METHODS)), 3.0), squeeze=False)
    for r, (name, i) in enumerate(picks):
        axes[r][0].imshow(image[i], interpolation="nearest")
        axes[r][0].set_xticks([]), axes[r][0].set_yticks([])
        axes[r][0].set_ylabel(f"{name}\nρ̄ {scores[i]:.2f}", fontsize=7)
        for c, m in enumerate(METHODS, start=1):
            overlay(axes[r][c], image[i], maps[m][i], mask[i], signed=not m.startswith("gradcam"))
            if r == 0:
                axes[r][c].set_title(LABELS[m], fontsize=7)
    axes[0][0].set_title("image", fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--scalars", nargs="+", default=["vx", "vy"])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--checks", nargs="+", default=["randomisation", "road", "agreement", "stability"])
    parser.add_argument("--out", default=str(REPO / "results"))
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.data, sets=["test_id"])
    visible = np.flatnonzero((frames.mask >= 0).reshape(len(frames), -1).any(1))
    idx = np.sort(np.random.default_rng(0).choice(visible, min(args.frames, len(visible)), replace=False))
    image, state, mask = frames.image[idx], frames.state[idx], frames.mask[idx]
    fill = frames.image.reshape(-1, 3).mean(0)
    std = {s: float(frames.mean[:, ["vx", "vy"].index(s)].std()) for s in args.scalars}
    print(f"{len(idx)} Test-ID frames with a box in view; std of mean vx/vy over Test-ID frames: {std}")
    net = SteeringNet(load_ppo().policy).eval()

    if "randomisation" in args.checks:
        df = sanity.randomisation_cascade(net, image, state, args.scalars, METHODS, steps=args.steps, fill=fill)
        df.to_csv(out / "sanity_randomisation.csv", index=False)
        fig_randomisation(df, out / "figures" / "sanity_randomisation.png")
        print(df.groupby(["scalar", "stage", "method"], sort=False)["spearman"].mean().unstack().round(2).to_string())

    maps = {s: {m: local_maps(net, image, state, s, m, steps=args.steps, fill=fill) for m in METHODS} for s in args.scalars}

    if "road" in args.checks:
        rows = []
        for s in args.scalars:
            orders = {**maps[s], "random": np.random.default_rng(1).uniform(size=image.shape[:3])}
            curves = sanity.road_curves(net, image, state, s, orders, std[s]).assign(scalar=s)
            aucs = sanity.road_auc(curves)
            fig_road(curves, aucs, s, out / "figures" / f"sanity_road_{s}.png")
            curves.to_csv(out / f"sanity_road_curves_{s}.csv", index=False)
            rows += [{"scalar": s, "order": k, "auc": v} for k, v in aucs.items()]
            print(f"ROAD AUC ({s}):", {k: round(v, 3) for k, v in aucs.items()})
        pd.DataFrame(rows).to_csv(out / "sanity_road_auc.csv", index=False)

    if "agreement" in args.checks:
        parts = []
        for s in args.scalars:
            ag = sanity.agreement(maps[s]).assign(scalar=s)
            parts.append(ag)
            per_frame = ag.groupby("frame")["spearman"].mean()
            fig_agreement_examples(image, mask, maps[s], per_frame, out / "figures" / f"sanity_agreement_{s}.png")
        ag = pd.concat(parts, ignore_index=True)
        ag.to_csv(out / "sanity_agreement.csv", index=False)
        print(ag.groupby(["scalar", "a", "b"])[["spearman", "top_overlap"]].mean().round(2).to_string())

    if "stability" in args.checks:
        sub = slice(0, min(len(idx), 200))
        parts, trans = [], []
        for s in args.scalars:
            parts.append(sanity.stability(net, image[sub], state[sub], s, METHODS, std[s], M=args.M, steps=args.steps, fill=fill))
            trans.append(sanity.translation_sensitivity(net, image[sub], state[sub], s, METHODS, std[s], steps=args.steps, fill=fill))
        st, tr = pd.concat(parts, ignore_index=True), pd.concat(trans, ignore_index=True)
        st.to_csv(out / "sanity_stability.csv", index=False)
        tr.to_csv(out / "sanity_translation.csv", index=False)
        rows = []
        for (s, m), g in st.groupby(["scalar", "method"]):
            row = {"scalar": s, "method": m, "frames": len(g)}
            for col in ("q_lle", "q_as", "q_lle_rel", "q_as_rel", "map_change_rel"):
                row[col], row[f"{col}_lo"], row[f"{col}_hi"] = mean_ci(g[col])
            t = tr[(tr.scalar == s) & (tr.method == m)]
            row["shift_map_change_rel"], row["shift_map_change_rel_lo"], row["shift_map_change_rel_hi"] = mean_ci(t.map_change_rel)
            row["shift_output_change_std"] = float(t.output_change_std.mean())
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(out / "sanity_stability_summary.csv", index=False)
        print(summary[["scalar", "method", "q_lle", "q_as", "map_change_rel", "shift_map_change_rel",
                       "shift_output_change_std"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"saved results/sanity_*.csv and results/figures/sanity_*.png under {out}")


if __name__ == "__main__":
    main()
