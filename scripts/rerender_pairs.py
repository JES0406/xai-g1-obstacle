"""ID/OOD pairs with the same geometry: logged Test-ID states re-rendered under
the training look and under each OOD variant (GPU: run on the lab machine).

    MUJOCO_GL=egl python scripts/rerender_pairs.py                 # ~60 states x 4 looks
    env/wsl_gl.sh python scripts/rerender_pairs.py --per-kind 3    # laptop check

States: before braking and mid-avoidance (from the episodes with image) and
the 3-5 steps before box hits (from any Test-ID episode: qpos is logged for
all). Every member of a pair is re-rendered; outputs and maps are recomputed
on the re-rendered images. Writes results/pairs.npz, results/pairs.csv and
results/figures/pairs_<scalar>.png.
"""

import argparse
import json
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
from src.attribution import load_frames, local_maps, scalar_outputs, select_frames  # noqa: E402
from src.load_policy import SteeringNet, load_ppo, make_env  # noqa: E402
from src.metrics import bootstrap_mean  # noqa: E402
from src.sanity import localisation  # noqa: E402
from src.scenarios import VARIANTS, Look, Segmenter, rerender  # noqa: E402

MAP_METHODS = ("gradcam_conv2", "ig_blur", "occlusion")
NAMES = {"id": "training look", "box_colour": "blue boxes", "floor_tint": "floor tint", "lighting": "lighting"}


def states(data, per_kind, seed=0):
    """(kind, episode path, step) for the states to re-render."""
    frames = load_frames(data, sets=["test_id"])
    picks = select_frames(frames, per_kind=per_kind, seed=seed)
    out = [(kind, frames.episodes[frames.episode[i]]["path"], int(frames.step[i]))
           for kind in ("onset", "avoiding") for i in picks[kind]]
    hits = []
    for path in sorted(Path(data, "test_id").glob("b*_s*.npz")):
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
        if meta["box_hit"] and meta["steps"] >= 5:
            hits.append((str(path), meta["steps"]))
    rng = np.random.default_rng(seed)
    for k in rng.permutation(len(hits))[:per_kind]:
        path, steps = hits[k]
        out.append(("collision", path, steps - 4))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--per-kind", type=int, default=20)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--out", default=str(REPO / "results"))
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    todo = states(args.data, args.per_kind)
    print(f"{len(todo)} states x {len(VARIANTS)} looks")
    env = make_env(boxes=0)
    look, seg = Look(env), Segmenter(env)
    images = np.zeros((len(todo), len(VARIANTS), 64, 64, 3), np.uint8)
    masks = np.zeros((len(todo), len(VARIANTS), 64, 64), np.int8)
    state = np.zeros((len(todo), 6), np.float32)
    for i, (kind, path, step) in enumerate(todo):
        with np.load(path) as f:
            qpos, centres, half = f["qpos"][step], f["box_centres"], f["box_half"]
            state[i] = f["state"][step]
        for v, variant in enumerate(VARIANTS):
            images[i, v], masks[i, v] = rerender(env, look, seg, qpos, centres, half, variant)
    look.apply("id")
    seg.close()
    env.close()

    net = SteeringNet(load_ppo().policy).eval()
    fill = images[:, 0].reshape(-1, 3).mean(0)
    rows, saved = [], {"images": images, "masks": masks, "state": state,
                       "kind": np.array([k for k, _, _ in todo]), "step": np.array([s for _, _, s in todo]),
                       "episode": np.array([p for _, p, _ in todo])}
    for v, variant in enumerate(VARIANTS):
        outs = scalar_outputs(net, images[:, v], state)
        saved[f"outputs/{variant}"] = outs
        per = {"variant": variant, "mu_vx": outs[:, 0], "mu_vy": outs[:, 1], "V": outs[:, 3]}
        for scalar in ("vx", "vy"):
            for method in MAP_METHODS:
                m = local_maps(net, images[:, v], state, scalar, method, steps=args.steps, fill=fill)
                saved[f"maps/{variant}/{scalar}/{method}"] = m
                loc = localisation(m, masks[:, v])
                per[f"q_rra_{scalar}_{method}"] = loc.q_rra.to_numpy()
                per[f"random_{scalar}"] = loc.random.to_numpy()
        df = pd.DataFrame(per)
        df.insert(0, "pair", np.arange(len(todo)))
        df.insert(1, "kind", saved["kind"])
        rows.append(df)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out / "pairs.csv", index=False)
    np.savez_compressed(out / "pairs.npz", **saved)

    print("\nChange from the training look on the same states (mean [95 % CI]):")
    base = table[table.variant == "id"].set_index("pair")
    for variant in VARIANTS[1:]:
        d = table[table.variant == variant].set_index("pair")
        parts = []
        for col in ("mu_vx", "mu_vy", "q_rra_vx_gradcam_conv2", "q_rra_vx_ig_blur", "q_rra_vx_occlusion"):
            m, lo, hi = bootstrap_mean((d[col] - base[col]).to_numpy())
            parts.append(f"{col} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {NAMES[variant]:<12} " + "  ".join(parts))
    for scalar in ("vx", "vy"):
        pair_figure(saved, scalar, out / "figures" / f"pairs_{scalar}.png")
    print(f"saved {out / 'pairs.csv'}, {out / 'pairs.npz'} and figures/pairs_*.png")


def pair_figure(saved, scalar, path, examples=3):
    """For a few states: the image and the Grad-CAM conv2 / occlusion maps under each look."""
    from src.plotstyle import overlay

    plotstyle.use()
    kinds = saved["kind"]
    picks = [int(np.flatnonzero(kinds == k)[0]) for k in ("onset", "avoiding", "collision") if (kinds == k).any()][:examples]
    methods = ("gradcam_conv2", "occlusion")
    fig, axes = plt.subplots(len(picks) * (1 + len(methods)), len(VARIANTS),
                             figsize=(1.35 * len(VARIANTS), 1.35 * len(picks) * (1 + len(methods)) + 0.3), squeeze=False)
    for r, i in enumerate(picks):
        for v, variant in enumerate(VARIANTS):
            top = r * (1 + len(methods))
            ax = axes[top][v]
            ax.imshow(saved["images"][i, v], interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            mu = saved[f"outputs/{variant}"][i]
            ax.set_xlabel(f"μvx {mu[0]:.2f}  μvy {mu[1]:.2f}", fontsize=6, labelpad=1)
            if r == 0:
                ax.set_title(NAMES[variant], fontsize=7)
            for k, method in enumerate(methods, start=1):
                overlay(axes[top + k][v], saved["images"][i, v], saved[f"maps/{variant}/{scalar}/{method}"][i],
                        saved["masks"][i, v], signed=not method.startswith("gradcam"))
        axes[r * (1 + len(methods))][0].set_ylabel(saved["kind"][i], fontsize=7)
        for k, method in enumerate(methods, start=1):
            axes[r * (1 + len(methods)) + k][0].set_ylabel(method.replace("gradcam_", "Grad-CAM "), fontsize=6)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
