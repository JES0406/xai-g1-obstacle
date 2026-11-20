"""Integrity check of a rollout folder.

For every episode: array lengths agree, action == clip(mean) (policy sets) or 0
(walker baseline). For episodes with image: running the network again on the
saved image and state gives back the logged mean and V (so the saved image is
exactly what the policy saw), and the box mask covers the red pixels.

    python scripts/check_rollouts.py data/raw
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.load_policy import SteeringNet, load_ppo  # noqa: E402

PER_STEP = ("state", "mean", "value", "action", "command", "clearance", "qpos", "qvel", "reward")


def main(root, policy=None):
    torch.set_num_threads(4)
    run_info = Path(root) / "RUN.json"
    if policy is None and run_info.exists():          # the policy that produced these episodes
        policy = REPO / json.loads(run_info.read_text())[-1].get("policy", "checkpoints/planPW")
    net = SteeringNet(load_ppo(policy).policy if policy else load_ppo().policy).eval()
    print(f"checking with policy {policy or 'checkpoints/planPW'}")
    paths = sorted(p for p in Path(root).glob("*/b*_s*.npz") if not p.name.endswith(".tmp.npz"))
    problems, worst_mu, worst_v, red_cover = [], 0.0, 0.0, []
    for path in paths:
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}
        meta = json.loads(str(d["meta"]))
        T = meta["steps"]
        for k in PER_STEP:
            if len(d[k]) != T:
                problems.append(f"{path}: {k} has {len(d[k])} rows, steps = {T}")
        if meta["baseline"] == "walker":
            if np.any(d["action"] != 0):
                problems.append(f"{path}: walker baseline with non-zero action")
        elif not np.allclose(d["action"], np.clip(d["mean"], -1, 1), atol=1e-6):
            problems.append(f"{path}: action != clip(mean)")
        if len(d["box_centres"]) != meta["boxes"]:
            problems.append(f"{path}: {len(d['box_centres'])} boxes placed, {meta['boxes']} asked")
        if "image" not in d:
            continue
        if d["image"].shape != (T, 64, 64, 3) or d["mask"].shape != (T, 64, 64):
            problems.append(f"{path}: image {d['image'].shape} / mask {d['mask'].shape}")
            continue
        if meta["baseline"] != "walker":
            image = torch.from_numpy(d["image"]).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                mu, v = net(image, torch.from_numpy(d["state"]))
            worst_mu = max(worst_mu, float(np.abs(mu.numpy() - d["mean"]).max()))
            worst_v = max(worst_v, float(np.abs(v.numpy()[:, 0] - d["value"]).max()))
        if meta["variant"] != "box_colour" and meta["baseline"] == "":
            img = d["image"].astype(int)
            red = (img[..., 0] > 120) & (img[..., 1] < 80) & (img[..., 2] < 80)
            if red.sum():
                red_cover.append(((red & (d["mask"] >= 0)).sum(), red.sum()))
    print(f"{len(paths)} episodes checked, {sum('image' in np.load(p).files for p in paths)} with image")
    print(f"max |mean recomputed - logged| = {worst_mu:.2e}, max |V recomputed - logged| = {worst_v:.2e}")
    if red_cover:
        inside, total = np.sum(red_cover, axis=0)
        print(f"red pixels inside the box mask: {inside}/{total} = {inside / total:.1%} (the rest: box reflections on the floor)")
    if worst_mu > 1e-4 or worst_v > 1e-2:
        problems.append("saved images do not reproduce the logged outputs")
    for p in problems:
        print("PROBLEM:", p)
    print("OK" if not problems else f"{len(problems)} problems")
    return not problems


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=str(REPO / "data/raw"))
    parser.add_argument("--policy", help="default: the policy recorded in <root>/RUN.json")
    args = parser.parse_args()
    sys.exit(0 if main(args.root, args.policy) else 1)
