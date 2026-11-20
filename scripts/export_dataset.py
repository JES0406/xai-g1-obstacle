"""Publish the rollout dataset: copy the episodes saved with image of each
evaluation set from data/raw/ to data/rollouts/ (tracked in git), write an
index of every episode, and generate data/README.md with the dataset
description (origin, size, types, preprocessing).

    python scripts/export_dataset.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import episode_frame  # noqa: E402

SETS = ["validation", "test_id", "test_id_0", "test_ood_box_colour", "test_ood_floor_tint", "test_ood_lighting"]
FIELDS = {
    "state": "commanded speed, sin and cos of the heading, previous action (3)",
    "mean": "raw policy mean (vx, vy, yaw) before the clip",
    "value": "critic value V(s)",
    "action": "action applied, clip(mean) to [-1, 1] (0 for the walker baseline)",
    "command": "velocity command the walker received (vx, vy, yaw rate) after the residual and the EMA",
    "clearance": "distance from the pelvis to the nearest box face, m (inf without boxes)",
    "qpos": "MuJoCo generalised positions of the robot (base pose + 29 joints)",
    "qvel": "base linear and angular velocity",
    "reward": "reward of the step",
    "image": "64x64 RGB image the policy saw (uint8)",
    "mask": "per-pixel box index from the geometry segmentation, -1 = no box (int8)",
    "box_centres": "centres of the boxes of the episode, m",
    "box_half": "half widths of the boxes, m",
    "meta": "JSON: set, variant, baseline, transform, boxes, seed, save_image, speed, steps and outcome",
}


def describe(npz_path):
    with np.load(npz_path) as f:
        return {k: (f[k].dtype.str, f[k].shape) for k in f.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--out", default=str(REPO / "data/rollouts"))
    args = parser.parse_args()
    src, out = Path(args.data), Path(args.out)

    copied, frames = [], 0
    for name in SETS:
        for path in sorted((src / name).glob("b*_s*.npz")):
            with np.load(path) as f:
                if "image" not in f.files:
                    continue
                steps = json.loads(str(f["meta"]))["steps"]
            target = out / name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append((name, target))
            frames += steps
    run = json.loads((src / "RUN.json").read_text())
    shutil.copy2(src / "RUN.json", out / "RUN.json")

    episodes = episode_frame(src)
    episodes = episodes[episodes.set.isin(SETS)]
    episodes.to_csv(out / "episodes_index.csv", index=False)
    size_mb = sum(p.stat().st_size for _, p in copied) / 1e6
    per_set = pd.DataFrame(copied, columns=["set", "path"]).groupby("set").size()
    fields = describe(copied[0][1])

    lines = [
        "# Rollout dataset",
        "",
        "Episodes of the PPO steering policy for the Unitree G1 (MuJoCo), logged step by step by `src/rollout.py`.",
        "",
        "## Origin",
        "",
        "Generated in simulation with the obstacle environment in `env/` (copied from JES0406/BecaDeepLearning, commit 434846e)",
        f"and the checkpoints in `checkpoints/`, on one machine ({run[0]['gpu'].split(',')[0]}, host {run[0]['host']}), commit {run[0]['commit']}.",
        "Each episode is fully determined by its seed (user speed and box layout) and its condition; `configs/sets.yaml` lists the seeds,",
        "box densities and visual variants. No external dataset is used.",
        "",
        "## Size",
        "",
        f"- Full evaluation: {len(episodes)} episodes over {len(SETS)} sets, {int(episodes.frames.sum())} control steps (10 Hz, at most 200 per episode). "
        "Their per-episode summary is `episodes_index.csv`; the step logs are regenerated with `scripts/run_rollouts.py` (seeds fixed).",
        f"- Published here: the {len(copied)} episodes saved with image and segmentation ({frames} frames, {size_mb:.0f} MB), "
        "the ones every per-frame explanation uses:",
        "",
        "| set | episodes with image |",
        "|---|---|",
        *[f"| {s} | {per_set.get(s, 0)} |" for s in SETS],
        "",
        "## Types",
        "",
        "One compressed NumPy archive (`.npz`) per episode, `<set>/b<boxes>_s<seed>.npz`. Per-step arrays have one row per control step.",
        "",
        "| field | dtype | shape (example) | content |",
        "|---|---|---|---|",
        *[f"| `{k}` | {fields[k][0]} | {fields[k][1]} | {FIELDS.get(k, '')} |" for k in fields],
        "",
        "## Preprocessing",
        "",
        "None beyond what the policy itself does: the image is divided by 255 and transposed HWC to CHW inside the network;",
        "the state is not normalised. The logged image is the exact input of the policy (checked by `scripts/check_rollouts.py`,",
        "which recomputes the outputs from it); the mask is rendered from the same simulator state.",
        "",
        "## Sets",
        "",
        "- `validation`: seeds 1000-1019 (used in earlier evaluations), 0/10/20/30 boxes.",
        "- `test_id`: seeds 2000+, never used, 10/20/30 boxes. `test_id_0`: same seeds, no boxes.",
        "- `test_ood_*`: the Test-ID seeds with blue boxes, a tinted floor, or other lighting (values in `src/scenarios.py`).",
        "",
        "Load an episode:",
        "",
        "```python",
        "import json, numpy as np",
        "ep = np.load('data/rollouts/test_id/b30_s2000.npz')",
        "meta = json.loads(str(ep['meta'])); image, mask, mean = ep['image'], ep['mask'], ep['mean']",
        "```",
        "",
    ]
    (REPO / "data" / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"copied {len(copied)} episodes ({frames} frames, {size_mb:.0f} MB) to {out}; wrote data/README.md")


if __name__ == "__main__":
    main()
