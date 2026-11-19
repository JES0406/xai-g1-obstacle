"""Evaluate a trained obstacle policy against the walker alone, on the same courses.

    MUJOCO_GL=egl python sims/mujoco/g1/evaluate_obstacles.py runs/obsP/final
    MUJOCO_GL=egl python sims/mujoco/g1/evaluate_obstacles.py runs/obsP/final --blind
    MUJOCO_GL=egl python sims/mujoco/g1/evaluate_obstacles.py runs/obsP/final --boxes 20 --episodes 1 --video

For each number of boxes, the same seeded courses are run by:

    policy   the trained policy, without exploration noise
    walker   no correction at all: the walker follows the user command straight
             into the boxes. The policy earns its keep only where it beats this row.
    blind    (--blind) the trained policy with a black image. A policy that
             scores the same without its camera is not using it.

Columns: success (20 s, no hit, no fall, at least half the commanded distance),
box hits and what hit (arm / leg / torso), falls, progress (share of the
commanded distance), lateral (farthest it strayed sideways, m), end y (mean
sideways position at the end, m: far from 0 = it always leaves on the same side).

The run's config.json names the algorithm and the walker (train_obstacles.py).
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO, SAC

from obstacle_env import STEER_HZ, ObstacleEnv
from robot import REPO
from runs import read_config

HIT_PARTS = ("arm", "leg", "torso")
LEGEND = "yellow: command the walker follows    white: the user's command"


def caption(frame, text):
    """Write two lines on a video frame: `text` at the top, the arrow legend at the bottom."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    for y, line in ((8, text), (image.height - 26, LEGEND)):
        draw.text((10, y), line, fill="white", font=font, stroke_width=2, stroke_fill="black")
    return np.asarray(image)


def run(env, model, seed, frames=None, blind=False):
    """One episode; returns the info of its last step. model None = no
    correction (the walker alone). blind: the policy gets a black image."""
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        if blind:
            obs = {**obs, "image": np.zeros_like(obs["image"])}
        action = np.zeros(3) if model is None else model.predict(obs, deterministic=True)[0]
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if frames is not None:
            frames.append(caption(env.render(), f"{len(env.box_centres)} boxes   t = {env.robot.data.time:4.1f} s"
                                  f"   v = {env.speed:.2f} m/s"))
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="saved policy folder, e.g. runs/obsP/final")
    parser.add_argument("--boxes", type=int, nargs="+", default=[0, 10, 20, 30])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--blind", action="store_true", help="also run the policy with a black image")
    parser.add_argument("--video", action="store_true", help="record the policy's first episode of each box count")
    args = parser.parse_args()

    folder = Path(args.folder)
    config = read_config(folder)
    model = {"ppo": PPO, "sac": SAC}[config["algorithm"]].load(folder / "model.zip", device="cpu")
    env = ObstacleEnv(str((REPO / config["walker"]).resolve()), render_mode="rgb_array")
    print(f"{'boxes':>5} {'':>8} {'success':>8} {'box hit':>8} {'arm/leg/torso':>14} {'fall':>6} {'progress':>9} {'lateral':>8} {'end y':>7}")
    for boxes in args.boxes:
        env.set_boxes(boxes, boxes)
        rows = [("policy", model, False), ("walker", None, False)] + ([("blind", model, True)] if args.blind else [])
        for name, policy, blind in rows:
            frames = [] if args.video and name == "policy" else None
            results = [run(env, policy, 1000 + episode, frames if episode == 0 else None, blind)
                       for episode in range(args.episodes)]

            def mean(key):
                return np.mean([r[key] for r in results])

            parts = "/".join(str(sum(r["box_hit_part"] == part for r in results)) for part in HIT_PARTS)
            print(f"{boxes:>5} {name:>8} {mean('success'):>8.0%} {mean('box_hit'):>8.0%} {parts:>14} {mean('fall'):>6.0%}"
                  f" {mean('progress'):>9.2f} {mean('lateral'):>7.2f}m {mean('end_y'):>+6.2f}m")
            if frames:
                path = folder / "videos" / f"boxes{boxes}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(path, frames, fps=STEER_HZ)
    env.close()


if __name__ == "__main__":
    main()
