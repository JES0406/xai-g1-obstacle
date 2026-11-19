"""Short fine-tune of the CNN only, with visual domain randomisation.

    MUJOCO_GL=egl python scripts/finetune.py --steps 1500000          # lab machine, ~1 h
    env/wsl_gl.sh python scripts/finetune.py --steps 512 --envs 2 --name ft_smoke   # laptop smoke test

Starts from checkpoints/planPW. Only the image CNN (features extractor) is
trained: actor and critic heads and the action std are frozen; the CNN is
shared, so V changes too. Constant low learning rate, density pinned at 30
boxes (episodes get 15-30), and src.scenarios.VisualRandomisation on every
reset, with ranges disjoint from the Test-OOD values.

Writes runs/<name>/ (config.json, checkpoints/, final/model.zip, tensorboard).
"""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.load_policy import PPO_DIR, WALKER_DIR  # noqa: E402  (also puts env/ on sys.path)

from callbacks import Checkpoint  # noqa: E402
from obstacle_env import MAX_BOXES, ObstacleEnv  # noqa: E402
from runs import save  # noqa: E402
from train_obstacles import DensityCurriculum  # noqa: E402

DR_SEED_OFFSET = 104_729


class Randomised(gym.Wrapper):
    """ObstacleEnv with a random look drawn before every reset."""

    def __init__(self, env, p=0.5):
        from src.scenarios import Look, VisualRandomisation

        super().__init__(env)
        self.dr = VisualRandomisation(Look(env), p=p)
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed + DR_SEED_OFFSET)
        self.dr.sample(self.rng)
        return self.env.reset(seed=seed, options=options)

    def set_boxes(self, low, high):
        self.env.set_boxes(low, high)


class PinnedDensity(DensityCurriculum):
    """The training curriculum's logging, with the level held at `level`."""

    def __init__(self, level):
        super().__init__(level=level)
        self.pinned = level

    def _on_step(self):
        result = super()._on_step()
        if self.level != self.pinned:
            self.level = self.pinned
            self._apply()
        return result


def make_env(walker, p):
    def build():
        return Randomised(ObstacleEnv(walker), p=p)
    return build


def freeze_all_but_cnn(policy):
    for module in (policy.mlp_extractor, policy.action_net, policy.value_net):
        module.requires_grad_(False)
    policy.log_std.requires_grad_(False)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    return trainable, total


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ft_dr")
    parser.add_argument("--steps", type=int, default=1_500_000)
    parser.add_argument("--envs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--p", type=float, default=0.5, help="probability of randomising each visual knob")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    args = parser.parse_args()

    folder = REPO / "runs" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({
        "task": "obstacles", "algorithm": "ppo", "walker": "checkpoints/g1_vel_v1",
        "finetune_of": "checkpoints/planPW", **vars(args)}, indent=1))

    walker = str(WALKER_DIR)
    venv = SubprocVecEnv([make_env(walker, args.p) for _ in range(args.envs)])
    venv.seed(args.seed)
    venv = VecTransposeImage(venv)
    model = PPO.load(PPO_DIR / "model.zip", env=venv, device="auto",
                     custom_objects={"learning_rate": args.lr, "lr_schedule": lambda _: args.lr},
                     tensorboard_log=str(folder))
    trainable, total = freeze_all_but_cnn(model.policy)
    print(f"fine-tuning {trainable} of {total} parameters (CNN only), lr {args.lr}, {args.envs} envs, "
          f"{args.steps} steps, density pinned at {MAX_BOXES}", flush=True)
    callbacks = CallbackList([PinnedDensity(MAX_BOXES), Checkpoint(folder, args.checkpoint_every)])
    model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name="ft", reset_num_timesteps=True,
                log_interval=1)
    save(folder / "final", model)
    print(f"saved {folder / 'final'}")


if __name__ == "__main__":
    main()
