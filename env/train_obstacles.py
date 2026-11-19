"""Train the obstacle policy (obstacle_env.py) with PPO or SAC (Stable-Baselines3).

    MUJOCO_GL=egl python sims/mujoco/g1/train_obstacles.py ppo --name obsP --steps 6000000
    MUJOCO_GL=egl python sims/mujoco/g1/train_obstacles.py sac --name obsS --steps 1500000 --envs 8

Same env, same reward, same curriculum for both algorithms: this is the PPO vs
SAC comparison. Steps are steps of this policy (10 Hz; an episode is 200).

Before training it checks the walker inside the env: 5 episodes with no boxes
and no correction must cover at least 80% of the commanded distance without a
fall. If they do not, the walker (or how it is fed) is broken and training on
top of it would waste the run.

Density curriculum: episodes start with no boxes. Every 50 finished episodes,
a success rate of 70% or more adds one box to the most an episode can get (up
to 30) and one below 30% takes one away; an episode gets between half the
level and the level. One line per 50 episodes goes to the log:

    grep ^steer runs/obsP.log | tail

Everything goes to runs/<name>/ (runs.py).
"""

import argparse
import json
import time

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from callbacks import AdaptiveLearningRate, Checkpoint
from obstacle_env import MAX_BOXES, ObstacleEnv
from robot import REPO
from runs import save

ALGORITHMS = {"ppo": PPO, "sac": SAC}
WINDOW = 50              # episodes per curriculum decision
PROMOTE, DEMOTE = 0.7, 0.3
CHECK_PROGRESS = 0.8     # the walker alone, no boxes: share of the commanded distance it must cover

KWARGS = {
    "ppo": dict(
        n_steps=256,            # per env: 12 envs -> 3072 steps (~15 episodes) per update
        batch_size=256,
        n_epochs=5,
        learning_rate=3e-4,     # the start; AdaptiveLearningRate takes over, as for the walker
        gamma=0.99,             # looks ~100 steps = 10 s ahead
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0),  # std 0.37: 0.18 m/s of noise on the command
    ),
    "sac": dict(
        buffer_size=300_000,    # each step stores 2 images of 64x64x3 bytes: 7.4 GB
        learning_starts=5_000,  # random steps before the first update
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        train_freq=1,           # after every step of the vector env ...
        gradient_steps=2,       # ... two updates
        policy_kwargs=dict(net_arch=[128, 128]),
    ),
}


class DensityCurriculum(BaseCallback):
    """Counts finished episodes (the same way for PPO and SAC) and moves the
    number of boxes on their success rate. Logs and prints each window."""

    def __init__(self, level=0):
        super().__init__()
        self.level = level
        self.episodes = []
        self.last = {}
        self.started = None

    def _apply(self):
        self.training_env.env_method("set_boxes", self.level // 2, self.level)

    def _on_training_start(self):
        self.started = (time.time(), self.num_timesteps)
        self._apply()

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes.append([info["success"], info["box_hit"], info["fall"], info["progress"]])
        if len(self.episodes) >= WINDOW:
            success, hit, fall, progress = np.mean(self.episodes, axis=0)
            self.episodes = []
            seconds, steps = time.time() - self.started[0], self.num_timesteps - self.started[1]
            print(f"steer steps {self.num_timesteps:>9d}  boxes {self.level // 2}-{self.level:<2d}  success {success:.2f}"
                  f"  hit {hit:.2f}  fall {fall:.2f}  progress {progress:.2f}  steps/s {steps / seconds:.0f}", flush=True)
            self.last = {"curriculum/boxes": self.level, "rollout/success_rate": success, "rollout/box_hit_rate": hit,
                         "rollout/fall_rate": fall, "rollout/progress": progress}
            if success >= PROMOTE and self.level < MAX_BOXES:
                self.level += 1
            elif success < DEMOTE and self.level > 0:
                self.level -= 1
            self._apply()
        for key, value in self.last.items():
            self.logger.record(key, value)
        return True


def check_walker(walker, episodes=5):
    """The walker alone (no correction, no boxes) must walk the commanded line."""
    env = ObstacleEnv(walker)
    progress, falls = [], 0
    for episode in range(episodes):
        env.reset(seed=episode)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            done = terminated or truncated
        progress.append(info["progress"])
        falls += info["fall"]
    env.close()
    print(f"walker check: progress {np.mean(progress):.2f} (need >= {CHECK_PROGRESS}), falls {falls}/{episodes}", flush=True)
    return np.mean(progress) >= CHECK_PROGRESS and falls == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=ALGORITHMS)
    parser.add_argument("--walker", default="checkpoints/g1_vel_v1", help="folder of the saved walking policy")
    parser.add_argument("--name", help="run folder under runs/ (default: obstacles_<algorithm>)")
    parser.add_argument("--steps", type=int, default=3_000_000, help="steps of this policy (10 Hz)")
    parser.add_argument("--envs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--buffer", type=int, default=KWARGS["sac"]["buffer_size"],
                        help="sac: replay buffer size in steps (24.6 kB of images each)")
    parser.add_argument("--learning-starts", type=int, default=KWARGS["sac"]["learning_starts"],
                        help="sac: random steps before the first update")
    parser.add_argument("--skip-check", action="store_true", help="train even if the walker fails its check (smoke tests)")
    args = parser.parse_args()

    walker = str((REPO / args.walker).resolve())
    folder = REPO / "runs" / (args.name or f"obstacles_{args.algorithm}")
    if not args.skip_check and not check_walker(walker):
        raise SystemExit("the walker does not walk inside the obstacle env: not training (see --skip-check)")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"task": "obstacles", **vars(args)}, indent=1))

    venv = make_vec_env(ObstacleEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                        env_kwargs={"walker": walker})
    kwargs = dict(KWARGS[args.algorithm])
    if args.algorithm == "sac":
        kwargs.update(buffer_size=args.buffer, learning_starts=args.learning_starts)
    model = ALGORITHMS[args.algorithm]("MultiInputPolicy", venv, seed=args.seed, verbose=1,
                                       tensorboard_log=str(folder), **kwargs)
    callbacks = [DensityCurriculum(), Checkpoint(folder, args.checkpoint_every)]
    if args.algorithm == "ppo":
        callbacks.append(AdaptiveLearningRate())
    model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks), tb_log_name=args.algorithm,
                log_interval=1 if args.algorithm == "ppo" else 100)
    save(folder / "final", model)
    print(f"saved {folder / 'final'}")


if __name__ == "__main__":
    main()
