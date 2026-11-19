"""Run folders: what a training run saves, and how to load a saved walker.

    runs/<name>/
        config.json              the arguments the run was started with
        checkpoints/<steps>/     a saved policy, every so many steps
        final/                   the saved policy at the end

A saved policy is a folder with model.zip and, for the walker, vecnormalize.pkl
(the running mean and variance its observations are scaled with) and
curriculum.json (where the curricula were, so a run can resume).
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

FRAME_STACK = 3   # the walker sees its last 3 observations


def save(folder, model, curricula=None):
    """curricula: {name: callback with .state()}, saved so a run can resume."""
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / "model.zip")
    normalizer = model.get_vec_normalize_env()
    if normalizer is not None:
        normalizer.save(str(folder / "vecnormalize.pkl"))
    if curricula:
        state = {name: curriculum.state() for name, curriculum in curricula.items()}
        (folder / "curriculum.json").write_text(json.dumps(state, indent=1))


def read_config(folder):
    """The config.json of the run a saved policy belongs to (the folder itself,
    or up to two levels above: runs/<name>/checkpoints/<steps>)."""
    folder = Path(folder)
    for parent in (folder, folder.parent, folder.parent.parent):
        for name in ("config.json", "planner.json"):   # planner.json: runs from before 2026-11-10
            if (parent / name).exists():
                return json.loads((parent / name).read_text())
    raise FileNotFoundError(f"no config.json in or above {folder}")


class Walker:
    """A saved walking policy, ready to act on raw G1WalkEnv observations.

    Training wraps the env in VecFrameStack (the last 3 observations side by
    side) and VecNormalize (subtract the running mean, divide by the running
    std, clip to +-10). This class does the same two things by hand, so a
    saved walker can be used without building a vector env around it.

        walker = Walker("pretrained/g1_vel_v1")
        obs, _ = env.reset()
        walker.reset()
        obs, *_ = env.step(walker.act(obs))
    """

    def __init__(self, folder):
        folder = Path(folder)
        torch.set_num_threads(1)  # one small forward pass per step: threads only add overhead
        self.policy = PPO.load(folder / "model.zip", device="cpu").policy
        self.policy.set_training_mode(False)
        # A pickle runs code when loaded: only ever load a file a run of ours wrote.
        with open(folder / "vecnormalize.pkl", "rb") as file:
            self.normalizer = pickle.load(file)
        self.frames = None

    def reset(self):
        """Call at the start of every episode: forget the earlier frames."""
        self.frames = None

    def act(self, observation):
        """The 29 joint actions for one observation, without exploration noise."""
        action, _ = self.policy.predict(self.prepare(observation), deterministic=True)
        return action

    def prepare(self, observation):
        """Stack the observation on the two before it and normalise: what the
        network gets. Call once per step (it remembers the frames)."""
        if self.frames is None:  # as VecFrameStack: the frames before the first are zeros
            self.frames = {key: [np.zeros_like(value)] * FRAME_STACK for key, value in observation.items()}
        stacked = {}
        for key, value in observation.items():
            self.frames[key] = self.frames[key][1:] + [value]
            stacked[key] = np.concatenate(self.frames[key], axis=-1)
        for key in self.normalizer.norm_obs_keys:
            rms = self.normalizer.obs_rms[key]
            scaled = (stacked[key] - rms.mean) / np.sqrt(rms.var + self.normalizer.epsilon)
            stacked[key] = np.clip(scaled, -self.normalizer.clip_obs, self.normalizer.clip_obs).astype(np.float32)
        return stacked
