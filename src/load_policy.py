"""Load the PPO steering policy as a differentiable torch module.

The forward pass, step by step:

    p = PPO.load(...).policy
    f = p.extract_features(obs)          # divides the image by 255, differentiably
    mean = p.action_net(p.mlp_extractor.forward_actor(f))
    V    = p.value_net(p.mlp_extractor.forward_critic(f))

Clip and EMA live outside the network, so attributions target the mean
*before* the clip. The CNN is shared by actor and critic.

    from src.load_policy import load_ppo, SteeringNet, to_inputs, make_env
    model = load_ppo()
    net = SteeringNet(model.policy)
    env = make_env(boxes=10)
    obs, _ = env.reset(seed=2000)
    image, state = to_inputs(model.policy, obs, requires_grad=True)
    mean, value = net(image, state)
"""

import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
ENV_DIR = REPO / "env"
PPO_DIR = REPO / "checkpoints" / "planPW"
WALKER_DIR = REPO / "checkpoints" / "g1_vel_v1"

# The env modules import each other as top-level modules, and unpickling the
# walker needs env/policy.py (AsymmetricPolicy) importable.
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

ACTION_SCALE = (0.5, 0.5, 1.0)   # m/s, m/s, rad/s per unit of clipped action
N_PARAMETERS = 439_207           # PPO policy (CNN + actor + critic)


def load_ppo(folder=PPO_DIR, device="cpu"):
    """The SB3 PPO model, in eval mode."""
    from stable_baselines3 import PPO

    model = PPO.load(Path(folder) / "model.zip", device=device)
    model.policy.set_training_mode(False)
    return model


class SteeringNet(nn.Module):
    """(image, state) -> (mean action before the clip, V).

    image: float (B, 3, 64, 64), values 0-255 (CHW). state: float (B, 6).
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image, state):
        features = self.policy.extract_features({"image": image, "state": state})
        mean = self.policy.action_net(self.policy.mlp_extractor.forward_actor(features))
        value = self.policy.value_net(self.policy.mlp_extractor.forward_critic(features))
        return mean, value


def to_inputs(policy, obs, requires_grad=False):
    """Env observation (HWC uint8 image) -> float tensors for SteeringNet."""
    tensors, _ = policy.obs_to_tensor(obs)          # transposes HWC -> CHW, adds the batch axis
    image = tensors["image"].float().detach().requires_grad_(requires_grad)
    state = tensors["state"].float().detach()
    return image, state


def gradcam_layers(policy):
    """Grad-CAM targets, after the ReLU: conv3 (4x4 maps) and conv2 (6x6)."""
    cnn = policy.features_extractor.extractors["image"].cnn
    return {"conv3": cnn[5], "conv2": cnn[3]}


def make_env(boxes=0, render_mode=None, walker=WALKER_DIR):
    """ObstacleEnv with the frozen walker; boxes: int or (fewest, most)."""
    from obstacle_env import ObstacleEnv

    low, high = (boxes, boxes) if isinstance(boxes, int) else boxes
    return ObstacleEnv(str(walker), boxes=(low, high), render_mode=render_mode)
