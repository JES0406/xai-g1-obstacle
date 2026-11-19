"""The walker's PPO policy: an actor that sees less than its critic.

Asymmetric actor-critic. The critic only exists during training, in the
simulator, so it may see what the real robot cannot measure well (true base
velocity, height, foot contacts: the `privileged` key). That makes the value
easier to learn. The actor is what would run on the robot, so it must not.
legged_gym, unitree_rl_gym and Humanoid-Gym all do this.

How: actor and critic get their own feature extractor (SB3's
`share_features_extractor=False`). Both are SB3's CombinedExtractor (a CNN for
the image, flatten for vectors); the actor's replaces `privileged` with zeros.

The name of this file and of AsymmetricPolicy are written inside every saved
model.zip: renaming either breaks loading the saved walkers.
"""

import torch
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor


class AsymmetricExtractor(CombinedExtractor):
    def __init__(self, observation_space, **kwargs):
        super().__init__(observation_space, **kwargs)
        self.hide_privileged = False

    def forward(self, observations):
        features = []
        for key, extractor in self.extractors.items():
            out = extractor(observations[key])
            if key == "privileged" and self.hide_privileged:
                out = torch.zeros_like(out)
            features.append(out)
        return torch.cat(features, dim=1)


class AsymmetricPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = AsymmetricExtractor
        super().__init__(*args, **kwargs)
        self.pi_features_extractor.hide_privileged = True  # the actor never sees it


def freeze_cnn(policy):
    """The walker's image is black: keep the CNN at its random start instead of
    training it on a constant."""
    for extractor in (policy.pi_features_extractor, policy.vf_features_extractor):
        extractor.extractors["image"].requires_grad_(False)
