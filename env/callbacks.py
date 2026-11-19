"""Training callbacks both trainers use (Stable-Baselines3 calls them during learn())."""

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import ConstantSchedule

from runs import save


class Checkpoint(BaseCallback):
    """Every `every` steps, save the policy (runs.save) under
    <folder>/checkpoints/<steps>. The lab machine is shared and a process can
    be killed at any time: a run resumes from its last checkpoint."""

    def __init__(self, folder, every, curricula=None):
        super().__init__()
        self.folder, self.every, self.curricula = folder, every, curricula
        self.last = 0

    def _on_training_start(self):
        self.last = self.num_timesteps  # a resumed run counts from where it was

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            save(self.folder / "checkpoints" / f"{self.num_timesteps:012d}", self.model, self.curricula)
        return True


class AdaptiveLearningRate(BaseCallback):
    """PPO only. After each update, look at how far the policy moved (the KL
    divergence between the old and the new policy): above twice the target,
    divide the learning rate by 1.5; below half, multiply it by 1.5. This is
    rsl_rl's "adaptive" schedule (unitree_rl_lab uses a target of 0.01).

    With a fixed 3e-4 the walker's updates were 15-70 times too large for
    20M steps and its gait never got refined."""

    def __init__(self, desired_kl=0.01, low=1e-5, high=1e-2, initial=None):
        super().__init__()
        self.desired_kl, self.low, self.high = desired_kl, low, high
        self.initial = initial

    def _on_training_start(self):
        # `initial`: a run that continues a saved policy starts at the rate that
        # policy had adapted to. Restarting at 3e-4 a policy tuned at 1e-5
        # destroys it in one update.
        self.lr = self.initial or self.model.lr_schedule(1.0)
        self.model.lr_schedule = ConstantSchedule(self.lr)

    def _on_rollout_start(self):
        kl = self.model.logger.name_to_value.get("train/approx_kl")  # from the update just done
        if kl is None:
            return
        if kl > 2 * self.desired_kl:
            self.lr = max(self.low, self.lr / 1.5)
        elif 0 < kl < self.desired_kl / 2:
            self.lr = min(self.high, self.lr * 1.5)
        self.model.lr_schedule = ConstantSchedule(self.lr)
        self.logger.record("train/adaptive_lr", self.lr)

    def _on_step(self):
        return True
