"""Smoke test of the setup: checkpoints load, the env builds, the differentiable
net matches SB3's own deterministic action, and gradients reach the image.

    MUJOCO_GL=egl python scripts/smoke_load.py            # native Linux
    env/wsl_gl.sh python scripts/smoke_load.py            # WSL2 (laptop)
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.load_policy import N_PARAMETERS, SteeringNet, gradcam_layers, load_ppo, make_env, to_inputs  # noqa: E402


def main():
    model = load_ppo()
    policy = model.policy
    n = sum(p.numel() for p in policy.parameters())
    print(f"PPO loaded: {n} parameters (expected {N_PARAMETERS})")
    assert n == N_PARAMETERS

    net = SteeringNet(policy).eval()
    env = make_env(boxes=10)
    obs, _ = env.reset(seed=1000)
    worst = 0.0
    for step in range(20):
        image, state = to_inputs(policy, obs, requires_grad=True)
        mean, value = net(image, state)
        sb3_action, _ = model.predict(obs, deterministic=True)
        ours = np.clip(mean.detach().numpy()[0], -1, 1)
        worst = max(worst, float(np.abs(ours - sb3_action).max()))
        if step == 0:
            mean[0, 0].backward()
            grad = image.grad.abs().sum().item()
            print(f"step 0: mean {mean.detach().numpy()[0].round(3)}  V {value.item():.3f}  |d mean_vx / d image| = {grad:.3e}")
            assert grad > 0, "no gradient reaches the image"
        obs, _, terminated, truncated, _ = env.step(sb3_action)
        if terminated or truncated:
            break
    print(f"max |clip(mean) - SB3 deterministic action| over {step + 1} steps: {worst:.2e}")
    assert worst < 1e-5
    print("Grad-CAM layers:", {k: type(v).__name__ for k, v in gradcam_layers(policy).items()})
    env.close()
    print("OK")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
