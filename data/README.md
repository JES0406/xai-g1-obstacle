# Rollout dataset

Episodes of the PPO steering policy for the Unitree G1 (MuJoCo), logged step by step by `src/rollout.py`.

## Origin

Generated in simulation with the obstacle environment in `env/` (copied from JES0406/BecaDeepLearning, commit 434846e)
and the checkpoints in `checkpoints/`, on one machine (NVIDIA GeForce RTX 4070 Ti, host AA25LABIAP02), commit 301530f.
Each episode is fully determined by its seed (user speed and box layout) and its condition; `configs/sets.yaml` lists the seeds,
box densities and visual variants. No external dataset is used.

## Size

- Full evaluation: 1330 episodes over 6 sets, 244320 control steps (10 Hz, at most 200 per episode). Their per-episode summary is `episodes_index.csv`; the step logs are regenerated with `scripts/run_rollouts.py` (seeds fixed).
- Published here: the 120 episodes saved with image and segmentation (22740 frames, 97 MB), the ones every per-frame explanation uses:

| set | episodes with image |
|---|---|
| validation | 20 |
| test_id | 20 |
| test_id_0 | 20 |
| test_ood_box_colour | 20 |
| test_ood_floor_tint | 20 |
| test_ood_lighting | 20 |

## Types

One compressed NumPy archive (`.npz`) per episode, `<set>/b<boxes>_s<seed>.npz`. Per-step arrays have one row per control step.

| field | dtype | shape (example) | content |
|---|---|---|---|
| `meta` | <U306 | () | JSON: set, variant, baseline, transform, boxes, seed, save_image, speed, steps and outcome |
| `state` | <f4 | (200, 6) | commanded speed, sin and cos of the heading, previous action (3) |
| `mean` | <f4 | (200, 3) | raw policy mean (vx, vy, yaw) before the clip |
| `value` | <f4 | (200,) | critic value V(s) |
| `action` | <f4 | (200, 3) | action applied, clip(mean) to [-1, 1] (0 for the walker baseline) |
| `command` | <f4 | (200, 3) | velocity command the walker received (vx, vy, yaw rate) after the residual and the EMA |
| `clearance` | <f4 | (200,) | distance from the pelvis to the nearest box face, m (inf without boxes) |
| `qpos` | <f4 | (200, 36) | MuJoCo generalised positions of the robot (base pose + 29 joints) |
| `qvel` | <f4 | (200, 6) | base linear and angular velocity |
| `reward` | <f4 | (200,) | reward of the step |
| `box_centres` | <f4 | (0, 2) | centres of the boxes of the episode, m |
| `box_half` | <f4 | (0,) | half widths of the boxes, m |
| `image` | |u1 | (200, 64, 64, 3) | 64x64 RGB image the policy saw (uint8) |
| `mask` | |i1 | (200, 64, 64) | per-pixel box index from the geometry segmentation, -1 = no box (int8) |

## Preprocessing

None beyond what the policy itself does: the image is divided by 255 and transposed HWC to CHW inside the network;
the state is not normalised. The logged image is the exact input of the policy (checked by `scripts/check_rollouts.py`,
which recomputes the outputs from it); the mask is rendered from the same simulator state.

## Sets

- `validation`: seeds 1000-1019 (used in earlier evaluations), 0/10/20/30 boxes.
- `test_id`: seeds 2000+, never used, 10/20/30 boxes. `test_id_0`: same seeds, no boxes.
- `test_ood_*`: the Test-ID seeds with blue boxes, a tinted floor, or other lighting (values in `src/scenarios.py`).

Load an episode:

```python
import json, numpy as np
ep = np.load('data/rollouts/test_id/b30_s2000.npz')
meta = json.loads(str(ep['meta'])); image, mask, mean = ep['image'], ep['mask'], ep['mean']
```
