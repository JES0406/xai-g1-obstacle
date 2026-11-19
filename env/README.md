# env/ — the obstacle environment, copied from its origin

Source: GitHub `JES0406/BecaDeepLearning` (private), branch `worktree-dl2-bench`,
commit `434846e`, folder `sims/mujoco/g1/`. The code is the author's own.
Files were extracted with `git show 434846e:sims/mujoco/g1/<file>`, so they are
byte-identical to that commit except for the patches below.

| File | Role |
| --- | --- |
| `robot.py` | G1 model (MuJoCo Menagerie) + torso camera `front` |
| `walk_env.py` | walking env the frozen walker was trained in |
| `obstacle_env.py` | the env under study: 0–30 boxes, 64x64 RGB + state(6), residual (vx, vy, yaw) at 10 Hz |
| `runs.py` | `Walker` (loads the frozen walker), `read_config` |
| `policy.py` | custom SB3 policy classes (needed to unpickle the walker) |
| `callbacks.py` | training callbacks (imported by `train_obstacles.py`) |
| `train_obstacles.py` | training script (base for the optional CNN fine-tune) |
| `evaluate_obstacles.py` | original evaluation (policy / walker / blind rows, seeds 1000+) |
| `wsl_gl.sh` | headless EGL rendering under WSL2 (laptop only) |

## Patches

1. `robot.py`: `REPO = parents[1]` instead of `parents[3]` — the file now lives
   at `env/robot.py`, so the repo root is one level up. Menagerie is still
   expected at `external/mujoco_menagerie` (`setup/setup.sh menagerie`).
2. `train_obstacles.py`: `--walker` default `checkpoints/g1_vel_v1` instead of
   `pretrained/g1_vel_v1`.
3. `../checkpoints/planPW/config.json`: `"walker": "checkpoints/g1_vel_v1"`
   (was `pretrained/g1_vel_v1`; `evaluate_obstacles.py` resolves it from the repo root).

Import with `env/` on `sys.path` (the modules import each other as top-level
modules): `src/load_policy.py` does this.
