# xai-g1-obstacle

Explainability case study (XAI final project, MSc Advanced AI, ICAI 2026/27, Option A):
**does the PPO policy that steers a Unitree G1 around boxes avoid them because it sees
the box, or because of a spurious shortcut?**

## Layout

```
configs/      evaluation sets (seeds, densities, OOD variants, N)
env/          obstacle environment, copied from JES0406/BecaDeepLearning @ 434846e (see env/README.md)
checkpoints/  PPO policy planPW + frozen walker g1_vel_v1 (see checkpoints/README.md)
data/         published subset of rollouts (.npz with image)
src/          load_policy, scenarios, rollout, metrics, attribution, sanity, intervene
scripts/      one script per experiment
notebooks/    xai_case_study.ipynb: load -> evaluate -> XAI -> figures
results/      metrics (CSV/JSON) and figures
report/       LaTeX report
```

## Setup

```bash
setup/setup.sh menagerie   # MuJoCo Menagerie G1 into external/ (pinned 8161bba)
setup/setup.sh env         # conda env xai-g1 from requirements.txt (python 3.11, torch cu126)
```

Rendering is offscreen: `MUJOCO_GL=egl` on native Linux; under WSL2 prefix commands
with `env/wsl_gl.sh` (laptop only, for debugging).

## Reproduce

```bash
MUJOCO_GL=egl python scripts/smoke_load.py   # checkpoints load, env builds, net == SB3 action, grads reach the image
MUJOCO_GL=egl python scripts/run_rollouts.py --dry-run   # list the evaluation sets (configs/sets.yaml)
MUJOCO_GL=egl python scripts/run_rollouts.py             # all rollouts into data/raw/ (resumable), then a summary
python scripts/check_rollouts.py data/raw                # integrity: saved images reproduce the logged outputs
```

Every number and figure in the report comes from one machine (the lab RTX 4070 Ti)
and this commit; the command for each figure is listed here as it is added.

## Provenance

Environment code and checkpoints are the author's, from the private repo
`JES0406/BecaDeepLearning`, branch `worktree-dl2-bench`, commit `434846e`
(`sims/mujoco/g1/`, `runs/planPW/`, `pretrained/g1_vel_v1/`). Patches to the
copied files are listed in `env/README.md`.

Built on Stable-Baselines3, MuJoCo, MuJoCo Menagerie and Captum.
