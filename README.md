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
python scripts/metrics_table.py                          # results/metrics_*.csv, metrics_summary.md (95 % CIs)
python scripts/group_pfi.py                              # results/group_pfi.csv, figures/group_pfi.png
python scripts/ig_groups.py                              # results/ig_groups.csv, figures/ig_image_share.png
python scripts/local_maps.py                             # Grad-CAM / IG / occlusion examples + q_RRA per set -> results/q_rra.csv
MUJOCO_GL=egl python scripts/rerender_pairs.py           # ID/OOD pairs on the same states -> results/pairs.csv, figures/pairs_*.png
python scripts/gradcam_video.py                          # policy-camera video with Grad-CAM -> results/figures/gradcam_*.mp4
python scripts/sanity_checks.py                          # randomisation, ROAD, agreement, q_LLE/q_AS, 1-px shifts -> results/sanity_*
MUJOCO_GL=egl python scripts/run_rollouts.py --sets baseline_mask_top baseline_mask_random   # behaviour test in simulation
python scripts/colour_reference.py                       # training floor colour -> configs/colour_reference.json
MUJOCO_GL=egl python scripts/run_rollouts.py --sets int_hue_box_colour int_hue_test_id int_norm_floor_tint int_norm_test_id   # test-time interventions
python scripts/local_maps.py --no-examples --sets int_hue_box_colour int_norm_floor_tint --out results/interventions
python scripts/local_maps.py --no-examples --policy runs/ft_dr/final --data data/raw_ft --out results/ft
python scripts/before_after.py                           # results/before_after.md: original / intervention / fine-tune
MUJOCO_GL=egl python scripts/finetune.py                 # optional: CNN-only fine-tune with visual randomisation -> runs/ft_dr/
MUJOCO_GL=egl python scripts/run_rollouts.py --policy runs/ft_dr/final --out data/raw_ft   # evaluate the fine-tuned policy
```

Every number and figure in the report comes from one machine (the lab RTX 4070 Ti)
and this commit; the command for each figure is listed here as it is added.

## Provenance

Environment code and checkpoints are the author's, from the private repo
`JES0406/BecaDeepLearning`, branch `worktree-dl2-bench`, commit `434846e`
(`sims/mujoco/g1/`, `runs/planPW/`, `pretrained/g1_vel_v1/`). Patches to the
copied files are listed in `env/README.md`.

Built on Stable-Baselines3, MuJoCo, MuJoCo Menagerie and Captum.
