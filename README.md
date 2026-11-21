# xai-g1-obstacle

Explainability case study (XAI final project, MSc Advanced AI, ICAI 2026/27, Option A):
**does the PPO policy that steers a Unitree G1 around boxes avoid them because it sees
the box, or because of a shortcut?**

- Notebook with the whole pipeline: [`notebooks/xai_case_study.ipynb`](notebooks/xai_case_study.ipynb)
- Report: [`report/main.tex`](report/main.tex) (build below)
- Dataset: [`data/README.md`](data/README.md) and `data/rollouts/`
- All numbers and figures: `results/`

## Layout

```
configs/      evaluation sets, baselines and interventions (seeds, densities, visual variants, N)
env/          obstacle environment, copied from JES0406/BecaDeepLearning @ 434846e (see env/README.md)
checkpoints/  PPO policy planPW + frozen walker g1_vel_v1 (see checkpoints/README.md)
data/         published rollout dataset (episodes with image) and its description
src/          load_policy, scenarios, rollout, metrics, attribution, sanity, plotstyle
scripts/      one script per experiment (listed below)
notebooks/    xai_case_study.ipynb: load -> evaluate -> explain -> sanity checks -> before/after
results/      metrics (CSV), figures and the video
report/       LaTeX report; tables generated from results/
```

## Setup

```bash
setup/setup.sh menagerie   # MuJoCo Menagerie G1 into external/ (pinned 8161bba)
setup/setup.sh env         # conda env xai-g1 from requirements.txt (python 3.11, torch cu126)
conda activate xai-g1
export PYTHONNOUSERSITE=1                   # keep packages from ~/.local out of the env
MUJOCO_GL=egl python scripts/smoke_load.py   # checkpoints load, env builds, net == SB3 action, grads reach the image
```

Rendering is offscreen: `MUJOCO_GL=egl` on native Linux; under WSL2 prefix commands with `env/wsl_gl.sh`.

## Reproduce

Everything, from a clean clone, in order (about 3 h on an RTX 4070 Ti, mostly CPU):

```bash
scripts/reproduce_all.sh                   # add --train-finetune to retrain the fine-tune too
```

All published numbers come from one machine (RTX 4070 Ti) and this repository. A clean anonymous clone rerun on
that machine reproduces every rate inside its published 95 % interval and the baselines exactly, but not bit for
bit: the closed loop amplifies tiny numerical differences after a few dozen steps (see
[`results/reproduction_check.md`](results/reproduction_check.md)). Other GPUs also render slightly differently.
Each step on its own:

| result | command | output |
|---|---|---|
| evaluation rollouts, baselines, behaviour test | `MUJOCO_GL=egl python scripts/run_rollouts.py` | `data/raw/` (resumable) |
| integrity of the logs | `python scripts/check_rollouts.py data/raw` | saved images reproduce the logged outputs |
| evaluation tables (95 % CIs) | `python scripts/metrics_table.py` | `results/metrics_*.csv`, `metrics_summary.md` |
| group permutation importance | `python scripts/group_pfi.py` | `results/group_pfi.csv`, `figures/group_pfi.png` |
| IG by input group vs distance | `python scripts/ig_groups.py` | `results/ig_groups.csv`, `figures/ig_image_share.png` |
| local maps and q_RRA | `python scripts/local_maps.py` | `results/q_rra.csv`, `figures/local_maps_*.png` |
| ID/OOD pairs on the same states | `MUJOCO_GL=egl python scripts/rerender_pairs.py` | `results/pairs.csv`, `figures/pairs_*.png` |
| policy-camera video with Grad-CAM | `python scripts/gradcam_video.py` | `results/figures/gradcam_*.mp4` |
| randomisation, ROAD, agreement, q_LLE/q_AS, 1-px shifts | `python scripts/sanity_checks.py` | `results/sanity_*`, `figures/sanity_*.png` |
| reference floor colour | `python scripts/colour_reference.py` | `configs/colour_reference.json` |
| test-time interventions | `MUJOCO_GL=egl python scripts/run_rollouts.py --sets int_hue_box_colour int_hue_test_id int_norm_floor_tint int_norm_test_id` | `data/raw/int_*` |
| CNN fine-tune (optional; the weights used are shipped in `checkpoints/ft_dr`) | `MUJOCO_GL=egl python scripts/finetune.py` | `runs/ft_dr/` |
| fine-tune evaluation | `MUJOCO_GL=egl python scripts/run_rollouts.py --policy checkpoints/ft_dr --out data/raw_ft --sets test_id test_id_0 test_ood_box_colour test_ood_floor_tint test_ood_lighting` | `data/raw_ft/` |
| q_RRA of interventions / fine-tune | `python scripts/local_maps.py --no-examples --sets int_hue_box_colour int_norm_floor_tint --out results/interventions` and `... --policy checkpoints/ft_dr --data data/raw_ft --out results/ft` | `results/{interventions,ft}/q_rra.csv` |
| before/after table | `python scripts/before_after.py` | `results/before_after.{csv,md}` |
| report tables | `python scripts/make_report_tables.py` | `report/tables/*.tex` |
| published dataset | `python scripts/export_dataset.py` | `data/rollouts/`, `data/README.md` |

## Report

```bash
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Provenance

Environment code and checkpoints are the author's, from the private repo
`JES0406/BecaDeepLearning`, branch `worktree-dl2-bench`, commit `434846e`
(`sims/mujoco/g1/`, `runs/planPW/`, `pretrained/g1_vel_v1/`). Patches to the
copied files are listed in `env/README.md`.

Built on Stable-Baselines3, MuJoCo, MuJoCo Menagerie and Captum.
