# checkpoints/

| Folder | What | Origin (JES0406/BecaDeepLearning @ 434846e) | sha256 of model.zip |
| --- | --- | --- | --- |
| `planPW/` | PPO steering policy under study (6 002 688 steps, 12 envs, seed 0) | `runs/planPW/final/` + `runs/planPW/config.json` | `bcf15c84f1f38c2d4b7d69c941b7398ea1eaaddb9aa59ac89c4a805492f25e5a` |
| `g1_vel_v1/` | frozen walker (walkL, curriculum stage 4) | `pretrained/g1_vel_v1/` | `e63fc67d5631ae6b8193a61ad61a0e83e1b78949c62eda82544da2524e4d6c86` |

| `ft_dr/` | CNN-only fine-tune of planPW with visual randomisation (1.5M steps), behind the fine-tune rows of the before/after table | `runs/ft_dr/final/` of `scripts/finetune.py` (lab machine) | see `git log -- checkpoints/ft_dr` |

`planSW/` (SAC) is only added if the optional SAC contrast is done.

`g1_vel_v1/vecnormalize.pkl` is a pickle: only load files from this repo.
