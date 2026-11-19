#!/usr/bin/env bash
# Fetch what the repo does not ship. Run from anywhere:
#
#   setup/setup.sh menagerie   # MuJoCo Menagerie's G1 into external/ (pinned commit, sparse: unitree_g1 only)
#   setup/setup.sh env         # conda env "xai-g1" from requirements.txt (python 3.11)
#
# Trimmed from setup/setup.sh of JES0406/BecaDeepLearning @ 434846e (only the
# MuJoCo parts this project needs). Pins from that repo's versions.toml.

set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MENAGERIE_COMMIT=8161bba264d7

case "${1:-}" in
  menagerie)
    DIR="$ROOT/external/mujoco_menagerie"
    if [ ! -d "$DIR/.git" ]; then
      git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie "$DIR"
      git -C "$DIR" sparse-checkout set unitree_g1
    fi
    git -C "$DIR" fetch -q origin "$MENAGERIE_COMMIT" 2>/dev/null || true
    git -C "$DIR" checkout -q "$MENAGERIE_COMMIT"
    test -f "$DIR/unitree_g1/scene_mjx.xml"
    echo "menagerie G1 ready at $DIR ($(git -C "$DIR" rev-parse --short HEAD))"
    ;;

  env)
    PY="$HOME/miniconda3/envs/xai-g1/bin/python"
    [ -x "$PY" ] || conda create -y -n xai-g1 python=3.11
    "$PY" -m pip install -r "$ROOT/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu126
    "$PY" -c "import torch, stable_baselines3, mujoco, captum; print('torch', torch.__version__, '| sb3', stable_baselines3.__version__, '| mujoco', mujoco.__version__, '| captum', captum.__version__)"
    ;;

  *)
    sed -n '2,6p' "$0"
    exit 1
    ;;
esac
