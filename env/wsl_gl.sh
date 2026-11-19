#!/usr/bin/env bash
# Run a command with headless GPU rendering under WSL2 (MuJoCo Renderer via EGL).
# Not needed on native Linux with an NVIDIA driver: there MUJOCO_GL=egl is enough.
#
#     sims/mujoco/g1/wsl_gl.sh python -m pytest sims/mujoco/g1
#
# WSL2 exposes the GPU to OpenGL only through Mesa's d3d12 driver (system Mesa;
# do not install conda-forge's mesalib into the env, it only renders in
# software and shadows the system libEGL). Needs the user in the render and
# video groups (/dev/dri).
set -euo pipefail
export MUJOCO_GL=egl
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA   # the Intel iGPU renders a different image
exec "$@"
