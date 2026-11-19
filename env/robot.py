"""The Unitree G1 (29 joints) as a MuJoCo model, and the numbers every env shares.

The model is MuJoCo Menagerie's G1 with one camera added on the torso. It is
loaded as a spec (an editable description) so that an env can add things to it
before compiling: obstacle_env.py adds the boxes.

    from robot import load_spec
    model = load_spec().compile()
"""

from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]   # XAI repo: env/robot.py -> repo root (was parents[3] in sims/mujoco/g1/)
SCENE = REPO / "external/mujoco_menagerie/unitree_g1/scene_mjx.xml"

CONTROL_HZ = 50      # the walking policy acts 50 times per second

# Unitree's FixStand pose: where the real robot stands before a policy takes
# over (unitree_rl_lab, deploy/robots/g1_29dof/config/config.yaml).
# Order: left leg 6, right leg 6, waist 3, left arm 7, right arm 7.
DEFAULT_POSE = np.array([
    -0.1, 0, 0, 0.3, -0.2, 0,
    -0.1, 0, 0, 0.3, -0.2, 0,
    0, 0, 0,
    0, 0.25, 0, 0.97, 0.15, 0, 0,
    0, -0.25, 0, 0.97, -0.15, 0, 0,
])
SWING_JOINTS = [0, 3, 4, 6, 9, 10]   # hip pitch, knee, ankle pitch (left, right): what a step moves
WAIST_JOINTS = [12, 13, 14]          # yaw, roll, pitch
ARM_JOINTS = list(range(15, 29))

# The camera, at Unitree's D435 mount on the torso (d435_joint in the G1 URDF).
# It looks 25 deg down with a 90 deg field of view: the floor from ~0.4 m ahead
# up to above the horizon, 45 deg to each side.
IMAGE_SIZE = 64
CAMERA_POS = (0.0576235, 0.01753, 0.42987)
CAMERA_PITCH = np.radians(25)
CAMERA_FOVY = 90.0


def load_spec():
    """The Menagerie G1 scene with the camera "front" added on the torso."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    # MuJoCo cameras look along their -z axis with +y up. Pitching the view
    # down by p: image right = -y_world, image up = (sin p, 0, cos p).
    s, c = np.sin(CAMERA_PITCH), np.cos(CAMERA_PITCH)
    camera = spec.body("torso_link").add_camera()
    camera.name = "front"
    camera.pos = CAMERA_POS
    camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    camera.alt.xyaxes = [0, -1, 0, s, 0, c]
    camera.fovy = CAMERA_FOVY
    return spec
