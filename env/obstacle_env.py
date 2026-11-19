"""Obstacle environment: steer a frozen walker around boxes, by camera.

Two policies at two rates. The walker (a saved walking policy, such as
pretrained/g1_vel_v1) moves the 29 joints at 50 Hz and never sees the boxes.
The policy trained here runs at 10 Hz: it looks through the camera and outputs
a correction to the velocity command the walker follows. It is a residual
policy: its output is added to the user's command, and 0 means "obey the user".

    task         start at the origin facing +x; walk at v m/s for 20 s. Up to
                 30 boxes stand in a +-70 deg cone ahead.
    observation  image  64x64 RGB from the camera on the torso
                 state  commanded speed, sin and cos of the heading, the previous action
    action       3 numbers in [-1, 1], scaled by RESIDUAL and added to the
                 command (v, 0, 0): corrections to vx, vy and the yaw rate
    reward       progress along +x, paid only up to where the user's command
                 would have taken the robot by now
                 - a penalty for being within MARGIN of a box
                 - a small cost per metre away from the line y = 0
                 a box hit or a fall ends the episode and pays END_REWARD

The policy and its critic see the image and the state only. The distance to the
nearest box is used by the reward alone.

    from obstacle_env import ObstacleEnv
    env = ObstacleEnv("pretrained/g1_vel_v1", boxes=(5, 10))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(3))

Design and first results: notes/2026-11-10-obstacles-rgb-plan.md.
"""

import gymnasium as gym
import mujoco
import numpy as np

from robot import CONTROL_HZ, IMAGE_SIZE, load_spec
from runs import Walker
from walk_env import COMMAND_LIMITS, G1WalkEnv

STEER_HZ = 10                # this policy acts once per 5 walker steps
VX_RANGE = (0.5, 1.0)        # m/s, the user's forward command per episode

# The correction. Largest per axis: enough to stop a 0.5 m/s walk, to sidestep
# at the walker's full 0.5 m/s and to turn at its full 1 rad/s.
RESIDUAL = np.array([0.5, 0.5, 1.0])
LIMITS = np.array([COMMAND_LIMITS[axis] for axis in ("vx", "vy", "yaw")])
# The walker was trained on one command per episode; a new, noisy correction
# every 0.1 s would shake it. Half of each change passes per step.
SMOOTHING = 0.5

# The reward.
MARGIN = 0.7                 # m from the pelvis to a box face where the proximity penalty starts (an arm's reach)
PROXIMITY_WEIGHT = 0.5
LATERAL_WEIGHT = 0.05        # per step, per metre from the line y = 0; without it, walking round the whole field is free
LATERAL_CAP = 0.5
CATCH_UP = 2.0               # most progress paid in one step, in multiples of the commanded speed
# Small next to the walker's -100 on purpose: with a large penalty the safest
# plan is to stand still in front of the first box.
END_REWARD = -10.0
SUCCESS_PROGRESS = 0.5       # success = 20 s without a hit or a fall, and half the commanded distance
USER_RGBA = (1.0, 1.0, 1.0, 0.9)  # the user's command in videos: a white arrow

# The boxes: red, 1.2 m tall (cannot be stepped over), 0.3-0.5 m square, placed
# at random in a cone ahead. The cone is wide (27 m across at 10 m) so that
# there is no edge to walk around: the only way forward is between boxes.
MAX_BOXES = 30
CONE_HALF_ANGLE = np.radians(70)
CONE_NEAR = 1.5              # m: no box closer to the start
CONE_FAR = 12.0              # m
BOX_SPACING = 1.25           # m between centres: leaves gaps of at least 0.75 m for a 0.57 m wide robot
BOX_HALF_WIDTH = (0.15, 0.25)
BOX_HALF_HEIGHT = 0.6
PARKED = np.array([0.0, 0.0, -10.0])  # unused boxes wait under the floor
# Robot collision geoms by part, to report what hit a box.
ARM_GEOMS = ("shoulder", "elbow", "wrist", "hand")
TORSO_GEOMS = ("pelvis", "torso", "head")


def add_boxes(spec, count):
    """Add `count` movable boxes to the robot's model, parked under the floor."""
    # The scene lists its contact pairs one by one and has automatic contacts
    # off. Collision bit 2 makes the robot's collision geoms collide with the
    # boxes, and only with the boxes.
    for geom in spec.geoms:
        if geom.name.endswith("_collision"):
            geom.contype = 2
    for i in range(count):
        body = spec.worldbody.add_body(name=f"box{i}", mocap=True, pos=PARKED)
        body.add_geom(name=f"box{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[BOX_HALF_WIDTH[1], BOX_HALF_WIDTH[1], BOX_HALF_HEIGHT],
                      rgba=[0.8, 0.1, 0.1, 1], contype=0, conaffinity=2)


class ObstacleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": STEER_HZ}

    def __init__(self, walker=None, boxes=(0, 0), render_mode=None):
        """walker: folder of the saved walking policy; None holds the default
        pose instead (tests). boxes: (fewest, most) boxes per episode."""
        self.render_mode = render_mode
        spec = load_spec()
        add_boxes(spec, MAX_BOXES)
        self.robot = G1WalkEnv(model=spec.compile(), render_mode=render_mode,
                               command_ranges={"vx": VX_RANGE, "vy": (0.0, 0.0), "yaw": (0.0, 0.0)})
        self.walker = Walker(walker) if walker is not None else None
        self.walker_steps = CONTROL_HZ // STEER_HZ
        self.max_steps = self.robot.max_steps // self.walker_steps
        self.dt = 1 / STEER_HZ
        self.renderer = mujoco.Renderer(self.robot.model, IMAGE_SIZE, IMAGE_SIZE)

        model = self.robot.model
        self.box_geoms = [model.geom(f"box{i}").id for i in range(MAX_BOXES)]
        self.box_mocap = [model.body_mocapid[model.body(f"box{i}").id] for i in range(MAX_BOXES)]
        self.set_boxes(*boxes)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        })

    def set_boxes(self, low, high):
        """Boxes per episode, low to high (both included), from the next reset.
        The density curriculum raises it as the policy succeeds."""
        self.boxes_per_episode = (int(low), int(high))

    # -- Gymnasium API ---------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_observation, _ = self.robot.reset(seed=seed)
        self.speed = float(self.robot.command[0])           # the user's command: (speed, 0, 0)
        self.user_command = self.robot.command.copy()
        self._place_boxes()
        self.correction = np.zeros(3)
        self.last_action = np.zeros(3, dtype=np.float32)
        if self.walker is not None:
            self.walker.reset()
        self.steps = 0
        self.credited = 0.0                                 # progress paid so far, m
        self.max_lateral = 0.0
        return self._observation(), {"speed": self.speed, "boxes": len(self.box_centres)}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.correction += SMOOTHING * (RESIDUAL * action - self.correction)
        self.robot.set_command(np.clip(self.user_command + self.correction, LIMITS[:, 0], LIMITS[:, 1]))
        fell, hit_part = False, None
        for _ in range(self.walker_steps):
            joints = self.walker.act(self.robot_observation) if self.walker is not None else np.zeros(29)
            self.robot_observation, _, fell, _, _ = self.robot.step(joints)
            hit_part = self._box_hit_part()
            if fell or hit_part:
                break
        ended = bool(fell or hit_part)
        self.steps += 1
        self.last_action = action.astype(np.float32)
        x, y = float(self.robot.position[0]), float(self.robot.position[1])
        self.max_lateral = max(self.max_lateral, abs(y))

        truncated = not ended and self.steps >= self.max_steps
        # Progress is paid up to the line the user's command draws: speed x time.
        # Running ahead of it earns nothing; catching up after a detour does.
        credited = min(x, self.speed * self.steps * self.dt)
        progress = float(np.clip((credited - self.credited) / (self.speed * self.dt), -1.0, CATCH_UP))
        self.credited = credited
        proximity = float(np.clip(1.0 - self.clearance() / MARGIN, 0.0, 1.0))
        lateral = min(LATERAL_WEIGHT * abs(y), LATERAL_CAP)
        reward = END_REWARD if ended else progress - PROXIMITY_WEIGHT * proximity - lateral
        info = {"progress_rate": progress, "proximity": proximity, "lateral_cost": lateral,
                "command": self.robot.command.copy()}
        if ended or truncated:
            fraction = float(x / (self.speed * self.max_steps * self.dt))
            info.update({
                "progress": fraction,                 # distance along +x / commanded distance
                "box_hit": hit_part is not None,
                "box_hit_part": hit_part or "",       # "arm", "leg", "torso"
                "fall": hit_part is None and ended,
                "success": bool(truncated and fraction >= SUCCESS_PROGRESS),
                "lateral": self.max_lateral,          # farthest it strayed sideways, m
                "end_y": y,
            })
        return self._observation(), reward, ended, truncated, info

    def render(self):
        """The walker env's video frame, with a second, thinner white arrow: the
        user's command. Where it shows beside the yellow one (the command the
        walker follows), the difference is this policy's correction."""
        return self.robot.render(arrows=[(self.user_command, USER_RGBA)])

    def close(self):
        self.renderer.close()
        self.robot.close()

    # -- Observation -----------------------------------------------------------

    def camera_image(self):
        self.renderer.update_scene(self.robot.data, camera="front")
        return self.renderer.render().copy()

    def _observation(self):
        yaw = self.robot.yaw()
        state = np.array([self.speed, np.sin(yaw), np.cos(yaw), *self.last_action], dtype=np.float32)
        return {"image": self.camera_image(), "state": state}

    # -- Boxes -----------------------------------------------------------------

    def _place_boxes(self):
        """Put this episode's boxes at random in the cone: none within CONE_NEAR
        of the start, centres at least BOX_SPACING apart."""
        # Drawn with the robot env's generator, right after it drew the command:
        # a seed gives the same course it gave in every recorded evaluation.
        rng = self.robot.np_random
        far = min(CONE_FAR, self.speed * self.robot.max_steps * self.robot.dt + 1.0)
        wanted = rng.integers(self.boxes_per_episode[0], self.boxes_per_episode[1] + 1)
        centres = []
        for _ in range(1000):  # rejection sampling: draw a point, keep it if it is far enough from the others
            if len(centres) == wanted:
                break
            r = np.sqrt(rng.uniform(CONE_NEAR ** 2, far ** 2))  # uniform over the cone's area
            angle = rng.uniform(-CONE_HALF_ANGLE, CONE_HALF_ANGLE)
            centre = r * np.array([np.cos(angle), np.sin(angle)])
            if all(np.linalg.norm(centre - c) >= BOX_SPACING for c in centres):
                centres.append(centre)
        model, data = self.robot.model, self.robot.data
        for i in range(MAX_BOXES):
            if i < len(centres):
                half = rng.uniform(*BOX_HALF_WIDTH)
                model.geom_size[self.box_geoms[i]] = [half, half, BOX_HALF_HEIGHT]
                model.geom_rbound[self.box_geoms[i]] = np.linalg.norm([half, half, BOX_HALF_HEIGHT])
                data.mocap_pos[self.box_mocap[i]] = [*centres[i], BOX_HALF_HEIGHT]
            else:
                data.mocap_pos[self.box_mocap[i]] = PARKED
        self.box_centres = np.array(centres).reshape(-1, 2)
        mujoco.mj_forward(model, data)

    def clearance(self):
        """Distance (m, horizontal) from the pelvis to the nearest box face; inf without boxes."""
        if len(self.box_centres) == 0:
            return np.inf
        half = self.robot.model.geom_size[self.box_geoms[:len(self.box_centres)], :2]
        outside = np.maximum(np.abs(self.robot.position - self.box_centres) - half, 0.0)
        return float(np.linalg.norm(outside, axis=1).min())

    def _box_hit_part(self):
        """Which part of the robot touches a box: "torso", "leg" or "arm"; None
        when nothing does. "arm" only when the arms alone touch it: the body
        itself would have cleared the box."""
        data, model = self.robot.data, self.robot.model
        boxes = set(self.box_geoms)
        parts = set()
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if g1 in boxes or g2 in boxes:
                name = model.geom(g2 if g1 in boxes else g1).name
                parts.add("arm" if any(p in name for p in ARM_GEOMS)
                          else "torso" if any(p in name for p in TORSO_GEOMS) else "leg")
        for part in ("torso", "leg", "arm"):
            if part in parts:
                return part
        return None
