"""G1 walking environment (Gymnasium): follow a velocity command without falling.

    command      (vx, vy, yaw rate), drawn once per episode
    action       29 numbers in [-1, 1]: joint target = default pose + 0.5 * action,
                 held by the model's PD controllers; 50 Hz
    observation  proprio     98 floats the real robot can measure (below)
                 privileged   6 floats only the simulator knows; the critic
                              gets them, the actor does not (policy.py)
                 image       64x64x3, always black (see IMAGE below)
    reward       in [0, 1] per step, three parts (see _reward); a fall pays -100
    episode      20 s; ends early on a fall

    from walk_env import G1WalkEnv
    env = G1WalkEnv()
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Why each number is what it is: notes/2026-10-28-t2-env-design.md. What was
tried and dropped: notes/2026-11-10-experiments-record.md.
"""

import gymnasium as gym
import mujoco
import numpy as np

from robot import ARM_JOINTS, CONTROL_HZ, DEFAULT_POSE, IMAGE_SIZE, SWING_JOINTS, WAIST_JOINTS, load_spec

EPISODE_SECONDS = 20
ACTION_SCALE = 0.5       # rad of joint target per unit of action (MuJoCo Playground)

# The largest commands; the curriculum opens the ranges up to these.
COMMAND_LIMITS = {"vx": (-0.5, 1.0), "vy": (-0.5, 0.5), "yaw": (-1.0, 1.0)}
STILL_COMMAND = 0.1      # below this command norm the robot should stand: clock off, both feet down

# A fall: the pelvis too low or too tilted, or any body part but a foot on the floor.
FALL_HEIGHT = 0.2        # m
FALL_TILT = 0.8          # rad
FALL_REWARD = -100.0     # = -1 / (1 - gamma) for gamma = 0.99: a whole perfect future

# Reward part 1, tracking: exp(-error^2 / sigma^2), the mean of a wide kernel
# (a gradient far from the command) and a sharp one (standing still under a
# 0.3 m/s command must not pay). Playground and unitree_rl_lab use the wide one.
TRACKING_SIGMA2 = 0.25
TRACKING_SHARP_SIGMA2 = 0.05
REWARD_FLOOR = 0.2       # share of the tracking part paid for just staying up (walk_curriculum.py removes it)
SUCCESS_TRACKING = 0.8   # success = a full episode with a mean wide kernel of at least this

# Reward part 1, style: each cost c becomes exp(-c / sigma), and the style is
# their mean. sigma is the median cost of a random policy (tools/calibrate.py),
# so a random policy scores 0.37 on every term.
STYLE_SIGMAS = {
    "orientation": 0.0215,   # |gravity_xy|^2 in the pelvis frame, ~8 deg of tilt
    "ang_vel_xy": 10.6,      # pelvis roll/pitch rate^2, (rad/s)^2
    "lin_vel_z": 0.0622,     # vertical pelvis velocity^2, (m/s)^2
    "foot_slip": 0.0656,     # horizontal velocity^2 of the feet on the ground, (m/s)^2
    "pose": 0.467,           # rad^2 from the default pose, the swing joints left out
    "joint_limits": 0.0425,  # rad beyond the soft limits (the inner 90% of each joint's range)
    "action_rate": 19.2,     # |a_t - a_(t-1)|^2
}

# Reward part 2, gait: the feet should touch the ground as a metronome says
# (unitree_rl_gym's feet_gait). The policy sees the same clock.
GAIT_PERIOD = 0.8          # s per full step cycle
GAIT_OFFSETS = (0.0, 0.5)  # left and right leg, half a cycle apart
GAIT_STANCE = 0.55         # share of the cycle a foot is down
GAIT_SHARE = 0.2

# Reward part 3, joint reference: during a leg's swing, hip pitch, knee and
# ankle pitch should follow a sine bump on top of the default pose that lifts
# the foot and keeps it level (humanoid-gym's ref_joint_pos). Unlike the gait
# part, it pays for every bit of a lift, so it rewards the first attempt at a step.
REF_AMPLITUDE = 0.17       # rad
REF_PATTERN = np.array([-1.0, 2.0, -1.0])   # hip pitch, knee, ankle pitch
REFERENCE_SHARE = 0.3

# Video camera (render()): third person, behind the robot's right shoulder and
# above it, turning with its heading so what lies ahead is in shot. From
# straight behind, the robot's own body hides the floor just ahead of it,
# where the command arrows are. The heading it follows is
# smoothed over ~0.5 s, so the gait's sway does not shake the picture.
VIDEO_SIZE = (480, 640)    # height, width: the scene's largest offscreen frame
CHASE_DISTANCE = 4.0       # m from the point it looks at
CHASE_ELEVATION = -30.0    # deg, looking down
CHASE_SIDE = 30.0          # deg the camera sits round to the right of straight behind
CHASE_LEAD = 1.0           # m: it looks at a point this far ahead of the pelvis ...
CHASE_HEIGHT = 0.6         # ... at this height
CHASE_TIME = 0.5           # s, time constant of the heading smoothing
# The command in the video: an arrow on the floor from just ahead of the feet, along
# the commanded (vx, vy), as long as the distance the command asks for in
# ARROW_SECONDS. On the floor the camera sees it from above, at full length.
# The yaw rate is not drawn.
ARROW_HEIGHT = 0.03        # m above the floor
ARROW_SECONDS = 1.5
ARROW_WIDTH = 0.045        # m
ARROW_LEAD = 0.3           # m: arrows start this far ahead of the pelvis, clear of the feet
COMMAND_RGBA = (0.95, 0.68, 0.0, 1.0)   # yellow: the command the walker follows

# IMAGE. The walker does not use a camera: obstacle_env.py steers it from
# outside. The key stays because the frozen walker (pretrained/g1_vel_v1) was
# trained with this observation layout, back when the plan was to switch a
# camera on inside the walker itself. A walker trained from scratch should drop it.


class G1WalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, render_mode=None, episode_seconds=EPISODE_SECONDS, command_ranges=None, model=None):
        """command_ranges: (low, high) for any of vx, vy, yaw; the rest start as
        forward walking only. model: a compiled model to use instead of the
        plain robot (obstacle_env.py passes one with boxes in it)."""
        self.render_mode = render_mode
        self.model = model if model is not None else load_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.dt = 1 / CONTROL_HZ
        self.physics_steps = round(self.dt / self.model.opt.timestep)   # physics steps per action
        self.max_steps = round(episode_seconds * CONTROL_HZ)
        self.video_renderer = None  # created on the first render()
        self.chase = mujoco.MjvCamera()  # a free camera render() moves along with the robot
        self.chase.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.chase.distance, self.chase.elevation = CHASE_DISTANCE, CHASE_ELEVATION
        self.chase_yaw = None

        self.pelvis = self.model.body("pelvis").id
        self.floor = self.model.geom("floor").id
        self.feet = [
            {self.model.geom(f"{side}_foot{part}_collision").id for part in ("_box", "1", "2", "3")}
            for side in ("left", "right")
        ]
        self.foot_sites = [self.model.site(f"{side}_foot").id for side in ("left", "right")]
        # Every other geom the scene lets touch the floor (hands, knees, pelvis ...).
        floor_pairs = zip(self.model.pair_geom1, self.model.pair_geom2)
        self.body_floor_geoms = {a if b == self.floor else b for a, b in floor_pairs
                                 if self.floor in (a, b)} - set().union(*self.feet)
        self.ctrl_low, self.ctrl_high = self.model.actuator_ctrlrange.T
        low, high = self.model.jnt_range[1:].T
        mid, half = (low + high) / 2, (high - low) / 2
        self.soft_low, self.soft_high = mid - 0.9 * half, mid + 0.9 * half
        self.pose_joints = [j for j in range(self.model.nu) if j not in SWING_JOINTS]

        self.command_ranges = {"vx": (0.0, 1.0), "vy": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.set_command_ranges(**(command_ranges or {}))
        self.set_reward()

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            "proprio": gym.spaces.Box(-np.inf, np.inf, shape=(98,), dtype=np.float32),
            "privileged": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        })

    # -- Knobs the curricula and the obstacle env turn -------------------------

    def set_command_ranges(self, **ranges):
        """Set the (low, high) range of any of vx, vy, yaw, clamped to
        COMMAND_LIMITS. Takes effect at the next reset."""
        for axis, (low, high) in ranges.items():
            limit_low, limit_high = COMMAND_LIMITS[axis]
            self.command_ranges[axis] = (max(low, limit_low), min(high, limit_high))

    def set_reward(self, floor=REWARD_FLOOR, kernel_scale=1.0):
        """The survival floor and a scale on both tracking variances (smaller =
        sharper). Takes effect at the next step."""
        self.reward_floor, self.kernel_scale = floor, kernel_scale

    def set_command(self, command):
        """Replace the command in the middle of an episode."""
        self.command = np.asarray(command, dtype=float).copy()

    # -- Gymnasium API ---------------------------------------------------------

    def reset(self, seed=None, options=None):
        """options={"command": (vx, vy, yaw)} starts the episode with that command."""
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.chase_yaw = None  # the video camera jumps to the new episode's start
        self.data.qpos[7:] = DEFAULT_POSE
        self.data.ctrl[:] = DEFAULT_POSE
        mujoco.mj_forward(self.model, self.data)
        self.command = np.array([self.np_random.uniform(*self.command_ranges[axis]) for axis in ("vx", "vy", "yaw")])
        if options and "command" in options:  # a given command instead of the drawn one
            self.set_command(options["command"])
        self.last_action = np.zeros(self.model.nu)
        self.steps = 0
        self.tracking_sum = 0.0
        self.feet_down, _ = self._floor_contacts()
        return self._observation(), {"command": self.command.copy()}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = DEFAULT_POSE + ACTION_SCALE * action
        self.data.ctrl[:] = np.clip(target, self.ctrl_low, self.ctrl_high)
        for _ in range(self.physics_steps):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        self.feet_down, body_down = self._floor_contacts()
        terminated = body_down or self._tipped()
        truncated = not terminated and self.steps >= self.max_steps
        if terminated:
            reward, info = FALL_REWARD, {}
        else:
            reward, info = self._reward(action)
            self.tracking_sum += info["tracking_wide"]
        self.last_action = action
        if terminated or truncated:
            # The steps lost to a fall count as 0 tracking: divide by the full length.
            mean_tracking = self.tracking_sum / self.max_steps
            info["episode_tracking"] = mean_tracking
            info["success"] = bool(truncated and mean_tracking >= SUCCESS_TRACKING)
            # Metric only: distance covered along the commanded direction.
            speed = np.linalg.norm(self.command[:2])
            direction = self.command[:2] / speed if speed > 1e-6 else np.zeros(2)
            info["distance"] = float(self.position @ direction)
        return self._observation(), float(reward), terminated, truncated, info

    def render(self, arrows=()):
        """A 480x640 frame from over the robot's shoulder, for videos (see
        VIDEO_SIZE), with the command as a yellow arrow over its head.
        arrows: more (command, rgba) pairs, drawn the same way but thinner and
        just under it, so an equal command hides under the yellow arrow."""
        if self.video_renderer is None:
            self.video_renderer = mujoco.Renderer(self.model, *VIDEO_SIZE)
        yaw = self.yaw()
        if self.chase_yaw is None:
            self.chase_yaw, self.chase_time = yaw, self.data.time
        # Turn part of the way towards the robot's heading: more the longer since the last frame.
        share = 1.0 - np.exp(-(self.data.time - self.chase_time) / CHASE_TIME)
        self.chase_yaw += share * np.angle(np.exp(1j * (yaw - self.chase_yaw)))  # the short way round
        self.chase_time = self.data.time
        ahead = np.array([np.cos(self.chase_yaw), np.sin(self.chase_yaw)])
        self.chase.lookat[:] = [*(self.position + CHASE_LEAD * ahead), CHASE_HEIGHT]
        self.chase.azimuth = np.degrees(self.chase_yaw) + CHASE_SIDE  # azimuth = heading: straight behind
        self.video_renderer.update_scene(self.data, self.chase)
        for i, (command, rgba) in enumerate([(self.command, COMMAND_RGBA), *arrows]):
            self._add_arrow(self.video_renderer.scene, command, rgba, ARROW_HEIGHT - 0.01 * i,
                            ARROW_WIDTH * (0.6 if i else 1.0))
        return self.video_renderer.render()

    def _add_arrow(self, scene, command, rgba, height, width):
        """Draw a command's (vx, vy), given in the robot's heading frame, as an arrow in the world."""
        c, s = np.cos(self.yaw()), np.sin(self.yaw())
        vx, vy = command[0], command[1]
        start = np.array([*(self.position + ARROW_LEAD * np.array([c, s])), height])
        end = start + ARROW_SECONDS * np.array([c * vx - s * vy, s * vx + c * vy, 0.0])
        if np.linalg.norm(end - start) < 0.02 or scene.ngeom >= scene.maxgeom:  # nothing to draw, or no room
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)
        scene.ngeom += 1

    def close(self):
        if self.video_renderer is not None:
            self.video_renderer.close()

    # -- Where the robot is ----------------------------------------------------

    @property
    def position(self):
        """(x, y) of the pelvis, m. Episodes start at the origin facing +x."""
        return self.data.qpos[:2]

    def yaw(self):
        """Heading of the pelvis, rad."""
        w, x, y, z = self.data.qpos[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # -- Observation -----------------------------------------------------------

    def _observation(self):
        proprio = np.concatenate([
            self.data.sensor("gyro_pelvis").data,      # angular velocity, pelvis frame (IMU)
            self._gravity_in_pelvis(),                 # which way is down, pelvis frame
            self.command,
            self._clock(),                             # gait metronome: where in the step cycle we are
            self.data.qpos[7:] - DEFAULT_POSE,         # joint angles
            self.data.qvel[6:],                        # joint velocities
            self.last_action,
        ]).astype(np.float32)
        privileged = np.concatenate([
            self._velocity_heading(),                  # true base velocity, heading frame
            [self.data.qpos[2]],                       # base height
            np.array(self.feet_down, dtype=float),
        ]).astype(np.float32)
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)  # see IMAGE at the top
        return {"image": image, "proprio": proprio, "privileged": privileged}

    def _clock(self):
        """sin and cos of the gait phase: a timer the controller runs itself, so
        the real robot has it too. Zero when the robot should stand still."""
        if np.linalg.norm(self.command) < STILL_COMMAND:
            return np.zeros(2)
        phase = 2 * np.pi * self.steps * self.dt / GAIT_PERIOD
        return np.array([np.sin(phase), np.cos(phase)])

    def _velocity_heading(self):
        """Base velocity (vx, vy, vz) in the heading frame: x forward, whatever the tilt."""
        yaw = self.yaw()
        v = self.data.sensor("global_linvel_pelvis").data
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])

    def _gravity_in_pelvis(self):
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)  # pelvis frame -> world
        return rotation.T @ np.array([0.0, 0.0, -1.0])

    # -- Falls and contacts ----------------------------------------------------

    def _floor_contacts(self):
        """One pass over the contacts: ([left foot down, right foot down],
        whether any other body part touches the floor)."""
        feet_down, body_down = [False, False], False
        for i in range(self.data.ncon):
            geoms = {self.data.contact[i].geom1, self.data.contact[i].geom2}
            if self.floor not in geoms:
                continue
            for foot, foot_geoms in enumerate(self.feet):
                if geoms & foot_geoms:
                    feet_down[foot] = True
            if geoms & self.body_floor_geoms:
                body_down = True
        return feet_down, body_down

    def _tipped(self):
        tilt = np.arccos(np.clip(-self._gravity_in_pelvis()[2], -1.0, 1.0))
        return bool(self.data.qpos[2] < FALL_HEIGHT or tilt > FALL_TILT)

    # -- Reward ----------------------------------------------------------------

    def _reward(self, action):
        """Per step, in [0, 1]:

            0.5 * (floor + (1 - floor) * tracking) * style  +  0.2 * gait  +  0.3 * reference

        The gait and reference parts are added, not multiplied in: they pay for
        stepping even while the robot is far from the commanded speed. Inside
        the product a first step earns almost nothing, and the policy learns to
        stand instead."""
        error2 = self._tracking_error2()
        wide = float(np.exp(-error2 / TRACKING_SIGMA2))
        tracking = 0.5 * float(np.exp(-error2 / (self.kernel_scale * TRACKING_SIGMA2))) \
            + 0.5 * float(np.exp(-error2 / (self.kernel_scale * TRACKING_SHARP_SIGMA2)))
        costs = self._costs(action)
        kernels = {name: np.exp(-costs[name] / sigma) for name, sigma in STYLE_SIGMAS.items()}
        style = float(np.mean(list(kernels.values())))
        gait = self._gait()
        reference = self._reference()
        track = self.reward_floor + (1 - self.reward_floor) * tracking
        reward = (1 - GAIT_SHARE - REFERENCE_SHARE) * track * style + GAIT_SHARE * gait + REFERENCE_SHARE * reference
        # tracking_wide is the metric: the wide kernel alone, whatever set_reward did.
        info = {"tracking": tracking, "tracking_wide": wide, "style": style, "gait": gait, "reference": reference}
        info.update({f"cost/{name}": cost for name, cost in costs.items()})
        info.update({f"kernel/{name}": value for name, value in kernels.items()})
        return reward, info

    def _tracking_error2(self):
        """Squared error of the planar velocity (heading frame) and the yaw rate."""
        v = self._velocity_heading()
        yaw_rate = self.data.sensor("global_angvel_pelvis").data[2]
        return float(np.sum((v[:2] - self.command[:2]) ** 2) + (yaw_rate - self.command[2]) ** 2)

    def _costs(self, action):
        """The seven style costs of STYLE_SIGMAS, then four readings of the upper
        body that are reported (training logs, tools/wobble.py) but not rewarded."""
        gravity = self._gravity_in_pelvis()
        gyro = self.data.sensor("gyro_pelvis").data
        offset = self.data.qpos[7:] - DEFAULT_POSE
        q = self.data.qpos[7:]
        beyond = np.clip(self.soft_low - q, 0, None) + np.clip(q - self.soft_high, 0, None)
        return {
            "orientation": float(np.sum(gravity[:2] ** 2)),
            "ang_vel_xy": float(np.sum(gyro[:2] ** 2)),
            "lin_vel_z": float(self.data.sensor("global_linvel_pelvis").data[2] ** 2),
            "foot_slip": float(sum(np.sum(self._foot_velocity(foot)[:2] ** 2)
                                   for foot, down in enumerate(self.feet_down) if down)),
            "pose": float(np.sum(offset[self.pose_joints] ** 2)),
            "joint_limits": float(np.sum(beyond)),
            "action_rate": float(np.sum((action - self.last_action) ** 2)),
            "torso_orientation": float(np.sum(self.data.sensor("upvector_torso").data[:2] ** 2)),  # sin^2 of the torso tilt
            "torso_ang_vel_xy": float(np.sum(self.data.sensor("gyro_torso").data[:2] ** 2)),
            "waist": float(np.sum(offset[WAIST_JOINTS] ** 2)),
            "arms": float(np.sum(offset[ARM_JOINTS] ** 2)),
        }

    def _gait(self):
        """Share of the feet doing what the clock says. With a still command the
        right thing is both feet down."""
        if np.linalg.norm(self.command) < STILL_COMMAND:
            return float(np.mean(self.feet_down))
        t = self.steps * self.dt
        stance = [((t / GAIT_PERIOD + offset) % 1.0) < GAIT_STANCE for offset in GAIT_OFFSETS]
        return float(np.mean([s == down for s, down in zip(stance, self.feet_down)]))

    def _reference_pose(self):
        """Where the swing joints (SWING_JOINTS order) should be at this step."""
        target = DEFAULT_POSE[SWING_JOINTS].copy()
        if np.linalg.norm(self.command) < STILL_COMMAND:
            return target
        t = self.steps * self.dt
        for leg, offset in enumerate(GAIT_OFFSETS):
            phase = (t / GAIT_PERIOD + offset) % 1.0
            if phase >= GAIT_STANCE:  # swing: one sine bump over the swing part of the cycle
                bump = np.sin(np.pi * (phase - GAIT_STANCE) / (1 - GAIT_STANCE))
                target[3 * leg:3 * leg + 3] += bump * REF_AMPLITUDE * REF_PATTERN
        return target

    def _reference(self):
        """exp(-2 |q - target|) over the swing joints, as humanoid-gym."""
        error = self.data.qpos[7:][SWING_JOINTS] - self._reference_pose()
        return float(np.exp(-2.0 * np.linalg.norm(error)))

    def _foot_velocity(self, foot):
        velocity = np.zeros(6)  # [angular, linear], world frame
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                 self.foot_sites[foot], velocity, 0)
        return velocity[3:]
