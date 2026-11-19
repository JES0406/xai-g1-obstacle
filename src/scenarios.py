"""Visual conditions of the obstacle env, changed at run time on env.robot.model.

The policy was trained with one look only: red boxes, blue-grey checker floor,
one spotlight with shadows plus the headlight. Every variant here is therefore
out of distribution. Variants never touch geometry, so the same seed gives the
same command and box layout in every condition.

    env = make_env(boxes=10)
    look = Look(env)
    look.apply("box_colour")      # until the next apply()
    obs, _ = env.reset(seed=2000)

Baselines replace the policy's input or the policy itself (see `blind_image`).
"""

import numpy as np

# The variants. Values are kept away from the training look on purpose; any
# later domain randomisation must use ranges disjoint from these.
BOX_COLOUR_OOD = (0.1, 0.3, 0.9, 1.0)        # blue instead of red (0.8, 0.1, 0.1)
FLOOR_TINT_OOD = (1.0, 0.75, 0.45, 1.0)      # warm tint multiplied onto the checker texture
SPOT_POS_OOD = (0.0, 2.0, 4.0)               # spotlight mirrored to the other side (was 0, -2, 4)
SPOT_DIFFUSE_SCALE_OOD = 0.6                 # dimmer spotlight
HEADLIGHT_DIFFUSE_OOD = (0.3, 0.3, 0.3)      # was 0.6
HEADLIGHT_AMBIENT_OOD = (0.25, 0.25, 0.25)   # was 0.1: flatter light, no shadows

VARIANTS = ("id", "box_colour", "floor_tint", "lighting")
BASELINES = ("walker", "black", "noise")


class Look:
    """Applies a visual variant to an env's model; remembers the training look
    so that variants never stack."""

    def __init__(self, env):
        self.env = env
        model = env.robot.model
        self.floor_material = model.geom_matid[model.geom("floor").id]
        self.spot = model.light("spotlight").id
        self.default = {
            "box_rgba": model.geom_rgba[env.box_geoms].copy(),
            "floor_rgba": model.mat_rgba[self.floor_material].copy(),
            "spot_pos": model.light_pos[self.spot].copy(),
            "spot_diffuse": model.light_diffuse[self.spot].copy(),
            "spot_castshadow": int(model.light_castshadow[self.spot]),
            "head_diffuse": model.vis.headlight.diffuse.copy(),
            "head_ambient": model.vis.headlight.ambient.copy(),
        }
        self.current = "id"

    def apply(self, variant):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; one of {VARIANTS}")
        self._restore()
        model = self.env.robot.model
        if variant == "box_colour":
            model.geom_rgba[self.env.box_geoms] = BOX_COLOUR_OOD
        elif variant == "floor_tint":
            model.mat_rgba[self.floor_material] = FLOOR_TINT_OOD
        elif variant == "lighting":
            model.light_pos[self.spot] = SPOT_POS_OOD
            model.light_diffuse[self.spot] = self.default["spot_diffuse"] * SPOT_DIFFUSE_SCALE_OOD
            model.light_castshadow[self.spot] = 0
            model.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE_OOD
            model.vis.headlight.ambient[:] = HEADLIGHT_AMBIENT_OOD
        self.current = variant

    def _restore(self):
        model, d = self.env.robot.model, self.default
        model.geom_rgba[self.env.box_geoms] = d["box_rgba"]
        model.mat_rgba[self.floor_material] = d["floor_rgba"]
        model.light_pos[self.spot] = d["spot_pos"]
        model.light_diffuse[self.spot] = d["spot_diffuse"]
        model.light_castshadow[self.spot] = d["spot_castshadow"]
        model.vis.headlight.diffuse[:] = d["head_diffuse"]
        model.vis.headlight.ambient[:] = d["head_ambient"]


def blind_image(kind, image, rng):
    """The image a baseline policy sees instead of the camera's."""
    if kind == "black":
        return np.zeros_like(image)
    if kind == "noise":
        return rng.integers(0, 256, size=image.shape, dtype=np.uint8)
    raise ValueError(kind)


class Segmenter:
    """Per-pixel box index (-1 = no box) for the camera "front", rendered from
    the same MjData as the observation image."""

    GEOM = 5   # mujoco.mjtObj.mjOBJ_GEOM

    def __init__(self, env):
        import mujoco

        from robot import IMAGE_SIZE

        self.env = env
        self.renderer = mujoco.Renderer(env.robot.model, IMAGE_SIZE, IMAGE_SIZE)
        self.renderer.enable_segmentation_rendering()
        self.box_index = {geom: i for i, geom in enumerate(env.box_geoms)}

    def mask(self):
        self.renderer.update_scene(self.env.robot.data, camera="front")
        seg = self.renderer.render()                  # (H, W, 2): object id, object type
        ids, types = seg[..., 0], seg[..., 1]
        out = np.full(ids.shape, -1, dtype=np.int8)
        for geom, i in self.box_index.items():
            out[(types == self.GEOM) & (ids == geom)] = i
        return out

    def close(self):
        self.renderer.close()
