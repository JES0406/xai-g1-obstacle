"""Multiprocess rollouts with fixed seeds. Per step: state, action, raw mean
(before the clip), V, clearance, qpos, collision/fall events. Image and
segmentation mask saved inline for 20 episodes per set. Writes data/."""
