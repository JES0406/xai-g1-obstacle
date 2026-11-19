"""Rollouts of the PPO policy, logged step by step. One .npz per episode.

A job is one episode: (set, visual variant, baseline, number of boxes, seed).
The seed fixes the user command and the box layout, so the same seed in two
sets gives the same course under two conditions.

Per step (arrays of length T, the observation at t and what followed):
    state (T,6)  mean (T,3) raw policy mean before the clip  value (T,)
    action (T,3) what was applied  command (T,3) command the walker got
    clearance (T,) m to the nearest box face  qpos (T,36)  qvel (T,6) base
    reward (T,)
    image (T,64,64,3) uint8 and mask (T,64,64) int8 box index (-1 none):
    only in episodes with save_image, rendered from the same MjData as the
    observation (the mask is never re-rendered later).
Per episode: box_centres (n,2), box_half (n,), and `meta` (JSON): the job,
the user's speed and the outcome (success, box_hit, box_hit_part, fall,
progress, lateral, end_y).

Episodes are written as <out>/<set>/b<boxes>_s<seed>.npz; existing files are
skipped, so an interrupted run resumes where it stopped.
"""

import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NOISE_SEED_OFFSET = 7919   # noise baseline: its own RNG stream, fixed by the episode seed


@dataclass(frozen=True)
class Job:
    set: str
    variant: str          # visual look, see scenarios.VARIANTS
    baseline: str         # "" = the policy; else scenarios.BASELINES
    boxes: int
    seed: int
    save_image: bool

    def path(self, root):
        return Path(root) / self.set / f"b{self.boxes:02d}_s{self.seed}.npz"


# -- Job list --------------------------------------------------------------------


def build_jobs(config, sets=None, episodes=None):
    """All jobs of configs/sets.yaml. sets: names to keep (None = all, including
    baselines as baseline_<kind>). episodes: override of the episodes per density
    (debugging with a few episodes)."""
    from src.scenarios import BASELINES, VARIANTS

    image_total = config["image_episodes_per_set"]
    jobs = []
    for name, spec in config["sets"].items():
        if "like" in spec:
            spec = {**config["sets"][spec["like"]], **{k: v for k, v in spec.items() if k != "like"}}
        variant = spec.get("variant", "id")
        if variant not in VARIANTS:
            raise ValueError(f"set {name}: unknown variant {variant!r}; one of {VARIANTS}")
        jobs += _set_jobs(name, variant, "", spec, image_total, episodes)
    base = config["baselines"]
    seeds_from = config["sets"][base.get("seeds_like", "test_id")]["seeds_from"]
    for kind in base["kinds"]:
        if kind not in BASELINES:
            raise ValueError(f"unknown baseline {kind!r}; one of {BASELINES}")
        spec = {"seeds_from": seeds_from, "boxes": base["boxes"], "episodes_per_density": base["episodes_per_density"]}
        jobs += _set_jobs(f"baseline_{kind}", "id", kind, spec, 0, episodes)
    if sets:
        unknown = set(sets) - {j.set for j in jobs}
        if unknown:
            raise ValueError(f"unknown sets {sorted(unknown)}")
        jobs = [j for j in jobs if j.set in sets]
    return jobs


def _set_jobs(name, variant, baseline, spec, image_total, episodes):
    densities = spec["boxes"]
    n = episodes or spec["episodes_per_density"]
    # image episodes spread over densities: the first k seeds of each density
    share = [image_total // len(densities) + (i < image_total % len(densities)) for i in range(len(densities))]
    jobs = []
    for boxes, k in zip(densities, share):
        for i in range(n):
            jobs.append(Job(name, variant, baseline, boxes, spec["seeds_from"] + i, i < min(k, n)))
    return jobs


# -- Worker ----------------------------------------------------------------------

_worker = {}


def _init_worker(policy_dir=None):
    import torch

    from src.load_policy import PPO_DIR, SteeringNet, load_ppo, make_env
    from src.scenarios import Look, Segmenter

    torch.set_num_threads(1)
    model = load_ppo(policy_dir or PPO_DIR)
    env = make_env()
    _worker.update(model=model, net=SteeringNet(model.policy).eval(), env=env,
                   look=Look(env), seg=Segmenter(env))


def run_episode(job, out):
    """Run one job and write its .npz. Returns a short summary dict."""
    import torch

    from src.attribution import local_maps
    from src.load_policy import to_inputs
    from src.scenarios import attributed_region, blind_image, hide_region, random_region_like

    path = job.path(out)
    if path.exists():
        return {"job": asdict(job), "skipped": True}
    w = _worker
    env, model, net = w["env"], w["model"], w["net"]
    start = time.time()

    w["look"].apply(job.variant)
    env.set_boxes(job.boxes, job.boxes)
    obs, _ = env.reset(seed=job.seed)
    rng = np.random.default_rng(job.seed + NOISE_SEED_OFFSET)
    log = {k: [] for k in ("state", "mean", "value", "action", "command", "clearance", "qpos", "qvel", "reward")}
    images, masks = [], []
    while True:
        image = obs["image"]
        if job.baseline in ("black", "noise"):
            image = blind_image(job.baseline, image, rng)
        elif job.baseline in ("mask_top", "mask_random"):
            # behaviour test: hide the region Grad-CAM (conv2, evidence for braking) points at, or a
            # region of the same shape elsewhere, before the policy sees the image
            cam = local_maps(net, image[None], obs["state"][None], "vx", "gradcam_conv2")[0]
            region = attributed_region(cam)
            if job.baseline == "mask_random":
                region = random_region_like(region, rng)
            image = hide_region(image, region)
        if job.save_image:
            images.append(image)
            masks.append(w["seg"].mask())
        if job.baseline == "walker":
            mean, value = np.full(3, np.nan, np.float32), np.nan
            action = np.zeros(3, np.float32)
        else:
            x, s = to_inputs(model.policy, {"image": image, "state": obs["state"]})
            with torch.no_grad():
                m, v = net(x, s)
            mean, value = m.numpy()[0], float(v)
            action = np.clip(mean, -1.0, 1.0)
        log["state"].append(obs["state"])
        log["mean"].append(mean)
        log["value"].append(value)
        log["action"].append(action)
        log["clearance"].append(env.clearance())
        log["qpos"].append(env.robot.data.qpos.copy())
        log["qvel"].append(env.robot.data.qvel[:6].copy())
        obs, reward, terminated, truncated, info = env.step(action)
        log["command"].append(info["command"])
        log["reward"].append(reward)
        if terminated or truncated:
            break

    n = len(env.box_centres)
    outcome = {k: info[k] for k in ("success", "box_hit", "box_hit_part", "fall", "progress", "lateral", "end_y")}
    meta = {**asdict(job), "speed": env.speed, "steps": len(log["reward"]), **outcome}
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in log.items()}
    arrays["box_centres"] = env.box_centres.astype(np.float32)
    arrays["box_half"] = env.robot.model.geom_size[env.box_geoms[:n], 0].astype(np.float32)
    if job.save_image:
        arrays["image"] = np.asarray(images, dtype=np.uint8)
        arrays["mask"] = np.asarray(masks, dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    os.replace(tmp, path)
    return {"job": asdict(job), "skipped": False, "seconds": time.time() - start, **outcome}


# -- Launcher --------------------------------------------------------------------


def run(jobs, out, processes=8, policy=None, log=print):
    """Run the jobs not yet on disk with a pool of `processes` workers.
    policy: folder with model.zip (default checkpoints/planPW)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_run_info(out, processes, policy)
    pending = [j for j in jobs if not j.path(out).exists()]
    log(f"{len(jobs)} episodes, {len(jobs) - len(pending)} already on disk, {len(pending)} to run on {processes} processes")
    if not pending:
        return
    # image episodes first: they are the slowest and the ones the analysis needs
    pending.sort(key=lambda j: not j.save_image)
    start, done = time.time(), 0
    ctx = mp.get_context("spawn")   # fresh processes: no GL context or torch state inherited
    with ctx.Pool(processes, initializer=_init_worker, initargs=(policy,)) as pool:
        for result in pool.imap_unordered(partial(run_episode, out=out), pending):
            done += 1
            if done % max(1, len(pending) // 50) == 0 or done == len(pending):
                elapsed = time.time() - start
                eta = elapsed / done * (len(pending) - done)
                log(f"  {done}/{len(pending)}  {elapsed / 60:5.1f} min  eta {eta / 60:5.1f} min")


def _write_run_info(out, processes, policy=None):
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO).stdout.strip()
        except OSError:
            return ""

    info = {
        "host": platform.node(),
        "gpu": sh("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
        "commit": sh("git", "rev-parse", "--short", "HEAD"),
        "dirty": bool(sh("git", "status", "--porcelain", "--untracked-files=no")),
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "processes": processes,
        "policy": str(Path(policy).resolve().relative_to(REPO)) if policy else "checkpoints/planPW",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    runs = out / "RUN.json"
    history = json.loads(runs.read_text()) if runs.exists() else []
    hosts = {h["host"] for h in history} | {info["host"]}
    if len(hosts) > 1:
        print(f"WARNING: {out} mixes episodes from several machines {sorted(hosts)}: renders differ between GPUs")
    policies = {h.get("policy", "checkpoints/planPW") for h in history} | {info["policy"]}
    if len(policies) > 1:
        raise SystemExit(f"{out} already holds episodes of another policy {sorted(policies)}: use another --out")
    runs.write_text(json.dumps(history + [info], indent=1))


def load_meta(out):
    """meta of every episode on disk, as a list of dicts (with 'path')."""
    rows = []
    for path in sorted(Path(out).glob("*/b*_s*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            rows.append({**json.loads(str(f["meta"])), "path": str(path)})
    return rows


def summary(out):
    """Success / box hit / fall rate per set and density (no CIs: see metrics.py)."""
    rows = load_meta(out)
    groups = {}
    for r in rows:
        groups.setdefault((r["set"], r["boxes"]), []).append(r)
    lines = [f"{'set':<22} {'boxes':>5} {'n':>4} {'success':>8} {'box hit':>8} {'fall':>6} {'progress':>9} {'end y':>7}"]
    for (name, boxes), g in sorted(groups.items()):
        mean = lambda k: float(np.mean([r[k] for r in g]))  # noqa: E731
        lines.append(f"{name:<22} {boxes:>5} {len(g):>4} {mean('success'):>8.0%} {mean('box_hit'):>8.0%} "
                     f"{mean('fall'):>6.0%} {mean('progress'):>9.2f} {mean('end_y'):>+7.2f}")
    return "\n".join(lines)
