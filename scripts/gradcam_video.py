"""Video of one episode from the policy's camera with Grad-CAM superimposed.

    python scripts/gradcam_video.py                          # a Test-ID episode with 30 boxes
    python scripts/gradcam_video.py --episode data/raw/test_ood_box_colour/b30_s2000.npz

Left: the 64x64 image the policy saw (box outlines from the logged mask).
Right: Grad-CAM (conv2) of s = -mean vx, evidence for braking. Captions: time,
mean vx and vy before the clip, distance to the nearest box. Offline, from
the logs. Writes results/figures/gradcam_<set>_<episode>.mp4 and a 6-frame
strip .png.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.attribution import local_maps  # noqa: E402
from src.load_policy import SteeringNet, load_ppo  # noqa: E402

SCALE = 4                     # 64 -> 256 px
ORANGE = np.array([235, 104, 52], dtype=float)


def pick_episode(data):
    """A Test-ID episode with image and 30 boxes, a box hit if there is one."""
    paths = sorted(Path(data, "test_id").glob("b30_s*.npz"))
    with_image = []
    for p in paths:
        with np.load(p) as f:
            if "image" in f.files:
                with_image.append((json.loads(str(f["meta"]))["box_hit"], p))
    if not with_image:
        raise FileNotFoundError("no Test-ID episode with image and 30 boxes")
    hits = [p for hit, p in with_image if hit]
    return hits[0] if hits else with_image[0][1]


def outline(mask):
    edge = np.zeros_like(mask, dtype=bool)
    box = mask >= 0
    edge[1:-1, 1:-1] = box[1:-1, 1:-1] & ~(box[:-2, 1:-1] & box[2:, 1:-1] & box[1:-1, :-2] & box[1:-1, 2:])
    return edge


def compose(image, cam, mask, caption):
    big = lambda a: np.kron(a, np.ones((SCALE, SCALE, 1) if a.ndim == 3 else (SCALE, SCALE)))  # noqa: E731
    left = big(image).astype(float)
    edge = big(outline(mask)) > 0
    left[edge] = 255
    grey = big(image.mean(-1, keepdims=True).repeat(3, -1)).astype(float) * 0.6
    heat = np.array(Image.fromarray((cam / (cam.max() or 1) * 255).astype(np.uint8)).resize((64 * SCALE,) * 2, Image.BILINEAR)) / 255.0
    right = grey * (1 - heat[..., None]) + ORANGE * heat[..., None]
    right[edge] = 255
    frame = np.concatenate([left, np.full((64 * SCALE, 16, 3), 252.0), right], axis=1).astype(np.uint8)
    canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 32), (252, 252, 251))   # 528 x 288: multiples of 16 for the codec
    canvas.paste(Image.fromarray(frame), (0, 32))
    ImageDraw.Draw(canvas).text((6, 10), caption, fill=(11, 11, 11))
    return np.array(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--episode", help="an .npz with image; default: a Test-ID episode with 30 boxes")
    parser.add_argument("--out", default=str(REPO / "results/figures"))
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    path = Path(args.episode) if args.episode else pick_episode(args.data)
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    meta = json.loads(str(d["meta"]))
    net = SteeringNet(load_ppo().policy).eval()
    cams = local_maps(net, d["image"], d["state"], "vx", "gradcam_conv2")
    frames = []
    for t in range(len(d["image"])):
        c = d["clearance"][t]
        caption = (f"t {t / 10:4.1f} s   mean vx {d['mean'][t, 0]:+.2f}   mean vy {d['mean'][t, 1]:+.2f}   "
                   f"nearest box {c:.2f} m" if np.isfinite(c) else "")
        frames.append(compose(d["image"][t], cams[t], d["mask"][t], caption))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"gradcam_{meta['set']}_{path.stem}"
    imageio.mimsave(out / f"{stem}.mp4", frames, fps=10)
    picks = np.linspace(0, len(frames) - 1, 6).astype(int)
    Image.fromarray(np.concatenate([frames[i] for i in picks], axis=0)).save(out / f"{stem}_strip.png")
    outcome = "box hit" if meta["box_hit"] else "success" if meta["success"] else "fail"
    print(f"{path} ({outcome}, {len(frames)} steps) -> {out / (stem + '.mp4')}")


if __name__ == "__main__":
    main()
