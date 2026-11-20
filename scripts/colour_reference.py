"""Reference floor colour of the training look, for the floor_norm intervention:
mean RGB of the floor pixels (src.scenarios.floor_pixels) of the Test-ID frames
saved with image (same machine as the rollouts).

    python scripts/colour_reference.py          # -> configs/colour_reference.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.attribution import load_frames  # noqa: E402
from src.scenarios import floor_pixels  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--out", default=str(REPO / "configs/colour_reference.json"))
    args = parser.parse_args()
    frames = load_frames(args.data, sets=["test_id", "test_id_0"])
    pixels = np.concatenate([im[floor_pixels(im)] for im in frames.image])
    mean = pixels.astype(np.float64).mean(0)
    Path(args.out).write_text(json.dumps({"floor_mean_rgb": [round(float(v), 3) for v in mean], "frames": len(frames),
                                          "pixels": int(len(pixels)), "sets": ["test_id", "test_id_0"]}, indent=1))
    print(f"floor mean RGB over {len(pixels)} pixels of {len(frames)} Test-ID frames: {mean.round(2)} -> {args.out}")


if __name__ == "__main__":
    main()
