"""Run the evaluation rollouts of configs/sets.yaml and print a summary.

    MUJOCO_GL=egl python scripts/run_rollouts.py                       # everything (lab machine)
    MUJOCO_GL=egl python scripts/run_rollouts.py --sets validation     # one set
    env/wsl_gl.sh python scripts/run_rollouts.py --episodes 2 --processes 2 --out data/debug   # laptop debug

Interrupted runs resume: episodes already on disk are skipped.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.rollout import build_jobs, run, summary  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs/sets.yaml"))
    parser.add_argument("--sets", nargs="+", help="only these sets (baselines are baseline_<kind>)")
    parser.add_argument("--episodes", type=int, help="episodes per density, overriding the config (debug)")
    parser.add_argument("--processes", type=int, default=None, help="default: the config's")
    parser.add_argument("--out", default=str(REPO / "data/raw"))
    parser.add_argument("--dry-run", action="store_true", help="list the job counts and stop")
    args = parser.parse_args()

    if not os.environ.get("MUJOCO_GL"):
        raise SystemExit("set MUJOCO_GL=egl (native Linux) or run through env/wsl_gl.sh (WSL)")
    config = yaml.safe_load(Path(args.config).read_text())
    jobs = build_jobs(config, sets=args.sets, episodes=args.episodes)
    counts = {}
    for j in jobs:
        n, k = counts.get(j.set, (0, 0))
        counts[j.set] = (n + 1, k + j.save_image)
    for name, (n, k) in counts.items():
        print(f"  {name:<22} {n:>4} episodes, {k:>2} with image")
    print(f"  {'total':<22} {len(jobs):>4} episodes")
    if args.dry_run:
        return
    start = time.time()
    run(jobs, args.out, processes=args.processes or config["processes"])
    print(f"done in {(time.time() - start) / 60:.1f} min\n")
    print(summary(args.out))


if __name__ == "__main__":
    main()
