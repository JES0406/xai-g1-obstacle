"""Is one rollout exactly repeatable? Runs the same episodes several times in
one process, in different orders (so other episodes run in between), and
compares the step logs bit for bit.

    MUJOCO_GL=egl python scripts/determinism_check.py                 # 3 Test-ID episodes, 30 boxes
    MUJOCO_GL=egl python scripts/determinism_check.py --fresh-env     # new env per episode

If repeats differ only when other episodes ran in between, state carries over
from one episode to the next inside a worker; if they differ even back to back,
the simulation or rendering itself is not deterministic.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src import rollout  # noqa: E402
from src.rollout import Job  # noqa: E402

FIELDS = ("mean", "qpos", "action", "clearance")


def run(jobs, out, fresh_env):
    for job in jobs:
        if fresh_env or not rollout._worker:
            if rollout._worker:
                rollout._worker["seg"].close()
                rollout._worker["env"].close()
                rollout._worker.clear()
            rollout._init_worker()
        rollout.run_episode(job, out)


def compare(a, b):
    with np.load(a) as x, np.load(b) as y:
        T = min(len(x["qpos"]), len(y["qpos"]))
        diverge = np.flatnonzero(np.abs(x["qpos"][:T] - y["qpos"][:T]).max(1) > 0)
        same = all(np.array_equal(x[f], y[f]) for f in FIELDS) and len(x["qpos"]) == len(y["qpos"])
        return same, (int(diverge[0]) if len(diverge) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002])
    parser.add_argument("--boxes", type=int, default=30)
    parser.add_argument("--fresh-env", action="store_true", help="build a new env for every episode")
    args = parser.parse_args()

    jobs = [Job("test_id", "id", "", args.boxes, s, False) for s in args.seeds]
    orders = {"A": jobs, "B": list(reversed(jobs)), "C": jobs[1:] + jobs[:1]}
    root = Path(tempfile.mkdtemp(prefix="determinism_"))
    try:
        for name, order in orders.items():
            run(order, root / name, args.fresh_env)
        print(f"{'seed':>6}  " + "  ".join(f"A vs {k}" for k in "BC"))
        all_same = True
        for job in jobs:
            cells = []
            for k in "BC":
                same, step = compare(job.path(root / "A"), job.path(root / k))
                all_same &= same
                cells.append("identical" if same else f"differs from step {step}")
            print(f"{job.seed:>6}  " + "  ".join(f"{c:<22}" for c in cells))
        print("ALL IDENTICAL" if all_same else "NOT DETERMINISTIC")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
