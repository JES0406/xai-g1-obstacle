"""Evaluation tables with 95 % CIs from the rollouts.

    python scripts/metrics_table.py                # data/raw -> results/metrics_*.csv + printed tables

Writes results/metrics_by_density.csv, results/metrics_by_set.csv,
results/command_deviation.csv and results/metrics_summary.md.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import command_deviation, episode_frame, fmt, table  # noqa: E402


def markdown(t, by):
    head = [*by, "n", "success", "box hit", "fall", "progress", "min clearance (m)", "vy saturated"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in t.iterrows():
        cells = [str(r[b]) for b in by] + [str(r["n"])]
        cells += [fmt(r[k], r[f"{k}_lo"], r[f"{k}_hi"]) for k in ("success", "box_hit", "fall")]
        cells += [fmt(r[k], r[f"{k}_lo"], r[f"{k}_hi"], pct=False) for k in ("progress", "min_clearance")]
        cells += ["–" if r["saturated_vy"] != r["saturated_vy"] else f"{r['saturated_vy']:.0%}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--out", default=str(REPO / "results"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = episode_frame(args.data)
    df.to_csv(out / "episodes.csv", index=False)
    by_density = table(df, by=("set", "boxes"))
    with_boxes = df[df["boxes"] > 0]
    by_set = table(with_boxes, by=("set",))
    deviation = command_deviation(df)
    by_density.to_csv(out / "metrics_by_density.csv", index=False)
    by_set.to_csv(out / "metrics_by_set.csv", index=False)
    deviation.to_csv(out / "command_deviation.csv", index=False)

    text = ["## By set and density", "", markdown(by_density, ("set", "boxes")), "",
            "## By set (densities with boxes pooled)", "", markdown(by_set, ("set",)), "",
            "## Test-ID-0: deviation from the user's command (no boxes)", "",
            "| quantity | mean [95 % CI] | n |", "|---|---|---|"]
    text += [f"| {r.quantity} | {fmt(r['mean'], r.lo, r.hi, pct=False)} | {r.n} |" for _, r in deviation.iterrows()]
    (out / "metrics_summary.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
