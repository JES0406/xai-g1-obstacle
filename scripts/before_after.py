"""Before/after table of the actionable part: the original policy, the test-time
interventions and the fine-tuned policy, on the same seeds (paired episodes).

    python scripts/before_after.py            # data/raw (+ interventions), data/raw_ft, results/*q_rra.csv

Per condition (Test-ID and each OOD variant): success and box-hit rates with
95 % Wilson CIs, q_RRA (Grad-CAM conv2 and occlusion, s = -mean vx) when
available, and the paired change in success against the original with a
bootstrap CI and an exact McNemar p-value. Writes results/before_after.csv
and results/before_after.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.metrics import episode_frame, fmt, wilson  # noqa: E402

CONDITIONS = {"test_id": "Test-ID", "test_ood_box_colour": "blue boxes", "test_ood_floor_tint": "floor tint",
              "test_ood_lighting": "lighting"}
INTERVENTIONS = {"int_hue_box_colour": ("test_ood_box_colour", "recolour to red"),
                 "int_hue_test_id": ("test_id", "recolour to red"),
                 "int_norm_floor_tint": ("test_ood_floor_tint", "floor colour normalisation"),
                 "int_norm_test_id": ("test_id", "floor colour normalisation")}
QRRA_METHODS = ("gradcam_conv2", "occlusion")


def mcnemar(before, after):
    """Exact two-sided McNemar p-value for paired 0/1 outcomes."""
    from scipy.stats import binomtest

    b = int(np.sum((before == 1) & (after == 0)))
    c = int(np.sum((before == 0) & (after == 1)))
    return 1.0 if b + c == 0 else float(binomtest(c, b + c, 0.5).pvalue)


def paired_diff(before, after, resamples=4000, seed=0):
    d = after.astype(float) - before.astype(float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(resamples, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def q_rra_lookup(path):
    if not Path(path).exists():
        return {}
    t = pd.read_csv(path)
    t = t[(t.frames == "all") & (t.scalar == "vx")]
    return {(r.set, r.method): (r.q_rra, r.q_rra_lo, r.q_rra_hi, r.random) for r in t.itertuples()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO / "data/raw"))
    parser.add_argument("--data-ft", default=str(REPO / "data/raw_ft"))
    parser.add_argument("--out", default=str(REPO / "results"))
    args = parser.parse_args()

    orig = episode_frame(args.data)
    ft = episode_frame(args.data_ft) if Path(args.data_ft).exists() else pd.DataFrame(columns=orig.columns)
    q = {"original": q_rra_lookup(REPO / "results/q_rra.csv"),
         "fine-tune": q_rra_lookup(REPO / "results/ft/q_rra.csv"),
         "intervention": q_rra_lookup(REPO / "results/interventions/q_rra.csv")}

    rows = []
    for cond, label in CONDITIONS.items():
        base = orig[orig.set == cond]
        candidates = [("original", "", base, cond)]
        for name, (target, what) in INTERVENTIONS.items():
            if target == cond and (orig.set == name).any():
                candidates.append(("intervention", what, orig[orig.set == name], name))
        if (ft.set == cond).any():
            candidates.append(("fine-tune", "CNN fine-tune with visual randomisation", ft[ft.set == cond], cond))
        for kind, what, df, qset in candidates:
            row = {"condition": label, "policy": kind, "detail": what, "n": len(df)}
            for r in ("success", "box_hit"):
                row[r], row[f"{r}_lo"], row[f"{r}_hi"] = wilson(int(df[r].sum()), len(df))
            for m in QRRA_METHODS:
                v = q[kind].get((qset, m))
                row[f"q_rra_{m}"] = v[0] if v else np.nan
                row[f"q_rra_{m}_random"] = v[3] if v else np.nan
            if kind != "original":
                paired = base.merge(df, on=["boxes", "seed"], suffixes=("_before", "_after"))
                row["paired_n"] = len(paired)
                if len(paired):
                    b, a = paired.success_before.to_numpy(int), paired.success_after.to_numpy(int)
                    row["d_success"], row["d_success_lo"], row["d_success_hi"] = paired_diff(b, a)
                    row["mcnemar_p"] = mcnemar(b, a)
                else:
                    row["d_success"] = row["d_success_lo"] = row["d_success_hi"] = row["mcnemar_p"] = np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    out = Path(args.out)
    table.to_csv(out / "before_after.csv", index=False)

    lines = ["| condition | policy | n | success | box hit | Δ success vs original (paired) | McNemar p | q_RRA Grad-CAM conv2 | q_RRA occlusion |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        delta = "" if r.policy == "original" or not r.paired_n else \
            f"{r.d_success * 100:+.0f} pp [{r.d_success_lo * 100:+.0f}, {r.d_success_hi * 100:+.0f}] (n={int(r.paired_n)})"
        p = "" if r.policy == "original" else f"{r.mcnemar_p:.2g}"
        qs = [("-" if np.isnan(getattr(r, f"q_rra_{m}")) else f"{getattr(r, f'q_rra_{m}'):.2f} (rand {getattr(r, f'q_rra_{m}_random'):.2f})")
              for m in QRRA_METHODS]
        name = r.policy if not r.detail or r.policy == "fine-tune" else f"{r.policy}: {r.detail}"
        lines.append(f"| {r.condition} | {name} | {r.n} | {fmt(r.success, r.success_lo, r.success_hi)} | "
                     f"{fmt(r.box_hit, r.box_hit_lo, r.box_hit_hi)} | {delta} | {p} | {qs[0]} | {qs[1]} |")
    text = "\n".join(lines) + "\n"
    (out / "before_after.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
