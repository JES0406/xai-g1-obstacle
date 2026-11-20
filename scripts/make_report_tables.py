"""LaTeX tables for the report, generated from results/ (so the report never
carries a number by hand).

    python scripts/make_report_tables.py        # -> report/tables/*.tex
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "report" / "tables"

SET_LABEL = {"validation": "Validation (seeds 1000+)", "test_id": "Test-ID (new layouts)",
             "test_ood_box_colour": "OOD: blue boxes", "test_ood_floor_tint": "OOD: floor tint",
             "test_ood_lighting": "OOD: lighting", "baseline_walker": "Baseline: walker only",
             "baseline_black": "Baseline: black image", "baseline_noise": "Baseline: noise image",
             "baseline_mask_top": "Behaviour: hide Grad-CAM region", "baseline_mask_random": "Behaviour: hide random region"}
METHOD_LABEL = {"gradcam_conv3": "Grad-CAM conv3", "gradcam_conv2": "Grad-CAM conv2", "ig_black": "IG (black)",
                "ig_blur": "IG (blur)", "occlusion": "Occlusion"}
METHODS = list(METHOD_LABEL)


def pct(p, lo=None, hi=None):
    s = f"{p * 100:.0f}"
    return s if lo is None else f"{s} [{lo * 100:.0f}, {hi * 100:.0f}]"


def table(path, colspec, header, rows, caption, label):
    # scaled to the text width only when wider than it
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule",
             *[" & ".join(r) + r" \\" for r in rows], r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(RES / "metrics_by_set.csv").set_index("set")
    rows = [[SET_LABEL[k], f"{int(s.loc[k, 'n'])}", pct(s.loc[k, 'success'], s.loc[k, 'success_lo'], s.loc[k, 'success_hi']),
             pct(s.loc[k, 'box_hit'], s.loc[k, 'box_hit_lo'], s.loc[k, 'box_hit_hi']), pct(s.loc[k, 'fall']),
             f"{s.loc[k, 'progress']:.2f}"] for k in SET_LABEL if k in s.index]
    table(OUT / "evaluation.tex", "lrllrr", ["Set", "$n$", r"Success \%", r"Box hit \%", r"Fall \%", "Progress"], rows,
          r"Closed-loop evaluation, 10/20/30 boxes pooled. Rates with 95\,\% Wilson intervals; progress = distance covered / commanded.",
          "tab:evaluation")

    ba = pd.read_csv(RES / "before_after.csv")
    rows = []
    for r in ba.itertuples():
        name = r.policy if r.policy != "intervention" else f"intervention: {r.detail}"
        delta = "" if r.policy == "original" else f"{r.d_success * 100:+.0f} [{r.d_success_lo * 100:+.0f}, {r.d_success_hi * 100:+.0f}]"
        p = "" if r.policy == "original" else (f"{r.mcnemar_p:.1g}" if r.mcnemar_p < 0.01 else f"{r.mcnemar_p:.2f}")
        q = "" if pd.isna(r.q_rra_gradcam_conv2) else f"{r.q_rra_gradcam_conv2 / r.q_rra_gradcam_conv2_random:.1f}"
        rows.append([r.condition, name, f"{r.n}", pct(r.success, r.success_lo, r.success_hi), pct(r.box_hit), delta, p, q])
    table(OUT / "before_after.tex", "llrlrlrr",
          ["Condition", "Policy", "$n$", r"Success \%", r"Hit \%", r"$\Delta$ success (pp)", "McNemar $p$", r"q\textsubscript{RRA}/chance"],
          rows, r"Before/after on the same seeds (paired). q\textsubscript{RRA} for Grad-CAM conv2 on $-\mu_{vx}$, divided by its random reference.",
          "tab:before-after")

    road = pd.read_csv(RES / "sanity_road_auc.csv").pivot(index="order", columns="scalar", values="auc")
    st = pd.read_csv(RES / "sanity_stability_summary.csv").set_index(["scalar", "method"])
    rnd = pd.read_csv(RES / "sanity_randomisation.csv")
    rnd = rnd[rnd.stage == "conv1"].groupby(["scalar", "method"]).spearman.mean()
    rows = [[METHOD_LABEL[m], f"{road.loc[m, 'vx']:.3f}", f"{road.loc[m, 'vy']:.3f}", f"{rnd.loc[('vx', m)]:.2f}",
             f"{st.loc[('vx', m), 'q_lle']:.3g}", f"{st.loc[('vx', m), 'q_as']:.2g}",
             f"{st.loc[('vx', m), 'map_change_rel'] * 100:.0f}", f"{st.loc[('vx', m), 'shift_map_change_rel'] * 100:.0f}"] for m in METHODS]
    rows.append(["Random order", f"{road.loc['random', 'vx']:.3f}", f"{road.loc['random', 'vy']:.3f}", "", "", "", "", ""])
    table(OUT / "sanity.tex", "lrrrrrrr",
          ["Method", r"ROAD $v_x$", r"ROAD $v_y$", r"Rand.\ $\rho$", r"q\textsubscript{LLE}", r"q\textsubscript{AS}",
           r"Noise \%", r"1-px \%"], rows,
          r"Sanity checks ($v_x$ unless stated). ROAD: AUC of $|\Delta s|/\mathrm{std}(s)$, higher = more faithful. "
          r"Rand.\ $\rho$: Spearman with the original map after randomising the whole network. Last two: relative map change "
          r"under noise/brightness and under 1-px shifts.", "tab:sanity")

    q = pd.read_csv(RES / "q_rra.csv")
    q = q[(q.frames == "all") & (q.scalar == "vx")].pivot(index="method", columns="set", values="ratio")
    cols = ["test_id", "test_ood_box_colour", "test_ood_floor_tint", "test_ood_lighting"]
    rows = [[METHOD_LABEL[m], *[f"{q.loc[m, c]:.2f}" for c in cols]] for m in METHODS]
    table(OUT / "qrra.tex", "lrrrr", ["Method", "Test-ID", "Blue boxes", "Floor tint", "Lighting"], rows,
          r"Localisation: q\textsubscript{RRA} divided by the random-map reference $|s|/d$ ($-\mu_{vx}$, 500 frames per set). 1 = chance.",
          "tab:qrra")
    print(f"wrote {sorted(p.name for p in OUT.glob('*.tex'))} to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
