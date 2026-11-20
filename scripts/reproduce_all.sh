#!/usr/bin/env bash
# Regenerate every result, table and figure from a clean clone, in order.
# Run on the machine that produced the published numbers (renders differ between GPUs).
#
#   scripts/reproduce_all.sh                  # uses the shipped fine-tuned weights if runs/ft_dr/final exists
#   scripts/reproduce_all.sh --train-finetune # also retrains the CNN fine-tune (~1 h on an RTX 4070 Ti)
#
# Total time on the lab machine (RTX 4070 Ti): about 3 h, mostly CPU-bound attribution.
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONUNBUFFERED=1

step() { echo; echo "== $* =="; }

step "MuJoCo Menagerie G1 (pinned)"
[ -f external/mujoco_menagerie/unitree_g1/scene_mjx.xml ] || setup/setup.sh menagerie

step "Smoke test"
python scripts/smoke_load.py

step "Evaluation rollouts, baselines and behaviour test (data/raw)"
python scripts/run_rollouts.py --sets validation test_id test_id_0 test_ood_box_colour test_ood_floor_tint test_ood_lighting \
    baseline_walker baseline_black baseline_noise baseline_mask_top baseline_mask_random
python scripts/check_rollouts.py data/raw

step "Metrics and global explanations"
python scripts/metrics_table.py
python scripts/group_pfi.py
python scripts/ig_groups.py

step "Local explanations, localisation, ID/OOD pairs, video"
python scripts/local_maps.py
python scripts/rerender_pairs.py
python scripts/gradcam_video.py

step "Sanity checks"
python scripts/sanity_checks.py

step "Test-time interventions"
python scripts/colour_reference.py
python scripts/run_rollouts.py --sets int_hue_box_colour int_hue_test_id int_norm_floor_tint int_norm_test_id
python scripts/check_rollouts.py data/raw
python scripts/local_maps.py --no-examples --sets int_hue_box_colour int_norm_floor_tint --out results/interventions

if [ "${1:-}" = "--train-finetune" ]; then
    step "CNN fine-tune with visual randomisation"
    python scripts/finetune.py
fi
if [ -f runs/ft_dr/final/model.zip ]; then
    step "Fine-tune evaluation"
    python scripts/run_rollouts.py --policy runs/ft_dr/final --out data/raw_ft \
        --sets test_id test_id_0 test_ood_box_colour test_ood_floor_tint test_ood_lighting
    python scripts/check_rollouts.py data/raw_ft
    python scripts/local_maps.py --no-examples --policy runs/ft_dr/final --data data/raw_ft --out results/ft
else
    echo "no runs/ft_dr/final: fine-tune rows of the before/after table stay empty (use --train-finetune)"
fi

step "Before/after table, metrics, report tables, dataset"
python scripts/metrics_table.py
python scripts/before_after.py
python scripts/make_report_tables.py
python scripts/export_dataset.py
echo; echo "done: results/, report/tables/, data/rollouts/, data/README.md"
