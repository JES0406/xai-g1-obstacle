# Reproduction check

A clean, anonymous clone (`git clone https://github.com/JES0406/xai-g1-obstacle.git`, commit `9f872b8`)
was run end to end with `scripts/reproduce_all.sh` on the same lab machine (RTX 4070 Ti) on 2026-11-21:
no errors, all outputs regenerated. The integrity checks (`check_rollouts.py`) passed.

## Result

The baselines reproduce exactly. The policy's closed-loop rollouts are not bit-identical: in a sample episode
(`test_id/b10_s2000`) the first image is identical (0 pixel difference) and the robot pose stays identical for
73 of 200 steps, then drifts (pose difference 0.03 at the end). Tiny numerical differences are amplified by the
closed loop (the policy reacts to sub-pixel changes), so borderline episodes can flip. Every reproduced rate lies
inside the published 95 % interval:

| set | published success | reproduced | published 95 % CI | published box hit | reproduced |
|---|---|---|---|---|---|
| validation | 95.0 % | 96.7 % | [86, 98] | 5.0 % | 3.3 % |
| test_id | 93.7 % | 93.3 % | [90, 96] | 4.7 % | 5.0 % |
| test_ood_box_colour | 61.7 % | 63.3 % | [56, 67] | 36.0 % | 34.3 % |
| test_ood_floor_tint | 70.7 % | 71.3 % | [65, 76] | 8.3 % | 7.0 % |
| test_ood_lighting | 92.7 % | 91.7 % | [89, 95] | 7.0 % | 7.3 % |
| baseline_walker | 19.6 % | 19.6 % | [11, 32] | 80.4 % | 80.4 % |
| baseline_black | 0.0 % | 0.0 % | [0, 7] | 25.5 % | 25.5 % |
| baseline_noise | 21.6 % | 21.6 % | [12, 35] | 78.4 % | 78.4 % |
| baseline_mask_top | 74.5 % | 74.5 % | [61, 84] | 25.5 % | 25.5 % |
| baseline_mask_random | 94.1 % | 94.1 % | [84, 98] | 5.9 % | 5.9 % |
| int_hue_box_colour | 93.3 % | 93.0 % | [90, 96] | 5.3 % | 5.7 % |
| int_norm_floor_tint | 92.3 % | 92.0 % | [89, 95] | 5.0 % | 6.3 % |
| int_hue_test_id | 98.0 % | 98.0 % | [90, 100] | 2.0 % | 2.0 % |
| int_norm_test_id | 98.0 % | 98.0 % | [90, 100] | 2.0 % | 0.0 % |

Before/after (paired change in success, percentage points):

| condition | change | published | reproduced |
|---|---|---|---|
| blue boxes | recolour to red | +31.7 | +29.7 |
| floor tint | floor colour normalisation | +21.7 | +20.7 |
| Test-ID | recolour to red / floor normalisation | +0.0 / +0.0 | +2.0 / +2.0 |
| Test-ID | fine-tune | −9.7 | −9.7 |
| blue boxes | fine-tune | −11.3 | −13.0 |
| floor tint | fine-tune | −70.7 | −71.3 |
| lighting | fine-tune | +0.0 | +1.3 |

The published `results/` are those of the original run; the conclusions do not depend on which run is used.

## Where the variation comes from

`scripts/determinism_check.py` runs the same episodes several times in one process, in different orders:

- The simulation is deterministic: two environments reset to the same seed and given the same actions stay
  bit-identical, and so do their rendered images.
- The networks are deterministic: the same image and state give bit-identical outputs, before and after other inputs.
- The rendering is not, across episodes: the first image of an episode can differ (up to 59 grey levels in some
  pixels on the development laptop) depending on what the GPU rendered before in the same process, and switching
  between the camera and the segmentation renderers leaves a small trace. With a fresh environment per episode the
  first image matches, but trajectories still drift after 20–80 steps.

The policy amplifies these pixel differences (a 1-pixel shift of the image already moves its output by about
0.1 standard deviations), so a single episode is not exactly repeatable on a GPU, while every rate is. Rendering on
the CPU would remove the variation but would change every image and therefore every published number.
