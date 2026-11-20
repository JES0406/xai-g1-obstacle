| condition | policy | n | success | box hit | Δ success vs original (paired) | McNemar p | q_RRA Grad-CAM conv2 | q_RRA occlusion |
|---|---|---|---|---|---|---|---|---|
| Test-ID | original | 300 | 94% [90, 96] | 5% [3, 8] |  |  | 0.23 (rand 0.08) | 0.24 (rand 0.08) |
| Test-ID | intervention: recolour to red | 51 | 98% [90, 100] | 2% [0, 10] | +0 pp [+0, +0] (n=51) | 1 | - | - |
| Test-ID | intervention: floor colour normalisation | 51 | 98% [90, 100] | 2% [0, 10] | +0 pp [+0, +0] (n=51) | 1 | - | - |
| Test-ID | fine-tune | 300 | 84% [79, 88] | 7% [4, 10] | -10 pp [-14, -6] (n=300) | 4.9e-06 | 0.30 (rand 0.07) | 0.18 (rand 0.07) |
| blue boxes | original | 300 | 62% [56, 67] | 36% [31, 42] |  |  | 0.24 (rand 0.08) | 0.18 (rand 0.08) |
| blue boxes | intervention: recolour to red | 300 | 93% [90, 96] | 5% [3, 8] | +32 pp [+26, +37] (n=300) | 1.4e-25 | 0.21 (rand 0.07) | 0.23 (rand 0.07) |
| blue boxes | fine-tune | 300 | 50% [45, 56] | 49% [43, 55] | -11 pp [-18, -5] (n=300) | 0.0012 | 0.13 (rand 0.07) | 0.17 (rand 0.07) |
| floor tint | original | 300 | 71% [65, 76] | 8% [6, 12] |  |  | 0.24 (rand 0.07) | 0.19 (rand 0.07) |
| floor tint | intervention: floor colour normalisation | 300 | 92% [89, 95] | 5% [3, 8] | +22 pp [+17, +27] (n=300) | 2.4e-16 | 0.22 (rand 0.07) | 0.26 (rand 0.07) |
| floor tint | fine-tune | 300 | 0% [0, 1] | 41% [36, 47] | -71 pp [-76, -65] (n=300) | 3e-64 | 0.01 (rand 0.06) | 0.09 (rand 0.06) |
| lighting | original | 300 | 93% [89, 95] | 7% [5, 10] |  |  | 0.19 (rand 0.07) | 0.28 (rand 0.07) |
| lighting | fine-tune | 300 | 93% [89, 95] | 5% [3, 8] | +0 pp [-4, +4] (n=300) | 1 | 0.30 (rand 0.07) | 0.22 (rand 0.07) |
