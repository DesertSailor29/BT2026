# Pancreas test cases: why the baseline misses

Written by `nnunet/pancreas_case_figures.py`. 238 MSD07 test cases, D102 `all` (f0) and `all_rep2` (f1), checkpoint_final predictions as saved. Synthetic = every 5th of the 999 training pairs (200); real val = the 43 MSD07 val cases. Gland = pancreas + tumour. HU medians: pancreas eroded 1 voxel in-plane; surroundings = 3-voxel in-plane shell around the gland.

## Synthetic vs real (median [IQR])

| property | synthetic | real val | real test |
|---|---|---|---|
| gland ml | 71.8 [57.9, 88.8] | 88.3 [64.2, 114.0] | 85.8 [63.8, 119.2] |
| pancreas ml | 66.3 [51.6, 82.9] | 78.6 [57.1, 103.7] | 76.8 [57.4, 104.2] |
| tumour ml | 5.3 [2.4, 8.2] | 5.8 [3.5, 9.2] | 5.7 [2.7, 10.6] |
| tumour share of gland | 0.1 [0.0, 0.1] | 0.1 [0.0, 0.1] | 0.1 [0.0, 0.1] |
| pancreas HU | 81.0 [70.8, 89.0] | 89.0 [62.5, 108.0] | 92.5 [71.2, 110.0] |
| tumour HU | 69.0 [53.8, 84.0] | 74.0 [31.0, 101.5] | 78.0 [50.2, 103.0] |
| surroundings HU | 40.0 [26.5, 56.0] | 27.0 [-6.5, 55.0] | 29.0 [-4.8, 63.8] |
| pancreas − surroundings HU | 38.5 [30.8, 47.0] | 57.0 [40.5, 86.5] | 58.0 [38.0, 79.0] |
| pancreas − tumour HU | 9.0 [-2.2, 22.0] | 19.0 [-14.0, 35.5] | 11.5 [-13.0, 36.0] |
| in-plane spacing mm | 0.8 [0.8, 0.8] | 0.8 [0.7, 0.9] | 0.8 [0.7, 0.9] |
| slice spacing mm | 1.0 [1.0, 1.0] | 2.5 [2.5, 2.5] | 2.5 [2.5, 2.5] |
| gland z extent mm | 84.0 [74.0, 93.0] | 85.0 [75.0, 96.2] | 82.5 [72.5, 92.5] |

## Where the reference gland goes (mean over test cases)

| run | → pancreas | → tumour | → background | gland recall | gland precision | z coverage | recall right (head) | middle | left (tail) | pred / ref gland ml (median) | pred gland within 10 mm (median) | ref pancreas → tumour | ref tumour → pancreas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.15 | 0.06 | 0.79 | 0.21 | 0.55 | 0.65 | 0.20 | 0.21 | 0.23 | 0.30 | 0.74 | 0.04 | 0.12 |
| all_rep2 | 0.12 | 0.06 | 0.82 | 0.18 | 0.54 | 0.59 | 0.16 | 0.18 | 0.21 | 0.21 | 0.77 | 0.05 | 0.09 |

## Failure types (both runs, rules in the script header)

| type | cases |
|---|---|
| gland_missed | 57 |
| none | 83 |
| pancreas_as_tumour | 2 |
| tumour_missed | 8 |
| under_segmented | 88 |

## Gland Dice (mean of runs) by slice spacing

| slice spacing | cases | gland Dice mean |
|---|---|---|
| < 2.5 mm | 7 | 0.116 |
| 2.5 mm | 183 | 0.299 |
| > 2.5 to < 5 mm | 10 | 0.169 |
| >= 5 mm | 38 | 0.157 |

## Gland Dice (mean of runs) by surroundings median HU (fat around the gland = negative)

| surroundings HU | cases | gland Dice mean | of these 2.5 mm slices | gland Dice mean (2.5 mm) |
|---|---|---|---|---|
| -999–0 | 62 | 0.153 | 45 | 0.159 |
| 0–40 | 68 | 0.303 | 54 | 0.348 |
| 40–999 | 108 | 0.306 | 84 | 0.342 |

Synthetic cases with surroundings < 0 HU: 2% (real test 26%).

## Gland Dice (mean of runs) by pancreas median HU

| pancreas HU | cases | gland Dice mean |
|---|---|---|
| -999–40 | 18 | 0.052 |
| 40–70 | 37 | 0.171 |
| 70–90 | 56 | 0.243 |
| 90–110 | 61 | 0.364 |
| 110–130 | 47 | 0.349 |
| 130–999 | 19 | 0.194 |

## Gland Dice (mean of runs) vs case properties, Spearman

| property | rho | p |
|---|---|---|
| gland volume (ml) | -0.05 | 0.44 |
| surroundings median HU | +0.16 | 0.015 |
| pancreas median HU | +0.28 | 1.5e-05 |
| pancreas − surroundings HU | +0.05 | 0.44 |
| slice spacing (mm) | -0.15 | 0.021 |
| gland cranio-caudal extent (mm) | -0.11 | 0.1 |

Charts: [synthetic_vs_real.svg](synthetic_vs_real.svg), [where_gland_goes.svg](where_gland_goes.svg), [dice_vs_properties.svg](dice_vs_properties.svg). CT figures: `reports/figures/pancreas_cases/`.
