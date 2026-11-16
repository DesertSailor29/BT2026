# The AUC rule applied to the predictions (ccAucPost) + a sweep of stricter cuts

111 real LiTS test cases, checkpoint_best predictions. ccPost = training cc rule on the prediction; ccAucPost<cut> = ccPost, then predicted tumour pieces inside the predicted main structure with sep < cut -> their surroundings' label (filter_metrics' separability with the predicted liver as organ). Decisions use image + prediction only. **Main cut 0.5475** (= the training cut, fixed before results); 0.60 / 0.637 / 0.70 = exploratory sweep. Written by `nnunet/auc_post.py`; exploratory, the main analysis stays predict_test.py.

## 1. Does sep tell false-positive pieces from hits? (pieces inside the predicted main structure)

| run | pieces / case (hit, FP) | unscored | sep median hit / FP | AUROC sep (FP lower) | < 0.5475: FP / hit pieces (hit ml) | < 0.6: FP / hit pieces (hit ml) | < 0.637: FP / hit pieces (hit ml) | < 0.7: FP / hit pieces (hit ml) |
|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 10.2 (3.2, 7.1) | 70 | 0.907 / 0.639 | 0.82 | 131 / 3 (162.7) | 267 / 10 (412.5) | 356 / 16 (568.3) | 466 / 29 (750.8) |
| D101 f1 all_rep2 | 11.6 (3.0, 8.6) | 69 | 0.897 / 0.615 | 0.82 | 205 / 2 (82.7) | 412 / 6 (128.4) | 509 / 14 (387.7) | 602 / 29 (520.1) |
| D101 f2 ccReject | 16.1 (2.4, 13.7) | 24 | 0.901 / 0.597 | 0.81 | 411 / 3 (0.9) | 776 / 7 (1.6) | 922 / 10 (11.5) | 1055 / 24 (164.8) |
| D103 f0 ccRepair | 12.9 (2.6, 10.4) | 46 | 0.905 / 0.624 | 0.78 | 235 / 4 (70.4) | 493 / 6 (103.4) | 592 / 14 (228.3) | 712 / 28 (361.8) |
| D103 f1 ccRepairDel | 14.8 (2.6, 12.2) | 40 | 0.892 / 0.588 | 0.83 | 361 / 3 (51.7) | 717 / 5 (65.7) | 861 / 15 (347.2) | 956 / 23 (461.0) |
| D104 f0 ccAucRepair | 13.4 (2.7, 10.7) | 61 | 0.894 / 0.629 | 0.78 | 248 / 2 (0.3) | 480 / 3 (0.6) | 594 / 10 (234.1) | 715 / 22 (366.8) |
| D104 f1 ccAucRepair_rep2 | 10.7 (2.7, 8.1) | 56 | 0.894 / 0.653 | 0.73 | 158 / 2 (11.3) | 329 / 4 (144.4) | 399 / 14 (344.4) | 487 / 20 (438.8) |

## 2. Each variant vs ccPost, per run

Tumour Dice mean (NaN = both empty, left out) and the paired difference to ccPost; lesion recall, FP pieces and reference lesions lost are totals over the cases.

| run | variant | tumour Dice | vs ccPost [95% CI] | p | liver Dice | lesion recall | FP pieces / case | lesions lost vs ccPost | pieces relabelled |
|---|---|---|---|---|---|---|---|---|---|
| D101 f0 all | none | 0.312 | -0.070 [-0.092, -0.051] | 4.6e-15 | 0.882 | 0.468 | 39.9 | – | – |
| D101 f0 all | ccPost | 0.382 | – | – | 0.898 | 0.456 | 7.1 | 0 | – |
| D101 f0 all | ccAucPost0.5475 | 0.396 | +0.014 [+0.001, +0.030] | 2.7e-10 | 0.900 | 0.451 | 5.9 | 3 | 134 |
| D101 f0 all | ccAucPost0.6 | 0.431 | +0.049 [+0.025, +0.076] | 1.6e-09 | 0.902 | 0.441 | 4.7 | 10 | 277 |
| D101 f0 all | ccAucPost0.637 | 0.431 | +0.049 [+0.022, +0.078] | 1.9e-08 | 0.903 | 0.436 | 3.9 | 14 | 372 |
| D101 f0 all | ccAucPost0.7 | 0.426 | +0.044 [+0.011, +0.078] | 3.8e-06 | 0.903 | 0.419 | 2.9 | 26 | 495 |
| D101 f1 all_rep2 | none | 0.319 | -0.071 [-0.092, -0.052] | 8.1e-15 | 0.875 | 0.457 | 60.9 | – | – |
| D101 f1 all_rep2 | ccPost | 0.390 | – | – | 0.903 | 0.449 | 8.6 | 0 | – |
| D101 f1 all_rep2 | ccAucPost0.5475 | 0.417 | +0.027 [+0.014, +0.044] | 3.3e-11 | 0.904 | 0.444 | 6.8 | 3 | 207 |
| D101 f1 all_rep2 | ccAucPost0.6 | 0.445 | +0.055 [+0.035, +0.078] | 3.6e-11 | 0.904 | 0.441 | 4.9 | 5 | 418 |
| D101 f1 all_rep2 | ccAucPost0.637 | 0.448 | +0.058 [+0.033, +0.085] | 4.2e-10 | 0.905 | 0.432 | 4.1 | 12 | 523 |
| D101 f1 all_rep2 | ccAucPost0.7 | 0.446 | +0.056 [+0.023, +0.089] | 1.8e-07 | 0.905 | 0.415 | 3.2 | 24 | 631 |
| D101 f2 ccReject | none | 0.333 | -0.010 [-0.013, -0.007] | 2.9e-12 | 0.896 | 0.329 | 23.7 | – | – |
| D101 f2 ccReject | ccPost | 0.343 | – | – | 0.930 | 0.324 | 13.7 | 0 | – |
| D101 f2 ccReject | ccAucPost0.5475 | 0.356 | +0.013 [+0.007, +0.023] | 1.7e-13 | 0.930 | 0.323 | 10.0 | 1 | 414 |
| D101 f2 ccReject | ccAucPost0.6 | 0.365 | +0.022 [+0.013, +0.034] | 1.6e-12 | 0.930 | 0.322 | 6.7 | 2 | 783 |
| D101 f2 ccReject | ccAucPost0.637 | 0.369 | +0.027 [+0.016, +0.039] | 4e-12 | 0.930 | 0.319 | 5.4 | 4 | 932 |
| D101 f2 ccReject | ccAucPost0.7 | 0.355 | +0.013 [-0.014, +0.037] | 2.3e-07 | 0.930 | 0.305 | 4.2 | 14 | 1079 |
| D103 f0 ccRepair | none | 0.330 | -0.009 [-0.015, -0.001] | 1.2e-10 | 0.848 | 0.389 | 16.5 | – | – |
| D103 f0 ccRepair | ccPost | 0.339 | – | – | 0.872 | 0.379 | 10.4 | 0 | – |
| D103 f0 ccRepair | ccAucPost0.5475 | 0.370 | +0.031 [+0.017, +0.049] | 2.1e-11 | 0.873 | 0.375 | 8.2 | 3 | 239 |
| D103 f0 ccRepair | ccAucPost0.6 | 0.402 | +0.063 [+0.041, +0.086] | 8.2e-12 | 0.874 | 0.374 | 5.9 | 4 | 499 |
| D103 f0 ccRepair | ccAucPost0.637 | 0.402 | +0.063 [+0.037, +0.090] | 1.7e-09 | 0.875 | 0.364 | 5.0 | 11 | 606 |
| D103 f0 ccRepair | ccAucPost0.7 | 0.395 | +0.056 [+0.026, +0.087] | 6.7e-07 | 0.875 | 0.344 | 3.9 | 25 | 740 |
| D103 f1 ccRepairDel | none | 0.349 | -0.014 [-0.020, -0.008] | 2.1e-13 | 0.873 | 0.395 | 20.1 | – | – |
| D103 f1 ccRepairDel | ccPost | 0.363 | – | – | 0.888 | 0.392 | 12.2 | 0 | – |
| D103 f1 ccRepairDel | ccAucPost0.5475 | 0.379 | +0.016 [+0.009, +0.027] | 3e-13 | 0.889 | 0.388 | 9.0 | 3 | 364 |
| D103 f1 ccRepairDel | ccAucPost0.6 | 0.411 | +0.048 [+0.031, +0.069] | 1.5e-13 | 0.890 | 0.385 | 5.8 | 5 | 722 |
| D103 f1 ccRepairDel | ccAucPost0.637 | 0.414 | +0.051 [+0.028, +0.075] | 4.7e-12 | 0.891 | 0.371 | 4.5 | 15 | 876 |
| D103 f1 ccRepairDel | ccAucPost0.7 | 0.409 | +0.046 [+0.020, +0.073] | 2.4e-09 | 0.891 | 0.360 | 3.6 | 23 | 979 |
| D104 f0 ccAucRepair | none | 0.378 | -0.022 [-0.031, -0.015] | 6e-15 | 0.873 | 0.420 | 20.6 | – | – |
| D104 f0 ccAucRepair | ccPost | 0.400 | – | – | 0.925 | 0.416 | 10.7 | 0 | – |
| D104 f0 ccAucRepair | ccAucPost0.5475 | 0.411 | +0.011 [+0.007, +0.016] | 7.4e-13 | 0.926 | 0.415 | 8.5 | 1 | 250 |
| D104 f0 ccAucRepair | ccAucPost0.6 | 0.428 | +0.028 [+0.018, +0.040] | 4.4e-14 | 0.926 | 0.415 | 6.4 | 1 | 483 |
| D104 f0 ccAucRepair | ccAucPost0.637 | 0.430 | +0.030 [+0.011, +0.049] | 1.7e-10 | 0.926 | 0.403 | 5.4 | 9 | 604 |
| D104 f0 ccAucRepair | ccAucPost0.7 | 0.421 | +0.021 [-0.007, +0.046] | 7.3e-08 | 0.926 | 0.386 | 4.3 | 21 | 737 |
| D104 f1 ccAucRepair_rep2 | none | 0.381 | -0.014 [-0.022, -0.008] | 3.4e-13 | 0.873 | 0.415 | 12.5 | – | – |
| D104 f1 ccAucRepair_rep2 | ccPost | 0.395 | – | – | 0.897 | 0.405 | 8.1 | 0 | – |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.5475 | 0.417 | +0.022 [+0.011, +0.038] | 4.5e-09 | 0.898 | 0.403 | 6.7 | 1 | 160 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.6 | 0.453 | +0.058 [+0.038, +0.081] | 1.4e-12 | 0.899 | 0.401 | 5.1 | 3 | 333 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.637 | 0.446 | +0.051 [+0.026, +0.077] | 1.8e-09 | 0.899 | 0.388 | 4.5 | 12 | 413 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.7 | 0.444 | +0.049 [+0.018, +0.081] | 4.6e-08 | 0.899 | 0.379 | 3.7 | 18 | 507 |

## 3. Post-processing vs training (the question of this script)

Baseline = per-case mean of D101 f0 all + D101 f1 all_rep2. y − x per case, paired bootstrap CI, Wilcoxon p (unadjusted).

| comparison | x | y | y − x [95% CI] | p |
|---|---|---|---|---|
| baseline: ccAucPost0.5475 vs ccPost (what AUC post-processing adds) | 0.386 | 0.407 | +0.021 [+0.009, +0.035] | 1.1e-11 |
| D104 f0 ccAucRepair ccPost − baseline ccAucPost0.5475 (training vs post-processing; ≈ 0 = post-processing does it) | 0.407 | 0.400 | -0.007 [-0.025, +0.011] | 0.11 |
| D104 f0 ccAucRepair: ccAucPost0.5475 vs ccPost (AUC post on top of AUC training) | 0.400 | 0.411 | +0.011 [+0.007, +0.016] | 7.4e-13 |
| D104 f1 ccAucRepair_rep2 ccPost − baseline ccAucPost0.5475 (training vs post-processing; ≈ 0 = post-processing does it) | 0.407 | 0.395 | -0.012 [-0.026, +0.002] | 0.08 |
| D104 f1 ccAucRepair_rep2: ccAucPost0.5475 vs ccPost (AUC post on top of AUC training) | 0.395 | 0.417 | +0.022 [+0.011, +0.038] | 4.5e-09 |

Arms vs baseline when both get the same post-processing (Holm over the 5 arms listed):

| arm | variant | arm | baseline | diff [95% CI] | p | Holm p |
|---|---|---|---|---|---|---|
| D101 f2 ccReject | ccPost | 0.343 | 0.386 | -0.043 [-0.073, -0.014] | 0.001 | 0.004 |
| D103 f0 ccRepair | ccPost | 0.339 | 0.386 | -0.047 [-0.063, -0.031] | 8.6e-09 | 4.3e-08 |
| D103 f1 ccRepairDel | ccPost | 0.363 | 0.386 | -0.023 [-0.042, -0.005] | 0.0049 | 0.015 |
| D104 f0 ccAucRepair | ccPost | 0.400 | 0.386 | +0.014 [-0.005, +0.034] | 0.69 | 0.69 |
| D104 f1 ccAucRepair_rep2 | ccPost | 0.395 | 0.386 | +0.009 [-0.001, +0.019] | 0.28 | 0.56 |
| D101 f2 ccReject | ccAucPost0.5475 | 0.356 | 0.407 | -0.051 [-0.080, -0.023] | 7.4e-05 | 0.0003 |
| D103 f0 ccRepair | ccAucPost0.5475 | 0.370 | 0.407 | -0.036 [-0.057, -0.017] | 5.4e-06 | 2.7e-05 |
| D103 f1 ccRepairDel | ccAucPost0.5475 | 0.379 | 0.407 | -0.027 [-0.048, -0.007] | 0.0023 | 0.0069 |
| D104 f0 ccAucRepair | ccAucPost0.5475 | 0.411 | 0.407 | +0.005 [-0.013, +0.022] | 0.55 | 0.68 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.5475 | 0.417 | 0.407 | +0.011 [-0.002, +0.024] | 0.34 | 0.68 |

Tumour-free test cases with no predicted tumour left (10 cases): D101 f0 all 0 / 0 / 0 / 0 / 0 / 0; D101 f1 all_rep2 0 / 0 / 0 / 0 / 0 / 0; D101 f2 ccReject 0 / 0 / 0 / 0 / 0 / 0; D103 f0 ccRepair 0 / 0 / 0 / 0 / 0 / 0; D103 f1 ccRepairDel 0 / 0 / 0 / 0 / 0 / 0; D104 f0 ccAucRepair 0 / 0 / 0 / 0 / 0 / 0; D104 f1 ccAucRepair_rep2 0 / 0 / 0 / 0 / 0 / 0 (variants in the order none, ccPost, ccAucPost0.5475, ccAucPost0.6, ccAucPost0.637, ccAucPost0.7).
