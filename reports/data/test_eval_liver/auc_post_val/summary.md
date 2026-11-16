# The AUC rule applied to the predictions (ccAucPost) + a sweep of stricter cuts

20 real LiTS val cases, checkpoint_final predictions. ccPost = training cc rule on the prediction; ccAucPost<cut> = ccPost, then predicted tumour pieces inside the predicted main structure with sep < cut -> their surroundings' label (filter_metrics' separability with the predicted liver as organ). Decisions use image + prediction only. **Main cut 0.5475** (= the training cut, fixed before results); 0.60 / 0.637 / 0.70 = exploratory sweep. Written by `nnunet/auc_post.py`; exploratory, the main analysis stays predict_test.py.

## 1. Does sep tell false-positive pieces from hits? (pieces inside the predicted main structure)

| run | pieces / case (hit, FP) | unscored | sep median hit / FP | AUROC sep (FP lower) | < 0.5475: FP / hit pieces (hit ml) | < 0.6: FP / hit pieces (hit ml) | < 0.637: FP / hit pieces (hit ml) | < 0.7: FP / hit pieces (hit ml) |
|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 9.9 (3.6, 6.3) | 12 | 0.910 / 0.630 | 0.80 | 21 / 1 (0.2) | 48 / 2 (3.0) | 60 / 2 (3.0) | 71 / 7 (77.6) |
| D101 f1 all_rep2 | 9.8 (4.3, 5.5) | 8 | 0.906 / 0.629 | 0.77 | 24 / 0 (0.0) | 44 / 3 (281.8) | 56 / 3 (281.8) | 65 / 6 (300.9) |
| D101 f2 ccReject | 8.2 (2.2, 5.9) | 15 | 0.869 / 0.635 | 0.77 | 22 / 0 (0.0) | 39 / 0 (0.0) | 52 / 0 (0.0) | 67 / 4 (18.4) |
| D103 f0 ccRepair | 10.9 (3.6, 7.3) | 9 | 0.902 / 0.607 | 0.81 | 35 / 0 (0.0) | 68 / 0 (0.0) | 80 / 1 (1.9) | 94 / 4 (24.2) |
| D103 f1 ccRepairDel | 11.2 (3.5, 7.8) | 11 | 0.893 / 0.617 | 0.76 | 35 / 2 (0.3) | 66 / 4 (18.5) | 77 / 4 (18.5) | 91 / 8 (43.5) |
| D104 f0 ccAucRepair | 8.9 (3.7, 5.2) | 6 | 0.904 / 0.702 | 0.70 | 16 / 0 (0.0) | 35 / 1 (0.1) | 40 / 1 (0.1) | 48 / 4 (19.3) |
| D104 f1 ccAucRepair_rep2 | 10.3 (3.7, 6.6) | 9 | 0.896 / 0.712 | 0.67 | 26 / 0 (0.0) | 40 / 2 (0.5) | 47 / 3 (2.4) | 61 / 7 (26.1) |

## 2. Each variant vs ccPost, per run

Tumour Dice mean (NaN = both empty, left out) and the paired difference to ccPost; lesion recall, FP pieces and reference lesions lost are totals over the cases.

| run | variant | tumour Dice | vs ccPost [95% CI] | p | liver Dice | lesion recall | FP pieces / case | lesions lost vs ccPost | pieces relabelled |
|---|---|---|---|---|---|---|---|---|---|
| D101 f0 all | none | 0.289 | -0.097 [-0.149, -0.050] | 0.00098 | 0.860 | 0.276 | 64.4 | – | – |
| D101 f0 all | ccPost | 0.386 | – | – | 0.882 | 0.261 | 6.3 | 0 | – |
| D101 f0 all | ccAucPost0.5475 | 0.395 | +0.009 [+0.000, +0.024] | 0.0077 | 0.882 | 0.256 | 5.2 | 1 | 22 |
| D101 f0 all | ccAucPost0.6 | 0.420 | +0.034 [+0.006, +0.074] | 0.0015 | 0.883 | 0.251 | 3.9 | 2 | 50 |
| D101 f0 all | ccAucPost0.637 | 0.421 | +0.035 [+0.006, +0.075] | 0.00098 | 0.883 | 0.251 | 3.3 | 2 | 62 |
| D101 f0 all | ccAucPost0.7 | 0.373 | -0.013 [-0.092, +0.057] | 0.11 | 0.883 | 0.236 | 2.8 | 5 | 78 |
| D101 f1 all_rep2 | none | 0.297 | -0.113 [-0.170, -0.061] | 0.00065 | 0.859 | 0.372 | 43.0 | – | – |
| D101 f1 all_rep2 | ccPost | 0.410 | – | – | 0.891 | 0.367 | 5.5 | 0 | – |
| D101 f1 all_rep2 | ccAucPost0.5475 | 0.429 | +0.019 [+0.001, +0.042] | 0.012 | 0.891 | 0.367 | 4.3 | 0 | 24 |
| D101 f1 all_rep2 | ccAucPost0.6 | 0.435 | +0.025 [-0.035, +0.077] | 0.05 | 0.892 | 0.352 | 3.4 | 3 | 47 |
| D101 f1 all_rep2 | ccAucPost0.637 | 0.436 | +0.026 [-0.034, +0.078] | 0.05 | 0.892 | 0.352 | 2.8 | 3 | 59 |
| D101 f1 all_rep2 | ccAucPost0.7 | 0.412 | +0.002 [-0.079, +0.071] | 0.17 | 0.892 | 0.342 | 2.3 | 5 | 71 |
| D101 f2 ccReject | none | 0.365 | -0.002 [-0.009, +0.007] | 0.11 | 0.831 | 0.271 | 8.6 | – | – |
| D101 f2 ccReject | ccPost | 0.367 | – | – | 0.837 | 0.216 | 5.9 | 0 | – |
| D101 f2 ccReject | ccAucPost0.5475 | 0.370 | +0.004 [+0.000, +0.008] | 0.0051 | 0.837 | 0.216 | 4.8 | 0 | 22 |
| D101 f2 ccReject | ccAucPost0.6 | 0.372 | +0.005 [+0.001, +0.011] | 0.0033 | 0.837 | 0.216 | 4.0 | 0 | 39 |
| D101 f2 ccReject | ccAucPost0.637 | 0.372 | +0.005 [+0.001, +0.011] | 0.0033 | 0.837 | 0.216 | 3.3 | 0 | 52 |
| D101 f2 ccReject | ccAucPost0.7 | 0.340 | -0.027 [-0.108, +0.019] | 0.05 | 0.836 | 0.201 | 2.5 | 3 | 71 |
| D103 f0 ccRepair | none | 0.384 | +0.003 [-0.008, +0.020] | 0.019 | 0.850 | 0.322 | 9.7 | – | – |
| D103 f0 ccRepair | ccPost | 0.381 | – | – | 0.880 | 0.307 | 7.3 | 0 | – |
| D103 f0 ccRepair | ccAucPost0.5475 | 0.394 | +0.013 [+0.001, +0.027] | 0.0051 | 0.882 | 0.307 | 5.6 | 0 | 35 |
| D103 f0 ccRepair | ccAucPost0.6 | 0.396 | +0.015 [+0.004, +0.028] | 0.0022 | 0.882 | 0.307 | 4.0 | 0 | 68 |
| D103 f0 ccRepair | ccAucPost0.637 | 0.410 | +0.028 [+0.005, +0.064] | 0.0022 | 0.882 | 0.302 | 3.4 | 1 | 81 |
| D103 f0 ccRepair | ccAucPost0.7 | 0.379 | -0.002 [-0.087, +0.059] | 0.034 | 0.882 | 0.291 | 2.6 | 3 | 98 |
| D103 f1 ccRepairDel | none | 0.364 | -0.006 [-0.016, +0.000] | 0.023 | 0.842 | 0.276 | 10.5 | – | – |
| D103 f1 ccRepairDel | ccPost | 0.370 | – | – | 0.869 | 0.261 | 7.8 | 0 | – |
| D103 f1 ccRepairDel | ccAucPost0.5475 | 0.380 | +0.010 [+0.002, +0.023] | 0.0022 | 0.869 | 0.256 | 6.0 | 1 | 37 |
| D103 f1 ccRepairDel | ccAucPost0.6 | 0.387 | +0.018 [+0.003, +0.035] | 0.0046 | 0.872 | 0.251 | 4.5 | 2 | 70 |
| D103 f1 ccRepairDel | ccAucPost0.637 | 0.401 | +0.031 [+0.004, +0.072] | 0.0037 | 0.872 | 0.251 | 4.0 | 2 | 81 |
| D103 f1 ccRepairDel | ccAucPost0.7 | 0.364 | -0.006 [-0.101, +0.064] | 0.046 | 0.872 | 0.236 | 3.2 | 5 | 99 |
| D104 f0 ccAucRepair | none | 0.397 | +0.004 [-0.019, +0.039] | 0.026 | 0.848 | 0.307 | 7.8 | – | – |
| D104 f0 ccAucRepair | ccPost | 0.393 | – | – | 0.875 | 0.296 | 5.2 | 0 | – |
| D104 f0 ccAucRepair | ccAucPost0.5475 | 0.394 | +0.002 [+0.000, +0.003] | 0.018 | 0.875 | 0.296 | 4.4 | 0 | 16 |
| D104 f0 ccAucRepair | ccAucPost0.6 | 0.395 | +0.002 [+0.001, +0.004] | 0.021 | 0.876 | 0.291 | 3.5 | 1 | 36 |
| D104 f0 ccAucRepair | ccAucPost0.637 | 0.395 | +0.003 [+0.001, +0.005] | 0.015 | 0.876 | 0.291 | 3.2 | 1 | 41 |
| D104 f0 ccAucRepair | ccAucPost0.7 | 0.383 | -0.009 [-0.057, +0.023] | 0.11 | 0.876 | 0.276 | 2.8 | 4 | 52 |
| D104 f1 ccAucRepair_rep2 | none | 0.389 | -0.001 [-0.011, +0.013] | 0.023 | 0.853 | 0.312 | 10.8 | – | – |
| D104 f1 ccAucRepair_rep2 | ccPost | 0.390 | – | – | 0.886 | 0.302 | 6.6 | 0 | – |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.5475 | 0.394 | +0.004 [+0.001, +0.007] | 0.0033 | 0.887 | 0.302 | 5.3 | 0 | 26 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.6 | 0.394 | +0.004 [+0.002, +0.007] | 0.0015 | 0.887 | 0.291 | 4.6 | 2 | 42 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.637 | 0.411 | +0.021 [+0.004, +0.053] | 0.0015 | 0.887 | 0.286 | 4.2 | 3 | 50 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.7 | 0.375 | -0.015 [-0.108, +0.050] | 0.023 | 0.887 | 0.271 | 3.5 | 6 | 68 |

## 3. Post-processing vs training (the question of this script)

Baseline = per-case mean of D101 f0 all + D101 f1 all_rep2. y − x per case, paired bootstrap CI, Wilcoxon p (unadjusted).

| comparison | x | y | y − x [95% CI] | p |
|---|---|---|---|---|
| baseline: ccAucPost0.5475 vs ccPost (what AUC post-processing adds) | 0.398 | 0.412 | +0.014 [+0.001, +0.032] | 0.0022 |
| D104 f0 ccAucRepair ccPost − baseline ccAucPost0.5475 (training vs post-processing; ≈ 0 = post-processing does it) | 0.412 | 0.393 | -0.020 [-0.080, +0.030] | 0.73 |
| D104 f0 ccAucRepair: ccAucPost0.5475 vs ccPost (AUC post on top of AUC training) | 0.393 | 0.394 | +0.002 [+0.000, +0.003] | 0.018 |
| D104 f1 ccAucRepair_rep2 ccPost − baseline ccAucPost0.5475 (training vs post-processing; ≈ 0 = post-processing does it) | 0.412 | 0.390 | -0.022 [-0.078, +0.021] | 0.91 |
| D104 f1 ccAucRepair_rep2: ccAucPost0.5475 vs ccPost (AUC post on top of AUC training) | 0.390 | 0.394 | +0.004 [+0.001, +0.007] | 0.0033 |

Arms vs baseline when both get the same post-processing (Holm over the 5 arms listed):

| arm | variant | arm | baseline | diff [95% CI] | p | Holm p |
|---|---|---|---|---|---|---|
| D101 f2 ccReject | ccPost | 0.367 | 0.398 | -0.031 [-0.141, +0.065] | 0.92 | 1 |
| D103 f0 ccRepair | ccPost | 0.381 | 0.398 | -0.017 [-0.073, +0.026] | 1 | 1 |
| D103 f1 ccRepairDel | ccPost | 0.370 | 0.398 | -0.028 [-0.087, +0.021] | 0.39 | 1 |
| D104 f0 ccAucRepair | ccPost | 0.393 | 0.398 | -0.006 [-0.068, +0.048] | 0.61 | 1 |
| D104 f1 ccAucRepair_rep2 | ccPost | 0.390 | 0.398 | -0.008 [-0.067, +0.039] | 0.5 | 1 |
| D101 f2 ccReject | ccAucPost0.5475 | 0.370 | 0.412 | -0.042 [-0.150, +0.052] | 0.92 | 1 |
| D103 f0 ccRepair | ccAucPost0.5475 | 0.394 | 0.412 | -0.018 [-0.074, +0.023] | 0.86 | 1 |
| D103 f1 ccRepairDel | ccAucPost0.5475 | 0.380 | 0.412 | -0.032 [-0.090, +0.015] | 0.39 | 1 |
| D104 f0 ccAucRepair | ccAucPost0.5475 | 0.394 | 0.412 | -0.018 [-0.079, +0.032] | 0.86 | 1 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.5475 | 0.394 | 0.412 | -0.018 [-0.075, +0.025] | 0.82 | 1 |

Tumour-free test cases with no predicted tumour left (3 cases): D101 f0 all 0 / 0 / 0 / 0 / 0 / 0; D101 f1 all_rep2 0 / 0 / 0 / 0 / 0 / 0; D101 f2 ccReject 0 / 0 / 0 / 0 / 0 / 0; D103 f0 ccRepair 0 / 0 / 0 / 0 / 0 / 0; D103 f1 ccRepairDel 0 / 0 / 0 / 0 / 0 / 0; D104 f0 ccAucRepair 0 / 0 / 0 / 0 / 0 / 0; D104 f1 ccAucRepair_rep2 0 / 0 / 0 / 0 / 0 / 0 (variants in the order none, ccPost, ccAucPost0.5475, ccAucPost0.6, ccAucPost0.637, ccAucPost0.7).
