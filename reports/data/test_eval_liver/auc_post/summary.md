# The AUC rule applied to the predictions (ccAucPost) + a sweep of stricter cuts

111 real LiTS test cases, checkpoint_final predictions. ccPost = training cc rule on the prediction; ccAucPost<cut> = ccPost, then predicted tumour pieces inside the predicted main structure with sep < cut -> their surroundings' label (filter_metrics' separability with the predicted liver as organ). Decisions use image + prediction only. **Main cut 0.5475** (= the training cut, fixed before results); 0.60 / 0.637 / 0.70 = exploratory sweep. Written by `nnunet/auc_post.py`; exploratory, the main analysis stays predict_test.py.

## 1. Does sep tell false-positive pieces from hits? (pieces inside the predicted main structure)

| run | pieces / case (hit, FP) | unscored | sep median hit / FP | AUROC sep (FP lower) | < 0.5475: FP / hit pieces (hit ml) | < 0.6: FP / hit pieces (hit ml) | < 0.637: FP / hit pieces (hit ml) | < 0.7: FP / hit pieces (hit ml) |
|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 10.5 (2.8, 7.7) | 50 | 0.903 / 0.614 | 0.83 | 189 / 2 (20.3) | 368 / 9 (116.1) | 464 / 14 (259.5) | 560 / 26 (411.8) |
| D101 f1 all_rep2 | 10.2 (2.8, 7.4) | 38 | 0.895 / 0.622 | 0.80 | 171 / 5 (49.8) | 349 / 11 (267.1) | 421 / 21 (681.1) | 512 / 33 (798.4) |
| D101 f2 ccReject | 9.2 (2.5, 6.8) | 71 | 0.903 / 0.624 | 0.82 | 139 / 0 (0.0) | 292 / 3 (0.3) | 364 / 9 (126.6) | 448 / 18 (225.7) |
| D103 f0 ccRepair | 11.4 (2.7, 8.7) | 55 | 0.891 / 0.605 | 0.81 | 223 / 4 (41.9) | 447 / 9 (110.3) | 529 / 15 (274.7) | 633 / 29 (468.0) |
| D103 f1 ccRepairDel | 11.7 (2.6, 9.0) | 66 | 0.895 / 0.601 | 0.82 | 228 / 2 (48.5) | 470 / 4 (168.8) | 564 / 13 (375.9) | 659 / 24 (592.1) |
| D104 f0 ccAucRepair | 8.9 (2.6, 6.3) | 53 | 0.904 / 0.641 | 0.79 | 138 / 2 (32.3) | 251 / 5 (45.9) | 320 / 10 (224.6) | 399 / 19 (338.6) |
| D104 f1 ccAucRepair_rep2 | 10.6 (2.6, 7.9) | 52 | 0.898 / 0.652 | 0.74 | 158 / 2 (11.4) | 325 / 4 (142.6) | 398 / 14 (343.3) | 483 / 21 (438.2) |

## 2. Each variant vs ccPost, per run

Tumour Dice mean (NaN = both empty, left out) and the paired difference to ccPost; lesion recall, FP pieces and reference lesions lost are totals over the cases.

| run | variant | tumour Dice | vs ccPost [95% CI] | p | liver Dice | lesion recall | FP pieces / case | lesions lost vs ccPost | pieces relabelled |
|---|---|---|---|---|---|---|---|---|---|
| D101 f0 all | none | 0.301 | -0.078 [-0.101, -0.058] | 2.2e-14 | 0.870 | 0.422 | 66.8 | – | – |
| D101 f0 all | ccPost | 0.379 | – | – | 0.885 | 0.408 | 7.7 | 0 | – |
| D101 f0 all | ccAucPost0.5475 | 0.403 | +0.024 [+0.012, +0.039] | 1.4e-09 | 0.886 | 0.406 | 6.0 | 1 | 191 |
| D101 f0 all | ccAucPost0.6 | 0.434 | +0.055 [+0.034, +0.079] | 3.6e-10 | 0.888 | 0.399 | 4.4 | 6 | 377 |
| D101 f0 all | ccAucPost0.637 | 0.435 | +0.056 [+0.030, +0.086] | 2e-08 | 0.888 | 0.391 | 3.6 | 12 | 478 |
| D101 f0 all | ccAucPost0.7 | 0.436 | +0.057 [+0.024, +0.092] | 1.2e-06 | 0.888 | 0.375 | 2.7 | 23 | 586 |
| D101 f1 all_rep2 | none | 0.305 | -0.078 [-0.102, -0.057] | 1.2e-14 | 0.876 | 0.443 | 46.3 | – | – |
| D101 f1 all_rep2 | ccPost | 0.383 | – | – | 0.903 | 0.437 | 7.4 | 0 | – |
| D101 f1 all_rep2 | ccAucPost0.5475 | 0.402 | +0.019 [+0.007, +0.034] | 1.6e-08 | 0.905 | 0.433 | 5.8 | 3 | 176 |
| D101 f1 all_rep2 | ccAucPost0.6 | 0.440 | +0.057 [+0.034, +0.084] | 9.3e-10 | 0.906 | 0.425 | 4.2 | 9 | 360 |
| D101 f1 all_rep2 | ccAucPost0.637 | 0.445 | +0.061 [+0.033, +0.093] | 2.8e-08 | 0.906 | 0.410 | 3.6 | 19 | 442 |
| D101 f1 all_rep2 | ccAucPost0.7 | 0.440 | +0.057 [+0.023, +0.092] | 2.1e-06 | 0.906 | 0.391 | 2.7 | 33 | 545 |
| D101 f2 ccReject | none | 0.382 | -0.004 [-0.007, -0.002] | 8.7e-08 | 0.857 | 0.354 | 9.5 | – | – |
| D101 f2 ccReject | ccPost | 0.386 | – | – | 0.874 | 0.350 | 6.8 | 0 | – |
| D101 f2 ccReject | ccAucPost0.5475 | 0.392 | +0.006 [+0.002, +0.010] | 2.4e-09 | 0.874 | 0.350 | 5.5 | 0 | 139 |
| D101 f2 ccReject | ccAucPost0.6 | 0.399 | +0.013 [+0.006, +0.022] | 4.7e-11 | 0.874 | 0.348 | 4.2 | 1 | 295 |
| D101 f2 ccReject | ccAucPost0.637 | 0.395 | +0.009 [-0.008, +0.024] | 1.5e-08 | 0.874 | 0.341 | 3.5 | 6 | 373 |
| D101 f2 ccReject | ccAucPost0.7 | 0.386 | -0.000 [-0.028, +0.024] | 5.8e-06 | 0.874 | 0.331 | 2.8 | 13 | 466 |
| D103 f0 ccRepair | none | 0.375 | -0.005 [-0.009, -0.002] | 1.3e-09 | 0.866 | 0.416 | 10.8 | – | – |
| D103 f0 ccRepair | ccPost | 0.380 | – | – | 0.883 | 0.408 | 8.7 | 0 | – |
| D103 f0 ccRepair | ccAucPost0.5475 | 0.407 | +0.027 [+0.014, +0.044] | 2.8e-09 | 0.884 | 0.406 | 6.7 | 1 | 227 |
| D103 f0 ccRepair | ccAucPost0.6 | 0.442 | +0.062 [+0.038, +0.089] | 4.4e-11 | 0.885 | 0.399 | 4.7 | 6 | 456 |
| D103 f0 ccRepair | ccAucPost0.637 | 0.440 | +0.060 [+0.032, +0.090] | 1.2e-09 | 0.885 | 0.391 | 4.0 | 12 | 544 |
| D103 f0 ccRepair | ccAucPost0.7 | 0.433 | +0.053 [+0.021, +0.087] | 1.2e-06 | 0.886 | 0.371 | 3.0 | 26 | 662 |
| D103 f1 ccRepairDel | none | 0.370 | -0.007 [-0.011, -0.004] | 2.9e-10 | 0.851 | 0.403 | 12.0 | – | – |
| D103 f1 ccRepairDel | ccPost | 0.377 | – | – | 0.870 | 0.399 | 9.0 | 0 | – |
| D103 f1 ccRepairDel | ccAucPost0.5475 | 0.402 | +0.025 [+0.014, +0.039] | 1.1e-10 | 0.871 | 0.396 | 7.0 | 2 | 230 |
| D103 f1 ccRepairDel | ccAucPost0.6 | 0.436 | +0.059 [+0.038, +0.084] | 4.9e-12 | 0.872 | 0.394 | 4.8 | 4 | 474 |
| D103 f1 ccRepairDel | ccAucPost0.637 | 0.436 | +0.059 [+0.032, +0.088] | 2.5e-10 | 0.873 | 0.379 | 3.9 | 14 | 577 |
| D103 f1 ccRepairDel | ccAucPost0.7 | 0.430 | +0.053 [+0.020, +0.086] | 1.5e-07 | 0.873 | 0.364 | 3.1 | 25 | 683 |
| D104 f0 ccAucRepair | none | 0.403 | -0.009 [-0.015, -0.004] | 8.4e-12 | 0.870 | 0.413 | 9.4 | – | – |
| D104 f0 ccAucRepair | ccPost | 0.412 | – | – | 0.892 | 0.410 | 6.3 | 0 | – |
| D104 f0 ccAucRepair | ccAucPost0.5475 | 0.426 | +0.014 [+0.006, +0.024] | 7.3e-09 | 0.893 | 0.409 | 5.0 | 1 | 140 |
| D104 f0 ccAucRepair | ccAucPost0.6 | 0.445 | +0.033 [+0.019, +0.050] | 8.6e-11 | 0.893 | 0.408 | 4.0 | 2 | 256 |
| D104 f0 ccAucRepair | ccAucPost0.637 | 0.446 | +0.034 [+0.014, +0.055] | 3.7e-09 | 0.893 | 0.399 | 3.4 | 8 | 330 |
| D104 f0 ccAucRepair | ccAucPost0.7 | 0.444 | +0.032 [+0.002, +0.061] | 1.5e-07 | 0.893 | 0.388 | 2.7 | 16 | 418 |
| D104 f1 ccAucRepair_rep2 | none | 0.381 | -0.013 [-0.021, -0.007] | 3.2e-13 | 0.873 | 0.413 | 12.2 | – | – |
| D104 f1 ccAucRepair_rep2 | ccPost | 0.395 | – | – | 0.896 | 0.403 | 7.9 | 0 | – |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.5475 | 0.417 | +0.022 [+0.011, +0.037] | 4.5e-09 | 0.898 | 0.402 | 6.5 | 1 | 160 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.6 | 0.452 | +0.058 [+0.037, +0.081] | 2e-12 | 0.898 | 0.399 | 5.0 | 3 | 329 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.637 | 0.446 | +0.051 [+0.026, +0.077] | 2.1e-09 | 0.899 | 0.385 | 4.4 | 13 | 412 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.7 | 0.444 | +0.049 [+0.018, +0.080] | 5.4e-08 | 0.899 | 0.375 | 3.6 | 20 | 504 |

## 3. Post-processing vs training (the question of this script)

Baseline = per-case mean of D101 f0 all + D101 f1 all_rep2. y − x per case, paired bootstrap CI, Wilcoxon p (unadjusted).

| comparison | x | y | y − x [95% CI] | p |
|---|---|---|---|---|
| baseline: ccAucPost0.5475 vs ccPost (what AUC post-processing adds) | 0.381 | 0.403 | +0.022 [+0.011, +0.035] | 7.8e-10 |
| D104 f0 ccAucRepair ccPost − baseline ccAucPost0.5475 (training vs post-processing; ≈ 0 = post-processing does it) | 0.403 | 0.412 | +0.009 [-0.008, +0.026] | 0.42 |
| D104 f0 ccAucRepair: ccAucPost0.5475 vs ccPost (AUC post on top of AUC training) | 0.412 | 0.426 | +0.014 [+0.006, +0.024] | 7.3e-09 |
| D104 f1 ccAucRepair_rep2 ccPost − baseline ccAucPost0.5475 (training vs post-processing; ≈ 0 = post-processing does it) | 0.403 | 0.395 | -0.008 [-0.021, +0.004] | 0.053 |
| D104 f1 ccAucRepair_rep2: ccAucPost0.5475 vs ccPost (AUC post on top of AUC training) | 0.395 | 0.417 | +0.022 [+0.011, +0.037] | 4.5e-09 |

Arms vs baseline when both get the same post-processing (Holm over the 5 arms listed):

| arm | variant | arm | baseline | diff [95% CI] | p | Holm p |
|---|---|---|---|---|---|---|
| D101 f2 ccReject | ccPost | 0.386 | 0.381 | +0.005 [-0.028, +0.039] | 0.89 | 1 |
| D103 f0 ccRepair | ccPost | 0.380 | 0.381 | -0.001 [-0.011, +0.008] | 0.86 | 1 |
| D103 f1 ccRepairDel | ccPost | 0.377 | 0.381 | -0.004 [-0.021, +0.012] | 0.62 | 1 |
| D104 f0 ccAucRepair | ccPost | 0.412 | 0.381 | +0.031 [+0.011, +0.051] | 0.00053 | 0.0027 |
| D104 f1 ccAucRepair_rep2 | ccPost | 0.395 | 0.381 | +0.014 [+0.003, +0.025] | 0.023 | 0.092 |
| D101 f2 ccReject | ccAucPost0.5475 | 0.392 | 0.403 | -0.011 [-0.046, +0.023] | 0.65 | 1 |
| D103 f0 ccRepair | ccAucPost0.5475 | 0.407 | 0.403 | +0.004 [-0.006, +0.017] | 0.86 | 1 |
| D103 f1 ccRepairDel | ccAucPost0.5475 | 0.402 | 0.403 | -0.001 [-0.016, +0.014] | 0.77 | 1 |
| D104 f0 ccAucRepair | ccAucPost0.5475 | 0.426 | 0.403 | +0.023 [+0.005, +0.043] | 0.0087 | 0.043 |
| D104 f1 ccAucRepair_rep2 | ccAucPost0.5475 | 0.417 | 0.403 | +0.014 [+0.002, +0.028] | 0.18 | 0.71 |

Tumour-free test cases with no predicted tumour left (10 cases): D101 f0 all 0 / 0 / 0 / 0 / 0 / 0; D101 f1 all_rep2 0 / 0 / 0 / 0 / 0 / 0; D101 f2 ccReject 0 / 0 / 0 / 0 / 0 / 0; D103 f0 ccRepair 0 / 0 / 0 / 0 / 0 / 0; D103 f1 ccRepairDel 0 / 0 / 0 / 0 / 0 / 0; D104 f0 ccAucRepair 0 / 0 / 0 / 0 / 0 / 0; D104 f1 ccAucRepair_rep2 0 / 0 / 0 / 0 / 0 / 0 (variants in the order none, ccPost, ccAucPost0.5475, ccAucPost0.6, ccAucPost0.637, ccAucPost0.7).
