# False-positive tumour pieces: where they sit, and the cc rule applied to the predictions

111 real LiTS test cases, checkpoint_best predictions. Piece = 26-connected predicted tumour component; FP = overlaps no reference lesion. *Predicted main structure* = the piece of predicted liver ∪ tumour that holds the largest predicted liver piece (the training cc rule). *Reference liver share* = fraction of the piece's voxels with reference label > 0. ccPost = everything outside the predicted main structure set to background. Written by `nnunet/fp_location.py` (exploratory; main analysis = predict_test.py).

## 1. Where the false positives sit

| run | FP pieces / case | outside pred. main | FP ml / case | ml outside pred. main | FP outside ref. liver (share < 0.5) | hit pieces outside pred. main | cases without predicted liver |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 39.9 | 32.9 (82 %) | 36.20 | 12.13 (34 %) | 85 % | 29 | 0 |
| D101 f1 all_rep2 | 60.9 | 52.3 (86 %) | 28.64 | 15.83 (55 %) | 89 % | 16 | 0 |
| D101 f2 ccReject | 23.7 | 9.9 (42 %) | 7.49 | 1.13 (15 %) | 59 % | 11 | 0 |
| D103 f0 ccRepair | 16.5 | 6.2 (37 %) | 16.35 | 2.48 (15 %) | 56 % | 15 | 0 |
| D103 f1 ccRepairDel | 20.1 | 7.9 (39 %) | 17.51 | 2.67 (15 %) | 52 % | 6 | 0 |
| D104 f0 ccAucRepair | 20.6 | 9.9 (48 %) | 9.33 | 2.28 (24 %) | 67 % | 11 | 0 |
| D104 f1 ccAucRepair_rep2 | 12.5 | 4.4 (35 %) | 14.21 | 2.39 (17 %) | 57 % | 15 | 0 |

## 2. Size of the false-positive pieces (count, share of the run's FP pieces)

| run | where | 0–0.01 ml | 0.01–0.1 ml | 0.1–1 ml | 1–10 ml | ≥ 10 ml | median ml |
|---|---|---|---|---|---|---|---|
| D101 f0 all | inside pred. main | 71 (2 %) | 292 (7 %) | 320 (7 %) | 72 (2 %) | 29 (1 %) | 0.113 |
| D101 f0 all | outside pred. main | 772 (17 %) | 1895 (43 %) | 790 (18 %) | 171 (4 %) | 22 (0 %) | 0.040 |
| D101 f1 all_rep2 | inside pred. main | 62 (1 %) | 299 (4 %) | 499 (7 %) | 75 (1 %) | 24 (0 %) | 0.151 |
| D101 f1 all_rep2 | outside pred. main | 1053 (16 %) | 2976 (44 %) | 1586 (23 %) | 165 (2 %) | 21 (0 %) | 0.048 |
| D101 f2 ccReject | inside pred. main | 26 (1 %) | 588 (22 %) | 888 (34 %) | 18 (1 %) | 3 (0 %) | 0.118 |
| D101 f2 ccReject | outside pred. main | 203 (8 %) | 601 (23 %) | 284 (11 %) | 14 (1 %) | 1 (0 %) | 0.043 |
| D103 f0 ccRepair | inside pred. main | 43 (2 %) | 257 (14 %) | 690 (38 %) | 126 (7 %) | 33 (2 %) | 0.250 |
| D103 f0 ccRepair | outside pred. main | 79 (4 %) | 261 (14 %) | 295 (16 %) | 45 (2 %) | 3 (0 %) | 0.101 |
| D103 f1 ccRepairDel | inside pred. main | 43 (2 %) | 240 (11 %) | 970 (43 %) | 84 (4 %) | 21 (1 %) | 0.224 |
| D103 f1 ccRepairDel | outside pred. main | 133 (6 %) | 381 (17 %) | 320 (14 %) | 36 (2 %) | 4 (0 %) | 0.072 |
| D104 f0 ccAucRepair | inside pred. main | 69 (3 %) | 346 (15 %) | 706 (31 %) | 55 (2 %) | 12 (1 %) | 0.157 |
| D104 f0 ccAucRepair | outside pred. main | 153 (7 %) | 466 (20 %) | 436 (19 %) | 39 (2 %) | 2 (0 %) | 0.076 |
| D104 f1 ccAucRepair_rep2 | inside pred. main | 59 (4 %) | 291 (21 %) | 456 (33 %) | 72 (5 %) | 20 (1 %) | 0.147 |
| D104 f1 ccAucRepair_rep2 | outside pred. main | 78 (6 %) | 196 (14 %) | 173 (12 %) | 39 (3 %) | 2 (0 %) | 0.074 |

## 3. ccPost vs the saved prediction, per run

Means as predict_test.py (NaN = both empty, left out); diff/CI/p over the cases scored in both variants. A tumour-free case that becomes clean turns from Dice 0 into NaN and drops out (counted in the last column).

| run | tumour Dice none → ccPost | diff [95% CI] | Wilcoxon p | liver Dice none → ccPost | lesion recall | FP pieces / case | FP ml / case | ref. lesions lost | tumour-free cases clean |
|---|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 0.312 → 0.382 | +0.070 [+0.051, +0.092] | 4.6e-15 | 0.882 → 0.898 | 0.468 → 0.456 | 39.9 → 7.1 | 36.20 → 24.07 | 9 | 0/10 |
| D101 f1 all_rep2 | 0.319 → 0.390 | +0.071 [+0.052, +0.092] | 8.1e-15 | 0.875 → 0.903 | 0.457 → 0.449 | 60.9 → 8.6 | 28.64 → 12.81 | 6 | 0/10 |
| D101 f2 ccReject | 0.333 → 0.343 | +0.010 [+0.007, +0.013] | 2.9e-12 | 0.896 → 0.930 | 0.329 → 0.324 | 23.7 → 13.7 | 7.49 → 6.36 | 3 | 0/10 |
| D103 f0 ccRepair | 0.330 → 0.339 | +0.009 [+0.001, +0.015] | 1.2e-10 | 0.848 → 0.872 | 0.389 → 0.379 | 16.5 → 10.4 | 16.35 → 13.87 | 7 | 0/10 |
| D103 f1 ccRepairDel | 0.349 → 0.363 | +0.014 [+0.008, +0.020] | 2.1e-13 | 0.873 → 0.888 | 0.395 → 0.392 | 20.1 → 12.2 | 17.51 → 14.83 | 2 | 0/10 |
| D104 f0 ccAucRepair | 0.378 → 0.400 | +0.022 [+0.015, +0.031] | 6e-15 | 0.873 → 0.925 | 0.420 → 0.416 | 20.6 → 10.7 | 9.33 → 7.05 | 3 | 0/10 |
| D104 f1 ccAucRepair_rep2 | 0.381 → 0.395 | +0.014 [+0.008, +0.022] | 3.4e-13 | 0.873 → 0.897 | 0.415 → 0.405 | 12.5 → 8.1 | 14.21 → 11.82 | 7 | 0/10 |

## 4. Arms vs baseline when both are post-processed (exploratory)

Baseline = per-case mean of D101 f0 all + D101 f1 all_rep2 (NaN only if all are NaN). Holm over the arms listed.

| arm | variant | arm tumour Dice | baseline | diff [95% CI] | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|
| D101 f2 ccReject | none | 0.333 | 0.315 | +0.018 [-0.014, +0.051] | 0.57 | 0.57 |
| D103 f0 ccRepair | none | 0.330 | 0.315 | +0.015 [-0.001, +0.032] | 0.21 | 0.41 |
| D103 f1 ccRepairDel | none | 0.349 | 0.315 | +0.034 [+0.015, +0.054] | 0.00031 | 0.00092 |
| D104 f0 ccAucRepair | none | 0.378 | 0.315 | +0.063 [+0.042, +0.085] | 3.1e-07 | 1.2e-06 |
| D104 f1 ccAucRepair_rep2 | none | 0.381 | 0.315 | +0.066 [+0.046, +0.086] | 2.6e-10 | 1.3e-09 |
| D101 f2 ccReject | ccPost | 0.343 | 0.386 | -0.043 [-0.073, -0.014] | 0.001 | 0.004 |
| D103 f0 ccRepair | ccPost | 0.339 | 0.386 | -0.047 [-0.063, -0.031] | 8.6e-09 | 4.3e-08 |
| D103 f1 ccRepairDel | ccPost | 0.363 | 0.386 | -0.023 [-0.042, -0.005] | 0.0049 | 0.015 |
| D104 f0 ccAucRepair | ccPost | 0.400 | 0.386 | +0.014 [-0.005, +0.034] | 0.69 | 0.69 |
| D104 f1 ccAucRepair_rep2 | ccPost | 0.395 | 0.386 | +0.009 [-0.001, +0.019] | 0.28 | 0.56 |
