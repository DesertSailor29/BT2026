# False-positive tumour pieces: where they sit, and the cc rule applied to the predictions

111 real LiTS test cases, checkpoint_final predictions. Piece = 26-connected predicted tumour component; FP = overlaps no reference lesion. *Predicted main structure* = the piece of predicted liver ∪ tumour that holds the largest predicted liver piece (the training cc rule). *Reference liver share* = fraction of the piece's voxels with reference label > 0. ccPost = everything outside the predicted main structure set to background. Written by `nnunet/fp_location.py` (exploratory; main analysis = predict_test.py).

## 1. Where the false positives sit

| run | FP pieces / case | outside pred. main | FP ml / case | ml outside pred. main | FP outside ref. liver (share < 0.5) | hit pieces outside pred. main | cases without predicted liver |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 66.8 | 59.1 (88 %) | 35.01 | 17.42 (50 %) | 90 % | 22 | 0 |
| D101 f1 all_rep2 | 46.3 | 38.9 (84 %) | 40.93 | 22.18 (54 %) | 88 % | 11 | 0 |
| D101 f2 ccReject | 9.5 | 2.7 (28 %) | 3.04 | 0.68 (22 %) | 44 % | 13 | 0 |
| D103 f0 ccRepair | 10.8 | 2.1 (19 %) | 13.41 | 0.63 (5 %) | 38 % | 10 | 0 |
| D103 f1 ccRepairDel | 12.0 | 3.0 (25 %) | 14.80 | 2.12 (14 %) | 42 % | 7 | 0 |
| D104 f0 ccAucRepair | 9.4 | 3.2 (33 %) | 9.17 | 2.47 (27 %) | 51 % | 11 | 0 |

## 2. Size of the false-positive pieces (count, share of the run's FP pieces)

| run | where | 0–0.01 ml | 0.01–0.1 ml | 0.1–1 ml | 1–10 ml | ≥ 10 ml | median ml |
|---|---|---|---|---|---|---|---|
| D101 f0 all | inside pred. main | 56 (1 %) | 266 (4 %) | 419 (6 %) | 90 (1 %) | 28 (0 %) | 0.156 |
| D101 f0 all | outside pred. main | 1079 (15 %) | 3411 (46 %) | 1748 (24 %) | 295 (4 %) | 24 (0 %) | 0.049 |
| D101 f1 all_rep2 | inside pred. main | 42 (1 %) | 252 (5 %) | 402 (8 %) | 94 (2 %) | 27 (1 %) | 0.172 |
| D101 f1 all_rep2 | outside pred. main | 746 (15 %) | 2041 (40 %) | 1246 (24 %) | 246 (5 %) | 41 (1 %) | 0.055 |
| D101 f2 ccReject | inside pred. main | 81 (8 %) | 411 (39 %) | 238 (23 %) | 21 (2 %) | 3 (0 %) | 0.065 |
| D101 f2 ccReject | outside pred. main | 93 (9 %) | 144 (14 %) | 55 (5 %) | 5 (0 %) | 1 (0 %) | 0.027 |
| D103 f0 ccRepair | inside pred. main | 65 (5 %) | 285 (24 %) | 510 (42 %) | 83 (7 %) | 25 (2 %) | 0.155 |
| D103 f0 ccRepair | outside pred. main | 27 (2 %) | 103 (9 %) | 89 (7 %) | 14 (1 %) | 0 (0 %) | 0.078 |
| D103 f1 ccRepairDel | inside pred. main | 73 (5 %) | 331 (25 %) | 500 (38 %) | 73 (5 %) | 25 (2 %) | 0.138 |
| D103 f1 ccRepairDel | outside pred. main | 67 (5 %) | 128 (10 %) | 107 (8 %) | 26 (2 %) | 2 (0 %) | 0.064 |
| D104 f0 ccAucRepair | inside pred. main | 50 (5 %) | 234 (22 %) | 352 (34 %) | 46 (4 %) | 15 (1 %) | 0.146 |
| D104 f0 ccAucRepair | outside pred. main | 79 (8 %) | 143 (14 %) | 110 (11 %) | 14 (1 %) | 4 (0 %) | 0.060 |

## 3. ccPost vs the saved prediction, per run

Means as predict_test.py (NaN = both empty, left out); diff/CI/p over the cases scored in both variants. A tumour-free case that becomes clean turns from Dice 0 into NaN and drops out (counted in the last column).

| run | tumour Dice none → ccPost | diff [95% CI] | Wilcoxon p | liver Dice none → ccPost | lesion recall | FP pieces / case | FP ml / case | ref. lesions lost | tumour-free cases clean |
|---|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 0.301 → 0.379 | +0.078 [+0.058, +0.101] | 2.2e-14 | 0.870 → 0.885 | 0.422 → 0.408 | 66.8 → 7.7 | 35.01 → 17.59 | 10 | 0/10 |
| D101 f1 all_rep2 | 0.305 → 0.383 | +0.078 [+0.057, +0.102] | 1.2e-14 | 0.876 → 0.903 | 0.443 → 0.437 | 46.3 → 7.4 | 40.93 → 18.75 | 4 | 0/10 |
| D101 f2 ccReject | 0.382 → 0.386 | +0.004 [+0.002, +0.007] | 8.7e-08 | 0.857 → 0.874 | 0.354 → 0.350 | 9.5 → 6.8 | 3.04 → 2.36 | 3 | 0/10 |
| D103 f0 ccRepair | 0.375 → 0.380 | +0.005 [+0.002, +0.009] | 1.3e-09 | 0.866 → 0.883 | 0.416 → 0.408 | 10.8 → 8.7 | 13.41 → 12.78 | 6 | 0/10 |
| D103 f1 ccRepairDel | 0.370 → 0.377 | +0.007 [+0.004, +0.011] | 2.9e-10 | 0.851 → 0.870 | 0.403 → 0.399 | 12.0 → 9.0 | 14.80 → 12.68 | 3 | 0/10 |
| D104 f0 ccAucRepair | 0.403 → 0.412 | +0.009 [+0.004, +0.015] | 8.4e-12 | 0.870 → 0.892 | 0.413 → 0.410 | 9.4 → 6.3 | 9.17 → 6.71 | 2 | 0/10 |

## 4. Arms vs baseline when both are post-processed (exploratory)

Baseline = per-case mean of D101 f0 all + D101 f1 all_rep2 (NaN only if all are NaN). Holm over the arms listed.

| arm | variant | arm tumour Dice | baseline | diff [95% CI] | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|
| D101 f2 ccReject | none | 0.382 | 0.303 | +0.079 [+0.038, +0.120] | 0.0011 | 0.0011 |
| D103 f0 ccRepair | none | 0.375 | 0.303 | +0.072 [+0.051, +0.094] | 8.1e-10 | 2.4e-09 |
| D103 f1 ccRepairDel | none | 0.370 | 0.303 | +0.067 [+0.043, +0.092] | 1.1e-07 | 2.2e-07 |
| D104 f0 ccAucRepair | none | 0.403 | 0.303 | +0.100 [+0.072, +0.129] | 6.1e-11 | 2.4e-10 |
| D101 f2 ccReject | ccPost | 0.386 | 0.381 | +0.005 [-0.028, +0.039] | 0.89 | 1 |
| D103 f0 ccRepair | ccPost | 0.380 | 0.381 | -0.001 [-0.011, +0.008] | 0.86 | 1 |
| D103 f1 ccRepairDel | ccPost | 0.377 | 0.381 | -0.004 [-0.021, +0.012] | 0.62 | 1 |
| D104 f0 ccAucRepair | ccPost | 0.412 | 0.381 | +0.031 [+0.011, +0.051] | 0.00053 | 0.0021 |
