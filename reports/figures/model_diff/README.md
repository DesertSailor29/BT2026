# Where the liver runs differ: CT | reference | every run

Written by `nnunet/model_diff_figures.py` from the checkpoint_final test predictions as saved (no post-processing) of the 7 liver runs and `reports/data/test_eval_liver/fp_location/per_case.csv`. Cases picked by fixed rules (largest tumour Dice range over the runs; largest difference in each direction for fixed arm pairs incl. the two noise pairs; the tumour-free case with the largest FP range), not by eye. Slice = where the runs disagree most (some but not all runs predict tumour, or a reference lesion voxel missed by at least one run). Top row full slice, bottom row zoomed on the disagreement. Style: W400/L40, radiological (patient right = image left). Green organ, red tumour.

| fig | case | slice | rule | why | all | all_rep2 | ccReject | ccRepair | ccRepairDel | ccAucRepair | ccAucRepair_rep2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [M01](M01_spread_lits_088.png) | lits_088 | 447 | spread | tumour Dice range 0.73 (ccReject 0.03 → ccRepair 0.76) | 0.54 | 0.71 | 0.03 | 0.76 | 0.69 | 0.63 | 0.73 |
| [M02](M02_spread_lits_069.png) | lits_069 | 212 | spread | tumour Dice range 0.73 (ccReject 0.00 → ccAucRepair 0.73) | 0.22 | 0.21 | 0.00 | 0.26 | 0.53 | 0.73 | 0.53 |
| [M03](M03_spread_lits_014.png) | lits_014 | 123 | spread | tumour Dice range 0.63 (all 0.02 → ccReject 0.65) | 0.02 | 0.04 | 0.65 | 0.18 | 0.33 | 0.45 | 0.35 |
| [M04](M04_spread_lits_068.png) | lits_068 | 45 | spread | tumour Dice range 0.62 (all_rep2 0.21 → ccReject 0.83) | 0.47 | 0.21 | 0.83 | 0.78 | 0.82 | 0.78 | 0.76 |
| [M05](M05_spread_lits_075.png) | lits_075 | 76 | spread | tumour Dice range 0.62 (all_rep2 0.17 → ccReject 0.79) | 0.24 | 0.17 | 0.79 | 0.60 | 0.67 | 0.77 | 0.72 |
| [M06](M06_spread_lits_061.png) | lits_061 | 167 | spread | tumour Dice range 0.58 (all 0.06 → ccReject 0.65) | 0.06 | 0.09 | 0.65 | 0.09 | 0.08 | 0.56 | 0.15 |
| [M07](M07_contrast_lits_003.png) | lits_003 | 122 | contrast | ccAucRepair − baseline = +0.50 (largest where ccAucRepair is better, case not shown yet) | 0.11 | 0.10 | 0.59 | 0.37 | 0.44 | 0.60 | 0.41 |
| [M08](M08_contrast_lits_104.png) | lits_104 | 355 | contrast | ccAucRepair − baseline = -0.38 (largest where ccAucRepair is worse, case not shown yet) | 0.36 | 0.51 | 0.01 | 0.38 | 0.05 | 0.05 | 0.30 |
| [M09](M09_contrast_lits_102.png) | lits_102 | 584 | contrast | ccReject − ccRepair = +0.45 (largest where ccReject is better, case not shown yet) | 0.29 | 0.25 | 0.78 | 0.32 | 0.38 | 0.53 | 0.36 |
| [M10](M10_contrast_lits_073.png) | lits_073 | 104 | contrast | ccReject − ccRepair = -0.31 (largest where ccReject is worse, case not shown yet) | 0.11 | 0.15 | 0.00 | 0.31 | 0.24 | 0.30 | 0.27 |
| [M11](M11_contrast_lits_082.png) | lits_082 | 400 | contrast | ccRepair − ccRepairDel = +0.23 (largest where ccRepair is better, case not shown yet) | 0.51 | 0.51 | 0.90 | 0.86 | 0.63 | 0.83 | 0.80 |
| [M12](M12_contrast_lits_065.png) | lits_065 | 444 | contrast | ccRepair − ccRepairDel = -0.21 (largest where ccRepair is worse, case not shown yet) | 0.05 | 0.07 | 0.32 | 0.11 | 0.32 | 0.22 | 0.17 |
| [M13](M13_contrast_lits_062.png) | lits_062 | 156 | contrast | ccAucRepair − ccRepair = +0.38 (largest where ccAucRepair is better, case not shown yet) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.38 | 0.12 |
| [M14](M14_contrast_lits_010.png) | lits_010 | 375 | contrast | ccAucRepair − ccRepair = -0.22 (largest where ccAucRepair is worse, case not shown yet) | 0.58 | 0.40 | 0.73 | 0.78 | 0.81 | 0.56 | 0.79 |
| [M15](M15_contrast_lits_076.png) | lits_076 | 146 | contrast | ccAucRepair_rep2 − ccAucRepair = +0.12 (largest where ccAucRepair_rep2 is better, case not shown yet) | 0.76 | 0.77 | 0.59 | 0.77 | 0.65 | 0.70 | 0.82 |
| [M16](M16_contrast_lits_111.png) | lits_111 | 525 | contrast | ccAucRepair_rep2 − ccAucRepair = -0.18 (largest where ccAucRepair_rep2 is worse, case not shown yet) | 0.06 | 0.07 | 0.46 | 0.22 | 0.32 | 0.36 | 0.18 |
| [M17](M17_contrast_lits_056.png) | lits_056 | 195 | contrast | all_rep2 − all = +0.15 (largest where all_rep2 is better, case not shown yet) | 0.45 | 0.59 | 0.18 | 0.47 | 0.54 | 0.40 | 0.48 |
| [M18](M18_contrast_lits_072.png) | lits_072 | 70 | contrast | all_rep2 − all = -0.15 (largest where all_rep2 is worse, case not shown yet) | 0.58 | 0.42 | 0.79 | 0.65 | 0.46 | 0.59 | 0.44 |
| [M19](M19_fp_lits_034.png) | lits_034 | 80 | fp | tumour-free case, FP ml range 16.5–440.6 over the runs | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
