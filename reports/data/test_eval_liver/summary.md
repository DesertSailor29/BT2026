# Test set evaluation (checkpoint_final.pth)

Test cases: 111 real LiTS (RAS), of which 40 are the thesis test split. Baseline = per-case mean of D101 f0 all + D101 f1 all_rep2.
Dice NaN (both empty) excluded from mean/median and from paired tests; n = cases counted. Lesion: 26-connected tumour components, detected = any overlap; 'big' = reference lesion >= 1 ml.

## Main comparison: tumour Dice per case, all 111, vs the baseline

Clear win = Holm-adjusted Wilcoxon p < 0.05 over the 1 arms of the family (ccReject, ccRepair, ccRepairDel, ccAucRepair) and mean difference > 0.01. Difference = arm - baseline, with a paired bootstrap 95% CI (10000 resamples). Arms outside the family: exploratory (no Holm).

| run | n | arm mean | baseline mean | mean diff [95% CI] | median diff | Wilcoxon p | Holm p | clear win |
|---|---|---|---|---|---|---|---|---|
| D101 f2 ccReject | 111 | 0.382 | 0.303 | +0.0791 [+0.0383, +0.1204] | +0.0000 | 0.0011 | 0.0011 | yes |
| noise: D101 f1 all_rep2 - D101 f0 all | 111 | 0.305 | 0.301 | +0.0041 [-0.0070, +0.0148] | +0.0000 | 0.0715 | - | - |

Each arm against each baseline run separately (tumour Dice, mean diff [95% CI], Wilcoxon p):

| run | vs D101 f0 all | vs D101 f1 all_rep2 |
|---|---|---|
| D101 f2 ccReject | +0.0812 [+0.0421, +0.1201], p 0.000467 | +0.0771 [+0.0331, +0.1208], p 0.00171 |

## Secondary (exploratory, no correction): vs the baseline, all 111

| run | metric | mean diff [95% CI] | median diff | Wilcoxon p |
|---|---|---|---|---|
| D101 f2 ccReject | liver_dice | -0.0154 [-0.0255, -0.0071] | -0.0025 | 0.00782 |
| D101 f2 ccReject | liver_tumour_dice | -0.0223 [-0.0350, -0.0115] | -0.0051 | 0.00153 |
| D101 f2 ccReject | les_fp | -47.0676 [-51.2928, -42.9278] | -46.5000 | 5.98e-20 |
| D101 f2 ccReject | fp_ml | -34.9296 [-45.5315, -26.3349] | -19.6380 | 5.99e-20 |

Tumour Dice on the 40 thesis test cases (exploratory):

| run | n | mean diff [95% CI] | Wilcoxon p |
|---|---|---|---|
| D101 f2 ccReject | 40 | +0.0567 [-0.0051, +0.1229] | 0.198 |

## all 111: Dice

| run | liver mean | liver median | liver+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 0.870 | 0.911 | 0.874 | 0.301 | 0.179 | 111 (0) | 0.582 |
| D101 f1 all_rep2 | 0.876 | 0.899 | 0.882 | 0.305 | 0.205 | 111 (0) | 0.590 |
| D101 f2 ccReject | 0.857 | 0.901 | 0.856 | 0.382 | 0.426 | 111 (0) | 0.594 |

## all 111: lesions

| run | ref lesions | recall | recall >= 1 ml | pred comps | precision | FP comps / case | FP ml / case | tumour-free cases with FP |
|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 709 | 0.422 | 0.778 | 7750 | 0.043 | 66.81 | 35.01 | 10/10 |
| D101 f1 all_rep2 | 709 | 0.443 | 0.781 | 5463 | 0.060 | 46.28 | 40.93 | 10/10 |
| D101 f2 ccReject | 709 | 0.354 | 0.681 | 1337 | 0.213 | 9.48 | 3.04 | 10/10 |

## thesis test 40: Dice

| run | liver mean | liver median | liver+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 0.848 | 0.907 | 0.855 | 0.274 | 0.101 | 40 (0) | 0.524 |
| D101 f1 all_rep2 | 0.856 | 0.889 | 0.870 | 0.281 | 0.154 | 40 (0) | 0.530 |
| D101 f2 ccReject | 0.833 | 0.898 | 0.830 | 0.334 | 0.117 | 40 (0) | 0.483 |

## thesis test 40: lesions

| run | ref lesions | recall | recall >= 1 ml | pred comps | precision | FP comps / case | FP ml / case | tumour-free cases with FP |
|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 256 | 0.391 | 0.800 | 2784 | 0.039 | 66.88 | 35.23 | 4/4 |
| D101 f1 all_rep2 | 256 | 0.418 | 0.822 | 1948 | 0.057 | 45.92 | 44.86 | 4/4 |
| D101 f2 ccReject | 256 | 0.301 | 0.644 | 425 | 0.205 | 8.45 | 4.95 | 4/4 |

Lowest liver Dice (D101 f0 all): lits_043 0.00, lits_013 0.25, lits_095 0.44, lits_023 0.53, lits_121 0.56
Lowest liver Dice (D101 f1 all_rep2): lits_043 0.04, lits_013 0.43, lits_023 0.65, lits_048 0.67, lits_019 0.70
Lowest liver Dice (D101 f2 ccReject): lits_043 0.00, lits_013 0.04, lits_095 0.38, lits_130 0.56, lits_121 0.59
