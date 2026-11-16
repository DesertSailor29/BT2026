# Test set evaluation (checkpoint_final.pth)

Test cases: 111 real LiTS (RAS), of which 40 are the thesis test split. Baseline = per-case mean of D101 f0 all + D101 f1 all_rep2.
Dice NaN (both empty) excluded from mean/median and from paired tests; n = cases counted. Lesion: 26-connected tumour components, detected = any overlap; 'big' = reference lesion >= 1 ml.

## Main comparison: tumour Dice per case, all 111, vs the baseline

Clear win = Holm-adjusted Wilcoxon p < 0.05 over the 4 arms of the family (ccReject, ccRepair, ccRepairDel, ccAucRepair) and mean difference > 0.01. Difference = arm - baseline, with a paired bootstrap 95% CI (10000 resamples). Arms outside the family: exploratory (no Holm).

| run | n | arm mean | baseline mean | mean diff [95% CI] | median diff | Wilcoxon p | Holm p | clear win |
|---|---|---|---|---|---|---|---|---|
| D101 f2 ccReject | 111 | 0.382 | 0.303 | +0.0791 [+0.0383, +0.1204] | +0.0000 | 0.0011 | 0.0011 | yes |
| D103 f0 ccRepair | 111 | 0.375 | 0.303 | +0.0719 [+0.0510, +0.0939] | +0.0284 | 8.14e-10 | 2.44e-09 | yes |
| D103 f1 ccRepairDel | 111 | 0.370 | 0.303 | +0.0672 [+0.0429, +0.0919] | +0.0171 | 1.08e-07 | 2.16e-07 | yes |
| D104 f0 ccAucRepair | 111 | 0.403 | 0.303 | +0.1001 [+0.0724, +0.1293] | +0.0543 | 6.08e-11 | 2.43e-10 | yes |
| D104 f1 ccAucRepair_rep2 | 111 | 0.381 | 0.303 | +0.0787 [+0.0565, +0.1014] | +0.0380 | 6.3e-11 | - | - |
| noise: D101 f1 all_rep2 - D101 f0 all | 111 | 0.305 | 0.301 | +0.0041 [-0.0070, +0.0148] | +0.0000 | 0.0715 | - | - |

Each arm against each baseline run separately (tumour Dice, mean diff [95% CI], Wilcoxon p):

| run | vs D101 f0 all | vs D101 f1 all_rep2 |
|---|---|---|
| D101 f2 ccReject | +0.0812 [+0.0421, +0.1201], p 0.000467 | +0.0771 [+0.0331, +0.1208], p 0.00171 |
| D103 f0 ccRepair | +0.0740 [+0.0546, +0.0946], p 8.13e-12 | +0.0699 [+0.0465, +0.0941], p 4.05e-08 |
| D103 f1 ccRepairDel | +0.0693 [+0.0466, +0.0926], p 3.33e-08 | +0.0652 [+0.0384, +0.0922], p 2.97e-06 |
| D104 f0 ccAucRepair | +0.1022 [+0.0754, +0.1303], p 1.12e-11 | +0.0981 [+0.0683, +0.1296], p 6.39e-09 |
| D104 f1 ccAucRepair_rep2 | +0.0807 [+0.0597, +0.1025], p 5.57e-12 | +0.0766 [+0.0521, +0.1014], p 3.78e-09 |

## Secondary (exploratory, no correction): vs the baseline, all 111

| run | metric | mean diff [95% CI] | median diff | Wilcoxon p |
|---|---|---|---|---|
| D101 f2 ccReject | liver_dice | -0.0154 [-0.0255, -0.0071] | -0.0025 | 0.00782 |
| D101 f2 ccReject | liver_tumour_dice | -0.0223 [-0.0350, -0.0115] | -0.0051 | 0.00153 |
| D101 f2 ccReject | les_fp | -47.0676 [-51.2928, -42.9278] | -46.5000 | 5.98e-20 |
| D101 f2 ccReject | fp_ml | -34.9296 [-45.5315, -26.3349] | -19.6380 | 5.99e-20 |
| D103 f0 ccRepair | liver_dice | -0.0070 [-0.0122, -0.0020] | -0.0020 | 0.00405 |
| D103 f0 ccRepair | liver_tumour_dice | -0.0067 [-0.0148, +0.0004] | +0.0004 | 0.377 |
| D103 f0 ccRepair | les_fp | -45.7252 [-50.0586, -41.4730] | -43.0000 | 5.97e-20 |
| D103 f0 ccRepair | fp_ml | -24.5621 [-34.4887, -16.9452] | -14.3160 | 1.36e-18 |
| D103 f1 ccRepairDel | liver_dice | -0.0220 [-0.0291, -0.0159] | -0.0156 | 2.36e-11 |
| D103 f1 ccRepairDel | liver_tumour_dice | -0.0220 [-0.0316, -0.0138] | -0.0084 | 2.65e-06 |
| D103 f1 ccRepairDel | les_fp | -44.5450 [-48.8018, -40.4053] | -43.0000 | 5.97e-20 |
| D103 f1 ccRepairDel | fp_ml | -23.1711 [-32.7290, -15.4774] | -12.6005 | 3.64e-14 |
| D104 f0 ccAucRepair | liver_dice | -0.0029 [-0.0060, +0.0002] | -0.0011 | 0.0235 |
| D104 f0 ccAucRepair | liver_tumour_dice | -0.0047 [-0.0099, -0.0006] | -0.0003 | 0.14 |
| D104 f0 ccAucRepair | les_fp | -47.1126 [-51.3471, -42.9955] | -47.0000 | 5.98e-20 |
| D104 f0 ccAucRepair | fp_ml | -28.7954 [-37.8916, -21.5082] | -17.0505 | 3.64e-19 |
| D104 f1 ccAucRepair_rep2 | liver_dice | +0.0002 [-0.0034, +0.0042] | +0.0007 | 0.58 |
| D104 f1 ccAucRepair_rep2 | liver_tumour_dice | +0.0003 [-0.0045, +0.0048] | +0.0023 | 0.392 |
| D104 f1 ccAucRepair_rep2 | les_fp | -44.3649 [-48.4776, -40.3693] | -42.5000 | 5.97e-20 |
| D104 f1 ccAucRepair_rep2 | fp_ml | -24.0234 [-31.6066, -17.7668] | -13.6815 | 1.26e-17 |

Tumour Dice on the 40 thesis test cases (exploratory):

| run | n | mean diff [95% CI] | Wilcoxon p |
|---|---|---|---|
| D101 f2 ccReject | 40 | +0.0567 [-0.0051, +0.1229] | 0.198 |
| D103 f0 ccRepair | 40 | +0.0554 [+0.0223, +0.0923] | 0.00567 |
| D103 f1 ccRepairDel | 40 | +0.0666 [+0.0278, +0.1100] | 0.00316 |
| D104 f0 ccAucRepair | 40 | +0.0993 [+0.0524, +0.1533] | 0.000168 |
| D104 f1 ccAucRepair_rep2 | 40 | +0.0730 [+0.0356, +0.1162] | 0.00042 |

## all 111: Dice

| run | liver mean | liver median | liver+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 0.870 | 0.911 | 0.874 | 0.301 | 0.179 | 111 (0) | 0.582 |
| D101 f1 all_rep2 | 0.876 | 0.899 | 0.882 | 0.305 | 0.205 | 111 (0) | 0.590 |
| D101 f2 ccReject | 0.857 | 0.901 | 0.856 | 0.382 | 0.426 | 111 (0) | 0.594 |
| D103 f0 ccRepair | 0.866 | 0.911 | 0.871 | 0.375 | 0.324 | 111 (0) | 0.651 |
| D103 f1 ccRepairDel | 0.851 | 0.890 | 0.856 | 0.370 | 0.323 | 111 (0) | 0.651 |
| D104 f0 ccAucRepair | 0.870 | 0.904 | 0.873 | 0.403 | 0.405 | 111 (0) | 0.675 |
| D104 f1 ccAucRepair_rep2 | 0.873 | 0.909 | 0.878 | 0.381 | 0.353 | 111 (0) | 0.663 |

## all 111: lesions

| run | ref lesions | recall | recall >= 1 ml | pred comps | precision | FP comps / case | FP ml / case | tumour-free cases with FP |
|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 709 | 0.422 | 0.778 | 7750 | 0.043 | 66.81 | 35.01 | 10/10 |
| D101 f1 all_rep2 | 709 | 0.443 | 0.781 | 5463 | 0.060 | 46.28 | 40.93 | 10/10 |
| D101 f2 ccReject | 709 | 0.354 | 0.681 | 1337 | 0.213 | 9.48 | 3.04 | 10/10 |
| D103 f0 ccRepair | 709 | 0.416 | 0.771 | 1510 | 0.205 | 10.82 | 13.41 | 10/10 |
| D103 f1 ccRepairDel | 709 | 0.403 | 0.753 | 1631 | 0.183 | 12.00 | 14.80 | 10/10 |
| D104 f0 ccAucRepair | 709 | 0.413 | 0.753 | 1352 | 0.226 | 9.43 | 9.17 | 10/10 |
| D104 f1 ccAucRepair_rep2 | 709 | 0.413 | 0.764 | 1659 | 0.185 | 12.18 | 13.95 | 10/10 |

## thesis test 40: Dice

| run | liver mean | liver median | liver+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 0.848 | 0.907 | 0.855 | 0.274 | 0.101 | 40 (0) | 0.524 |
| D101 f1 all_rep2 | 0.856 | 0.889 | 0.870 | 0.281 | 0.154 | 40 (0) | 0.530 |
| D101 f2 ccReject | 0.833 | 0.898 | 0.830 | 0.334 | 0.117 | 40 (0) | 0.483 |
| D103 f0 ccRepair | 0.845 | 0.901 | 0.851 | 0.333 | 0.204 | 40 (0) | 0.568 |
| D103 f1 ccRepairDel | 0.830 | 0.887 | 0.835 | 0.344 | 0.223 | 40 (0) | 0.584 |
| D104 f0 ccAucRepair | 0.848 | 0.894 | 0.853 | 0.377 | 0.326 | 40 (0) | 0.595 |
| D104 f1 ccAucRepair_rep2 | 0.852 | 0.906 | 0.859 | 0.350 | 0.154 | 40 (0) | 0.586 |

## thesis test 40: lesions

| run | ref lesions | recall | recall >= 1 ml | pred comps | precision | FP comps / case | FP ml / case | tumour-free cases with FP |
|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 256 | 0.391 | 0.800 | 2784 | 0.039 | 66.88 | 35.23 | 4/4 |
| D101 f1 all_rep2 | 256 | 0.418 | 0.822 | 1948 | 0.057 | 45.92 | 44.86 | 4/4 |
| D101 f2 ccReject | 256 | 0.301 | 0.644 | 425 | 0.205 | 8.45 | 4.95 | 4/4 |
| D103 f0 ccRepair | 256 | 0.367 | 0.789 | 492 | 0.193 | 9.93 | 16.21 | 4/4 |
| D103 f1 ccRepairDel | 256 | 0.375 | 0.778 | 541 | 0.177 | 11.12 | 13.25 | 4/4 |
| D104 f0 ccAucRepair | 256 | 0.371 | 0.767 | 449 | 0.212 | 8.85 | 12.92 | 4/4 |
| D104 f1 ccAucRepair_rep2 | 256 | 0.367 | 0.778 | 544 | 0.175 | 11.22 | 16.88 | 4/4 |

Lowest liver Dice (D101 f0 all): lits_043 0.00, lits_013 0.25, lits_095 0.44, lits_023 0.53, lits_121 0.56
Lowest liver Dice (D101 f1 all_rep2): lits_043 0.04, lits_013 0.43, lits_023 0.65, lits_048 0.67, lits_019 0.70
Lowest liver Dice (D101 f2 ccReject): lits_043 0.00, lits_013 0.04, lits_095 0.38, lits_130 0.56, lits_121 0.59
Lowest liver Dice (D103 f0 ccRepair): lits_043 0.00, lits_013 0.26, lits_095 0.50, lits_121 0.57, lits_023 0.63
Lowest liver Dice (D103 f1 ccRepairDel): lits_043 0.00, lits_013 0.25, lits_095 0.39, lits_121 0.54, lits_023 0.58
Lowest liver Dice (D104 f0 ccAucRepair): lits_043 0.00, lits_013 0.33, lits_023 0.64, lits_095 0.64, lits_048 0.65
Lowest liver Dice (D104 f1 ccAucRepair_rep2): lits_043 0.01, lits_013 0.32, lits_095 0.59, lits_121 0.64, lits_023 0.66
