# Test set evaluation (checkpoint_best.pth)

Test cases: 111 real LiTS (RAS), of which 40 are the thesis test split. Baseline = per-case mean of D101 f0 all + D101 f1 all_rep2.
Dice NaN (both empty) excluded from mean/median and from paired tests; n = cases counted. Lesion: 26-connected tumour components, detected = any overlap; 'big' = reference lesion >= 1 ml.

## Main comparison: tumour Dice per case, all 111, vs the baseline

Clear win = Holm-adjusted Wilcoxon p < 0.05 over the 4 arms of the family (ccReject, ccRepair, ccRepairDel, ccAucRepair) and mean difference > 0.01. Difference = arm - baseline, with a paired bootstrap 95% CI (10000 resamples). Arms outside the family: exploratory (no Holm).

| run | n | arm mean | baseline mean | mean diff [95% CI] | median diff | Wilcoxon p | Holm p | clear win |
|---|---|---|---|---|---|---|---|---|
| D101 f2 ccReject | 111 | 0.333 | 0.315 | +0.0180 [-0.0143, +0.0507] | +0.0000 | 0.569 | 0.569 | no |
| D103 f0 ccRepair | 111 | 0.330 | 0.315 | +0.0151 [-0.0014, +0.0323] | +0.0000 | 0.205 | 0.411 | no |
| D103 f1 ccRepairDel | 111 | 0.349 | 0.315 | +0.0339 [+0.0147, +0.0540] | +0.0043 | 0.000306 | 0.000917 | yes |
| D104 f0 ccAucRepair | 111 | 0.378 | 0.315 | +0.0626 [+0.0415, +0.0852] | +0.0180 | 3.12e-07 | 1.25e-06 | yes |
| D104 f1 ccAucRepair_rep2 | 111 | 0.381 | 0.315 | +0.0656 [+0.0464, +0.0857] | +0.0209 | 2.63e-10 | - | - |
| noise: D101 f1 all_rep2 - D101 f0 all | 111 | 0.319 | 0.312 | +0.0067 [-0.0063, +0.0207] | +0.0000 | 0.419 | - | - |

Each arm against each baseline run separately (tumour Dice, mean diff [95% CI], Wilcoxon p):

| run | vs D101 f0 all | vs D101 f1 all_rep2 |
|---|---|---|
| D101 f2 ccReject | +0.0213 [-0.0102, +0.0531], p 0.564 | +0.0146 [-0.0198, +0.0482], p 0.552 |
| D103 f0 ccRepair | +0.0185 [-0.0000, +0.0373], p 0.164 | +0.0117 [-0.0055, +0.0297], p 0.286 |
| D103 f1 ccRepairDel | +0.0373 [+0.0155, +0.0594], p 0.000584 | +0.0306 [+0.0118, +0.0502], p 0.00112 |
| D104 f0 ccAucRepair | +0.0660 [+0.0427, +0.0904], p 4.76e-07 | +0.0592 [+0.0381, +0.0820], p 5.22e-07 |
| D104 f1 ccAucRepair_rep2 | +0.0690 [+0.0482, +0.0904], p 3.24e-10 | +0.0622 [+0.0426, +0.0833], p 2.67e-09 |

## Secondary (exploratory, no correction): vs the baseline, all 111

| run | metric | mean diff [95% CI] | median diff | Wilcoxon p |
|---|---|---|---|---|
| D101 f2 ccReject | liver_dice | +0.0169 [+0.0055, +0.0301] | +0.0026 | 0.0718 |
| D101 f2 ccReject | liver_tumour_dice | +0.0100 [+0.0014, +0.0196] | +0.0026 | 0.169 |
| D101 f2 ccReject | les_fp | -26.7658 [-29.9550, -23.6081] | -26.5000 | 2.89e-19 |
| D101 f2 ccReject | fp_ml | -24.9325 [-33.1643, -18.1767] | -11.3285 | 6.15e-20 |
| D103 f0 ccRepair | liver_dice | -0.0308 [-0.0403, -0.0226] | -0.0178 | 1.27e-14 |
| D103 f0 ccRepair | liver_tumour_dice | -0.0322 [-0.0444, -0.0217] | -0.0143 | 2.66e-11 |
| D103 f0 ccRepair | les_fp | -33.9189 [-37.1171, -30.7792] | -32.5000 | 6.13e-20 |
| D103 f0 ccRepair | fp_ml | -16.0659 [-26.6282, -7.9203] | -4.5080 | 1.55e-07 |
| D103 f1 ccRepairDel | liver_dice | -0.0061 [-0.0123, -0.0007] | -0.0002 | 0.349 |
| D103 f1 ccRepairDel | liver_tumour_dice | -0.0086 [-0.0183, -0.0006] | +0.0024 | 0.674 |
| D103 f1 ccRepairDel | les_fp | -30.3153 [-33.5315, -27.0811] | -30.0000 | 1.33e-19 |
| D103 f1 ccRepairDel | fp_ml | -14.9117 [-24.6456, -7.7641] | -5.3765 | 1.04e-10 |
| D104 f0 ccAucRepair | liver_dice | -0.0052 [-0.0148, +0.0049] | -0.0126 | 0.0108 |
| D104 f0 ccAucRepair | liver_tumour_dice | -0.0077 [-0.0154, +0.0006] | -0.0111 | 0.00393 |
| D104 f0 ccAucRepair | les_fp | -29.8468 [-32.9685, -26.7297] | -29.0000 | 1.26e-19 |
| D104 f0 ccAucRepair | fp_ml | -23.0883 [-32.9746, -15.1360] | -7.6615 | 4.52e-18 |
| D104 f1 ccAucRepair_rep2 | liver_dice | -0.0058 [-0.0109, -0.0009] | -0.0023 | 0.00342 |
| D104 f1 ccAucRepair_rep2 | liver_tumour_dice | -0.0067 [-0.0135, -0.0005] | -0.0006 | 0.0852 |
| D104 f1 ccAucRepair_rep2 | les_fp | -37.9369 [-41.3200, -34.5676] | -36.5000 | 6.13e-20 |
| D104 f1 ccAucRepair_rep2 | fp_ml | -18.2054 [-25.6673, -12.1164] | -7.4460 | 1.51e-16 |

Tumour Dice on the 40 thesis test cases (exploratory):

| run | n | mean diff [95% CI] | Wilcoxon p |
|---|---|---|---|
| D101 f2 ccReject | 40 | +0.0133 [-0.0351, +0.0609] | 0.638 |
| D103 f0 ccRepair | 40 | +0.0085 [-0.0194, +0.0381] | 0.829 |
| D103 f1 ccRepairDel | 40 | +0.0259 [-0.0056, +0.0581] | 0.0859 |
| D104 f0 ccAucRepair | 40 | +0.0656 [+0.0286, +0.1040] | 0.00477 |
| D104 f1 ccAucRepair_rep2 | 40 | +0.0567 [+0.0254, +0.0932] | 0.00272 |

## all 111: Dice

| run | liver mean | liver median | liver+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 0.882 | 0.921 | 0.890 | 0.312 | 0.221 | 111 (0) | 0.573 |
| D101 f1 all_rep2 | 0.875 | 0.899 | 0.880 | 0.319 | 0.212 | 111 (0) | 0.612 |
| D101 f2 ccReject | 0.896 | 0.916 | 0.895 | 0.333 | 0.283 | 111 (0) | 0.558 |
| D103 f0 ccRepair | 0.848 | 0.883 | 0.853 | 0.330 | 0.255 | 111 (0) | 0.615 |
| D103 f1 ccRepairDel | 0.873 | 0.915 | 0.876 | 0.349 | 0.251 | 111 (0) | 0.653 |
| D104 f0 ccAucRepair | 0.873 | 0.887 | 0.877 | 0.378 | 0.338 | 111 (0) | 0.652 |
| D104 f1 ccAucRepair_rep2 | 0.873 | 0.908 | 0.878 | 0.381 | 0.358 | 111 (0) | 0.662 |

## all 111: lesions

| run | ref lesions | recall | recall >= 1 ml | pred comps | precision | FP comps / case | FP ml / case | tumour-free cases with FP |
|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 709 | 0.468 | 0.812 | 4815 | 0.079 | 39.95 | 36.20 | 10/10 |
| D101 f1 all_rep2 | 709 | 0.457 | 0.788 | 7109 | 0.049 | 60.90 | 28.64 | 10/10 |
| D101 f2 ccReject | 709 | 0.329 | 0.698 | 2902 | 0.095 | 23.66 | 7.49 | 10/10 |
| D103 f0 ccRepair | 709 | 0.389 | 0.757 | 2133 | 0.141 | 16.50 | 16.35 | 10/10 |
| D103 f1 ccRepairDel | 709 | 0.395 | 0.733 | 2523 | 0.115 | 20.11 | 17.51 | 10/10 |
| D104 f0 ccAucRepair | 709 | 0.420 | 0.788 | 2597 | 0.121 | 20.58 | 9.33 | 10/10 |
| D104 f1 ccAucRepair_rep2 | 709 | 0.415 | 0.767 | 1696 | 0.183 | 12.49 | 14.21 | 10/10 |

## thesis test 40: Dice

| run | liver mean | liver median | liver+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 0.861 | 0.920 | 0.873 | 0.295 | 0.180 | 40 (0) | 0.517 |
| D101 f1 all_rep2 | 0.855 | 0.890 | 0.866 | 0.293 | 0.127 | 40 (0) | 0.543 |
| D101 f2 ccReject | 0.882 | 0.897 | 0.882 | 0.307 | 0.304 | 40 (0) | 0.453 |
| D103 f0 ccRepair | 0.827 | 0.879 | 0.831 | 0.302 | 0.142 | 40 (0) | 0.529 |
| D103 f1 ccRepairDel | 0.852 | 0.916 | 0.856 | 0.320 | 0.120 | 40 (0) | 0.580 |
| D104 f0 ccAucRepair | 0.861 | 0.893 | 0.864 | 0.360 | 0.305 | 40 (0) | 0.556 |
| D104 f1 ccAucRepair_rep2 | 0.852 | 0.906 | 0.859 | 0.351 | 0.166 | 40 (0) | 0.586 |

## thesis test 40: lesions

| run | ref lesions | recall | recall >= 1 ml | pred comps | precision | FP comps / case | FP ml / case | tumour-free cases with FP |
|---|---|---|---|---|---|---|---|---|
| D101 f0 all | 256 | 0.461 | 0.856 | 1698 | 0.079 | 39.10 | 39.54 | 4/4 |
| D101 f1 all_rep2 | 256 | 0.438 | 0.822 | 2533 | 0.047 | 60.35 | 32.63 | 4/4 |
| D101 f2 ccReject | 256 | 0.312 | 0.722 | 1003 | 0.095 | 22.70 | 10.40 | 4/4 |
| D103 f0 ccRepair | 256 | 0.344 | 0.778 | 736 | 0.137 | 15.88 | 16.94 | 4/4 |
| D103 f1 ccRepairDel | 256 | 0.348 | 0.733 | 861 | 0.107 | 19.23 | 17.33 | 4/4 |
| D104 f0 ccAucRepair | 256 | 0.398 | 0.822 | 905 | 0.123 | 19.85 | 10.39 | 4/4 |
| D104 f1 ccAucRepair_rep2 | 256 | 0.371 | 0.789 | 558 | 0.174 | 11.53 | 17.08 | 4/4 |

Lowest liver Dice (D101 f0 all): lits_043 0.01, lits_013 0.37, lits_023 0.58, lits_095 0.61, lits_121 0.68
Lowest liver Dice (D101 f1 all_rep2): lits_043 0.04, lits_013 0.43, lits_023 0.66, lits_048 0.67, lits_054 0.70
Lowest liver Dice (D101 f2 ccReject): lits_043 0.41, lits_048 0.70, lits_013 0.75, lits_085 0.75, lits_023 0.75
Lowest liver Dice (D103 f0 ccRepair): lits_043 0.00, lits_013 0.11, lits_095 0.47, lits_121 0.54, lits_019 0.56
Lowest liver Dice (D103 f1 ccRepairDel): lits_043 0.00, lits_013 0.30, lits_095 0.55, lits_121 0.60, lits_023 0.63
Lowest liver Dice (D104 f0 ccAucRepair): lits_043 0.24, lits_013 0.56, lits_023 0.69, lits_048 0.70, lits_011 0.75
Lowest liver Dice (D104 f1 ccAucRepair_rep2): lits_043 0.01, lits_013 0.32, lits_095 0.59, lits_121 0.64, lits_023 0.66
