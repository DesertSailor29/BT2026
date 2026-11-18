# Test set evaluation (checkpoint_final.pth)

Test cases: 238 real MSD07 (RAS). Baseline = per-case mean of D102 f0 all + D102 f1 all_rep2.
Dice NaN (both empty) excluded from mean/median and from paired tests; n = cases counted. Lesion: 26-connected tumour components, detected = any overlap; 'big' = reference lesion >= 1 ml.

## Main comparison: tumour Dice per case, all 238, vs the baseline

Clear win = Holm-adjusted Wilcoxon p < 0.05 over the 0 arms of the family (ccReject, ccRepair, ccRepairDel, ccAucRepair) and mean difference > 0.01. Difference = arm - baseline, with a paired bootstrap 95% CI (10000 resamples). Arms outside the family: exploratory (no Holm).

| run | n | arm mean | baseline mean | mean diff [95% CI] | median diff | Wilcoxon p | Holm p | clear win |
|---|---|---|---|---|---|---|---|---|
| noise: D102 f1 all_rep2 - D102 f0 all | 238 | 0.172 | 0.183 | -0.0105 [-0.0239, +0.0017] | +0.0000 | 0.843 | - | - |

## all 238: Dice

| run | pancreas mean | pancreas median | pancreas+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |
|---|---|---|---|---|---|---|---|
| D102 f0 all | 0.209 | 0.126 | 0.281 | 0.183 | 0.016 | 238 (0) | 0.178 |
| D102 f1 all_rep2 | 0.173 | 0.087 | 0.250 | 0.172 | 0.002 | 238 (0) | 0.169 |

## all 238: lesions

| run | ref lesions | recall | recall >= 1 ml | pred comps | precision | FP comps / case | FP ml / case | tumour-free cases with FP |
|---|---|---|---|---|---|---|---|---|
| D102 f0 all | 240 | 0.550 | 0.561 | 4181 | 0.035 | 16.95 | 5.21 | 0/0 |
| D102 f1 all_rep2 | 240 | 0.521 | 0.530 | 1403 | 0.096 | 5.33 | 4.19 | 0/0 |

Lowest pancreas Dice (D102 f0 all): msd_005 0.00, msd_018 0.00, msd_050 0.00, msd_055 0.00, msd_069 0.00
Lowest pancreas Dice (D102 f1 all_rep2): msd_005 0.00, msd_018 0.00, msd_037 0.00, msd_041 0.00, msd_043 0.00
