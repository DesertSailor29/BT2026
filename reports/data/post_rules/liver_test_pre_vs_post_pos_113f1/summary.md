# Pre vs post for pos / ccPost: posRepair (D113 f1) vs the baseline, liver test (111 cases)

Comparisons declared in `notes/AUTOPILOT.md` s10 before this run (`nnunet/pre_vs_post_liver.py`). Baseline = per-case mean of 101:0 / 101:1. b − a per case, bootstrap 95 % CI, Wilcoxon p. Equivalence (Q1, R1): CI within ±0.02. Exploratory.

| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | liver Dice b − a [95% CI] |
|---|---|---|---|---|---|---|---|
| Q1 | posRepair as saved vs baseline + pos | 0.315 | 0.320 | +0.004 [-0.010, +0.018] | 0.18 | yes | -0.006 [-0.015, +0.002] |
| Q2 | posRepair + pos vs baseline + pos | 0.315 | 0.331 | +0.016 [+0.003, +0.029] | 0.0019 | no | +0.011 [+0.007, +0.016] |
| Q3 | posRepair as saved vs baseline | 0.303 | 0.320 | +0.017 [+0.004, +0.030] | 0.00065 | no | +0.010 [+0.007, +0.015] |
| R1 | posRepair as saved vs baseline + ccPost (reference) | 0.381 | 0.320 | -0.061 [-0.082, -0.042] | 7.8e-09 | no | -0.011 [-0.020, -0.002] |
| R2 | posRepair + ccPost vs baseline + ccPost (reference) | 0.381 | 0.390 | +0.009 [-0.005, +0.024] | 0.13 | no | +0.012 [+0.008, +0.016] |
