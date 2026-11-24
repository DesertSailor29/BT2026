# Pre vs post for tumTouch / ccPost: ccRepair (D103 f0) vs the baseline, liver test (111 cases)

Comparisons declared in `notes/AUTOPILOT.md` s10 before this run (`nnunet/pre_vs_post_liver.py`). Baseline = per-case mean of 101:0 / 101:1. b − a per case, bootstrap 95 % CI, Wilcoxon p. Equivalence (P1, R1): CI within ±0.02. Exploratory.

| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | liver Dice b − a [95% CI] |
|---|---|---|---|---|---|---|---|
| P1 | ccRepair as saved vs baseline + tumTouch | 0.365 | 0.375 | +0.009 [-0.002, +0.021] | 0.14 | no | -0.007 [-0.012, -0.002] |
| P2 | ccRepair + tumTouch vs baseline + tumTouch | 0.365 | 0.377 | +0.012 [+0.000, +0.023] | 0.086 | no | -0.007 [-0.012, -0.002] |
| P3 | ccRepair as saved vs baseline | 0.303 | 0.375 | +0.072 [+0.051, +0.094] | 8.1e-10 | no | -0.007 [-0.012, -0.002] |
| R1 | ccRepair as saved vs baseline + ccPost | 0.381 | 0.375 | -0.006 [-0.017, +0.003] | 0.65 | yes | -0.028 [-0.037, -0.020] |
| R2 | ccRepair + ccPost vs baseline + ccPost | 0.381 | 0.380 | -0.001 [-0.011, +0.008] | 0.86 | yes | -0.011 [-0.016, -0.006] |
