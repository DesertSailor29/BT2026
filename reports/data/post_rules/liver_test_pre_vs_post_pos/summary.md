# Pre vs post for pos / ccPost: posRepair (D113 f0) vs the baseline, liver test (111 cases)

Comparisons declared in `notes/AUTOPILOT.md` s10 before this run (`nnunet/pre_vs_post_liver.py`). Baseline = per-case mean of 101:0 / 101:1. b − a per case, bootstrap 95 % CI, Wilcoxon p. Equivalence (Q1, R1): CI within ±0.02. Exploratory.

| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | liver Dice b − a [95% CI] |
|---|---|---|---|---|---|---|---|
| Q1 | posRepair as saved vs baseline + pos | 0.315 | 0.291 | -0.024 [-0.039, -0.010] | 0.00013 | no | -0.025 [-0.035, -0.017] |
| Q2 | posRepair + pos vs baseline + pos | 0.315 | 0.301 | -0.014 [-0.027, -0.002] | 0.0082 | no | -0.013 [-0.020, -0.008] |
| Q3 | posRepair as saved vs baseline | 0.303 | 0.291 | -0.011 [-0.024, +0.001] | 0.059 | no | -0.009 [-0.016, -0.003] |
| R1 | posRepair as saved vs baseline + ccPost (reference) | 0.381 | 0.291 | -0.090 [-0.115, -0.065] | 1.9e-11 | no | -0.030 [-0.040, -0.022] |
| R2 | posRepair + ccPost vs baseline + ccPost (reference) | 0.381 | 0.378 | -0.003 [-0.018, +0.011] | 0.93 | yes | -0.012 [-0.019, -0.007] |
