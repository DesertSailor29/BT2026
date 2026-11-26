# Pre vs post for pos / ccPost: posRepair (per-case mean of 113:0 / 113:1) vs the baseline, liver test (111 cases)

Comparisons declared in `notes/AUTOPILOT.md` s10 before this run (`nnunet/pre_vs_post_liver.py`). Baseline = per-case mean of 101:0 / 101:1. b − a per case, bootstrap 95 % CI, Wilcoxon p. Equivalence (Q1, R1): CI within ±0.02. Exploratory.

| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | liver Dice b − a [95% CI] |
|---|---|---|---|---|---|---|---|
| Q1 | posRepair as saved vs baseline + pos | 0.315 | 0.306 | -0.010 [-0.023, +0.002] | 0.15 | no | -0.016 [-0.024, -0.009] |
| Q2 | posRepair + pos vs baseline + pos | 0.315 | 0.316 | +0.001 [-0.011, +0.012] | 0.51 | yes | -0.001 [-0.004, +0.002] |
| Q3 | posRepair as saved vs baseline | 0.303 | 0.306 | +0.003 [-0.009, +0.013] | 0.13 | yes | +0.001 [-0.002, +0.004] |
| R1 | posRepair as saved vs baseline + ccPost (reference) | 0.381 | 0.306 | -0.076 [-0.098, -0.054] | 2.2e-10 | no | -0.020 [-0.029, -0.013] |
| R2 | posRepair + ccPost vs baseline + ccPost (reference) | 0.381 | 0.384 | +0.003 [-0.010, +0.015] | 0.25 | yes | -0.000 [-0.004, +0.003] |
