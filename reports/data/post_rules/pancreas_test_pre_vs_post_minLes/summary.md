# Pre vs post for minLes: minLesRepair (D114 f0) vs the baseline, MSD07 test (238 cases)

Comparisons declared in `notes/AUTOPILOT.md` s14 before any count or training (`nnunet/pre_vs_post_pancreas.py`). Baseline = per-case mean of 102:0 / 102:1. b − a per case, bootstrap 95 % CI, Wilcoxon p. Equivalence (M1): CI within ±0.02. R rows are references, not declared. Exploratory.

| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | pancreas Dice b − a [95% CI] | lesion recall a → b |
|---|---|---|---|---|---|---|---|---|
| M1 | minLesRepair as saved vs baseline + minLes | 0.183 | 0.169 | -0.014 [-0.028, -0.000] | 0.039 | no | -0.010 [-0.016, -0.004] | 0.450 → 0.429 |
| M2 | minLesRepair + grow0.1+minLes vs baseline + grow0.1+minLes | 0.197 | 0.182 | -0.015 [-0.031, -0.000] | 0.075 | no | -0.014 [-0.021, -0.007] | 0.519 → 0.425 |
| M3 | minLesRepair as saved vs baseline | 0.177 | 0.169 | -0.008 [-0.022, +0.006] | 0.35 | no | -0.010 [-0.016, -0.004] | 0.535 → 0.429 |
| R1 | minLesRepair + minLes vs baseline + minLes (reference) | 0.183 | 0.167 | -0.016 [-0.031, -0.003] | 0.013 | no | -0.010 [-0.016, -0.004] | 0.450 → 0.379 |
| R2 | minLesRepair + thr0.1+minLes vs baseline + thr0.1+minLes (reference) | 0.196 | 0.185 | -0.011 [-0.026, +0.002] | 0.14 | no | -0.013 [-0.020, -0.006] | 0.519 → 0.433 |
