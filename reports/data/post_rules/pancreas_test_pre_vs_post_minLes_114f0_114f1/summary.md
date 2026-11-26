# Pre vs post for minLes: minLesRepair (per-case mean of 114:0 / 114:1) vs the baseline, MSD07 test (238 cases)

Comparisons declared in `notes/AUTOPILOT.md` s14 before any count or training (`nnunet/pre_vs_post_pancreas.py`). Baseline = per-case mean of 102:0 / 102:1. b − a per case, bootstrap 95 % CI, Wilcoxon p. Equivalence (M1): CI within ±0.02. R rows are references, not declared. Exploratory.

| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | pancreas Dice b − a [95% CI] | lesion recall a → b |
|---|---|---|---|---|---|---|---|---|
| M1 | minLesRepair as saved vs baseline + minLes | 0.183 | 0.188 | +0.005 [-0.008, +0.017] | 0.62 | yes | -0.008 [-0.014, -0.003] | 0.450 → 0.479 |
| M2 | minLesRepair + grow0.1+minLes vs baseline + grow0.1+minLes | 0.197 | 0.200 | +0.003 [-0.009, +0.016] | 0.57 | yes | -0.011 [-0.017, -0.005] | 0.519 → 0.481 |
| M3 | minLesRepair as saved vs baseline | 0.177 | 0.188 | +0.011 [-0.001, +0.022] | 0.19 | no | -0.008 [-0.014, -0.003] | 0.535 → 0.479 |
| R1 | minLesRepair + minLes vs baseline + minLes (reference) | 0.183 | 0.187 | +0.003 [-0.009, +0.015] | 0.72 | yes | -0.008 [-0.014, -0.003] | 0.450 → 0.425 |
| R2 | minLesRepair + thr0.1+minLes vs baseline + thr0.1+minLes (reference) | 0.196 | 0.202 | +0.005 [-0.007, +0.017] | 0.49 | yes | -0.011 [-0.017, -0.005] | 0.519 → 0.485 |
