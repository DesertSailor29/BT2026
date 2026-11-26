# Pre vs post for minLes: minLesRepair (D114 f1) vs the baseline, MSD07 test (238 cases)

Comparisons declared in `notes/AUTOPILOT.md` s14 before any count or training (`nnunet/pre_vs_post_pancreas.py`). Baseline = per-case mean of 102:0 / 102:1. b − a per case, bootstrap 95 % CI, Wilcoxon p. Equivalence (M1): CI within ±0.02. R rows are references, not declared. Exploratory.

| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | pancreas Dice b − a [95% CI] | lesion recall a → b |
|---|---|---|---|---|---|---|---|---|
| M1 | minLesRepair as saved vs baseline + minLes | 0.183 | 0.207 | +0.023 [+0.008, +0.039] | 0.026 | no | -0.007 [-0.014, -0.000] | 0.450 → 0.529 |
| M2 | minLesRepair + grow0.1+minLes vs baseline + grow0.1+minLes | 0.197 | 0.219 | +0.022 [+0.008, +0.036] | 0.0068 | no | -0.007 [-0.015, -0.000] | 0.519 → 0.537 |
| M3 | minLesRepair as saved vs baseline | 0.177 | 0.207 | +0.029 [+0.015, +0.044] | 2.7e-05 | no | -0.007 [-0.014, -0.000] | 0.535 → 0.529 |
| R1 | minLesRepair + minLes vs baseline + minLes (reference) | 0.183 | 0.207 | +0.023 [+0.007, +0.040] | 0.024 | no | -0.007 [-0.014, -0.000] | 0.450 → 0.471 |
| R2 | minLesRepair + thr0.1+minLes vs baseline + thr0.1+minLes (reference) | 0.196 | 0.218 | +0.022 [+0.007, +0.036] | 0.0067 | no | -0.008 [-0.016, -0.001] | 0.519 → 0.537 |
