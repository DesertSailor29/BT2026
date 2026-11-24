# The AUC rule applied to the predictions (ccAucPost) + a sweep of stricter cuts

43 real MSD07 val cases, checkpoint_final predictions. ccPost = training cc rule on the prediction; ccAucPost<cut> = ccPost, then predicted tumour pieces inside the predicted main structure with sep < cut -> their surroundings' label (filter_metrics' separability with the predicted pancreas as organ). Decisions use image + prediction only. **Main cut 0.509** (= the training cut, fixed before results); 0.55 / 0.6 / 0.637 / 0.7 = exploratory sweep. Written by `nnunet/auc_post.py`; exploratory, the main analysis stays predict_test.py.

## 1. Does sep tell false-positive pieces from hits? (pieces inside the predicted main structure)

| run | pieces / case (hit, FP) | unscored | sep median hit / FP | AUROC sep (FP lower) | < 0.509: FP / hit pieces (hit ml) | < 0.55: FP / hit pieces (hit ml) | < 0.6: FP / hit pieces (hit ml) | < 0.637: FP / hit pieces (hit ml) | < 0.7: FP / hit pieces (hit ml) |
|---|---|---|---|---|---|---|---|---|---|
| D102 f0 all | 0.9 (0.3, 0.6) | 2 | 0.680 / 0.570 | 0.75 | 1 / 0 (0.0) | 6 / 1 (11.8) | 15 / 4 (48.4) | 17 / 5 (49.0) | 19 / 8 (73.5) |
| D102 f1 all_rep2 | 0.7 (0.3, 0.4) | 0 | 0.664 / 0.579 | 0.58 | 0 / 0 (0.0) | 5 / 1 (13.0) | 9 / 3 (40.3) | 10 / 4 (50.2) | 12 / 7 (59.5) |

## 2. Each variant vs ccPost, per run

Tumour Dice mean (NaN = both empty, left out) and the paired difference to ccPost; lesion recall, FP pieces and reference lesions lost are totals over the cases.

| run | variant | tumour Dice | vs ccPost [95% CI] | p | pancreas Dice | lesion recall | FP pieces / case | lesions lost vs ccPost | pieces relabelled |
|---|---|---|---|---|---|---|---|---|---|
| D102 f0 all | none | 0.141 | -0.005 [-0.027, +0.028] | 0.04 | 0.162 | 0.465 | 17.2 | – | – |
| D102 f0 all | ccPost | 0.145 | – | – | 0.151 | 0.349 | 0.6 | 0 | – |
| D102 f0 all | ccAucPost0.509 | 0.145 | +0.000 [+0.000, +0.000] | 0.32 | 0.151 | 0.349 | 0.5 | 0 | 1 |
| D102 f0 all | ccAucPost0.55 | 0.143 | -0.002 [-0.007, +0.001] | 0.72 | 0.151 | 0.326 | 0.4 | 1 | 7 |
| D102 f0 all | ccAucPost0.6 | 0.112 | -0.033 [-0.085, +0.007] | 0.44 | 0.156 | 0.256 | 0.2 | 4 | 19 |
| D102 f0 all | ccAucPost0.637 | 0.109 | -0.036 [-0.087, +0.005] | 0.21 | 0.156 | 0.233 | 0.2 | 5 | 22 |
| D102 f0 all | ccAucPost0.7 | 0.095 | -0.050 [-0.105, -0.006] | 0.037 | 0.157 | 0.163 | 0.1 | 8 | 27 |
| D102 f1 all_rep2 | none | 0.133 | +0.046 [+0.009, +0.093] | 0.21 | 0.118 | 0.465 | 5.9 | – | – |
| D102 f1 all_rep2 | ccPost | 0.087 | – | – | 0.105 | 0.233 | 0.4 | 0 | – |
| D102 f1 all_rep2 | ccAucPost0.509 | 0.087 | +0.000 [+0.000, +0.000] | nan | 0.105 | 0.233 | 0.4 | 0 | 0 |
| D102 f1 all_rep2 | ccAucPost0.55 | 0.084 | -0.003 [-0.009, +0.000] | 0.65 | 0.106 | 0.209 | 0.3 | 1 | 6 |
| D102 f1 all_rep2 | ccAucPost0.6 | 0.056 | -0.030 [-0.077, +0.001] | 0.35 | 0.108 | 0.163 | 0.2 | 3 | 12 |
| D102 f1 all_rep2 | ccAucPost0.637 | 0.046 | -0.041 [-0.089, -0.002] | 0.12 | 0.109 | 0.140 | 0.2 | 4 | 14 |
| D102 f1 all_rep2 | ccAucPost0.7 | 0.034 | -0.053 [-0.104, -0.013] | 0.028 | 0.109 | 0.093 | 0.1 | 6 | 19 |

## 3. Post-processing vs training (the question of this script)

Baseline = per-case mean of D102 f0 all + D102 f1 all_rep2. y − x per case, paired bootstrap CI, Wilcoxon p (unadjusted).

| comparison | x | y | y − x [95% CI] | p |
|---|---|---|---|---|
| baseline: ccAucPost0.509 vs ccPost (what AUC post-processing adds) | 0.116 | 0.116 | +0.000 [+0.000, +0.000] | 0.32 |

Tumour-free test cases with no predicted tumour left (0 cases): D102 f0 all 0 / 0 / 0 / 0 / 0 / 0 / 0; D102 f1 all_rep2 0 / 0 / 0 / 0 / 0 / 0 / 0 (variants in the order none, ccPost, ccAucPost0.509, ccAucPost0.55, ccAucPost0.6, ccAucPost0.637, ccAucPost0.7).
