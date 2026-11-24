# The AUC rule applied to the predictions (ccAucPost) + a sweep of stricter cuts

238 real MSD07 test cases, checkpoint_final predictions. ccPost = training cc rule on the prediction; ccAucPost<cut> = ccPost, then predicted tumour pieces inside the predicted main structure with sep < cut -> their surroundings' label (filter_metrics' separability with the predicted pancreas as organ). Decisions use image + prediction only. **Main cut 0.509** (= the training cut, fixed before results); none (test: fixed cut only) = exploratory sweep. Written by `nnunet/auc_post.py`; exploratory, the main analysis stays predict_test.py.

## 1. Does sep tell false-positive pieces from hits? (pieces inside the predicted main structure)

| run | pieces / case (hit, FP) | unscored | sep median hit / FP | AUROC sep (FP lower) | < 0.509: FP / hit pieces (hit ml) |
|---|---|---|---|---|---|
| D102 f0 all | 1.0 (0.4, 0.7) | 17 | 0.685 / 0.616 | 0.58 | 7 / 2 (10.6) |
| D102 f1 all_rep2 | 0.8 (0.3, 0.5) | 4 | 0.691 / 0.642 | 0.55 | 1 / 3 (35.0) |

## 2. Each variant vs ccPost, per run

Tumour Dice mean (NaN = both empty, left out) and the paired difference to ccPost; lesion recall, FP pieces and reference lesions lost are totals over the cases.

| run | variant | tumour Dice | vs ccPost [95% CI] | p | pancreas Dice | lesion recall | FP pieces / case | lesions lost vs ccPost | pieces relabelled |
|---|---|---|---|---|---|---|---|---|---|
| D102 f0 all | none | 0.183 | +0.020 [+0.000, +0.041] | 0.28 | 0.209 | 0.550 | 16.9 | – | – |
| D102 f0 all | ccPost | 0.163 | – | – | 0.189 | 0.354 | 0.7 | 0 | – |
| D102 f0 all | ccAucPost0.509 | 0.161 | -0.001 [-0.006, +0.002] | 0.72 | 0.189 | 0.346 | 0.6 | 2 | 9 |
| D102 f1 all_rep2 | none | 0.172 | +0.033 [+0.014, +0.052] | 0.42 | 0.173 | 0.521 | 5.3 | – | – |
| D102 f1 all_rep2 | ccPost | 0.139 | – | – | 0.157 | 0.304 | 0.5 | 0 | – |
| D102 f1 all_rep2 | ccAucPost0.509 | 0.138 | -0.002 [-0.005, +0.001] | 0.27 | 0.158 | 0.292 | 0.5 | 3 | 4 |

## 3. Post-processing vs training (the question of this script)

Baseline = per-case mean of D102 f0 all + D102 f1 all_rep2. y − x per case, paired bootstrap CI, Wilcoxon p (unadjusted).

| comparison | x | y | y − x [95% CI] | p |
|---|---|---|---|---|
| baseline: ccAucPost0.509 vs ccPost (what AUC post-processing adds) | 0.151 | 0.150 | -0.001 [-0.004, +0.001] | 0.4 |

Tumour-free test cases with no predicted tumour left (0 cases): D102 f0 all 0 / 0 / 0; D102 f1 all_rep2 0 / 0 / 0 (variants in the order none, ccPost, ccAucPost0.509).
