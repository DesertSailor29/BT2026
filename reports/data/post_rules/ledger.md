# Ledger of post-hoc rules tried (autopilot A2; every rule and parameter, failures included)

Val only (43 real MSD07 val cases, runs 102:0 `all` / 102:1 `all_rep2`) unless marked. Effect = mean per-case
tumour Dice vs the saved prediction (f0 / f1); details `pancreas_val/summary.md`. Code `nnunet/post_rules.py`.
Val-label parameters (`pancreas_val/val_label_params.json`): pancreas ∪ tumour HU p0.5-p99.5 = [-95, 213]
(2.9 % of voxels < -50 HU, 0.2 % > 300), 43 lesions, smallest 1.19 ml.

| # | rule | anatomical fact | parameter (source) | val effect f0 / f1 | verdict |
|---|---|---|---|---|---|
| 0 | ccPost (A1, fixed) | one organ | none (training cc rule) | +0.004 / -0.046 (A1) | hurts: fragmented gland |
| 0 | ccAucPost 0.509-0.70 (A1) | tumour visible | cut 0.509 (rules.json); sweep val | 0 / 0 at 0.509, worse stricter | nothing |
| 1 | ccTol5/10/20 | one organ, split by thick slices | d 5/10/20 mm (1-4 thick slices) | pending (s3 job) | |
| 2 | ccK2/3 | one organ in <= k pieces | k 2/3 | pending | |
| 3 | tumTouch | MSD07 tumours are pancreatic | adjacency | pending | |
| 4 | tumNear10 | same, tolerant | 10 mm | pending | |
| 5 | tissueTxt | soft tissue, not fat/bone | -50 / 300 HU (textbook) | pending | |
| 6 | tissueVal | same | -95 / 213 HU (val labels p0.5/p99.5) | pending | |
| 7 | minLes | no lesion smaller than real ones | 1.19 ml (smallest val lesion) | pending | |
| 8 | grow0.3/0.2/0.1 | gland continues where the model was nearly sure | t; 10 mm, -50/300 HU | pending | |
| 9 | grow0.2+ccTol10 | 8 then 1 | | pending | |
| C | rand_ccPost / rand_ccTol10 / rand_tumTouch | control: random piece removal, matched count | seeded per case | pending | |
| C | open1 | control: generic opening 3x3x3 | | pending | |
| C | dilate1 | control: growth without probabilities | 1 voxel | pending | |
