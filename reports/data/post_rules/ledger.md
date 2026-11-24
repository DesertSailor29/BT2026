# Ledger of post-hoc rules tried (autopilot A2; every rule and parameter, failures included)

Val only (43 real MSD07 val cases, runs 102:0 `all` / 102:1 `all_rep2`) unless marked. Effect = mean per-case
tumour Dice vs the saved prediction (f0 / f1); details `pancreas_val/summary.md`. Code `nnunet/post_rules.py`.
Val-label parameters (`pancreas_val/val_label_params.json`): pancreas ∪ tumour HU p0.5-p99.5 = [-95, 213]
(2.9 % of voxels < -50 HU, 0.2 % > 300), 43 lesions, smallest 1.19 ml.

| # | rule | anatomical fact | parameter (source) | val effect f0 / f1 | verdict |
|---|---|---|---|---|---|
| 0 | ccPost (A1, fixed) | one organ | none (training cc rule) | +0.004 / -0.046 (A1) | hurts: fragmented gland |
| 0 | ccAucPost 0.509-0.70 (A1) | tumour visible | cut 0.509 (rules.json); sweep val | 0 / 0 at 0.509, worse stricter | nothing |
| 1 | ccTol5/10/20 | one organ, split by thick slices | d 5/10/20 mm (1-4 thick slices) | +0.005/-0.044, +0.004/-0.044, +0.002/-0.030 | hurts f1 (lesions lost 14-11) |
| 2 | ccK2/3 | one organ in <= k pieces | k 2/3 | +0.018/-0.020, +0.017/-0.018 | f0 only (lost 7/6) |
| 3 | tumTouch | MSD07 tumours are pancreatic | adjacency | +0.017/-0.006 | f0 only (lost 5) |
| 4 | tumNear10 | same, tolerant | 10 mm | +0.016/-0.007 | f0 only (lost 3) |
| 5 | tissueTxt | soft tissue, not fat/bone | -50 / 300 HU (textbook) | +0.001/+0.001 | nothing (few voxels) |
| 6 | tissueVal | same | -95 / 213 HU (val labels p0.5/p99.5) | +0.001/+0.001 | nothing |
| 7 | minLes | no lesion smaller than real ones | 1.19 ml (smallest val lesion) | +0.013/+0.005 (lost 4 of 86) | near miss (mean 0.009); open1 control +0.009/+0.001 |
| 8 | grow0.3/0.2/0.1 | gland continues where the model was nearly sure | t; 10 mm, -50/300 HU | tumour -0.003/+0.003, -0.003/+0.006, -0.004/+0.009; **gland +0.021/+0.018, +0.034/+0.029, +0.053/+0.047** | gland gain, tumour ±0 |
| 9 | grow0.2+ccTol10 | 8 then 1 | | -0.007/-0.005 | hurts (lost 10) |
| C | rand_ccPost / rand_ccTol10 / rand_tumTouch | control: random piece removal, matched count | seeded per case | -0.14/-0.09, -0.14/-0.09, -0.12/-0.08 | random removal destroys lesions (most tumour pieces are the hits in few-piece cases) |
| C | open1 | control: generic opening 3x3x3 | | +0.009/+0.001, gland -0.009/-0.007 | as good as minLes on tumour |
| C | dilate1 | control: growth without probabilities | 1 voxel | tumour -0.035/-0.018, gland +0.047/+0.040 | gland gain ≈ grow0.1, but tumour hurt |

First screen (s3 code, run s4-s5, `pancreas_val/summary_screen1.md`): **no rule promising**. f0 (17 FP pieces/case)
gains from any FP removal, f1 (6/case) loses real lesions from it; grow rules lift gland Dice (+0.05 at t 0.1) but
dilation does about as much for the gland (under-segmentation: any growth helps), while hurting tumour.

Second screen (s5 code, run s5-s6, `postrules2`, `pancreas_val/summary_screen2.md`; first-screen rules recomputed,
same numbers):

| # | rule | anatomical fact | parameter (source) | val effect f0 / f1 | verdict |
|---|---|---|---|---|---|
| 10 | grow0.05 / grow0.02 | as 8, lower t | t (sweep) | tumour -0.005/+0.009, -0.007/+0.007; gland +0.072/+0.064, +0.097/+0.088 | gland gain grows as t drops; tumour ±0 |
| 11 | grow0.1d20 / grow0.05d20 | as 8, wider window | 20 mm | = d10 (±0.002) | window not binding |
| 12 | growT0.2 / growT0.1 | the tumour continues where the model was nearly sure | t; 10 mm, -50/300 HU | -0.002/+0.004, -0.004/+0.005 | nothing |
| 13 | minLes0.5 / minLes1.75 | as 7, sensitivity | 0.5 ml; 1.75 ml (p5 val lesions) | +0.011/+0.004 (lost 2), +0.015/+0.008 (lost 5) | 0.5: mean < 0.01; 1.75: lost 5 (> 5 %) |
| 14 | pos | pancreas lies just anterior to the spine | spine from bone HU > 200; LOO 2D prior of the other 42 val labels, 10 mm margin, 50 % inside | +0.005/+0.002, 0 lesions lost, FP pieces 17 → 11 / 6 → 4.7 | safe but small |
| 15a | minLes+tumNear10 | 7 + 4 | | +0.017/-0.005 | f0 only |
| 15b | **pos+minLes** | 14 + 7 | | +0.015/+0.006 (lost 4 of 86) | **promising** → frozen |
| 15c | **grow0.1+minLes** | 8 + 7 | | +0.007/+0.018 (lost 2); gland +0.053/+0.047 | **promising** → frozen |
| 15d | **grow0.1+tumNear10** | 8 + 4 | | +0.014/+0.019 (lost 0); gland +0.053/+0.047 | **promising** (best mean) → frozen |
| 15e | grow0.05+minLes | 10 + 7 | | +0.004/+0.017 | promising; variant of 15c, not frozen |
| 15f | grow0.1+growT0.2+minLes | 8 + 12 + 7 | | +0.008/+0.020 | promising; growT adds ~0.001, not frozen |
| 15g | **grow0.1+pos+minLes** | 8 + 14 + 7 | | +0.009/+0.019 (lost 2) | **promising** → frozen |
| C | thr0.1 / thr0.05 | control: lower threshold without anatomy | | tumour -0.005/+0.008, -0.006/+0.009; **gland +0.058/+0.049, +0.081/+0.068** | **plain thresholding lifts the gland as much as grow** (grow0.1 +0.053/+0.047): the anatomical limits of grow add nothing on val |
| C | rand_minLes / rand_pos | random removal matched to minLes / pos | seeded | -0.115/-0.079, -0.007/-0.041 | random removal hurts |

Reading: the tumour gains of the winners are roughly additive parts: removal of small pieces (minLes) / pieces far from
the gland (tumNear10) / outside the position prior (pos), plus growing the gland, which reconnects tumours (f1).
The gland gain of grow is a lower threshold, not anatomy (thr0.1 ≥ grow0.1). Frozen set (s6, `notes/AUTOPILOT.md`):
pos+minLes, grow0.1+minLes, grow0.1+tumNear10, grow0.1+pos+minLes; controls incl. new thr0.1+X twins (A3,
`nnunet/post_rules_test.py`, `pancreas_{val,test}_frozen/`).

**A3a test (s7, 2026-11-24, no-probability rules only):** F1 pos+minLes 2-run mean +0.007 [+0.002, +0.012] (f0 +0.012,
f1 +0.002), below the 0.01 floor → **not confirmed**; lesion recall 0.55 → 0.46 / 0.52 → 0.44. minLes alone +0.006,
tumNear10 +0.011 (f1 +0.003), pos +0.004. `pancreas_test_frozen_noprobs/summary.md`. F2-F4 pending (test probs).
