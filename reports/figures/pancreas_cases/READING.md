## Reading (written by hand, 2026-11-18, after looking at the figures and `case_diagnostics/summary.md`)

**The baseline mostly fails by predicting too little gland, not by mixing up labels.** On average about 80 % of the
reference gland voxels are predicted as background (`all` 0.79, `all_rep2` 0.82). Only ~6 % become tumour, and
reference pancreas called tumour is rare (4-5 %). Precision is moderate (0.55), and most predicted gland lies within
10 mm of the reference (median 0.74). So the model looks in the right place but claims only small pieces: median
predicted gland volume is 0.30 / 0.21 of the reference, and on average only 59-65 % of the gland's slices get any
prediction. Recall is about the same in the head, body and tail thirds (0.16-0.23), so no single part of the gland
is the problem. Failure types (both runs): gland missed 57, under-segmented 88, tumour missed with the gland found 8,
pancreas labelled as tumour 2, none of these 83.

**Why: the real scans show things the synthetic scans almost never have.**
- *Fat around and inside the gland.* In real MSD07 the 3-voxel shell around the gland is below 0 HU in 26 % of the
  test cases; in the synthetic training scans it is 2 %. Synthetic pancreases sit in uniform grey soft tissue
  (surroundings median 40 HU, narrow IQR 27-56) with smooth edges and no lobulation (P00, top rows). Real glands are
  often lobulated, with fat between the lobules and around the gland (P00 bottom; P08 msd_005, P09 msd_255). The 62
  test cases with fatty surroundings reach gland Dice 0.15 against 0.30 for the rest (2.5 mm-slice cases only: 0.16
  vs 0.35), so it is not a slice-thickness effect. A very low pancreas HU (< 40 HU, fatty or
  atrophic gland, 18 cases) gives gland Dice 0.05.
- *Thick slices.* The synthetic scans all have 1.0 mm slices; 38 test cases have 5 mm slices (gland Dice 0.16 vs
  0.30 at 2.5 mm). nnU-Net resamples them to 1 mm, but the partial-volume blur is something the model never saw.
- *Missing even on clear scans.* Some well-enhanced, sharply outlined glands are still missed entirely (P09 msd_255:
  pancreas 85 HU, thin fat rim; Dice 0.00-0.06). This points to a general appearance gap (the synthetic texture is
  smoother and blurrier, P00), not only the fat cases.
- *Large tumours.* Big cystic or low-density tumours are found only in part (P05 msd_414: 92 ml tumour, ~10 % of the
  gland found). Small tumours inside a found gland are also missed (P13 msd_316, 1.4 ml).

**Not the cause:** gland size (Spearman −0.05), tumour share of the gland (0.00) and the gland's cranio-caudal
extent (−0.11) barely relate to Dice. Synthetic gland volumes are a little smaller than real ones (median 72 vs
86 ml), with tumour volumes the same (5.3 vs 5.7 ml). In-plane spacing matches (0.8 mm).

**Successes (P01-P04)** are brightly enhanced glands (pancreas often > 100 HU) with clear borders and 2.5 mm slices,
i.e. the real cases that look most like a clean version of the synthetic data.

**What this means for the study.** The liver filters removed MAISI's *label* errors. Here the problem is the
*image*: the synthetic pancreas scans lack the variety of real appearance (fat, lobulation, slice thickness). A
volume filter cannot add what is missing from every synthetic volume. Options to discuss: augmentation (thick-slice
resampling and blur), checking whether MAISI can produce fatty or lobulated glands, or mixing in real data (outside
the current synthetic-only design). These are observations and correlations, not causal tests.
