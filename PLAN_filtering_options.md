# Filtering options for the MAISI pool (draft 2026-11-08)

Candidate filters (and repairs) for Option B, with arguments for and against each. Nothing chosen yet; the pick goes
into `PLAN_training.md` and `maisi/filter_metrics.py`.

## Context (as of 2026-11-08)

- Pool: 999 liver + 999 pancreas, all generated. MAISI QC passes 993/999 liver; thesis 15 mm rule (≥ 90% of tumour
  voxels within 15 mm of the liver) passes 621/999 liver (62%), ~92% pancreas.
- Arm `all` (999 synthetic liver) trained twice (f0, f1), tested on 111 real LiTS: liver Dice 0.870 / 0.876,
  tumour 0.301 / 0.305 (thesis LM0, trained on real LiTS: tumour 0.653).
- **Noise floor:** run-to-run differences ≈ 0.03 Dice per case, < 0.01 in the mean. A filter has to beat that.
- Error pattern of `all`: tumour over-predicted in most validation cases, false-positive tumours in tumour-free cases,
  one test case (lits_043) with no liver at all.
- A filter arm is only a new fold in `splits_final.json` (`nnunet/make_splits.py --ids <filter>.json`): no new
  preprocessing. ≈ 4.8 GPU h per training, one at a time → ~8–12 runs possible before 3 Dec.

## 1. Mask geometry (mask only)

Rules: thesis 15 mm rule, tumour fraction strictly inside the liver, number of tumour components, largest-component
share, liver in one component without holes (Euler number), liver volume in a realistic range, tumour present.

- **For:** cheap, interpretable, aimed at the dominant failure (trial10: 4/10 liver tumours in pelvis or lung).
  The 15 mm rule replicates the thesis. Mask-only, so it can run *before* image generation: redo-until-pass costs
  only ≈ +36% generation time instead of 3.3× (trial10 report).
- **Against:** blind to a bad image under a good mask. Thresholds (15 mm, 90%, component count) are arbitrary.
  Real LiTS is often multifocal, so a "≤ 3 components" rule can drop realistic volumes.

## 2. Label–image consistency (top candidate for tumour Dice)

Metrics:
- tumour–liver contrast: ΔHU, CNR = |μ_liver − μ_tumour| / σ_liver;
- tumour median HU < 0 (label on air/fat; trial10 had −234 and −71 HU);
- **separability AUC:** how well HU separates tumour voxels from a ring of liver around the tumour (0.5 = lesion
  not visible in the image);
- edge strength: mean image gradient along the label boundary vs inside.

- **For:** most plausible mechanism behind the over-predicted tumours: a tumour label on normal-looking liver teaches
  "normal liver = tumour" → false positives. trial10 already had 3 pancreas tumours with ≤ 6 HU contrast. The AUC is
  nearly threshold-free.
- **Against:** real tumours can be subtle or hyperdense; keeping only obvious hypodense lesions may cost sensitivity on
  subtle real lesions → use |ΔHU| and check sensitivity per lesion size afterwards. Contrast depends on the contrast
  phase, which MAISI doesn't control.

## 3. Global image realism

- MAISI QC (organ median HU): removes 6/999, too few for an arm of its own.
- Organ HU vs a real reference distribution (liver, spleen, …).
- Tile-seam score: intensity jumps at the VAE decode tile period (64/64/48 voxels), e.g. an FFT peak.
- Feature-space distance to real CT: pretrained 3D CT encoder + per-volume kNN / realism score
  (Kynkäänniemi et al. 2019, "Improved precision and recall metric for assessing generative models").

- **For:** generic; the feature distance is established in the literature; the seam score is MAISI-specific.
- **Against:** global realism says little about label correctness (MAISI QC passing 99% shows it barely separates
  volumes). The encoder choice is arbitrary and hard to interpret, and global features can miss small tumours. Seams
  probably hardly affect segmentation.
- **Reference data:** LiTS is now (almost) all test data, so reference statistics must come from the 20 validation
  cases or an external set (e.g. MSD07 images with TotalSegmentator liver masks). Otherwise it's test-set leakage.

## 4. Model-based

A judge model segments each synthetic volume; drop the volumes where it disagrees with MAISI's mask.

- **(a) TotalSegmentator** liver Dice vs the MAISI liver mask. Independent and trained on real data, but it judges only
  the liver, which isn't the problem. (Its `liver_vessels` task may also output a liver-tumour class: to check.)
- **(b) Per-volume loss of the existing `all` models** (small-loss criterion from noisy-label learning: mislabelled
  samples are fitted last/worst). No new training, ~999 predictions ≈ 1–2 GPU h. Circular (the models saw these
  volumes), and it also removes hard but correct samples (large, irregular tumours).
- **(c) Cross-fitting:** train on half the pool, score the other half. Clean, but ~2 extra trainings.
- **(d) Disagreement between f0 and f1** per volume. Free, but likely flags hard volumes rather than wrong ones.

Full data valuation (Data Shapley, TracIn) is out of scope at this budget.

## 5. Distribution matching

Resample the pool so tumour size, lesion count and location match real statistics (not a per-volume pass/fail).

- **For:** fixes dataset bias that per-volume rules can't (e.g. MAISI's weak tumour-size control).
- **Against:** needs real label statistics outside the test set; answers a different question than "remove bad
  volumes".

## 6. Controls

- **Train on the rejects** of the best filter: if they do worse than `all`, strong evidence the filter targets real
  faults. One run.
- **Size-matched random subset:** only needed if a filtered arm wins with far fewer volumes (backlog).

## Repair instead of reject (preprocessing)

- **Delete off-liver tumour components:** relabel tumour pieces outside liver ± 15 mm to background. LiTS only labels
  liver tumours, so "lesion-like thing in the pelvis = background" is the correct target. Keeps all 999 volumes.
- **nnU-Net v2 ignore label:** mark doubtful regions (stray pieces, invisible tumours) as `ignore`, so they give no
  training signal. Avoids teaching anything wrong there; needs an `ignore` entry in `dataset.json` (must be the
  highest label value).
- **Cost:** labels change → a new dataset + preprocessing (≈ 1 CPU h, as for `all`). A different intervention from
  filtering; frame it as "repair vs reject" in the paper.

## Setting thresholds

1. Fixed in advance from external or validation-case statistics (most defensible).
2. Percentile cut (simple, but the cut point has no principle behind it).
3. Choose between filters on the 20 validation cases (legitimate, but noisy).

Never tune on the 111 test cases. Diagnose the `all` model's errors (are false tumours inside or outside the liver?)
on the validation cases, not the test cases.

## Recommendation (Claude, 2026-11-08)

1. **CPU first:** compute families 1–3 + 4b/4d for all 999 into one CSV. Look at ~50 random volumes and label each
   good/bad by eye → how well each metric catches what a human calls bad, before any GPU time. Useful in the paper too.
2. **GPU arms:** 15 mm rule (thesis replication); label–image consistency (AUC or contrast); 4b or a combination;
   rejects of the best filter; optionally "delete off-liver components" as the repair arm. That's 4–5 runs
   (≈ 20–24 GPU h), which fits before 3 Dec.
