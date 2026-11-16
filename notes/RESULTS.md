# Results digest (finished experiments, pipeline order)

One paragraph per finished experiment with the key numbers and where the full output lives. The numbers' source of
truth is the `summary.md` named in each entry; details and dates are in `notes/LOG.md`. Liver = main study, pancreas =
generalisation check (measured only, no arms trained).

## Generation

- **trial10** (2026-10-20, full B200, 4 workers): 20/20 ok, MAISI QC 20/20, 88 s/vol. Only 3/10 liver volumes pass the
  thesis filter (≥ 90 % of tumour within 15 mm of the liver; tumours in pelvis/lung); pancreas 10/10. Size control weak.
  `reports/trial10_quality.md`.
- **Pool** (2026-11-05 → 11-07 20:54): 1998/1998 ok (999 liver + 999 pancreas, sizes 0.1–0.9 × 111, seeds
  100000–101997, 512³ @ 0.8×0.8×1.0, patched Path A), MAISI QC liver 993, pancreas 985; `$MAISI_RUNS/pool`, 249 GB.
  Thesis 15 mm filter passes liver 621/999. Tumour size control weak (Spearman 0.38 with the size setting); pool tumour
  burden far below real LiTS (in-liver median 2.3 ml per volume vs 16 ml per real case; 34 % of in-liver components
  < 0.05 ml vs 10.7 % of real lesions).

## Baseline `all` (999 synthetic, 2 runs)

- 2026-11-08, checkpoint_final, 111 LiTS test: liver Dice 0.870/0.876, **tumour 0.301/0.305**, run-to-run f1−f0
  +0.004 (p 0.072). Lesion-wise: recall 0.42/0.44, precision 0.04/0.06, 46–67 FP components per case, FPs in 10/10
  tumour-free cases. lits_045 liver Dice 0 on val (thick slices). `reports/data/test_eval_liver/summary.md`.

## Filter measurements (liver pool vs real LiTS)

- **Measure session** (2026-11-13): real val lesions touch the liver 199/199 (test 707/709). Touch rule robust to
  tolerance (keeps 401/416/431/453 at 0/3/5/15 mm; near misses are specks). AUC cut set by lits_049 (only real case
  below 0.547; its lesions show nothing under the label) → excluded → **cut 0.5475**; real test lesions below it 0/681.
  Real sep val ≈ test (p5 0.667/0.663, median 0.90); synthetic median 0.71. MAISI errors seen: invisible tumour blobs in
  the pelvis (37–95 ml), invisible labels inside the liver (≤ 49 ml, ≤ 5 HU contrast), stray specks.
  `$MAISI_RUNS/pool/filter_metrics/liver/summary.md`, `reports/data/filter_metrics/`, `reports/figures/filter_review/`,
  report `Research/2 - …`.
- **cc rule counts** (cut 0.5475): cc unchanged 343 / liver only 58 / tumour only 477 / both 121, tumour-free after
  repair 157; ccAuc unchanged 300, 162 in-liver components relabelled in 144 volumes, tumour-free 177. Detached liver
  pieces: pool 179/999 (max 2 ml), real test 19/111 (max 0.96 ml).
- **Liver plausibility check: no-go as a filter** (2026-11-14). v3 border measures flag pool 214/999, real test
  30/111; blind review of 60: flagged 30 → 28 ok, 2 unsure, 0 bad; the only bad liver (B21 = liver_0595) unflagged.
  The border-step flag measures edge softness, not wrong labels. `reports/data/liver_check/`,
  `reports/figures/liver_review/`.
- **Contrast phase of the pool** (descriptive): mostly weak enhancement (aorta median 94 HU; rule bins non-contrast
  612, late 340, arterial 45, portal venous 2); pool liver core HU 78 vs real 101.

## Liver arms (checkpoint_final, 111 LiTS test, baseline = per-case mean of the two `all` runs 0.303)

Main family, Holm over 4; all four are clear wins. `reports/data/test_eval_liver/summary.md`, report
`Research/4 - Liver filter arms - test results.pdf`.

| arm | volumes | tumour Dice | diff [95 % CI] |
|---|---|---|---|
| ccReject (D101 f2) | 343 | 0.382 | +0.079 [0.038, 0.120] |
| ccRepair (D103 f0) | 999 | 0.375 | +0.072 [0.051, 0.094] |
| ccRepairDel (D103 f1) | 842 | 0.370 | +0.067 [0.043, 0.092] |
| ccAucRepair (D104 f0) | 999 | **0.403** | +0.100 [0.072, 0.129] |
| ccAucRepair_rep2 (D104 f1) | 999 | 0.381 | +0.079 [0.057, 0.101] |

FP components per case 67/46 → 9–12; liver Dice −0.003 to −0.022. Arm vs arm (exploratory): ccAucRepair − ccRepair
+0.028 [0.010, 0.048]; ccRepair − ccReject −0.007 (n.s.); ccRepair − ccRepairDel +0.005 (n.s.). Gain concentrated in
small/medium tumour burden; on the largest third ccReject loses (−0.07, misses e.g. the lits_088 hilar tumour). ccReject
val loss rises after epoch ~756 (mild overfit to 343 volumes); best epochs: all 915, all_rep2 988, ccReject 775,
ccRepair 879, ccRepairDel 938, ccAucRepair 871.

## Test-time postprocessing (exploratory, decided after seeing results)

- **ccPost** = the training cc rule applied to the predictions (`nnunet/fp_location.py`,
  `reports/data/test_eval_liver/fp_location/summary.md`): 84–88 % of `all`'s FP pieces lie outside the predicted liver
  (mostly specks); ccPost lifts `all` 0.301/0.305 → 0.379/0.383 (+0.078), FP pieces/case 67/46 → 8/7. Filter arms gain
  only +0.004 to +0.009. **Under ccPost only ccAucRepair beats the baseline: f0 +0.031 [0.011, 0.051] Holm 0.002; rep2
  +0.014 [0.003, 0.025] Holm 0.092.** ccPost fails when the predicted liver is broken (lits_043: spleen kept as liver).
  Figures `reports/figures/ccpost_review/`.
- **ccAucPost** = ccPost + the AUC rule on predicted in-liver pieces (`nnunet/auc_post.py`,
  `reports/data/test_eval_liver/auc_post[_val]/summary.md`): sep separates in-liver FPs from hits (AUROC 0.79–0.83).
  Pre-fixed readout ccAucRepair+ccPost vs baseline+ccAucPost0.5475: f0 +0.009 [−0.008, 0.026] p 0.42, rep2 −0.008 p 0.053
  → **post-processing gives the baseline most of what AUC training gives; the AUC-training edge is not robust across
  the two runs.** Stricter cuts: 0.60 helps every run on test (+0.033 to +0.062 vs ccPost, 2–9 lesions lost), plateau
  to 0.637, 0.70 loses 16–33 lesions; val would pick ~0.60.
- **checkpoint_best** (sensitivity check): predictions + postproc running on UCloud (see NOTES.md Current state);
  output `reports/data/test_eval_liver_best/`.

## Pancreas (measured only)

2026-11-14, MSD07 split 43 val / 238 test: AUC cut 0.509 (msd_025, isodense real tumour, kept); real lesions touch the
pancreas 43/43 + 240/240. cc unchanged 911, tumour-free 75 (= the volumes whose single tumour lies off the pancreas),
repair-delete 924; ccAuc unchanged 883 (30 components). Real MSD07 has detached pancreas pieces in 31/281 (≤ 1.3 ml).
AUC rule near its floor (real pancreas lesions are low-contrast too). Branch `pancreas`, report
`Research/3 - Pancreas filter rules - measurement findings.pdf`.

## Size split (ids only, no training)

D104 by measured in-liver burden, median split 2.29 ml: sizeLarge 590, sizeSmall 589, sizeRand 590 (tumour-free in
every list). `$MAISI_RUNS/pool/filter_metrics/liver/size/`.
