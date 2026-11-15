# ccPost review: what the cc rule on the predictions catches, harms, can't reach, and missed lesions

Written by `nnunet/ccpost_figures.py` from the checkpoint_final test predictions (111 LiTS) and `reports/data/test_eval_liver/fp_location/`. **ccPost** = the training-label cc rule applied to a prediction: keep the piece of predicted liver ∪ tumour that holds the largest predicted liver piece, everything else → background. Examples are picked by fixed rules (largest of each kind, a new case per example where possible), not by eye. Each figure: top = full axial slice (CT | reference | as saved | after ccPost), the dashed ring marks the piece of interest; bottom = the same zoomed. Panel titles give whole-case tumour Dice, FP pieces and lesions found. Style: W400/L40, radiological (patient right = image left). Green organ, red tumour.

## A. ccPost catches: false-positive tumour outside the predicted liver, removed

- **A1** `A1_caught_lits_115_all.png`: lits_115, all, 32.36 ml, slice 673: largest FP piece of `all` outside the predicted liver (and outside the real liver).
- **A2** `A2_caught_lits_094_all.png`: lits_094, all, 9.05 ml, slice 332: `all`'s case with the most FP pieces outside the predicted liver (141 pieces; largest shown).
- **A3** `A3_caught_lits_043_all_rep2.png`: lits_043, all_rep2, 562.42 ml, slice 117: largest FP piece outside the predicted main structure but inside the real liver (predicted liver is broken into pieces).
- **A4** `A4_caught_lits_034_ccAucRepair.png`: lits_034, ccAucRepair, 24.11 ml, slice 76: largest FP piece of ccAucRepair outside the predicted liver.
- **A5** `A5_caught_lits_091_all.png`: lits_091, all, 16.27 ml, slice 310: largest outside-FP piece of `all` in a tumour-free test case.

## B. ccPost harms: something real removed

- **B1** `B1_harm_lits_130_all_rep2.png`: lits_130, all_rep2, 25.35 ml, slice 525: largest correctly found tumour piece that ccPost removes (lies outside the predicted main structure).
- **B2** `B2_harm_lits_023_all.png`: lits_023, all, 8.33 ml, slice 339: second-largest correctly found tumour piece removed by ccPost (another case).
- **B3** `B3_harm_lits_053_ccRepair.png`: lits_053, ccRepair, 17.63 ml, slice 64: largest predicted liver piece removed by ccPost that lies in the real liver (97 % inside).
- **B4** `B4_harm_lits_109_ccRepair.png`: lits_109, ccRepair, 4.13 ml, slice 446: largest tumour Dice loss under ccPost (-0.067); the removed piece is shown.

## C. ccPost can't reach: false-positive tumour inside the predicted liver, kept

- **C1** `C1_kept_lits_020_all.png`: lits_020, all, 148.83 ml, slice 487: largest FP piece of `all` inside the predicted liver (kept).
- **C2** `C2_kept_lits_120_ccAucRepair.png`: lits_120, ccAucRepair, 118.02 ml, slice 257: largest FP piece of ccAucRepair inside the predicted liver (kept).
- **C3** `C3_kept_lits_032_ccAucRepair.png`: lits_032, ccAucRepair, 31.28 ml, slice 115: largest kept FP piece of ccAucRepair in a tumour-free case.
- **C4** `C4_kept_lits_082_ccRepairDel.png`: lits_082, ccRepairDel, 36.35 ml, slice 304: largest kept FP piece that lies outside the real liver (the predicted liver reaches beyond it).

## D. Missed lesions (false negatives; ccPost can't add tumour)

- **D1** `D1_missed_lits_110_all.png`: lits_110, all, 12.71 ml, slice 494: largest real lesion `all` misses.
- **D2** `D2_missed_lits_039_ccAucRepair.png`: lits_039, ccAucRepair, 161.22 ml, slice 50: largest real lesion `ccAucRepair` misses.
- **D3** `D3_missed_lits_098_ccAucRepair.png`: lits_098, ccAucRepair, 8.21 ml, slice 560: largest real lesion that every run misses.
- **D4** `D4_missed_lits_074_ccAucRepair.png`: lits_074, ccAucRepair, 125.33 ml, slice 79: largest real lesion both `all` runs find but ccAucRepair misses.
- **D5** `D5_missed_lits_046_all.png`: lits_046, all, 9.04 ml, slice 40: largest real lesion ccAucRepair finds but both `all` runs miss (shown: `all`).

## E. One case, every run: saved vs ccPost

- **E1** `E1_runs_lits_075.png`: lits_075, slice 70: largest tumour Dice gain of `all` under ccPost (+0.456).
- **E2** `E2_runs_lits_088.png`: lits_088, slice 447: largest tumour Dice spread between the runs (0.731).

## Reading (by eye, Claude on UCloud, 2026-11-15; check before quoting)

**A. Caught (ccPost does what it should).** The off-liver FPs of the baseline are not random: the models paint
tumour on round, low-contrast blobs anywhere in the body. A1 a lung/pleura-based mass in the right chest (32 ml),
A2 bowel wall in the right colon with 140 more specks along the bowel, A4 a large right renal or retroperitoneal mass
(ccAucRepair), A5 faeces in the right colon of a tumour-free case. All are removed. This matches the MAISI training
error seen in the pool (tumour labels in the pelvis, lung and bowel).

**A3 is really a failure, not a catch:** in lits_043 `all_rep2` labels most of the liver as tumour (562 ml), and the
largest predicted *liver* piece is the spleen. ccPost keeps the spleen and deletes the real liver plus the tumour
(lesions found 1/1 → 0/1). So ccPost assumes the largest liver piece is the liver; a model that breaks down on a
case can make it remove the wrong thing.

**B. Harm.** B1 (lits_130) and B2 (lits_023): correctly found tumours in the left lobe / near the caudate lobe whose
predicted liver around them is missing, so the tumour piece is not connected to the main structure and ccPost
deletes the true positive (B2: Dice 0.07 → 0.00). B3: a real liver segment (left lobe tip, 18 ml) predicted as a
separate piece is removed (tumour unaffected). B4 (lits_109, largest Dice loss, −0.067): a tumour at the liver hilum
with a thin gap to the predicted liver. Common pattern: when the predicted liver has holes or gaps, ccPost removes
real liver and tumour beyond them. Small in total (2–10 lesions lost per run, see summary.md).

**C. Kept (ccPost can't reach).** C1 (lits_020): `all` calls large parts of a heterogeneous right lobe tumour
(149 ml), reference says liver. C2 (lits_120): a right pleural effusion labelled tumour by ccAucRepair (118 ml), kept
because it touches the predicted liver on other slices. C3 (lits_032, tumour-free): a 31 ml piece at the liver's
posterior edge near the IVC. C4 (lits_082): a fluid-filled structure below the liver (bowel or gallbladder,
36 ml) connected to the predicted liver on other slices. → The FPs left after ccPost are mostly *fluid or
low-contrast structures touching the liver*. A cc rule can't fix those; an HU or shape check might.

**D. Missed.** D1 (lits_110): a faint, almost isodense lesion; `all` leaves a hole in the liver there (not tumour,
not liver). D2 (lits_039): a 161 ml exophytic lesion hanging below the liver into the abdomen, missed by
ccAucRepair. D3 (lits_098): a lesion next to the IVC at the caudate lobe; every run leaves it out of the liver and
calls it background. D4 (lits_074): a 125 ml **isodense** lesion in the left lobe that both `all` runs find but
ccAucRepair misses. Possible reason (not tested): the AUC repair turned invisible synthetic tumour labels into liver,
so the model may have learned not to call low-contrast regions tumour. D5 (lits_046): the opposite, a clearly
hypodense lesion that `all` leaves as a hole and ccAucRepair finds.

**E. One case, every run.** E1 (lits_075): `all` gains +0.46 Dice from ccPost (46 → 6 FP pieces; at this slice
the visible change is a spleen piece predicted as liver being removed); the filter arms are already at 0.60–0.79 and
ccPost changes nothing for them. E2 (lits_088, a large hilar tumour): ccReject almost misses it (Dice 0.03) while
every repair arm finds it (0.63–0.76), matching ccReject's weakness on large tumours in the test report.
