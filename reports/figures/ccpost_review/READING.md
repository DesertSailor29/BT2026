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
