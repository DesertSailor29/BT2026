# Follow-ups after the liver test results (2026-11-15)

Four ideas raised after reading the results report (`Research/4 - Liver filter arms - test results.pdf`). All are
**exploratory**; the main analysis (predict_test.py on the saved checkpoint_final predictions, Holm over the 4 arms)
stays as decided before the results.

| # | idea | question | cost | status |
|---|---|---|---|---|
| 1 | AUC rule on the predictions (ccAucPost) | does post-processing do what training on the ccAuc labels did? | CPU, ~1 h | script ready (`nnunet/auc_post.py`) |
| 2 | stricter filters | does a stricter visibility cut keep helping? | CPU (as part of 1) | in `auc_post.py` as a cut sweep |
| 3 | larger tumours | does training on volumes with larger tumour burden help? | GPU, ~14–21 h | fold sizes measured, ids script ready (`nnunet/size_split.py`); not started |
| 4 | tumour-free generated volumes | do clean negatives lower false positives? | GPU generation | backlog (reasons below) |

## 1. ccAucPost: the AUC rule applied to the predictions

**Why.** Under ccPost (the training cc rule applied to the predictions) the cc filter arms lose their lead; only
ccAucRepair still beats the baseline (+0.031 [+0.011, +0.051], Holm p 0.002). Either the AUC repair taught the network
something a post-processing step can't do, or the same rule applied at test time would give the baseline the same gain.

**Rule (fixed before running).** On each prediction: ccPost, then every predicted tumour piece inside the predicted main
structure with sep < **0.5475** (the training cut, from `rules.json`, checked by the script) is relabelled to its
surroundings' label (liver if ≥ 50 % of its outer shell is predicted liver, else background). sep, shell, contact and
the relabel value are `filter_metrics.py`'s functions, unchanged, with the *predicted* liver as the organ. Decisions use
only the CT image and the prediction; the reference is read afterwards, for scoring.

**Readout (decided now, before the numbers).**
- Main: `ccAucRepair + ccPost` − `baseline + ccAucPost0.5475` (per-case tumour Dice, paired bootstrap CI, Wilcoxon).
  CI covering 0 with |diff| < 0.01 → post-processing reproduces the training effect. Clearly > 0 → training adds
  something a test-time rule can't.
- Also: baseline ccAucPost vs ccPost (what the rule adds to `all`), ccAucRepair with ccAucPost vs ccPost (does it add
  on top of training), arms vs baseline under ccAucPost (Holm over the arms listed).
- Harm check: real lesions lost vs ccPost, and how well sep tells FP pieces from hits (AUROC, table 1). Expectation:
  the FPs left after ccPost are mostly fluid touching the liver (ccpost_review), which is *visible*, so the rule may
  remove little; faint real lesions (lits_074, isodense 125 ml) are at risk.

**Run (UCloud, CPU only, alongside the training; ~30–60 min, ~4 GB RAM per worker).**
```bash
cd "$PROJ/BT2026" && git pull --rebase --autostash
setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && nice -n 10 python "$PROJ/BT2026/nnunet/auc_post.py" --workers 8 && nice -n 10 python "$PROJ/BT2026/nnunet/auc_post.py" --split val --workers 8' \
  > "$PROJ/logs/auc_post_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
AP_PID=$!
setsid nohup bash -c "source \"$PROJ/BT2026/maisi/env.sh\" && python \"$PROJ/BT2026/job_status.py\" --name LIVER_auc_post --glob \"$PROJ/nnunet_data/results/test_eval_liver/auc_post/cache/*/*.json\" --total 666 --pid $AP_PID" \
  > "$PROJ/logs/status_auc_post_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```
Watch the training's s/epoch (ccAucRepair_rep2 shares the GPU with the best-checkpoint predictions, ~38 s/epoch); if it
rises, restart with `--workers 4` (resumable). Outputs: `$nnUNet_results/test_eval_liver/auc_post/{summary.md,
per_case.csv, components.csv}` (+ `auc_post_val/`), copied to `reports/data/test_eval_liver/auc_post[_val]/`; commit +
push them with NOTES (small). Once ccAucRepair_rep2 is tested: rerun with `--runs 101:0 101:1 101:2 103:0 103:1 104:0 104:1`
(only the new pairs are measured).

## 2. Stricter filters

The cc rule already has **no tolerance**: a tumour piece must be 26-connected to the main liver piece (the 0/3/5/15 mm
numbers of 11-13 were only a robustness check of the old touch rule). So "stricter" means a higher visibility cut,
or a contact rule. Measured on the pool (in-liver components after cc, 2169 in 839 volumes) and on real LiTS lesions:

| rule | synthetic in-liver pieces relabelled | real val lesions failing | real test lesions failing (of 709) |
|---|---|---|---|
| sep < 0.5475 (current) | 162 (147 ml) in 144 vols | 1 (lits_049, excluded) | 0 |
| sep < 0.60 | 338 (501 ml) in 268 vols | 3 | 5 |
| sep < 0.637 | 479 (924 ml) in 354 vols | 6 | 22 |
| sep < 0.70 | 715 (2.7 L) in 459 vols | 15 | 54 |
| contact < 0.25 | 268 (174 ml) in 204 vols | 0 | 5 |
| contact < 0.5 | 1037 (737 ml) in 530 vols | 1 | 25 |

Any cut above 0.5475 also calls real lesions invisible, so it stops being "less visible than every real lesion" and
becomes a tuned setting. Contact rules hit real exophytic lesions. **Plan:** the cut sweep 0.60 / 0.637 / 0.70 runs
inside `auc_post.py` as a dose-response check on the predictions. A stricter *training* arm (D104 rebuilt at a new cut,
~1 h CPU + ~7 GPU h) only if the sweep shows a clear trend, and then with the cut chosen on the **val** predictions
(`--split val`), never on test.

## 3. Larger tumours

**Not overfitting**, as long as the target comes from the val cases or the literature and the split is defined on the
synthetic labels only: matching the training data to the clinical population is domain matching. Report Dice per
tumour-burden tercile, so a gain on large tumours paid for by small ones is visible.

Measured (ccAucRepair labels = what the network sees; from `reports/data/` CSVs, to be confirmed by `size_split.py` on
the real labels):

| | volumes | tumour ml per tumour volume p10 / 25 / 50 / 75 / 90 | total tumour |
|---|---|---|---|
| pool after ccAuc | 999 (819 with tumour, 177 tumour-free, 3 never had tumour) | 0.0 / 0.3 / 2.3 / 20.0 / 75.7 | 21.7 L |
| **sizeLarge** (upper half) | 410 + 180 tumour-free = **590** | median 20.0, p10 3.6 | 21.4 L |
| **sizeSmall** (lower half) | 409 + 180 = **589** | median 0.33 | 0.25 L |
| sizeRand (random half, control) | 410 + 180 = 590 | ≈ pool | ≈ 10.8 L |
| real val (reference, 17 cases with tumour) | | 1.6 / 6.3 / 18.3 / 29.4 / 132 | |
| real test (not used) | 101 with tumour | 0.9 / 3.4 / 14.9 / 111 / 266 | |

- Per *lesion* the pool is not small (median 0.2 ml vs real val 0.3 ml); the gap is per *case*: real cases have more
  lesions (median 6 in val vs 2) and the large ones. The large half matches the real per-case burden (median 20 vs 18 ml).
- Split by measured burden, not by MAISI's size setting (it barely controls the outcome; the large half still holds
  23–32 volumes of each low size setting).
- Confound: the large half also has ~85× more tumour voxels. "Larger tumours" and "more tumour voxels" can't be told
  apart in this design; say so.

**Arms** (new folds of Dataset104, no generation, no new preprocessing): `ccAucRepair_sizeLarge`, `ccAucRepair_sizeSmall`,
optional `ccAucRepair_sizeRand`. Compare large vs small directly, and each vs ccAucRepair (999). ~7 h each
(train 4.9 + predict 2, sequential: overlapping train and predict gave no gain on one GPU). Two arms ≈ 14 h, three ≈ 21 h.
The GPU is busy until ~Mon 16 Nov 12–14 (ccAucRepair_rep2 + best-checkpoint predictions); the session ends Tue 17 Nov
~13:24 → two arms fit, three only in a new session.

**Prepare now (CPU, ~5 min; writes id lists only, registers nothing):**
```bash
setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && nice -n 10 python "$PROJ/BT2026/nnunet/size_split.py" --workers 8' \
  > "$PROJ/logs/size_split_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```
Check the log against the table above (819 / 177+3 / split ≈ 2.29 ml / 590 / 589). **After the decision**, per arm:
`make_splits.py --target liver --dataset-id 104 --arm ccAucRepair_sizeLarge --ids $MAISI_RUNS/pool/filter_metrics/liver/size/ids_sizeLarge.json`,
then train + predict + eval as the other arms (`run_queue.py` follow-up or by hand).

## 4. Tumour-free generated volumes (backlog)

- Natural experiment already run: ccRepair vs ccRepairDel differ by 157 tumour-free volumes → −1.2 FP pieces per case,
  tumour Dice +0.005 (n.s.).
- nnU-Net already samples mostly tumour-free patches; the FPs left after ccPost are fluid structures (effusion,
  gallbladder, bowel) that the tumour-bearing volumes also show as background.
- Only 10 tumour-free test cases → little power for the question it targets.
- If done: regenerate the image from the *same* mask with the tumour label removed (matched twins, only the tumour
  differs); MAISI image stage only, ~8 GPU h per 300 volumes. Needs its own GPU time, not alongside training.
