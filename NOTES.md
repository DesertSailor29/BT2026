# Project notes

Loaded every session (via CLAUDE.md): keep it short and true. Everything else is in `notes/` and read only when needed
(index at the bottom). Finished results → `notes/RESULTS.md`; old log entries → `notes/LOG.md`.

## Current state (2026-11-18)
- **Liver study done on checkpoint_final** (111 LiTS test, tumour Dice; baseline `all` 0.303): ccReject 0.382,
  ccRepair 0.375, ccRepairDel 0.370, ccAucRepair 0.403 (rep2 0.381); all four Holm-significant. But a test-time cc
  postprocessing (ccPost) gives the baseline the same gain (0.381), and ccAucPost most of the AUC gain; under ccPost
  only ccAucRepair stays ahead (f0 +0.031 Holm 0.002, rep2 +0.014 Holm 0.09) → the AUC-training edge is not robust
  across runs. Details `notes/RESULTS.md`, `reports/data/test_eval_liver/`, report `Research/4 - Liver filter arms -
  test results.pdf` (19 pages, VM only; rep2 in every chart/table since 2026-11-16, checkpoint_best not yet).
  The two ccAucRepair runs differ by 0.022 (p 7e-5; `all` pair 0.004) → noise floor too small for arm-vs-arm claims.
- **checkpoint_best done** (secondary): no late overfitting of ccReject (best 0.333 vs final 0.382); only ccAucRepair
  (0.378, rep2 0.381; baseline 0.315) beats the baseline at both checkpoints; under ccPost at best the cc arms fall below
  the baseline and the AUC arms are n.s. (+0.014 / +0.009). Numbers `notes/RESULTS.md`. Not yet in report 4.
- model_diff figures (`nnunet/model_diff_figures.py`, `reports/figures/model_diff/`) committed 2026-11-16.
- **New experiments paused** until the findings are presented to the supervisor (size test ids ready, stricter cut,
  tumour-free generation, pancreas arms → `notes/BACKLOG.md`). Pancreas: measured only (`notes/RESULTS.md`).
- **Supervisor meeting held**: agreed to run the pancreas baseline (`all` + `all_rep2`), test it, then decide the
  next step from how pancreas behaves (its synthetic scans lack the liver's obvious visual faults).
- **Pancreas baseline done** (2026-11-18 13:57; 238 MSD07 test): tumour Dice 0.183 / 0.172 (rep2 − f0 −0.011, p 0.84),
  pancreas Dice only 0.209 / 0.173 (medians 0.13 / 0.09); much weaker than liver. Numbers `notes/RESULTS.md`.
  **Why it misses** (`reports/figures/pancreas_cases/README.md`, numbers `case_diagnostics/summary.md`): ~80 % of
  the reference gland predicted as background (under-segmentation, no label swap); fails on fat around/in the gland
  (26 % of real test, 2 % of synthetic; gland Dice 0.15 vs 0.30) and 5 mm slices (0.16 vs 0.30); synthetic glands
  are smooth, unlobulated, in uniform grey tissue → an image-appearance gap a volume filter cannot fix.
  All committed (runner, eval summary + per-case CSVs in `reports/data/test_eval_pancreas/`, figures).
- **Pancreas ccPost hurts** (A1, test 238): tumour Dice 0.183 → 0.163 / 0.172 → 0.139, lesion recall 0.55 → 0.35
  (fragmented predicted gland); ccAucPost adds nothing. Numbers `notes/RESULTS.md`.
- **Autopilot running since 2026-11-24 19:00 (UCloud; s6: A2 val screens done, 6 combos promising; A3 frozen
  2026-11-24 20:35: pos+minLes, grow0.1+minLes, grow0.1+tumNear10, grow0.1+pos+minLes + controls; s7: test F1
  pos+minLes not confirmed (+0.007 < 0.01); s8: liver val screen → froze tumTouch, pos, tissueVal; s9: liver test: tumTouch +0.063 and pos +0.013 confirmed,
  tissueVal +0.009 not, nothing adds to ccPost; s10: on the 999 synthetic labels tumTouch changes 598 (twin = ccRepair,
  no new arm), pos 84; s11: B1 pre vs post: ccRepair ≈ baseline+ccPost (−0.006, equivalent), vs baseline+tumTouch +0.009 (CI
  just misses ±0.02); D113 posRepair built; gpu G1 thickSlice (D112 f0, test pred ~03:00 Wed), then test probs for
  F2-F4, then `nnunet/liver_arm.sh posRepair 113 --all-synth`)**: `autopilot/supervisor.sh` (gpu + cpu lane) wakes a new
  headless Claude session whenever a watched job ends; handoff + plan `autopilot/MISSION.md`, journal
  `notes/AUTOPILOT.md`, how-to GUIDE "Optional: autopilot". Direction (user): anatomical rules on the predictions
  first (pancreas ccPost/ccAucPost, then explore on val, freeze, test once), pre-training arms only for confirmed rules;
  GPU meanwhile val probabilities + `thickSlice` (D112, `nnunet/build_thick_dataset.py`). Runs until Fri 27 Nov.
  Fake-tested on the VM; autopilot work goes to branch `autopilot`. Rule ledger `reports/data/post_rules/ledger.md`.
## Next steps
0. **Autopilot running** (since 2026-11-24 19:00, until Fri 27 Nov 05:20): journal `notes/AUTOPILOT.md`, branch
   `autopilot`. Friday: review the branch (summary on top of the journal), then merge into main. Pending: pancreas F2-F4 (need test probs after thickSlice), liver posRepair arm (D113 ready, `nnunet/liver_arm.sh`).
1. **Supervisor deck built** (`Research/5 - Liver filter arms - supervisor slides.pptx`, VM only; 13 main + 8 backup
   slides, speaker notes): user reviews it in PowerPoint, then present. Rebuild: `reports/slides/make_assets.py <dir>`
   (charts + CT crops, scratchpad venv) → `NODE_PATH=… node reports/slides/build_deck.js <dir> <out.pptx>
   <pptx skill>/scripts/apply_theme.js`; needs `nodejs npm libreoffice-impress` (installed on the VM 2026-11-16).
2. **Pancreas baseline done + diagnosed**: discuss with the user (and supervisor) the appearance gap (fat, lobulation,
   thick slices) and options (thick-slice/blur augmentation, MAISI fatty glands, real data) vs training filter arms.
3. Add the checkpoint_best results to report 4 (`reports/liver_results/build.py`; rep2 is in since 2026-11-16).
   Builds need a venv with pandas/matplotlib/Pillow (Kali VM has none; one-off venv in the session scratchpad).

## Decisions in force (full records + reasons: `notes/DECISIONS.md`)
- Option B (automatic filtering of MAISI volumes); training on synthetic only; LiTS = test (111) + real val (20, for
  checkpoint selection); pancreas (MSD07, 43 val / 238 test) as a secondary generalisation check with the liver rules.
- Pool: 999 liver + 999 pancreas, Path A size sweep, 512³ @ 0.8×0.8×1.0, MAISI out of the box (no QC retries).
- Arms: ccReject (343) → ccRepair (999) → ccRepairDel (842) → ccAucRepair (999); AUC cut 0.5475 (lits_049 excluded from
  the real reference); liver HU check dropped as a filter.
- Analysis: exploratory; per-case tumour Dice, baseline = mean of the two `all` runs, Holm over 4 + 0.01 noise floor;
  checkpoint_final primary, checkpoint_best secondary; ccPost/ccAucPost exploratory (declared after the results).
- Supervisor meeting (2026-11-17): run + test the pancreas baseline first; next steps decided from its behaviour.
- Autopilot (2026-11-24): rules on predictions first (val screen, test confirms frozen rules), then pre arms for winners;
  `autopilot/MISSION.md`, branch `autopilot`.

## Open questions
- Report the main results with or without ccPost (main analysis was fixed as 'saved predictions' before results)?
- Pancreas arms: which to train; msd_025 (isodense, sets cut 0.509): keep (recommended) or exclude?
- MIDL full paper (~3 Dec, soft) or short paper (~15 Apr)?
- UCloud job time limit (unknown).

## Where things are
- `notes/RESULTS.md` finished experiments + numbers · `notes/DECISIONS.md` all decisions with reasons ·
  `notes/LOG.md` full history (grep it) · `notes/BACKLOG.md` parked ideas · `notes/PAPER.md` what the paper must
  say/defend · `notes/MAISI.md` MAISI gotchas, versions, thesis numbers, data paths.
- Plans: `plans/PLAN_training.md` (filter rules, evaluation), `plans/PLAN_followups.md` (ccAucPost, size split).
- UCloud how-to: `maisi/GUIDE.md`. Figure styles: `maisi/viz.py` (CT), `reports/style/` (charts, diagrams, reports).
- Reports (PDFs in `Research/`, VM/Mac only, numbered `N - Title.pdf`): 1 plan (claude.ai/artifact/UBHmWwPop8ZisdTPimgMoG),
  2 liver measurement findings (claude.ai/artifact/M6LL95w17E8qF9hyekgFyE), 3 pancreas findings
  (claude.ai/artifact/8BbNZK8t6NuPZzd88W1xt9), 4 liver test results (`reports/liver_results/`). Blind review page:
  claude.ai/artifact/DARpa2rdB7a6fYBsu5PBbv.

## Log
- 2026-11-24 23:55 (UCloud autopilot s11): B1 done (exit 0): ccRepair vs baseline+ccPost −0.006 [−0.017, +0.003] (equivalent ±0.02), vs baseline+tumTouch +0.009 [−0.002, +0.021] (not quite), vs baseline +0.072 → pre ≈ post. D113 posRepair built + preprocessed (84 changed). New `nnunet/liver_arm.sh`. Nothing started (cpu: no waiting work). thickSlice epoch 812, train ETA 00:43. Next: thickSlice end → test probs 102 0 1 → posRepair.
- 2026-11-24 22:55 (UCloud autopilot s10): B0 done (exit 0): synthetic liver labels changed by tumTouch 598/999 (= ccRepair's tumour part, no new arm), pos 84 (0.35 % tumour voxels). New `nnunet/pre_vs_post_liver.py`, `nnunet/build_rule_repair_dataset.py`; cpu `B1_liver_prepost_D113` (`$PROJ/logs/B1_liver_prepost_D113_20261124_224625.log`: ccRepair vs baseline+tumTouch, then D113 posRepair build + preprocess, ~2 h). thickSlice epoch 573. Next: B1 result; write liver_arm.sh; thickSlice end → test probs.
- 2026-11-24 22:35 (UCloud autopilot s9): A3L liver test done (exit 0): tumTouch +0.063 [+0.046, +0.081] and pos +0.013 [+0.007, +0.019] confirmed (Holm), tissueVal +0.009 not; ccPost+pos = ccPost; open1 +0.011, rand_tumTouch −0.187. New `nnunet/label_rule_counts.py`; cpu `B0_liver_label_counts` (`$PROJ/logs/B0_liver_label_counts_20261124_222754.log`, ~1 h). thickSlice epoch 525, train ETA 00:40. Next: counts → liver repair arm or 'generator obeys'; thickSlice end → test probs (F2-F4).
- 2026-11-24 21:20 (UCloud autopilot s8): A2L liver val done (exit 0): ccPost +0.097/+0.113, tumTouch +0.084/+0.097, pos +0.027/+0.024, tissueVal +0.018/+0.020 promising; ccPost+X = ccPost (nothing adds to the cc rule); controls ≤ +0.009 or negative. Frozen FL1-3 tumTouch / pos / tissueVal (journal s8); new `nnunet/post_rules_test_liver.py`; cpu `A3L_liver_test` (`$PROJ/logs/A3L_liver_test_20261124_211829.log`). Next: liver test result; thickSlice end → test probs (F2-F4).
- 2026-11-24 20:45 (UCloud autopilot s7): A3a done (exit 0): val reproduces the screen; test F1 pos+minLes +0.007 [+0.002, +0.012] (f0 +0.012, f1 +0.002) < 0.01 floor → **not confirmed** (lesion recall 0.55 → 0.46). `post_rules.py --target liver` (new); cpu `A2L_liver_val` (`$PROJ/logs/A2L_liver_val_20261124_204456.log`). Next: liver screen result; thickSlice end → test probs (F2-F4).
- 2026-11-24 20:35 (UCloud autopilot s6): A2b done (exit 0): 6 combos promising (pos+minLes +0.015/+0.006, grow0.1+tumNear10 +0.014/+0.019, ...); thr0.1 control = grow for the gland. A3 frozen (journal s6, e6b5945): 4 rules + controls; new `nnunet/post_rules_test.py`, `predict_test_probs.sh`; cpu `A3a_frozen` (`$PROJ/logs/A3a_frozen_20261124_201746.log`). Next: A3a result; after thickSlice: test probs on gpu.
