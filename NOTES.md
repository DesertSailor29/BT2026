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
  `nnunet/pancreas_all.sh` (sequential edit) and `reports/data/test_eval_pancreas/` still uncommitted.
  GPU idle: UCloud session can be stopped.

## Next steps
1. **Supervisor deck built** (`Research/5 - Liver filter arms - supervisor slides.pptx`, VM only; 13 main + 8 backup
   slides, speaker notes): user reviews it in PowerPoint, then present. Rebuild: `reports/slides/make_assets.py <dir>`
   (charts + CT crops, scratchpad venv) → `NODE_PATH=… node reports/slides/build_deck.js <dir> <out.pptx>
   <pptx skill>/scripts/apply_theme.js`; needs `nodejs npm libreoffice-impress` (installed on the VM 2026-11-16).
2. **Pancreas baseline done**: discuss with the user (and supervisor) what the weak pancreas organ Dice (0.19) means
   and whether to train filter arms; commit `nnunet/pancreas_all.sh` + `reports/data/test_eval_pancreas/` (ask).
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
- 2026-11-18 15:02 (UCloud): Per user: figures of succeeding/failing pancreas test cases + why they miss. New `nnunet/pancreas_case_figures.py` (CPU): per-case diagnostics (where the reference gland goes, recall by gland third, HU/size/spacing vs synthetic), charts → `reports/data/test_eval_pancreas/case_diagnostics/`, CT figures → `reports/figures/pancreas_cases/`. Log `/work/Research/logs/pancreas_case_figures_20261118_150139.log`, status `PANCREAS_case_figures__*`.
- 2026-11-18 14:55 (UCloud): Pancreas `all` + `all_rep2` done 13:57 (f0 train done 04:07, f1 09:25; val Dice 0.300/0.294). Test 238: tumour Dice 0.183/0.172 (diff −0.011 [−0.024, +0.002], p 0.84), pancreas Dice 0.209/0.173, lesion recall 0.55/0.52, FP comps/case 17.0/5.3. No errors in log. GPU idle, session can be stopped.
- 2026-11-17 22:41 (UCloud): Per user: supervisor meeting agreed pancreas baseline → test → decide. Stopped the first run during build (399/999 written; 8 partial `.tmp.nii.gz` removed) and changed `pancreas_all.sh` so nothing shares the GPU (it predicted fold 0 during fold-1 training, which slowed liver training 2.2×): train both folds, then predict both, then eval. Restarted, resumed with 399 skipped. Log `/work/Research/logs/pancreas_all_20261117_224044.log`. ETA ~Wed 10:30-11:00.
- 2026-11-17 22:38 (UCloud): Per user started pancreas `all` + `all_rep2` (Dataset102, folds 0/1): `setsid nohup … nnunet/pancreas_all.sh`, log `/work/Research/logs/pancreas_all_20261117_223800.log`; alive, step 1 build (999 synth + 43 val, 238 test) running, status `PANCREAS_all__0of2562`. ~12-14 h on the full B200.
- 2026-11-17 21:50 (Kali VM): Per user merged branch `pancreas` into main (clean; replayed linearly as ac8e2b0..f0b21f5 by pull --rebase, pushed 2026-11-17 with 5f7ab2c) and made the pancreas `all` training ready: `build_dataset.py` real split for pancreas = `maisi/msd07_split.json` (43 val / 238 test, cases `msd_NNN`), shared helpers `real_case` / `real_files` / `target_of`; `make_splits.py` val names per target; `predict_test.py --target pancreas` (Dataset102, organ metrics `pancreas_*`, no thesis subset, baseline 102:0 + 102:1, out `test_eval_pancreas`; liver stays default); `training_curves.py` one chart per target; new runner `nnunet/pancreas_all.sh`. Fake-tested in the scratchpad (fake MSD07 + pool + fake nnUNetv2_*): full run, rerun skips all, partial resume (5 raw cases, fold 1 --c, missing predictions); liver regression old vs new: split, dataset.json, summary/confusion/per-case CSVs, training curves identical. Uncommitted.
- 2026-11-16 16:25 (Kali VM): Per user, interviewed on report 4 (supervisor only, ~15 min + discussion; goals: progress, why these filters, what led to post-processing and what it showed, decisions: size-split arms, ccPost in main results + MIDL track, how interesting "post-processing lifts the raw set to filtered quality" is; dislikes in report 4: small chart text, hollow/dashed rep2 markers). Built the deck: `reports/slides/make_assets.py` (simplified charts in reports/style colours, repeat runs as solid labelled rows, checkpoint_best chart new) + `build_deck.js` (pptxgenjs, Arial) → `Research/5 - Liver filter arms - supervisor slides.pptx`. Numbers re-derived from the CSVs, match RESULTS.md; deck uses 0.021 for the ccAucRepair run-to-run gap (exact −0.0215; report 4 text says 0.022). Uncommitted.
