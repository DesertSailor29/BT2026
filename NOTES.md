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
  just misses ±0.02); D113 posRepair built; s12: thickSlice test −0.016 [−0.030, −0.002] vs baseline (pancreas
  Dice ±0; thick-slice scans' gland +0.03 only) → no gain, no rep2; s13: G2 test probs done; s14: pancreas test
  F2-F4 (grow0.1 + minLes / tumNear10 / pos+minLes) confirmed +0.020/+0.020/+0.021, pancreas Dice +0.05, but grow ≈
  plain thr0.1 (anatomy adds nothing: a threshold fix + small-piece removal); s15: B0p pancreas label counts: minLes changes 131 (all become tumour-free) → minLesRepair arm, pos 5 (generator obeys); s16: D114 built + preprocessed (131 changed, 1042 pkl); s17: liver posRepair test −0.011 [−0.024, +0.001] vs baseline (n.s.); s18: posRepair pre vs post Q1 −0.024 [−0.039, −0.010] vs baseline+pos (post better, not equivalent; one run); s20: pancreas minLesRepair test −0.008 [−0.022, +0.006] vs baseline n.s. (pancreas Dice −0.010, recall 0.43, FP pieces 1.3/case); s21: pancreas M1 minLesRepair vs baseline+minLes −0.014 [−0.028, −0.000] (post better, like liver pos Q1; one run) → minLesRepair_rep2 queued; s22: liver posRepair_rep2 test +0.017 [+0.004, +0.030] vs baseline (f0 −0.011: runs differ 0.028, not reproducible); s23: liver pos Q1 on the 2-run mean −0.010 [−0.023, +0.002], runs −0.024/+0.004 → "pre ≠ post" does not stand (pre ≈ post on the mean); s24: pancreas minLesRepair_rep2 test +0.029 [+0.015, +0.044] vs baseline (f0 −0.008: runs differ 0.037); s25: pancreas minLes pre vs post rep2: M1 runs −0.014/+0.023, 2-run mean +0.005 [−0.008, +0.017] equivalent → "pre ≠ post" does not stand on either organ; s26: liver all_rep3 0.301 vs baseline 0.303 (−0.001 [−0.009, +0.006]) → liver baseline stable (3 runs range 0.004), repaired arms vary more; gpu pancreas all_rep3 (3rd pancreas baseline, D102 f2) → ~02:55 Fri)**: `autopilot/supervisor.sh` (gpu + cpu lane) wakes a new
  headless Claude session whenever a watched job ends; handoff + plan `autopilot/MISSION.md`, journal
  `notes/AUTOPILOT.md`, how-to GUIDE "Optional: autopilot". Direction (user): anatomical rules on the predictions
  first (pancreas ccPost/ccAucPost, then explore on val, freeze, test once), pre-training arms only for confirmed rules;
  GPU meanwhile val probabilities + `thickSlice` (D112, `nnunet/build_thick_dataset.py`). **Extended by 48 h (user,
  2026-11-26 21:00) to Sun 29 Nov 05:20 for the liver** (MISSION "Liver extension": L1 failure analysis + 3-run vote,
  L2 liver probability rules, L3 tumour-size arms Large/Rand/Small + rep2 of Large/Rand).
  Fake-tested on the VM; autopilot work goes to branch `autopilot`. Rule ledger `reports/data/post_rules/ledger.md`.
## Next steps
0. **Autopilot running** (since 2026-11-24 19:00, extended: UCloud job ends ~Sun 29 Nov 06:40, supervisor deadline 05:20; more-time requests go to
   `TIME_REQUEST.md` at the repo root): journal `notes/AUTOPILOT.md`,
   branch `autopilot`; plan `autopilot/MISSION.md` "Liver extension". Running: gpu pancreas `all_rep3` (~02:55 Fri,
   s26 verdict), cpu `L1_liver_failure` (`$PROJ/logs/L1_liver_failure_20261126_210047.log`). Then gpu L2 (liver val +
   test probabilities) → L3 size arms (~7 h each, serial). Sunday: review the branch (summary on top), merge into main.
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
- 2026-11-26 21:05 (UCloud, with user): user added 48 h for liver work. MISSION.md "Liver extension" (L1-L3, comparisons declared); new `nnunet/liver_failure.py` (fake-tested on 3 cases, matches the eval's per-case Dice); started cpu `L1_liver_failure` (`$PROJ/logs/L1_liver_failure_20261126_210047.log`); supervisor restarted with deadline Sun 29 Nov 05:20, AP_MAX_SESSIONS 90 (`$PROJ/logs/autopilot_20261126_210106.log`).
- 2026-11-26 20:50 (UCloud autopilot s27): check: gpu pancreas_all_rep3 healthy (epoch 304, train ETA 00:10, end ~02:55); cpu idle; wake 00:40.
- 2026-11-26 19:25 (UCloud autopilot s26): liver_all_rep3 done (exit 0): D101 f3 val liver/tumour 0.856/0.305; test tumour 0.301 vs baseline 0.303, −0.001 [−0.009, +0.006] p 0.82 → liver baseline stable (s24 rule). Started gpu `pancreas_all_rep3` (`$PROJ/logs/pancreas_all_rep3_20261126_191859.log`, ~7.4 h).
- 2026-11-26 12:35 (UCloud autopilot s25): B1p_pancreas_prepost_minLes_rep2 done (exit 0): 114:1 M1 +0.023 [+0.008, +0.039]; 2-run mean M1 +0.005 [−0.008, +0.017] p 0.62, M2 +0.003 (both within ±0.02) → pancreas "pre ≠ post for minLes" does not stand (s21 rule). gpu all_rep3 healthy (epoch 56, train ETA 17:04). Nothing started.
- 2026-11-26 12:20 (UCloud autopilot s24): pancreas_minLesRepair_rep2 done (exit 0): D114 f1 val pancreas/tumour 0.136/0.160; test tumour 0.207 vs 0.177, +0.029 [+0.015, +0.044] p 3e-5 (f0 −0.008 → runs differ 0.037). Started gpu `liver_all_rep3` (`$PROJ/logs/liver_all_rep3_20261126_121504.log`, ~6.3 h), cpu `B1p_pancreas_prepost_minLes_rep2` (`$PROJ/logs/B1p_pancreas_prepost_minLes_rep2_20261126_121504.log`).
- 2026-11-26 05:25 (UCloud autopilot s23): B1_liver_prepost_pos_rep2 done (exit 0): 113:1 Q1 +0.004 [−0.010, +0.018]; 2-run mean Q1 −0.010 [−0.023, +0.002] p 0.15, Q2/Q3/R2 within ±0.02 → liver "pre ≠ post for pos" does not stand (s20 rule). gpu minLesRepair_rep2 healthy (epoch 99, ETA ~12:00). Nothing started.
