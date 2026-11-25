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
  plain thr0.1 (anatomy adds nothing: a threshold fix + small-piece removal); s15: B0p pancreas label counts: minLes changes 131 (all become tumour-free) → minLesRepair arm, pos 5 (generator obeys); s16: D114 built + preprocessed (131 changed, 1042 pkl); gpu liver posRepair arm (D113) running, ~13:15; then gpu pancreas minLesRepair (D114, ~7.5 h, test probs in the same pass))**: `autopilot/supervisor.sh` (gpu + cpu lane) wakes a new
  headless Claude session whenever a watched job ends; handoff + plan `autopilot/MISSION.md`, journal
  `notes/AUTOPILOT.md`, how-to GUIDE "Optional: autopilot". Direction (user): anatomical rules on the predictions
  first (pancreas ccPost/ccAucPost, then explore on val, freeze, test once), pre-training arms only for confirmed rules;
  GPU meanwhile val probabilities + `thickSlice` (D112, `nnunet/build_thick_dataset.py`). Runs until Fri 27 Nov.
  Fake-tested on the VM; autopilot work goes to branch `autopilot`. Rule ledger `reports/data/post_rules/ledger.md`.
## Next steps
0. **Autopilot running** (since 2026-11-24 19:00, until Fri 27 Nov 05:20): journal `notes/AUTOPILOT.md`, branch
   `autopilot`. Friday: review the branch (summary on top of the journal), then merge into main. Running: gpu liver posRepair arm (D113, `nnunet/liver_arm.sh`); then cpu `pre_vs_post_liver.py --rule pos` + gpu `PANCREAS_TEST_PROBS=1 pancreas_arm.sh minLesRepair 114 --all-synth` → `pre_vs_post_pancreas.py --pre 114:0` (M1-M3 declared s14).
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
- 2026-11-25 08:55 (UCloud autopilot s16): B2p done (exit 0): D114 PancreasSynthRepairMinLes 131/999 changed (tumour −0.90 %, 131 tumour-free), preprocessed 1042/1042. Code: `predict_test.py predict --probs` / `PANCREAS_TEST_PROBS=1` (test probs in the arm's own prediction pass), `pre_vs_post_liver.py --rule pos` (tumTouch output byte-identical), new `pre_vs_post_pancreas.py` (M1-M3). Nothing started (gpu busy: posRepair epoch 265, ETA 12:12). Next: posRepair eval + gpu minLesRepair.
- 2026-11-25 08:15 (UCloud autopilot s15): B0p done (exit 0): on the 999 synthetic pancreas labels minLes changes 131 (each a single tumour < 1.19 ml → tumour-free), pos 5 (generator obeys), ccPost 88. `build_rule_repair_dataset.py` extended (`--rule minLes` → D114, liver output byte-identical); cpu `B2p_build_D114` (`$PROJ/logs/B2p_build_D114_20261125_081111.log`, ~1-1.5 h). posRepair training epoch ~150, ETA 12:11. Next: posRepair eval + gpu `pancreas_arm.sh minLesRepair 114 --all-synth`.
- 2026-11-25 07:55 (UCloud autopilot s14): A3b done (exit 0): pancreas test F2 grow0.1+minLes +0.020 [+0.013, +0.028], F3 +0.020, F4 +0.021 confirmed (Holm, both runs), pancreas Dice +0.050; F1 +0.007 not; grow − thr ±0.001 (the plain threshold does the same). `label_rule_counts.py --target pancreas`; cpu `B0p_pancreas_label_counts` (`$PROJ/logs/B0p_pancreas_label_counts_20261125_075415.log`). Next: counts → minLesRepair arm (declared M1-M3, journal s14) or 'generator obeys'; posRepair eval.
- 2026-11-25 07:25 (UCloud autopilot s13): G2 done (exit 0, 238/238 probs per fold). cpu `A3b_frozen_full` (`$PROJ/logs/A3b_frozen_full_20261125_072131.log`, F1-F4 + controls on test, Holm over 4); gpu `liver_posRepair` (`$PROJ/logs/liver_posRepair_20261125_072131.log`, D113 train + test, ~7 h). Next: A3b verdicts; posRepair eval (s10 declarations).
- 2026-11-25 03:15 (UCloud autopilot s12): thickSlice done (exit 0): test tumour 0.161 vs 0.177, −0.016 [−0.030, −0.002] p 0.023; pancreas +0.000; val 0.151 (baseline 0.151/0.125) → no rep2. gpu `G2_testprobs_102` (`$PROJ/logs/G2_testprobs_102_20261125_031312.log`, ~4.4 h). Next: probs done → post_rules_test.py full (F2-F4) on cpu + liver_arm.sh posRepair on gpu.
- 2026-11-24 23:55 (UCloud autopilot s11): B1 done (exit 0): ccRepair vs baseline+ccPost −0.006 [−0.017, +0.003] (equivalent ±0.02), vs baseline+tumTouch +0.009 [−0.002, +0.021] (not quite), vs baseline +0.072 → pre ≈ post. D113 posRepair built + preprocessed (84 changed). New `nnunet/liver_arm.sh`. Nothing started (cpu: no waiting work). thickSlice epoch 812, train ETA 00:43. Next: thickSlice end → test probs 102 0 1 → posRepair.
