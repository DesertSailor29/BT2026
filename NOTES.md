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
  plain thr0.1 (anatomy adds nothing: a threshold fix + small-piece removal); s15: B0p pancreas label counts: minLes changes 131 (all become tumour-free) → minLesRepair arm, pos 5 (generator obeys); s16: D114 built + preprocessed (131 changed, 1042 pkl); s17: liver posRepair test −0.011 [−0.024, +0.001] vs baseline (n.s.); s18: posRepair pre vs post Q1 −0.024 [−0.039, −0.010] vs baseline+pos (post better, not equivalent; one run); s20: pancreas minLesRepair test −0.008 [−0.022, +0.006] vs baseline n.s. (pancreas Dice −0.010, recall 0.43, FP pieces 1.3/case); s21: pancreas M1 minLesRepair vs baseline+minLes −0.014 [−0.028, −0.000] (post better, like liver pos Q1; one run) → minLesRepair_rep2 queued; s22: liver posRepair_rep2 test +0.017 [+0.004, +0.030] vs baseline (f0 −0.011: runs differ 0.028, not reproducible); s23: liver pos Q1 on the 2-run mean −0.010 [−0.023, +0.002], runs −0.024/+0.004 → "pre ≠ post" does not stand (pre ≈ post on the mean); s24: pancreas minLesRepair_rep2 test +0.029 [+0.015, +0.044] vs baseline (f0 −0.008: runs differ 0.037); s25: pancreas minLes pre vs post rep2: M1 runs −0.014/+0.023, 2-run mean +0.005 [−0.008, +0.017] equivalent → "pre ≠ post" does not stand on either organ; s26: liver all_rep3 0.301 vs baseline 0.303 (−0.001 [−0.009, +0.006]) → liver baseline stable (3 runs range 0.004), repaired arms vary more; gpu pancreas all_rep3 (3rd pancreas baseline, D102 f2) → ~02:55 Fri; s29: L1 liver: misses are small lesions (recall 0.1-1 ml ~0.2, ≥ 10 ml ~0.95), ccPost gains in small/medium-burden cases, 3-run vote +0.009 < +0.01 → not met; s32: L2 liver probability rules null; s35/36: L3 sizeLarge test 0.419 (one run; vs ccAucRepair +0.027 [+0.005, +0.049], same lesion recall, far fewer FP pieces); s37: sizeRand test 0.426 (one run, ≥ sizeLarge); s38: Large − Rand −0.006 [−0.027, +0.013], 1 run each; s39: sizeSmall test 0.106 (−0.197 vs baseline; misses lesions of every size); s40: sizeSmall vs sizeRand −0.320, covers ~0 of large lesions; s41: Large_rep2 test 0.423 (f2 0.419, runs differ 0.004); s42: Large(2) − Rand(1) −0.005 [−0.021, +0.012], not met yet; s43: Rand_rep2 test 0.395 (f3 0.426, runs differ 0.031); s44: final primary Large − Rand (2 runs each) +0.011 [−0.010, +0.031] saved / +0.015 [−0.006, +0.036] ccPost → not met; Small_rep2 (last arm) running; s45: cpu installs TotalSegmentator for optional L1c; s46: install ok, L1c phase running on CPU; s47: L1c done: test 0 non-contrast/3 art/45 PV/63 late vs pool 612/45/2/340 → contrast-phase gap, descriptive)**: `autopilot/supervisor.sh` (gpu + cpu lane) wakes a new
  headless Claude session whenever a watched job ends; handoff + plan `autopilot/MISSION.md`, journal
  `notes/AUTOPILOT.md`, how-to GUIDE "Optional: autopilot". Direction (user): anatomical rules on the predictions
  first (pancreas ccPost/ccAucPost, then explore on val, freeze, test once), pre-training arms only for confirmed rules;
  GPU meanwhile val probabilities + `thickSlice` (D112, `nnunet/build_thick_dataset.py`). **Extended by 48 h (user,
  2026-11-26 21:00) to Sun 29 Nov 05:20 for the liver** (MISSION "Liver extension": L1 failure analysis + 3-run vote,
  L2 liver probability rules, L3 tumour-size arms Large/Rand/Small + rep2 of Large/Rand).
  Fake-tested on the VM; autopilot work goes to branch `autopilot`. Rule ledger `reports/data/post_rules/ledger.md`.
## Next steps
0. **Autopilot running** (since 2026-11-24 19:00, extended: UCloud job ends ~Sun 29 Nov 06:30, supervisor deadline 05:20; more-time requests go to
   `TIME_REQUEST.md` at the repo root): journal `notes/AUTOPILOT.md`,
   branch `autopilot`; plan `autopilot/MISSION.md` "Liver extension". L2 done (null, s32). L3 sizeLarge 0.419 / rep2 0.423, sizeRand 0.426, sizeSmall 0.106 (baseline 0.303). sizeRand_rep2 0.395. Final primary (s44): Large − Rand +0.011 [−0.010, +0.031], not met. Running: gpu `L3_sizeSmall_rep2` (D104 f7, last arm, training ETA 20:29, test → ~21:15; no further training arm after it, s44); then cpu final size eval + re-run `nnunet/phase_eval.py`, final summary. L1c done (s47, `reports/data/test_eval_liver/phase/`): real test all contrast-enhanced, pool mostly non-contrast (RESULTS). Sunday: review the branch (summary on top), merge into main.
1. **Supervisor deck built** (`Research/5 - Liver filter arms - supervisor slides.pptx`, VM only; 13 main + 8 backup
   slides, speaker notes): user reviews it in PowerPoint, then present. Rebuild: `reports/slides/make_assets.py <dir>`
   (charts + CT crops, scratchpad venv) → `NODE_PATH=… node reports/slides/build_deck.js <dir> <out.pptx>
   <pptx skill>/scripts/apply_theme.js`; needs `nodejs npm libreoffice-impress` (installed on the VM 2026-11-16).
2. **Pancreas baseline done + diagnosed**: discuss with the user (and supervisor) the appearance gap (fat, lobulation,
   thick slices) and options (thick-slice/blur augmentation, MAISI fatty glands, real data) vs training filter arms.
3. **Follow-up "fewer synthetic volumes beat all 999"** (user, 2026-11-28, parked for later): design E1-E4 in
   `notes/BACKLOG.md` (top entry); user decides scope / autopilot / supervisor first, then write a declared plan.
4. Add the checkpoint_best results to report 4 (`reports/liver_results/build.py`; rep2 is in since 2026-11-16).
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
- 2026-11-28 20:20 (UCloud autopilot s47): cpu L1c_phase done (exit 0, 111/111): test aorta median 143 HU (pool 94); phase test 0 NC / 3 art / 45 PV / 63 late vs pool 612 / 45 / 2 / 340 → contrast-phase gap (descriptive). gpu Small_rep2 epoch ~920, test ~21:20. Nothing new started.
- 2026-11-28 19:06 (UCloud autopilot s46): TotalSegmentator install ok (1.4 GB weights, 22 s/case on CPU). Started cpu `L1c_phase` (`$PROJ/logs/L1c_phase_20261128_190619.log`; new `nnunet/phase_segment.sh` + `phase_eval.py`, descriptive). gpu Small_rep2 healthy, ETA 20:34 / test ~21:20.
- 2026-11-28 19:00 (UCloud autopilot s45): gpu Small_rep2 healthy (epoch 666, train ETA 20:34, test ~21:20). Started cpu `L1c_ts_install` (`$PROJ/logs/L1c_ts_install_20261128_190107.log`: venv-ts + TotalSegmentator weights, no GPU) so L1c can use the idle gpu after 21:20.
- 2026-11-28 17:55 (UCloud, user session): user wants to prove "590 subsets > full 999 ccAucRepair" (s44 hint);
  E1-E4 design (incl. confounds: run noise ≈ gap, tumour-free share 30 % vs 18 %) parked in BACKLOG, Next steps 3.
- 2026-11-28 15:50 (UCloud autopilot s44): cpu L3_size_eval done (exit 0): final primary sizeLarge (2) − sizeRand (2) +0.011 [−0.010, +0.031] p 0.14 saved / +0.015 [−0.006, +0.036] p 0.058 ccPost → not met; both subsets > ccAucRepair (+0.029 / +0.018). gpu Small_rep2 epoch 13, ETA 20:29. Nothing new started; wake 19:00.
- 2026-11-28 15:45 (UCloud autopilot s43): gpu L3_sizeRand_rep2 done (exit 0; D104 f6, val tumour 0.348): test tumour 0.395 vs baseline 0.303, +0.093 [+0.063, +0.124]; sizeRand runs 0.426/0.395. Started gpu `L3_sizeSmall_rep2` (`$PROJ/logs/L3_sizeSmall_rep2_20261128_153926.log`, last arm, declared default), cpu `L3_size_eval` (`$PROJ/logs/L3_size_eval_20261128_153926.log`).
