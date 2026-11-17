# Project notes

Loaded every session (via CLAUDE.md): keep it short and true. Everything else is in `notes/` and read only when needed
(index at the bottom). Finished results → `notes/RESULTS.md`; old log entries → `notes/LOG.md`.

## Current state (2026-11-16)
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
- UCloud GPU session idle, all jobs done: can be stopped (ends anyway Tue 17 Nov ~13:24).
- **New experiments paused** until the findings are presented to the supervisor (size test ids ready, stricter cut,
  tumour-free generation, pancreas arms → `notes/BACKLOG.md`). Pancreas: measured only (`notes/RESULTS.md`).
- **Pancreas `all` + `all_rep2` ready to run** (2026-11-17, VM, uncommitted): branch `pancreas` merged into main
  (734302d, not pushed); `nnunet/pancreas_all.sh` = build Dataset102 (999 synth + 43 MSD07 val) → preprocess →
  folds 0/1 → train (as liver) → predict 238 MSD07 test → `predict_test.py eval --target pancreas`. Fake-tested
  (full run, resume, partial resume; liver eval/curves/split byte-identical to before). Not started.

## Next steps
1. **Supervisor deck built** (`Research/5 - Liver filter arms - supervisor slides.pptx`, VM only; 13 main + 8 backup
   slides, speaker notes): user reviews it in PowerPoint, then present. Rebuild: `reports/slides/make_assets.py <dir>`
   (charts + CT crops, scratchpad venv) → `NODE_PATH=… node reports/slides/build_deck.js <dir> <out.pptx>
   <pptx skill>/scripts/apply_theme.js`; needs `nodejs npm libreoffice-impress` (installed on the VM 2026-11-16).
2. **Pancreas baseline** (when the user decides to run it; experiments were paused for the supervisor meeting):
   needs main pushed, then on UCloud (full B200, ~12-14 h: build ~0.2 h, preprocess ~0.7 h, 2 × ~5 h train, predict
   overlaps the next training): `git pull`, then detached
   `setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && bash "$PROJ/BT2026/nnunet/pancreas_all.sh"' >
   "$PROJ/logs/pancreas_all_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &` (status files `PANCREAS_all__*`,
   `NNUNET_D102_*`; same command resumes). `PANCREAS_FOLDS=0` = one run only. Result: `$nnUNet_results/test_eval_pancreas/`.
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
- No new experiments until the supervisor meeting (2026-11-15).

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
- 2026-11-17 21:50 (Kali VM): Per user merged branch `pancreas` into main (734302d, clean auto-merge, not pushed) and made the pancreas `all` training ready: `build_dataset.py` real split for pancreas = `maisi/msd07_split.json` (43 val / 238 test, cases `msd_NNN`), shared helpers `real_case` / `real_files` / `target_of`; `make_splits.py` val names per target; `predict_test.py --target pancreas` (Dataset102, organ metrics `pancreas_*`, no thesis subset, baseline 102:0 + 102:1, out `test_eval_pancreas`; liver stays default); `training_curves.py` one chart per target; new runner `nnunet/pancreas_all.sh`. Fake-tested in the scratchpad (fake MSD07 + pool + fake nnUNetv2_*): full run, rerun skips all, partial resume (5 raw cases, fold 1 --c, missing predictions); liver regression old vs new: split, dataset.json, summary/confusion/per-case CSVs, training curves identical. Uncommitted.
- 2026-11-16 16:25 (Kali VM): Per user, interviewed on report 4 (supervisor only, ~15 min + discussion; goals: progress, why these filters, what led to post-processing and what it showed, decisions: size-split arms, ccPost in main results + MIDL track, how interesting "post-processing lifts the raw set to filtered quality" is; dislikes in report 4: small chart text, hollow/dashed rep2 markers). Built the deck: `reports/slides/make_assets.py` (simplified charts in reports/style colours, repeat runs as solid labelled rows, checkpoint_best chart new) + `build_deck.js` (pptxgenjs, Arial) → `Research/5 - Liver filter arms - supervisor slides.pptx`. Numbers re-derived from the CSVs, match RESULTS.md; deck uses 0.021 for the ccAucRepair run-to-run gap (exact −0.0215; report 4 text says 0.022). Uncommitted.
- 2026-11-16 (UCloud): Compared checkpoint_best vs final (numbers → `notes/RESULTS.md`): ccReject drops 0.382 → 0.333 at best, i.e. no late overfitting, late training helps the cc arms; baseline 0.303 → 0.315; AUC arms beat the baseline at both checkpoints (best +0.063 / +0.066 Holm 1e-6), under ccPost at best cc arms below baseline (Holm ≤ 0.015), AUC arms n.s. Committed model_diff figures per user. GPU idle, session can be stopped.
- 2026-11-16 13:34 (UCloud, auto): checkpoint_best predictions done; plain eval + ccPost + ccAucPost on best (runs: 101:0 101:1 101:2 103:0 103:1 104:0 104:1; missing: none; ok=1) → `reports/data/test_eval_liver_best/`, log `/work/Research/logs/final_then_best_postproc_20261115_224005.log`. GPU idle: session can be stopped once checked.
- 2026-11-16 08:35 (Kali VM): Per user added ccAucRepair_rep2 + its post-processing to report 4 (`make_charts.py`: rep2 in every per-run chart, hollow/dashed, + a ccAucRepair noise row in the gain chart; `report.html`: tiles, main findings (new item: run-to-run noise), tables in sections 2, 4, 5, 8, sweep, caveats, next steps; Holm in the postproc tables now over 5 as in summary.md: f0 ccPost 0.003, ccAucPost 0.043). Rebuilt `Research/4 - Liver filter arms - test results.pdf` (19 pages). New derived numbers → `notes/RESULTS.md` (rep2 − f0 −0.022 p 7e-5; mean-of-runs readout +0.000). Uncommitted.
- 2026-11-16 (Kali VM): Per user split the notes (NOTES.md 143 KB → 7 KB, CLAUDE.md 11 → 4.5 KB; ~38k → ~3k tokens per session): details in `notes/` (RESULTS, DECISIONS, LOG, BACKLOG, PAPER, MAISI), plans in `plans/`, status-file conventions in GUIDE "Job conventions". Removed: done/superseded Next steps (incl. 00a), stale Current state bullets, superseded decisions (MIG 10-19, 11-12 next-GPU-session, touch v2, 11-07 Q answers), resolved open questions, ~36 noise/duplicate log lines (October MIG/OOM/bench summarised in one line), `PLAN_training_questions.md`, `PLAN_filtering_options.md`, `CLAUDE.worklog.md`. Standing UCloud push permission now covers `NOTES.md` + `notes/`.
