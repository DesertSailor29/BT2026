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

## Next steps
1. **Slide deck for the supervisor meeting** (before anything else): (1) user gives feedback on report 4; (2) Claude
   interviews the user (audience, length, which findings, format); (3) build it.
2. Add the checkpoint_best results to report 4 (`reports/liver_results/build.py`; rep2 is in since 2026-11-16).
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
- 2026-11-16 (UCloud): Compared checkpoint_best vs final (numbers → `notes/RESULTS.md`): ccReject drops 0.382 → 0.333 at best, i.e. no late overfitting, late training helps the cc arms; baseline 0.303 → 0.315; AUC arms beat the baseline at both checkpoints (best +0.063 / +0.066 Holm 1e-6), under ccPost at best cc arms below baseline (Holm ≤ 0.015), AUC arms n.s. Committed model_diff figures per user. GPU idle, session can be stopped.
- 2026-11-16 13:34 (UCloud, auto): checkpoint_best predictions done; plain eval + ccPost + ccAucPost on best (runs: 101:0 101:1 101:2 103:0 103:1 104:0 104:1; missing: none; ok=1) → `reports/data/test_eval_liver_best/`, log `/work/Research/logs/final_then_best_postproc_20261115_224005.log`. GPU idle: session can be stopped once checked.
- 2026-11-16 08:35 (Kali VM): Per user added ccAucRepair_rep2 + its post-processing to report 4 (`make_charts.py`: rep2 in every per-run chart, hollow/dashed, + a ccAucRepair noise row in the gain chart; `report.html`: tiles, main findings (new item: run-to-run noise), tables in sections 2, 4, 5, 8, sweep, caveats, next steps; Holm in the postproc tables now over 5 as in summary.md: f0 ccPost 0.003, ccAucPost 0.043). Rebuilt `Research/4 - Liver filter arms - test results.pdf` (19 pages). New derived numbers → `notes/RESULTS.md` (rep2 − f0 −0.022 p 7e-5; mean-of-runs readout +0.000). Uncommitted.
- 2026-11-16 (Kali VM): Per user split the notes (NOTES.md 143 KB → 7 KB, CLAUDE.md 11 → 4.5 KB; ~38k → ~3k tokens per session): details in `notes/` (RESULTS, DECISIONS, LOG, BACKLOG, PAPER, MAISI), plans in `plans/`, status-file conventions in GUIDE "Job conventions". Removed: done/superseded Next steps (incl. 00a), stale Current state bullets, superseded decisions (MIG 10-19, 11-12 next-GPU-session, touch v2, 11-07 Q answers), resolved open questions, ~36 noise/duplicate log lines (October MIG/OOM/bench summarised in one line), `PLAN_training_questions.md`, `PLAN_filtering_options.md`, `CLAUDE.worklog.md`. Standing UCloud push permission now covers `NOTES.md` + `notes/`.
- 2026-11-16 08:15 (UCloud): Checked: final-checkpoint postproc (ccPost + ccAucPost, 7 runs incl. 104:1) done 03:29 and pushed (96cd183). rep2 confirms the direction but weaker: saved 0.381 (+0.079 vs baseline, like ccReject); under ccPost +0.014 [0.003, 0.025] Holm p 0.092 (ccAucRepair f0 +0.031, Holm 0.0027); under ccAucPost0.5475 +0.014 p 0.18 (f0 +0.023 Holm 0.043); readout rep2 ccPost − baseline ccAucPost0.5475 = −0.008 p 0.053 (f0 +0.009 p 0.42) → the AUC-training edge over AUC post-processing is not robust across the two runs. Per user, new `nnunet/model_diff_figures.py` (uncommitted): one figure per case, CT | reference | all 7 runs as saved, full slice + zoom on the largest disagreement blob; cases by fixed rules (6 largest Dice range, largest ± diff for 6 arm pairs incl. 2 noise pairs, tumour-free case with largest FP range) → `reports/figures/model_diff/` (19 PNGs + README, 9.5 MB, uncommitted). Log `$PROJ/logs/model_diff_figures_20261116_081253.log`. Seen: lits_088 ccReject misses the large hilar tumour (0.03 vs 0.54–0.76); lits_104 most runs label a large hypodense lesion as nothing; lits_034 (tumour-free) right renal mass called tumour by all runs, 440 ml in `all_rep2`. Best-ckpt chain 485/777.
- 2026-11-16 03:29 (UCloud, auto): ccPost + ccAucPost (checkpoint_final) rerun with all 7 runs incl. ccAucRepair_rep2 (ok=1); `reports/data/test_eval_liver/{fp_location,auc_post,auc_post_val}/summary.md`, log `/work/Research/logs/final_then_best_postproc_20261115_224005.log`. Best-checkpoint chain resumed.
