# Autopilot mission (handoff for every unattended session)

Written 2026-11-24 on the Kali VM by the planning session, at the user's request, revised the same day with the user.
The user is away until **Friday 27 Nov** (then reviews what you found and builds on it) and gives **one full B200**
until then. You are one of a chain of short headless sessions. The supervisor (`autopilot/supervisor.sh`) starts you
when a watched job ends, when a wake-up you asked for is due, or when nothing runs. You decide the next step, start
it, record why, push, and end. The next session knows only what you wrote down.

## Authority (what the user granted; overrides CLAUDE.md where they differ)
- You may explore: decide and start experiments within the direction below, change the plan when the evidence says
  so (record why), write and change code, and **commit code + notes and push to branch `autopilot` without asking**.
  Never touch `main` (no checkout, merge or push to it); the user reviews and merges the branch.
- Start every job with `autopilot/ap start [--lane gpu|cpu] <name> '<command>'` (detached, logged, exit code
  recorded, watched). Two lanes: **gpu** (training, prediction; one at a time, nothing shares the GPU) and **cpu**
  (analysis, dataset builds; never touches the GPU; use `nice -n 10` and <= 12 workers while a training runs, it
  needs ~24 cores for data loading). This replaces the raw `setsid nohup` line of CLAUDE.md; status files are still
  expected for long jobs (`job_status.py`).
- Nobody will answer questions. When unsure, take the cautious option that keeps the GPU doing something useful, and
  record the assumption.

## The research direction (from the user, 2026-11-24)
**Working thesis for the paper:** for these filters it does not matter much whether a rule is applied to the
*training labels* (before training) or to the *predictions* (after training); what matters is whether the rule
encodes **real anatomy**. Evidence so far (liver, `notes/RESULTS.md`): the cc rule on the predictions (ccPost) lifts
the baseline 0.301/0.305 → 0.379/0.383, the same gain the cc filter arms got by training; ccAucRepair+ccPost vs
baseline+ccAucPost is n.s. (+0.009 / −0.008). A rule that is not anatomy (the liver HU/border check) did not even
flag bad labels.

A corollary worth testing: a rule can only change something where it is *violated*: in the training labels
(generator errors) or in the predictions (model errors). Pancreas is the contrast case: its synthetic labels mostly
obey the cc rule (911/999 unchanged), but the predictions are poor (gland recall ~0.2, 17 / 5 false-positive pieces
per case). So: find which anatomical rules on the predictions make the results significantly better, and train only
what that tells us is worth training. Post-hoc analysis is cheap (CPU); training is ~8 GPU h per arm.

## Facts you need (details: `NOTES.md`, `notes/RESULTS.md`, `reports/figures/pancreas_cases/README.md`)
- Pancreas baseline `all` / `all_rep2` (D102 folds 0/1, 999 synthetic, checkpoint_final): test (238) tumour Dice
  0.183 / 0.172, pancreas 0.209 / 0.173; val (43 real) Dice 0.300 / 0.294. Main failure: under-segmentation (~80 % of
  the reference gland predicted as background); precision 0.55. Image gaps: fat around the gland, thick slices,
  lobulation (correlations).
- Predictions on disk: test `fold_X/test_final/` (238), val `fold_X/validation/` (43, nnU-Net's own, checkpoint_final),
  under `$nnUNet_results/Dataset102_PancreasSynth/nnUNetTrainer__nnUNetPlans__3d_fullres/`. Real images: D102
  `imagesTr/msd_*` (val), `imagesTs_real/` (test). Liver: the same layout for D101/103/104 (20 val, 111 test), runs
  listed in `nnunet/fp_location.py` RUNS (+ 104:1 = ccAucRepair_rep2).
- Existing post-hoc code (liver only): `nnunet/fp_location.py` (ccPost), `nnunet/auc_post.py` (ccAucPost, cut sweep,
  `--split val`). Training-label rules: `maisi/cc_rules.py` (has `--target pancreas`), `maisi/filter_metrics.py`
  (separability), `nnunet/build_repair_dataset.py` (liver only; pancreas cc caches were measured on the *uncropped*
  pool labels, D102 labels are cropped: mind the offsets). The full 132-class MAISI masks of the synthetic volumes
  (neighbouring organs, vertebrae) are in the pool's raw outputs (`notes/MAISI.md` for paths).
- Pancreas rules.json: AUC cut 0.509 (set by msd_025, an isodense real tumour; keep it, as recommended in NOTES).
- Timings, full B200: 1000-epoch training ~5.3 h, predicting 238 test cases ~2.2 h → **one pancreas arm ~7.5 h**,
  +1-1.5 h to build + preprocess a new dataset (can run on the cpu lane during the previous training). Val
  probabilities (43 cases) ~25 min per run.
- Decisions in force (NOTES.md) hold: synthetic-only training, fixed splits, per-case tumour Dice main metric,
  baseline = per-case mean of the two `all` runs, **test never read for a decision**.

## Plan (~60 h, until Fri 27 Nov ~07:00; GPU and CPU lanes run in parallel)
| when | gpu lane | cpu lane |
|---|---|---|
| start | G0 `predict_val_probs.sh 102 0 1` (~1 h): probabilities for rules that grow the gland | A1 port ccPost + ccAucPost to pancreas, run on val + test (fixed replication of the liver result) |
| ~1 h | G1 `pancreas_arm.sh thickSlice` (~9 h): the image-gap question, independent of the filter question; the GPU would idle otherwise | A2 explore anatomical rules on the **val** predictions (below) |
| ~10-14 h | | A3 freeze the winners, one test evaluation; B-prep: apply the winners to the 999 training labels, count what changes, build + preprocess the datasets |
| ~10-60 h | B: pre-training arms of the confirmed rules (`pancreas_arm.sh <arm> <id> ...`, ~8 h each), most promising first; then second runs of the arms the paper's claim rests on | analysis of each finished arm (pre vs post), liver rules (A2/A3 for liver), report |

**A1 (fixed, no choices):** pancreas ccPost (keep the 26-connected piece of predicted pancreas ∪ tumour that holds the
largest predicted pancreas piece) and ccAucPost (ccPost + tumour pieces with sep < 0.509 → surroundings' label),
same definitions as liver, both baseline runs, val and test. Extend `fp_location.py` / `auc_post.py` with
`--target pancreas` (keep the liver outputs byte-identical; check by re-running a liver table into a scratch dir).
Because the rules are fixed in advance, reading their test result is evaluation, not selection.

**A2 (exploration, val only):** propose and test anatomical rules on the 43 val predictions of both runs (liver:
the 20 val cases of every run, lower priority: smaller). Every rule must name the anatomical fact it encodes and
where its parameters come from (real val labels, the synthetic masks, or anatomy textbooks; never test). Starting
points, not limits:
- *one organ*: ccPost; with a tolerance (thick slices can split the gland into 2 pieces along z).
- *tumour belongs to the organ*: tumour pieces must touch / lie in the predicted gland (MSD07: PDAC, cysts, PNETs
  of the pancreas).
- *tumour is visible*: the AUC rule (ccAucPost) and stricter cuts *chosen on val*.
- *position*: the pancreas lies retroperitoneally, anterior to the spine and aorta around L1-L2, head in the
  duodenal C-loop, tail towards the splenic hilum: remove predicted pieces outside a location prior (from the 43 val
  labels relative to an image landmark such as the spine, which bone HU finds without any model, or from the
  synthetic masks' vertebra labels).
- *tissue*: gland / tumour voxels are soft tissue: drop predicted voxels in fat (< −50 HU) or bone (> 300 HU);
  thresholds from the val labels' HU range.
- *size*: drop tumour pieces smaller than the smallest real val lesion; plausibility of gland volume.
- *grow where the anatomy says the gland continues* (needs G0's probabilities): around the kept gland piece, accept
  pancreas where p(pancreas) exceeds a lower threshold, within a distance / HU window. The only kind of rule that can
  fix under-segmentation, which is the main failure. A post-hoc rule with a pre-training twin is preferred, but a
  post-only rule that helps is still a finding.
- **controls (non-anatomical)** for the same predictions: random removal of the same number of predicted pieces (or
  voxels) as the best rule removes, and a generic morphological opening. The thesis needs them: if they help as much,
  it is not the anatomy.
Keep a ledger of **every** rule and parameter tried, including failures (`reports/data/post_rules/ledger.md`: rule,
anatomical fact, parameter source, val effect per run). That keeps the forking paths visible for the paper.
"Promising" on val = improves per-case tumour Dice (pancreas: also report pancreas Dice) in **both** runs, mean gain
>= 0.01, and loses <= 5 % of the real val lesions.

**A3 (confirmation):** write a journal entry that freezes <= 4 rules per organ + the controls with exact parameters
**before** any test run of them. Then one test evaluation (both runs, Holm over the frozen set, bootstrap CIs, as
`predict_test.py`), everything reported. If a frozen rule needs probabilities on test, predict the 238 test cases with
`--save_probabilities` on the gpu lane (~2.2 h per run, ~50 GB; check `df` first).

**B (pre-training, only for confirmed rules):** apply the rule to the 999 synthetic training labels (as a repair:
same N; a reject arm only if a repair is not definable). If it changes **< 50 volumes**, do not train: record "the
generator already obeys this rule", which is itself an answer about pre vs post. Otherwise build a new dataset
(new id >= 113, images linked, plans copied from D102 like `build_thick_dataset.py`; check the counts), train one
run, predict, evaluate. Comparisons, declared in the journal before the result: pre arm as saved vs baseline + post
rule (equivalence if the 95 % CI of the per-case difference lies within ±0.02), pre arm + post rule vs baseline +
post rule, pre arm vs baseline. A second run (`--same-as`) for any arm whose pre-vs-post comparison carries the claim.
Liver: a pre arm only for a *new* liver rule that is confirmed (cc and AUC are done).

**G1 thickSlice** (`nnunet/build_thick_dataset.py`, D112): the 999 synthetic images re-sliced to the MSD07 val slice
thicknesses, labels at the slab centres, plans from D102. After it: eval vs baseline, and by slice spacing / fat
groups (`reports/data/test_eval_pancreas/case_diagnostics/test_cases.csv`). A second run only if its val score
(mean of pancreas + tumour val Dice) beats the baseline's by >= 0.02 and nothing from B is waiting.

Not in this plan on purpose (record if you reconsider): pancreas ccReject (cc flags only 88/999 pancreas volumes: it
tests a rule against an error this generator rarely makes, before knowing if the rule matters at all; dropped with
the user); new MAISI generation (~24 GPU h per 1000); real data in training; custom trainers / on-the-fly
augmentation (the arms must differ only in data); TotalSegmentator organ masks on the real scans (a model trained on
real data; possible later, but then never use its pancreas output).

## Every session
1. The repo is on branch `autopilot`. `git pull --rebase --autostash origin autopilot` (ignore if offline).
   `autopilot/ap status`. Read the wake-up report in your prompt and the newest entries of `notes/AUTOPILOT.md`.
2. If a job ended: read its log (errors, `VAL Dice` lines, summaries), check it produced what it should, and write the
   numbers into the NOTES Log, `notes/RESULTS.md` (final results, marked exploratory) and the journal.
3. If jobs still run (`check`): status files + log tail (epochs moving? loss not NaN? ETA sane?). Healthy → set the
   next wake-up or let the job end wake you. Hung (no progress > 1 h, GPU idle) → `kill <pid>`, restart (runners
   resume).
4. Keep **both lanes busy** when there is useful work: an idle lane with work waiting is a failure.
5. Decide. Write the journal entry (format below) **before** starting anything. Start with `ap start`, check after a
   few seconds (`ap status`, log tail). Do not sleep or wait for jobs; short CPU analyses (< ~10 min) may run inside
   the session, anything longer goes on the cpu lane. `ap wake-at '+90 minutes'` once after a training starts.
6. Update `NOTES.md` (Current state, Next steps, a Log line `YYYY-MM-DD HH:MM (UCloud autopilot s<N>): ...`; short),
   commit everything you changed (`autopilot: <what>`), `git push origin autopilot`. End with a 3-5 line message
   (the next session sees it).
Keep sessions short (aim < 45 min, hard limit 2 h). New code: small, documented like the existing scripts,
fake-tested on tiny input before a long run; figures follow the CLAUDE.md figure rules (`maisi/viz.py`,
`reports/style/`).

## Journal entry format (`notes/AUTOPILOT.md`, newest first)
```
### s<N> YYYY-MM-DD HH:MM, event <event>
Saw: <what the jobs produced: numbers, errors>
Decision: <what you start or do next, per lane>
Reasons: <why this, from the direction / evidence>
Assumptions: <what you take as true without checking, or could not check>
Alternatives considered: <and why not>
Started: <lane, ap job name, log path, expected end>   (or: nothing, because ...)
```

## Hard rules
- Never delete or overwrite existing results, datasets, predictions, the pool or raw data (`$nnUNet_results`,
  `$nnUNet_raw`/`_preprocessed` of existing datasets, `$MAISI_RUNS`, `$PROJ/raw_data`). New work goes into new
  dataset ids / folds / output dirs. Removing your own partial `.tmp` files is fine.
- Folds are append-only (make_splits.py). Never re-split. **Never use test results to choose** a rule, a parameter
  or what to train: selection on val, test only for frozen rules and finished arms.
- One GPU process at a time. No `sudo`; pip installs only into `$PROJ/venv-*` and only if needed.
- No Claude co-author or "Generated with" lines in commits. Don't commit data, credentials or patient files.
- GPU unusable (CUDA errors on a fresh start, no device) or disk free < 100 GB with no safe way to free it: record
  it and `autopilot/ap stop '<reason>'`.
- If everything worth doing is done before the deadline, write the final summary early and `ap stop` (say so: the
  user can then end the UCloud job to save GPU hours).

## Final session (event `final`, ~1 h before the deadline)
Do not start GPU work. Write the **summary for the user** at the top of `notes/AUTOPILOT.md`: what ran (log paths),
the numbers (val + test, vs baseline, CIs), what they mean for the thesis (pre vs post, anatomical vs controls,
thickSlice), the rule ledger's highlights, what failed, decisions the user should check, and the next steps you
recommend for Friday. Update NOTES.md Current state and Next steps, push, say whether jobs still run and whether the
UCloud session can be stopped, then `autopilot/ap stop final`.

## Assumptions behind this plan (made by the planning session; correct them in the journal if wrong)
- The UCloud job runs until ~Fri 27 Nov 08:00 and the supervisor's deadline is ~1 h before; GPU = one full B200.
- Claude Code is installed and logged in on UCloud (`CLAUDE_CONFIG_DIR=$PROJ/.claude-config`); git can push.
- D102 raw + preprocessed, the baseline folds' `validation/` and `test_final/` predictions, and the pancreas filter
  metrics (rules.json, caches) are on `/work`.
- 43 val cases (20 for liver) are enough to *screen* rules; the test confirmation, not the val screen, carries any
  claim. Everything here is exploratory and will be presented as such.
