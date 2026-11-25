# Autopilot journal (newest first)

Unattended sessions on UCloud (`autopilot/supervisor.sh`, handoff `autopilot/MISSION.md`). Each entry: what the
session saw, what it decided, reasons, assumptions, alternatives, what it started. The final session writes a
summary for the user at the top.

### s16 2026-11-25 08:55, event job_finished (B2p_build_D114)
Saw: B2p exit 0 after 0.51 h (`$PROJ/logs/B2p_build_D114_20261125_081111.log`, 0 error lines): Dataset114
PancreasSynthRepairMinLes built, 131/999 labels changed (counts match label_rule_counts), tumour voxels 9685523 →
9598260 (−0.90 %), pancreas voxels removed 0, 131 tumour-free; plans copied from D102; preprocessed 1042/1042 pkl.
gpu: liver_posRepair healthy (epoch 265 at 08:40, 17.3 s/epoch, EMA pseudo Dice 0.71, no NaN; train ETA 12:12,
then test prediction ~1 h).
Decision: no new job (gpu busy, cpu has no waiting work). Prepared code for the next two evaluations:
(1) `predict_test.py predict --probs` + `PANCREAS_TEST_PROBS=1` in `pancreas_arm.sh`: the arm's test prediction
saves probabilities in the same pass (fold_X/test_probs/) and copies the segmentations into test_final/, so M2
(grow0.1+minLes needs probabilities) costs no extra ~2 h GPU prediction. Default behaviour unchanged.
(2) `pre_vs_post_liver.py --rule pos` (posRepair, D113 f0: Q1-Q3 = the s10 declarations, + ccPost references);
the tumTouch default re-run reproduces the committed summary byte-identically.
(3) new `nnunet/pre_vs_post_pancreas.py --pre 114:<f>` (M1-M3 as declared in s14, + references minLes / thr0.1+minLes
marked "not declared"; reuses post_rules_test.py's frozen measure and the A3b baseline caches).
**Next gpu (the session woken by posRepair's end):** cpu `pre_vs_post_liver.py --rule pos --workers 12` (posRepair
eval, ~15 min); gpu `ap start --lane gpu pancreas_minLesRepair 'PANCREAS_TEST_PROBS=1 bash "$REPO/nnunet/pancreas_arm.sh"
minLesRepair 114 --all-synth'` (~7.5 h → ~21:00). Then `pre_vs_post_pancreas.py --pre 114:0` (fold from arms.json).
With ~8 h left after that, the last gpu slot fits one more arm only if started by ~21:30 (train 5.3 h + predict
2 h): candidate = minLesRepair_rep2 (`--same-as minLesRepair`) if M1/M2 carry a claim, else posRepair_rep2 if Q1
carries one, else nothing (write the summary early).
Reasons: MISSION B (declared comparisons, finished arms are evaluation); saving a GPU pass makes a second run fit
before the deadline.
Assumptions: nnUNetv2_predict with --save_probabilities writes the same segmentation as without (G2 vs test_final
argmax agreement 1.0000 over 476 cases, s14); post_rules_test.measure asserts agreement > 0.995 anyway.
Alternatives considered: building a minLes reject dataset now as fallback (not needed unless M1 says the repair
hurts; builds in ~0.5 h then); a separate predict_test_probs.sh run after the arm (~2 h GPU more).
Fake test: predict --probs with a stubbed nnUNetv2_predict (2 cases: probs + copies, resume, default path unchanged);
pre_vs_post_liver --rule pos with ARMS patched to 103:0 on 3 cases; pre_vs_post_pancreas --pre 102:0 --arm all on
238 (caches) and --no-probs on 2 cases. Subset outputs deleted.
Started: nothing (gpu busy until ~13:15; cpu: no waiting work).

### s15 2026-11-25 08:30, event job_finished (B0p_pancreas_label_counts)
Saw: B0p exit 0 after 0.24 h (`$PROJ/logs/B0p_pancreas_label_counts_20261125_075415.log`; spine found 999/999).
On the 999 synthetic D102 labels: **minLes changes 131 volumes** (87263 tumour voxels = 0.90 %, 131 pieces; median
536 changed voxels), pos 5 ("generator obeys"), pos+minLes = minLes (131), ccPost reference 88. Every one of the 131
has a single tumour < 1.19 ml, so the repair makes them **tumour-free** (tumour-free volumes 0 → 131).
`reports/data/post_rules/pancreas_label_counts/`. gpu: liver_posRepair healthy (epoch ~150, 17.3 s/epoch, EMA pseudo
Dice ~0.70, train ETA 12:11, then test prediction ~1 h → ~13:30).
Decision: cpu: build **Dataset114_PancreasSynthRepairMinLes** (`build_rule_repair_dataset.py --rule minLes`, extended
this session: target-aware, plans from D102; `build_repair_dataset.copy_plans` got optional src args, liver default
unchanged) + `nnUNetv2_preprocess -d 114` (~1-1.5 h). gpu: when posRepair ends, the session it wakes starts
`pancreas_arm.sh minLesRepair 114 --all-synth` (~7.5 h) and evaluates posRepair (s10 declarations). minLesRepair
comparisons M1-M3 as declared in s14 (unchanged).
Reasons: MISSION B: minLes is the label-definable part of the confirmed F2 and changes >= 50 volumes → one repair arm.
pos (5) → "the generator obeys pos" on pancreas (a recorded answer, no arm).
Assumptions: relabelling a small *drawn* tumour as background leaves the image showing a lesion labelled background
(the same image/label conflict as ccRepair's removed tumour pieces on liver); MISSION says repair, not reject, when a
repair is definable, so repair. A reject arm (drop the 131) would avoid that conflict: an option if M1 says the
repair hurts. Preprocessing 1042 cases niced next to the liver training does not slow it much.
Alternatives considered: reject arm (MISSION: repair first); waiting with the build until the GPU is free (lane idle).
Fake test: pancreas_0000/0005/0007 → 2 changed (both become tumour-free), counts match B0p, uint8, same affine; liver
`--rule pos` on liver_0000/0014 → labels byte-identical to D113.
Started: cpu `B2p_build_D114` (log path in the NOTES Log; ~1-1.5 h).

### s14 2026-11-25 07:55, event job_finished (A3b_frozen_full)
Saw: A3b exit 0 after 0.5 h (`$PROJ/logs/A3b_frozen_full_20261125_072131.log`; argmax agreement probs vs test_final
1.0000 over 476). **Pancreas test 238, frozen s6 set (Holm over 4, 2-run mean vs saved, none 0.183 / 0.172):**
- F1 pos+minLes NOT confirmed (as s7): +0.007 [+0.002, +0.012] (f1 +0.002).
- **F2 grow0.1+minLes CONFIRMED** +0.022 / +0.018, mean +0.020 [+0.013, +0.028], Holm 1e-10; pancreas Dice +0.050.
- **F3 grow0.1+tumNear10 CONFIRMED** +0.023 / +0.017, mean +0.020 [+0.013, +0.027], Holm 1e-9; pancreas +0.050.
- **F4 grow0.1+pos+minLes CONFIRMED** +0.023 / +0.019, mean +0.021 [+0.014, +0.030], Holm 2e-11; pancreas +0.050.
- But the anatomical part of grow does nothing (declared contrast, expectation "grow ≈ thr" written before test held):
  grow0.1+X − thr0.1+X tumour +0.000 / −0.001 / +0.000, pancreas Dice −0.002 (the plain threshold is slightly
  *better* for the gland); thr0.1+minLes +0.019, thr0.1+tumNear10 +0.021, thr0.1+pos+minLes +0.021. Components alone:
  thr0.1 +0.004 (n.s.), grow0.1 +0.005 (n.s.), minLes +0.006, tumNear10 +0.011 (f1 +0.003), pos +0.004; open1 +0.005
  (pancreas −0.010); random matched removal −0.14. Lesion recall 0.55 → 0.53 / 0.52 → 0.51 (F2/F4).
Meaning: on pancreas the gain is a **calibration fix** (lower the decision threshold to 0.1: gland +0.05, the main
failure is under-segmentation) **plus removing small / far tumour pieces** (which the lower threshold creates more of:
together +0.02, each alone ≤ +0.006). Anatomical limits on the growth (distance, HU window, connectivity to the gland)
add nothing over the threshold. The position prior adds ~0.001. So on pancreas, unlike liver, the confirmed effect is
not anatomy-specific: the anatomical rule (minLes = "smaller than any real lesion", a size fact) only works together
with a generic threshold change; a matched random removal is still strongly harmful (which pieces matter).
Decision: cpu: B-prep for the label-definable parts of the confirmed rules: `label_rule_counts.py --target pancreas`
(extended this session; minLes, pos, pos+minLes, ccPost reference on the 999 D102 synthetic labels). grow / thr act
on probabilities and have no label twin (a twin would be a loss re-weighting = custom trainer, excluded by MISSION).
gpu: unchanged (liver_posRepair, train ETA 12:10, then test prediction; ~14:30). Next gpu job is decided by the
session that sees the counts: if minLes changes >= 50 synthetic labels → pancreas minLesRepair arm (the twin of F2's
label part; declared comparisons below); else "generator obeys minLes" and the GPU goes to a posRepair second run
only if its pre-vs-post comparison carries a claim, otherwise to the best remaining option recorded then.
**Declared now (before any count or arm), if a pancreas minLesRepair arm is trained** (238 test, baseline = per-case
mean 102:0/102:1): M1 arm saved vs baseline+minLes (equivalence ±0.02), M2 arm+grow0.1+minLes vs baseline+grow0.1+minLes
(= F2), M3 arm vs baseline; pancreas Dice reported.
Reasons: MISSION B ("apply the rule to the 999 labels, < 50 changed → no arm"); counting selects nothing. Expectation:
MAISI pancreas tumours are single blobs (cc flagged only 88/999), so minLes likely changes few labels.
Assumptions: the val_label_params / pos_bins of the A2 screen (43 real val labels) are the frozen test-time prior and
size cut; D102 synthetic labels use 0/1/2 like the predictions; spine finder works on the synthetic pancreas images
(3/3 in the fake test).
Alternatives considered: training a "thr0.1 twin" (no label twin exists); a reject arm for minLes (repair definable).
Fake test: pancreas_0000-0002 → runs, 0 changed by any rule, spine found 3/3; liver code path unchanged (RULES for
liver identical, same function calls).
Started: cpu `B0p_pancreas_label_counts` (log path in the NOTES Log; ~15-20 min).

### s13 2026-11-25 07:25, event job_finished (G2_testprobs_102)
Saw: G2 exit 0 after 4.12 h (`$PROJ/logs/G2_testprobs_102_20261125_031312.log`, 0 error lines): fold 0 238/238 and
fold 1 238/238 probability files in `fold_X/test_probs/`. Disk 2.1 PB free.
Decision: cpu: `A3b_frozen_full` = `post_rules_test.py --split test --workers 12` (all frozen F1-F4 + controls, Holm
over the 4, as frozen in s6; nice -n 10 next to training). gpu: `liver_arm.sh posRepair 113 --all-synth` (D113, s10
declarations: posRepair saved vs baseline+pos (equivalence ±0.02), posRepair+pos vs baseline+pos, posRepair vs
baseline; liver Dice reported). Both start now.
Reasons: s6/s12 plan; F2-F4 need exactly these probabilities; the GPU would idle otherwise. A pancreas B arm can only
follow a confirmed rule; F2-F4 all contain grow (needs probabilities, no label twin), so even a confirmation would
not give a pancreas pre arm directly → posRepair does not block anything more useful.
Assumptions: post_rules_test.py asserts probs vs test_final argmax agreement (> 0.995) itself; running it with
12 workers niced next to the liver training slows training little.
Alternatives considered: pancreas thr0.1-label arm (no confirmed rule yet); thickSlice_rep2 (bar not met, s12).
Started: cpu `A3b_frozen_full`, gpu `liver_posRepair` (log paths in the NOTES Log; ~1-2 h / ~7 h).

### s12 2026-11-25 03:15, event job_finished (pancreas_thickSlice)
Saw: G1 thickSlice exit 0 after 7.38 h (`$PROJ/logs/pancreas_thickSlice_20261124_194809.log`; the 2 "error" lines
are table text). nnU-Net val (43 real, pancreas / tumour): 0.151 / 0.150 (baseline mean Validation Dice 0.151 / 0.125
for f0 / f1). **Test 238 (D112 f0 vs baseline mean of 102:0/102:1): tumour Dice 0.161 vs 0.177, −0.016 [−0.030,
−0.002], Wilcoxon/Holm p 0.023 (median diff 0) → worse, not a win**; vs f0 −0.021, vs f1 −0.011 (n.s.); noise pair
f1 − f0 −0.011. Pancreas Dice +0.000 [−0.007, +0.008]; FP tumour pieces/case 8.4 (vs 17.0 / 5.3), lesion recall
0.504 (vs 0.550 / 0.521). Subgroups (exploratory, in-session, `test_cases.csv`): slices >= 4 mm (n 40) pancreas
+0.030, tumour −0.002; < 4 mm (198) pancreas −0.006, tumour −0.019; fat around the gland (hu_shell < 0, n 62)
pancreas −0.003, tumour −0.026. → re-slicing the synthetic images to real slice thicknesses helps the gland a little
on thick-slice scans only and costs tumour Dice on thin ones; it does not close the image gap (fat untouched, as
expected: the re-slicing does not add fat).
Second thickSlice run: no (val score mean of pancreas + tumour 0.151 vs baseline ~0.138: +0.013 < the 0.02 bar;
by the EMA pseudo Dice 0.286 vs 0.300 / 0.294 it is lower).
Decision: gpu: `predict_test_probs.sh 102 0 1` (s6 plan; F2-F4 need test probabilities; ~4.4 h). cpu: nothing (no
analysis waits; the full frozen test run needs the probs). Next: when the probs are done, cpu `post_rules_test.py`
full test run (F1-F4, Holm over 4); gpu right after: `liver_arm.sh posRepair 113 --all-synth` (s10/s11), unless
F2-F4 confirm a pancreas rule with a B twin first (grow needs probabilities: no label twin; so posRepair likely next).
Reasons: MISSION A3 / s6 freeze; the GPU would idle otherwise.
Assumptions: disk fine (2.1 PB free); probs reproduce test_final argmax (asserted later by post_rules_test.py).
Alternatives considered: posRepair first (rejected in s11: pancreas F2-F4 were frozen first; order kept);
thickSlice_rep2 (bar not met).
Started: gpu `G2_testprobs_102` (log path in the NOTES Log; expected end ~07:45).

### s11 2026-11-24 23:55, event job_finished (B1_liver_prepost_D113)
Saw: B1 exit 0 after 1.04 h (`$PROJ/logs/B1_liver_prepost_D113_20261124_224625.log`). Declared comparisons (s10),
liver test 111: P1 ccRepair saved 0.375 vs baseline+tumTouch 0.365 +0.009 [−0.002, +0.021] p 0.14 (not within ±0.02,
by 0.001); P2 +0.012 [+0.000, +0.023]; P3 vs baseline +0.072; R1 ccRepair vs baseline+ccPost −0.006 [−0.017, +0.003]
**equivalent**, R2 (ccPost on both) −0.001 equivalent. ccRepair liver Dice −0.007 vs baseline. → pre ≈ post for the
one-organ / tumour-in-organ rule (strongest evidence yet for the working thesis; tumTouch alone is slightly weaker
than the training-time repair, the full cc rule matches it). D113 LiverSynthRepairPos built: 84 changed (= B0),
tumour voxels −0.35 %, 14 tumour-free (was 3); preprocessed 1019/1019.
gpu: thickSlice healthy (epoch 812, 17.2 s/epoch, EMA pseudo Dice 0.286, train ETA 00:43, then its test prediction
~03:00).
Decision: wrote `nnunet/liver_arm.sh` (the liver twin of `pancreas_arm.sh`: 20 val / 111 test / baseline 101:0+101:1,
dataset must be preprocessed; `pancreas_arm.sh` untouched because it is being executed). GPU queue unchanged:
thickSlice (→ ~03:00) → `predict_test_probs.sh 102 0 1` (F2-F4, ~4.4 h) → `liver_arm.sh posRepair 113 --all-synth`
(~6-7 h), unless a pancreas B arm is confirmed by F2-F4 first. cpu: nothing to start now (no analysis waits: thickSlice
eval needs its test predictions; F2-F4 need test probs).
Reasons: MISSION B / s10 declarations; no queue in `ap`, so the next gpu job starts when thickSlice's end wakes a
session.
Assumptions: the steps 2-5 of liver_arm.sh work as in pancreas_arm.sh (same code with liver constants; make_splits
--list on D113 and the missing-dataset error checked; train/predict not run here).
Alternatives considered: posRepair before the pancreas test probs (rejected: pancreas is the contrast case and F2-F4
were frozen first); idle cpu work (report 4 checkpoint_best is VM work).
Started: nothing (both lanes: gpu busy, cpu has no waiting work).

### s10 2026-11-24 22:55, event job_finished (B0_liver_label_counts)
Saw: B0 exit 0 after 0.22 h (`$PROJ/logs/B0_liver_label_counts_20261124_222754.log`; spine centre found 999/999).
On the 999 synthetic D101 labels: **tumTouch changes 598 volumes** (all lose tumour; 1659281 tumour voxels = 4.64 %,
1701 pieces), **pos changes 84** (79 lose tumour; 124429 voxels = 0.35 %, 123 pieces; median 223 changed voxels),
ccPost (reference) 656 (598 lose tumour, 1659548 voxels, 1704 pieces). `reports/data/post_rules/liver_label_counts/`.
Both are >= 50 → "arm possible" by MISSION B. But tumTouch's label twin = ccRepair's tumour part (same 598 volumes,
tumour voxels differ by 267 of 1.66 M); ccRepair (D103 f0) only adds the 58 liver-only volumes. The generator
violates tumour-in-organ often (60 % of volumes), and the twin of that rule is already trained.
gpu: thickSlice healthy (epoch 573 at 22:38, 17.2 s/epoch, EMA pseudo Dice 0.279, no NaN; train ETA 00:40, then its
test prediction, ~03:00).
Decision: cpu: one chained job `B1_liver_prepost_D113`: (1) new `nnunet/pre_vs_post_liver.py`: frozen A3L variants on
ccRepair's test predictions + the declared comparisons below (finished arm: evaluation, not selection); (2) new
`nnunet/build_rule_repair_dataset.py --rule pos` → **Dataset113_LiverSynthRepairPos** (999 labels, pos repair, plans
copied from D101; the build asserts it changes exactly B0's per-volume counts), then `nnUNetv2_preprocess -d 113`.
No tumTouch arm (D106 "touch" in build_repair_dataset.py would be its twin): ccRepair is it.
gpu: unchanged. After thickSlice: `predict_test_probs.sh 102 0 1` (pancreas F2-F4) as planned; then the liver posRepair
arm (D113, one run; needs a liver runner like pancreas_arm.sh — next session writes `nnunet/liver_arm.sh`), unless a
pancreas B arm is confirmed first (pancreas has priority: the contrast case).
**Declared before reading (tumTouch pre vs post, LiTS test 111, baseline = per-case mean 101:0/101:1):** P1 ccRepair
saved vs baseline+tumTouch (equivalent if 95 % CI within ±0.02), P2 ccRepair+tumTouch vs baseline+tumTouch, P3
ccRepair vs baseline; references R1/R2 = P1/P2 with ccPost. **posRepair (D113) comparisons, declared now:** posRepair
saved vs baseline+pos (equivalence ±0.02), posRepair+pos vs baseline+pos, posRepair vs baseline; liver Dice reported.
Reasons: MISSION B (rule changes >= 50 → twin); training a tumTouch twin would duplicate ccRepair (8 GPU h for a known
answer). pos changes few labels and little tumour, so the arm is a weak but cheap test of the corollary "a rule changes
something only where it is violated" (expected: posRepair ≈ baseline, while pos post-hoc +0.013); the GPU is free
after ~07:30 Wed otherwise.
Assumptions: ccRepair's extra liver-only removals (58 vols) do not matter for the tumour comparison; the val-label pos
prior (union of 20) is the right prior for synthetic labels too; liver preprocessing with 12 procs takes ~1 h.
Alternatives considered: tumTouch arm D106 (rejected: near-duplicate of ccRepair); no pos arm (rejected: GPU would
idle, MISSION B says train when >= 50); a reject arm for pos (repair is definable, so no).
Fake test: build on liver_0000/0014/0023 → 2 changed, counts match B0 (liver_0014 becomes tumour-free: its whole 9768-
voxel tumour lies outside the prior), uint8, same affine; pre_vs_post on lits_000/001 runs.
Started: cpu `B1_liver_prepost_D113` (log path in the NOTES Log; ~1.5-2 h).

### s9 2026-11-24 22:30, event job_finished (A3L_liver_test)
Saw: A3L exit 0 after 1.12 h (`$PROJ/logs/A3L_liver_test_20261124_211829.log`; the 6 "error" lines are table rows
with "rand_"/"nan"). Val sanity reproduces the s8 screen exactly. **Liver test (111 x 101:0 / 101:1, none 0.301 / 0.305):**
- **FL1 tumTouch CONFIRMED**: +0.065 / +0.061, mean +0.063 [+0.046, +0.081], Holm 1e-14; liver Dice unchanged; FP
  tumour pieces 67 → 9 / 46 → 8; lesion recall 0.422 → 0.408 / 0.443 → 0.439.
- **FL2 pos CONFIRMED**: +0.011 / +0.014, mean +0.013 [+0.007, +0.019], Holm 8e-16; liver Dice +0.017.
- **FL3 tissueVal NOT confirmed**: mean +0.009 [+0.002, +0.016] (Holm 5e-5, but < the 0.01 floor).
- References/controls: ccPost +0.078 (0.381, as known), ccTol10 +0.070, ccPost+pos = ccPost exactly,
  ccPost+tissueTxt +0.000; tumNear10 +0.056; tissueTxt +0.004; **open1 +0.011 [+0.007, +0.016] ≈ pos** (a generic
  opening does as much as the position prior for tumour Dice; pos also lifts liver Dice, open1 does not);
  rand_tumTouch −0.187, rand_pos −0.035. Contrast tumTouch − ccPost −0.016 [−0.025, −0.008] (f0 −0.014, f1 −0.018):
  the tumour-in-organ rule alone gets ~80 % of the one-organ gain; nothing adds to ccPost on test either.
Meaning for the thesis: on liver, the anatomical rules that help are those against *disconnected false-positive
pieces* (one organ, tumour in organ); a matched random removal of the same number of pieces is strongly harmful, so it
is which pieces are removed (anatomy), not how many. The position prior is real but small and ≈ a generic opening.
gpu: thickSlice healthy (epoch 525 at 22:24, 17.2 s/epoch, EMA pseudo Dice 0.279; train ETA ~00:40, then its test
prediction in the same runner).
Decision: cpu: B-prep for the confirmed rules (MISSION B): new `nnunet/label_rule_counts.py` applies tumTouch and pos
(frozen s8 parameters, test-time prior = union of 20 val labels) + ccPost as reference to the 999 D101 synthetic
labels and counts changed volumes. Next session: if tumTouch or pos changes >= 50 volumes, build a repair dataset
(new id >= 113) on the cpu lane and queue its training after the pancreas test probs; else record "generator obeys".
gpu: unchanged (thickSlice; then `predict_test_probs.sh 102 0 1` for F2-F4).
Reasons: MISSION B ("only for confirmed rules"; < 50 changed → no arm). tumTouch is a new liver rule distinct from cc
(a tumour piece outside the liver is always also outside the main structure, so its label twin is a subset of
ccRepair's changes: expected small-to-moderate count; the count decides).
Assumptions: the spine-centre finder works on synthetic MAISI images (3/3 found in the fake test); the pos prior from
real val labels is the right one for synthetic labels too (it is the frozen test-time prior).
Alternatives considered: a tumTouch pre arm without counting (MISSION requires the count); a liver pre arm for pos
(≈ open1 control, small gain: low priority even if the count allows); waiting for pancreas F2-F4 before any liver B
(independent lanes).
Fake test: liver_0000-0002: runs, 10 s for 3 volumes; tumTouch = ccPost on these 3 (2 changed), pos 0 changed.
Started: cpu `B0_liver_label_counts` (log path in the NOTES Log, ~15-20 min).

### s8 2026-11-24 21:25, event job_finished (A2L_liver_val)
Saw: A2L liver val screen exit 0 after 0.45 h (`$PROJ/logs/A2L_liver_val_20261124_204456.log`; 20 LiTS val x 101:0 /
101:1, 199 real val lesions; the 4 "error" lines were table rows containing "rand_"/"nan"). none 0.289 / 0.297.
Promising (gain > 0 both runs, mean >= 0.01, <= 5 % of 398 lesion-runs lost): ccPost +0.097/+0.113 (lost 4),
ccTol5/10/20 (+0.092-0.107, lost 2/1/0), ccK2/K3 (~ccPost), **tumTouch +0.084/+0.097 (lost 2)**, tumNear10
+0.083/+0.076 (lost 0), **pos +0.027/+0.024 (liver Dice +0.019/+0.025, lost 0)**, **tissueVal (1..210 HU)
+0.018/+0.020 (lost 0)**. Not promising: tissueTxt +0.007/+0.009, minLes no-op (smallest real val lesion 0.0003 ml),
minLes0.5 lost 22, open1 +0.009/+0.006, dilate1 −0.04, random controls −0.06 to −0.25. **ccPost+minLes / +pos /
+tissueTxt = ccPost exactly**: nothing tried adds to the one-organ rule on liver val. (Table header says "pancreas
Dice": it is liver Dice for this target; cosmetic, the data key is liver_dice.)
gpu: thickSlice healthy (epoch 265 at 21:07, 17.2 s/epoch, EMA pseudo Dice 0.279, no NaN; train ETA ~00:40).

**FREEZE liver (A3, before any test run of these rules; nothing below may change after reading test):**
- Frozen rules, Holm over these 3 (per run and for the per-case mean of the 2 runs), main metric per-case tumour Dice
  vs the saved prediction, liver baseline runs 101:0 / 101:1, checkpoint_final, 111 LiTS test cases:
  FL1 `tumTouch` (liver tumours lie in the liver: drop tumour pieces without a 26-neighbour predicted liver voxel),
  FL2 `pos` (spine-relative 2D position prior, union of all 20 real val liver labels, 2 mm bins, 10 mm margin;
  pieces < 50 % inside -> bg), FL3 `tissueVal` (predicted liver/tumour voxels outside 1..210 HU, the p0.5-p99.5 of the
  real val labels -> bg). Code `nnunet/post_rules.py` functions unchanged, wrapper `nnunet/post_rules_test_liver.py`
  VERSION liverfrozen1.
- Reference / controls (reported, not in Holm): ccPost and ccTol10 (one-organ rule; ccPost's test effect is already
  known from the liver study, so they are references, not confirmations), ccPost+pos, ccPost+tissueTxt (does anatomy
  add to ccPost?), tumNear10, tissueTxt, open1, rand_tumTouch, rand_pos (random tumour pieces matched in count).
- Declared secondary contrasts: tumTouch − ccPost (does the tumour-in-organ rule alone reach the one-organ gain?),
  ccPost+pos − ccPost, ccPost+tissueTxt − ccPost (expectation from val: ≈ 0).
- Confirmed on test = Holm p < 0.05 for the 2-run mean, gain > 0 in both runs and mean gain >= 0.01 (as pancreas).
Decision: cpu: `A3L_liver_test` = post_rules_test_liver.py on val (sanity: reproduces the screen) then test (111 x 2).
gpu: nothing new until thickSlice ends (~00:40 train, then its own test prediction); then `predict_test_probs.sh 102 0
1` (s6 plan) for F2-F4. B-prep for liver: if FL1/FL2/FL3 confirm, count how many of the 999 synthetic liver labels the
rule would change (MISSION B: < 50 → no arm).
Reasons: MISSION A3 (<= 4 frozen per organ + controls, one test run). Three distinct anatomical facts not yet tested on
liver test: tumour-in-organ, position, tissue HU. ccTol10 not in the family: a variant of ccPost whose test result is
known (0.381), so it cannot be a clean confirmation; tumNear10 is a near-duplicate of tumTouch.
Assumptions: pos prior on test = union of all 20 val label bin sets (cached `pos_bins.npz`); liver HU window 1..210 from
`liver_val/val_label_params.json`; the test reference is read only after the rules are applied (as pancreas).
Alternatives considered: freezing ccTol10 as FL4 (rejected: known result, near-duplicate); tumNear10 instead of
tumTouch (tumTouch has the larger val gain and the stricter anatomical statement); waiting for the pancreas test
probabilities before any liver test (no reason: independent lane, CPU only).
Fake test: --split val --cases lits_007 reproduces the screen exactly (f0 tumTouch 0.684, pos 0.590, tissueVal 0.607;
f1 0.520 / 0.585 / 0.614).
Started: cpu `A3L_liver_test` (`$PROJ/logs/A3L_liver_test_20261124_211829.log`; val sanity then test, ~0.5-1 h).

### s7 2026-11-24 20:45, event job_finished (A3a_frozen)
Saw: A3a exit 0 after 0.29 h (`$PROJ/logs/A3a_frozen_20261124_201746.log`). Val (43) reproduces the s6 screen exactly
(pos+minLes +0.015/+0.006, grow0.1+tumNear10 +0.014/+0.019; grow0.1+X − thr0.1+X ≈ 0). Test (238, no-prob rules):
**F1 pos+minLes NOT confirmed**: 2-run mean +0.007 [+0.002, +0.012] (p 1.3e-7, but < the 0.01 floor), f0 +0.012,
f1 +0.002 [−0.005, +0.007]; lesion recall 0.55 → 0.46 / 0.52 → 0.44; FP pieces 16.9 → 0.8 / 5.3 → 0.8. Controls:
minLes +0.006 (pos adds ~0.001), pos +0.004, tumNear10 +0.011 (f1 +0.003), open1 +0.005, rand −0.14. (Holm in this run
was over F1 alone; the full run with F2-F4 recomputes Holm over 4, which does not change F1's verdict: gain < 0.01.)
gpu: thickSlice healthy (epoch 147 at 20:32, 17.3 s/epoch, EMA pseudo Dice 0.273, no NaN; train ETA ~00:40, then
test prediction ~2.2 h → ~03:00-05:30).
Decision: cpu: liver A2 val screen (`post_rules.py --target liver`, new: same rules on the 20 LiTS val cases of
101:0 / 101:1, no probs, + ccPost+minLes / ccPost+pos / ccPost+tissueTxt). No B arm / B-prep for F1 (not confirmed;
MISSION B is for confirmed rules only). gpu: nothing new until thickSlice ends; then `predict_test_probs.sh 102 0 1`
for F2-F4 as planned in s6.
Reasons: MISSION plan puts liver A2/A3 on the cpu lane; the pancreas val screen is finished and frozen, and after
reading test any new pancreas rule would be chosen with test knowledge. Liver asks the thesis question directly:
does anatomy beyond the cc rule add to ccPost (the liver's known winner), and do non-cc anatomical rules (pos,
tissue, tumTouch) reach the cc rule's gain alone?
Assumptions: liver val label minLes = 0.0003 ml (LiTS val has 1-voxel lesions) → minLes is a no-op for liver (so
minLes0.5 is a fixed textbook-free size cut: flagged); spine-centre code works on LiTS (all 20 val labels got bins);
the 4 fake-test caches (lits_007/008) are reused by the real run (same VERSION, same code).
Fake test: lits_007 + lits_008: runs; none 0.420/0.412 → ccPost-type rules ~+0.3, pos +0.13, tissueVal +0.11,
rand controls −0.37 (2 cases only, not a result).
Alternatives considered: counting F1's training-label changes anyway (cheap but MISSION ties B-prep to a confirmed
rule; skipped); liver with all 7 runs (the promising criterion is defined on the 2 baseline runs; arms later, for
frozen rules); more pancreas screening (test seen: no).
Started: cpu `A2L_liver_val` (`$PROJ/logs/A2L_liver_val_20261124_204456.log`, ~15-30 min).

### s6 2026-11-24 20:35, event job_finished (A2b_postrules_val)
Saw: A2b second val screen exit 0 after 0.21 h (`$PROJ/logs/A2b_postrules_val_20261124_200057.log`, argmax
agreement 1.0000 over 86). **6 combinations promising**, no single rule: pos+minLes +0.015/+0.006 (lost 4 of 86),
grow0.1+minLes +0.007/+0.018, grow0.1+tumNear10 +0.014/+0.019 (lost 0), grow0.05+minLes +0.004/+0.017,
grow0.1+growT0.2+minLes +0.008/+0.020, grow0.1+pos+minLes +0.009/+0.019. pos alone +0.005/+0.002 (0 lost, FP pieces
17 → 11), minLes1.75 +0.015/+0.008 but lost 5 (> 5 %), growT ±0, d20 = d10. **Control thr0.1 lifts gland Dice
+0.058/+0.049, as much as grow0.1 (+0.053/+0.047)**: the gland gain is a lower threshold, not the anatomical limits.
Ledger filled (rows 10-15, `pancreas_val/summary_screen2.md`). gpu: G1 thickSlice healthy (epoch 78 at 20:11,
17.3 s/epoch, EMA pseudo Dice 0.233, no NaN; train ETA ~00:40, then test prediction ~2.2 h).

**FREEZE (A3, before any test run of these rules; nothing below may change after reading test):**
- Frozen rules, Holm over these 4 (per run and for the per-case mean of the 2 runs), main metric per-case tumour Dice
  vs the saved prediction, both baseline runs 102:0 / 102:1, checkpoint_final, 238 MSD07 test cases:
  F1 `pos+minLes`, F2 `grow0.1+minLes`, F3 `grow0.1+tumNear10`, F4 `grow0.1+pos+minLes`.
- Exact parameters (code `nnunet/post_rules.py` at commit of this entry, wrapper `nnunet/post_rules_test.py`
  VERSION frozen1): minLes 1.19 ml (smallest real val lesion); tumNear 10 mm; grow t 0.1 on p(pancreas)+p(tumour),
  10 mm, 26-connected to the predicted organ, -50..300 HU, label argmax(p1, p2); pos: spine centre from bone > 200 HU
  (40 mm half-width, posterior half of the body, 20 mm z median), prior = union of all 43 real val labels'
  spine-relative 2D occupancy (2 mm bins, 10 mm margin), pieces < 50 % inside → bg.
- Controls / components (reported, not in Holm): minLes, pos, grow0.1, tumNear10, thr0.1, thr0.1+minLes,
  thr0.1+tumNear10, thr0.1+pos+minLes (new: the combos with a plain threshold instead of grow), open1, rand_minLes,
  rand_pos+minLes (random tumour pieces matched in count).
- Declared secondary contrasts: grow0.1+X − thr0.1+X (tumour + pancreas Dice; tests the anatomical part of grow);
  pancreas Dice of every rule. Expectation from val (written before test): grow ≈ thr.
- Confirmed on test = Holm p < 0.05 for the 2-run mean and mean gain > 0 in both runs and gain >= 0.01 (noise
  floor). Only confirmed rules go to B.
Decision: cpu: `A3a_frozen` = post_rules_test.py on val (all frozen + new thr controls; sanity: must reproduce the
screen) then on test with `--no-probs` (F1 + no-prob controls, now). gpu: after thickSlice ends (~05:30), the next
session starts `predict_test_probs.sh 102 0 1` (new; ~4.4 h, ~64 GB; disk 2.1 PB free) for F2-F4, then the full
test run of post_rules_test.py. B-prep (cpu, after A3a): count how many of the 999 synthetic training labels minLes
and pos would change (if F1 confirms; counting itself selects nothing).
Reasons: MISSION A3 (<= 4 frozen, controls, one test evaluation). The 4 chosen are the promising ones that are
distinct: one without probabilities (pos+minLes, has a pre-training twin), the simplest grow combo (minLes), the
best mean (tumNear10), and the full combination; grow0.05+minLes and +growT0.2 are near-duplicates (growT adds
~0.001). No gland-only rule frozen: grow0.1 does not beat thr0.1 on val (my s5 condition not met); grow0.1 and
thr0.1 are still reported as controls with pancreas Dice. The thr0.1+X twins make the grow-vs-threshold question
testable inside the combinations.
Assumptions: pos prior on test from all 43 val labels (no test information; on val it stays leave-one-out). Probs
on test reproduce the saved test_final label maps (asserted > 0.995 argmax agreement per case; the same
checkpoint and nnU-Net settings). GPU after thickSlice: test probs before any B arm, because B needs a confirmed
rule and F1 may already confirm from the CPU run tonight; if F1 confirms and changes >= 50 training labels, its B
arm could go first instead (decided by the session that sees both).
Alternatives considered: interrupting thickSlice for test probs (no: it is half-way, and G1 is the image-gap
question); freezing grow0.05+minLes (near-duplicate); minLes1.75 (lost > 5 %); Holm over 8 incl. per-run (the
2-run mean is the primary, per-run reported).
Fake test: post_rules_test.py --split val on msd_006 + msd_025: per-case values identical to the screen (e.g.
msd_006 f0 pos+minLes 0.3823, grow0.1+tumNear10 0.3673).
Started: cpu `A3a_frozen` (log in the NOTES Log), expected ~20-30 min.

### s5 2026-11-24 20:00, event job_finished (A2_postrules_val)
Saw: A2 first screen exit 0 after 0.3 h (`$PROJ/logs/A2_postrules_val_20261124_193604.log`; 86 run/case pairs, prob
argmax agreement 1.0000). **No rule promising** (gain > 0 in both runs, mean >= 0.01, <= 5 % lesions lost). Val
tumour Dice none 0.141 / 0.133 (pancreas 0.162 / 0.118). FP-removal rules help f0 (17 FP pieces/case) and hurt f1
(6/case): ccK2 +0.018/−0.020, tumTouch +0.017/−0.006, tumNear10 +0.016/−0.007 (lost 3); minLes +0.013/+0.005
(lost 4 of 86: near miss, but the open1 control does +0.009/+0.001); tissue rules ±0.001; ccTol* hurt f1 like ccPost.
Grow: gland Dice +0.021/+0.018 (t 0.3) → +0.053/+0.047 (t 0.1), tumour ±0.01; but dilate1 (no probabilities) gives
the gland +0.047/+0.040 too (and tumour −0.035/−0.018). Random removal controls −0.08 to −0.14 (in few-piece cases
the pieces removed are the hits). Numbers `reports/data/post_rules/pancreas_val/summary_screen1.md`, ledger updated.
gpu: G1 thickSlice healthy (epoch ~16 at 19:53, 17.3 s/epoch, EMA pseudo Dice rising; ETA train ~00:40 Wed).
Decision: cpu: second val screen (`post_rules.py` VERSION postrules2, all first-screen rules recomputed + new):
lower grow thresholds (0.05 / 0.02) and a 20 mm window; growT (tumour growth on p(tumour) 0.2 / 0.1); minLes
sensitivity (0.5 ml, 1.75 ml = p5 of val lesions); `pos` (spine-relative 2D position prior from the other 42 val
labels, leave-one-out; spine centre per slice from bone HU only); combinations (grow + minLes / tumNear10 / growT /
pos); controls thr0.1 / thr0.05 (the same threshold without anatomy: tests whether grow is just a lower threshold),
rand_minLes / rand_pos. gpu: nothing new (G1 running; s4's wake-up at 21:18 checks it).
Reasons: the first screen says the two failure modes need different rules: under-segmentation (grow; the question is
whether the anatomical limits beat plain thresholding / dilation, hence thr controls) and FP pieces in f0 vs lost
lesions in f1 (combine growth, which reconnects lesions to the gland, with tolerant tumour rules; a position prior
that removes FPs by location rather than by connectivity). Fake test on msd_006 + msd_025: runs, ~1 min per case;
pos centre sane (msd_006 gland 52-109 mm anterior of the spine centre, tail to the left; HU 106 at the centre).
Assumptions: the LOO prior keeps the val-label parameter out of the case it scores (the 42 others still overlap the
scored set across cases: in-sample at the set level, flagged; test confirms). The promising criterion stays as
written (tumour Dice); a rule that helps only the gland (grow) is recorded as a gland finding, not promoted on
tumour Dice: for A3 I will also freeze the best gland rule if it beats its thr/dilate controls, because the MISSION
says pancreas Dice must be reported and under-segmentation is the main failure.
Alternatives considered: z-position (vertebral level) prior: needs vertebra labelling, postponed; liver A2: lower
priority, after pancreas; freezing now (first screen has no winner, nothing to freeze).
Started: cpu `A2b_postrules_val` (log in the NOTES Log), expected ~25-30 min. Its end wakes the next session: fill
the ledger, then A3 freeze (<= 4 rules + controls) and the test run.

### s4 2026-11-24 19:50, event job_finished (G0_valprobs)
Saw: G0 exit 0 after 0.8 h (`$PROJ/logs/G0_valprobs_20261124_185954.log`): fold 0 and fold 1 each 43/43 val
probability files. A2_postrules_val (cpu) started its real run right after (it polled for "fold 1: 43 / 43"),
still running at 19:50 (the files in `reports/data/post_rules/pancreas_val/` dated 19:35 are from s3's fake test and
will be overwritten by the run).
Decision: gpu: start G1 `pancreas_arm.sh thickSlice` (D112 built + preprocessed in s3; the runner skips build and
preprocessing, registers the fold, trains 1000 epochs, predicts 238 test, evals vs baseline). Wake-up +90 min to
check the training is healthy. cpu: A2 keeps running; its end wakes the next session (ledger + second screen).
Reasons: MISSION plan (G1 after G0); the GPU would idle otherwise; independent of the filter question.
Assumptions: the runner's build step only re-checks existing files (build_thick_dataset.py skips existing outputs,
plans exist) so it costs minutes of CPU next to A2 (376 cores, 2 TB RAM: no contention worth avoiding).
Alternatives considered: none (planned, no free choices).
Started: gpu `pancreas_thickSlice` (log path in the NOTES Log), expected end ~05:30 Wed (5.3 h train + 2.2 h test).

### s3 2026-11-24 19:40, event job_finished (D112_build)
Saw: D112_build exit 0 (`$PROJ/logs/D112_build_20261124_190840.log`): 999 thick volumes (1.25 mm 29, 2.5 mm 823,
3 mm 19, 5 mm 101, 7.5 mm 27) + 43 real val linked; preprocessing 1042/1042 (3126 files, same count as D102).
G0 fold 0 probabilities 43/43 done (19:26), fold 1 at ~20/43 at 19:36 (ends ~19:50).
Decision: gpu: nothing new (G0 still running; one GPU process at a time); the session woken by G0's end starts G1
`pancreas_arm.sh thickSlice` (build + preprocess are done, the runner should skip them). cpu: A2 first screen,
`nnunet/post_rules.py` (new): 17 rules + 5 controls on the 43 val predictions of both runs (definitions in its
docstring, ledger `reports/data/post_rules/ledger.md`); the job waits for G0's fold-1 probabilities (polls the G0
log, max 1 h, then falls back to `--no-probs`).
Reasons: MISSION A2; A1 showed the plain one-organ rule removes real lesions in fragmented glands, so the first
screen covers tolerant one-organ variants (distance, k pieces), tumour-belongs-to-gland, tissue HU, size, and grow
rules (the only kind that can fix under-segmentation), each with a non-anatomical control (random removal matched to
ccPost / ccTol10 / tumTouch, opening, plain dilation as growth without probabilities).
Assumptions: val-label parameters (HU window [-95, 213] = p0.5/p99.5 of val pancreas ∪ tumour; smallest val lesion
1.19 ml) are allowed by MISSION ("real val labels") although the same cases score the rule: in-sample, flagged in
the ledger; test confirms. Probabilities: npz (C, z, y, x) transposed to the NIfTI order; argmax agrees with the
saved label map (1.0000 on msd_006; the script asserts > 0.995 per case).
Fake test: msd_006 f0: ccPost 0.341 → 0.382 (as s1); grow0.1 pancreas Dice 0.076 → 0.124; dilate1 0.165 (!: plain
dilation lifts gland Dice too: under-segmentation; the control matters); ~1 min per case.
Alternatives considered: position prior (spine landmark) postponed to the second screen (more code; first see whether
the simple rules help at all); liver A2 later (lower priority).
Started: cpu `A2_postrules_val`, log `$PROJ/logs/A2_postrules_val_20261124_193604.log`, expected end ~20:05.
Next: G0 end → start G1 thickSlice (gpu) + `ap wake-at '+90 minutes'`; A2 end → fill the ledger, choose the second
screen (combinations of the promising ones, position prior, grow thresholds); pancreas Dice counts too.

### s2 2026-11-24 19:15, event job_finished (A1)
Saw: A1 exit 0 (log `$PROJ/logs/A1_pancreas_aucpost_20261124_190320.log`; the 3 "error-looking" lines are table
text, no traceback). Outputs `reports/data/test_eval_pancreas/auc_post{,_val}/` (summary, per_case, components).
**Pancreas ccPost hurts, unlike liver.** Test (238), tumour Dice none → ccPost: f0 0.183 → 0.163 (none − ccPost
+0.020 [+0.000, +0.041]), f1 0.172 → 0.139 (+0.033 [+0.014, +0.052]); pancreas Dice 0.209 → 0.189 / 0.173 → 0.157;
lesion recall 0.550 → 0.354 / 0.521 → 0.304; FP pieces/case 16.9 → 0.7 / 5.3 → 0.5. ccAucPost0.509 vs ccPost:
−0.001 / −0.002 (baseline −0.001 [−0.004, +0.001], p 0.4): adds nothing. Val (43): none → ccPost f0 0.141 → 0.145,
f1 0.133 → 0.087 (lesion recall 0.465 → 0.349 / 0.233); stricter cuts (val-only sweep) all worse: 0.55 −0.002/−0.003,
0.6 −0.033/−0.030, 0.7 −0.050/−0.053; sep AUROC (FP lower) 0.75 / 0.58.
Reading: the cc rule removes FP pieces (17 → <1 per case) but also ~1/3 of the real lesions, because the predicted
gland is fragmented (under-segmentation): tumours sitting in gland parts the model missed are not connected to the
largest predicted pancreas piece. Consistent with the corollary: the rule acts where predictions violate it, but here
the violation is mostly the model's *missing* gland, not anatomically wrong tumours. Points A2 to rules with
tolerance (tumour near / adjacent to the gland, distance-based keep) and to grow rules (G0 probabilities).
Decision: cpu lane: build + preprocess D112 (thickSlice) now (`build_thick_dataset.py --workers 16 --plans` +
`nnUNetv2_preprocess -d 112 -np 16`), so G1 training can start right after G0 (~20:00). gpu lane: G0 still running
(predicting, ~25 % at 19:08), no action. A2 exploration starts in the next session (needs the cpu lane free or runs
in-session as short analyses; G0's probabilities for grow rules).
Reasons: MISSION plan order; the GPU should not idle after G0; the build is ~1-1.5 h of CPU.
Assumptions: 16 workers + nice is fine while G0 predicts (376 cores, 2 TB RAM, nothing else running).
Alternatives considered: starting A2 on the cpu lane first (would delay G1 by the build time and idle the GPU).
Started: cpu `D112_build` (log path below in NOTES Log). Wake-up: G0's end wakes the next session.

### s1 2026-11-24 19:00, event idle
Saw: first session; no job yet; B200 idle (0 MiB), /work 2.1 PB free. D102 has 43 msd_* val images, fold_0/1
`validation/` (43 cases) and `test_final/` present.
Decision: gpu lane G0 `predict_val_probs.sh 102 0 1` (as planned). cpu lane: A1 (port ccPost / ccAucPost to
pancreas, `--target pancreas`), written and fake-tested in this session, then run on the cpu lane.
Reasons: MISSION plan, fixed; G0 is short and needed for grow rules; A1 has no free choices.
Assumptions: nnU-Net's `validation/` is checkpoint_final (it is written at the end of training).
Alternatives considered: starting thickSlice first (G0 is 1 h and unblocks A2's grow rules; order as planned).
Started: gpu `G0_valprobs`, log `$PROJ/logs/G0_valprobs_20261124_185954.log`, expected end ~20:00 (at 19:07 still
loading: process in disk sleep, GPU 0 %, no output since "building the font cache"; check at the wake-up).
A1 code: `nnunet/auc_post.py --target pancreas` (runs 102:0/1, MSD07 val/test, main cut 0.509 from pancreas
rules.json; stricter-cut sweep 0.55/0.60/0.637/0.70 on **val only**, test gets the fixed cut alone so no cut can be
picked from test); `fp_location.metrics` takes the organ name from `fl.ORG`. Liver unchanged: tables from a copy of
the liver val cache with old (HEAD) and new code are byte-identical (summary, per_case, components; scratch
`/work/Research/scratch_ap/s1_verify`). Smoke test msd_006 fold 0: ccPost tumour Dice 0.341 → 0.382, FP pieces 18 → 0.
Started: cpu `A1_pancreas_aucpost` (val then test, 12 workers, nice), log
`$PROJ/logs/A1_pancreas_aucpost_20261124_190320.log`, ~15 min. Wake-up set 19:23.
Next (planned): when A1 ends, cpu lane builds + preprocesses D112 (`build_thick_dataset.py --plans` +
`nnUNetv2_preprocess`, the same commands as step 1 of `pancreas_arm.sh`, which then skips them) so the GPU can start
thickSlice training right after G0; do not start `pancreas_arm.sh thickSlice` while that build runs.

### s0b 2026-11-24 18:30, plan revised with the user (Kali VM)
Saw: the user rejected pancreas ccReject as E2 and set the direction: the paper's main finding is that for these
filters it hardly matters whether a rule is applied before training (labels) or after (predictions); what matters is
whether the rule reflects real anatomy. The UCloud sessions may explore. Time: until Friday 27 Nov (~60 h), when the
user reviews and builds on the results.
Decision 1: **drop pancreas ccReject.** Reasons: cc flags only 88/999 pancreas volumes, so it tests a rule against
an error this generator rarely makes; it spends ~7.5 GPU h on a pre-training test before knowing whether the rule
matters at all (I chose it for being the planned check and needing no code: wrong criteria here).
Decision 2: **Phase A first: anatomical rules on the predictions (CPU)**: A1 the fixed pancreas replication of
ccPost / ccAucPost (val + test), A2 open exploration of anatomical rules on the val predictions with non-anatomical
controls and a ledger of everything tried, A3 freeze <= 4 rules per organ, one test confirmation.
Reasons: tests the thesis directly and cheaply; post-hoc results tell which rules are worth a ~8 GPU h pre-training
arm; the liver already shows pre ≈ post for cc and AUC, so pancreas (labels fine, predictions poor) is the
informative contrast. Selection on val keeps the decision in force "test never decides"; controls separate anatomy
from "any removal helps".
Assumptions: 43 val cases (20 liver) suffice to screen; the val predictions in `fold_X/validation` are
checkpoint_final (nnU-Net writes them with the final weights).
Alternatives: splitting the 238 test cases into explore / confirm halves (more power for screening, but it amends a
decision in force and halves the confirmation set: left for the user); TotalSegmentator organ masks for position
rules (a real-trained model; later, if image landmarks are not enough).
Decision 3: **Phase B only for confirmed rules, and only if the rule changes >= 50 of the 999 training labels**;
pre vs post compared with a ±0.02 equivalence margin declared in advance. Reasons: fewer than 50 changed volumes
cannot move a 999-volume training set measurably (liver ccRepair vs ccRepairDel differed by 157 volumes and ~0.005);
"the generator already obeys the rule" is itself the pre-vs-post answer. ±0.02 = twice the noise floor, the size of
the liver run-to-run gap.
Decision 4: **keep thickSlice (E1) on the GPU lane during Phase A**, after a 1 h export of val probabilities
(needed for rules that grow the gland, the only kind that can fix under-segmentation). Reason: Phase A needs no GPU
for ~10-14 h; the image gap is the other open pancreas question.
Decision 5: **two lanes (gpu, cpu)** in the supervisor, so analysis and dataset builds run while a training runs;
either lane's job end wakes a session. Tested with a fake `claude` (both lanes, exit codes 0/7, busy-lane refusal).
Time estimate: G0 1 h + G1 9 h; A ~10-14 h (parallel); B 1-3 arms x ~8 h; reps 1-2 x 7.5 h → ~35-55 GPU h, fits
the ~60 h until Friday morning with slack for failures.

### s0 2026-11-24 17:50, planning session (Kali VM, with the user)
Saw: pancreas baseline done and diagnosed (NOTES.md, `reports/figures/pancreas_cases/README.md`): tumour Dice
0.183 / 0.172, pancreas 0.209 / 0.173; the model under-segments; real scans show fat around the gland (26 % vs 2 %
synthetic), thick slices (38 test cases >= 5 mm; synthetic all 1.0 mm) and lobulation the synthetic ones lack.
The user is away for a few days and gives ~2 days of one full B200; asked for a supervisor that wakes a new Claude
session whenever a job ends, and for a plan whose decisions all carry reasons and assumptions.

Decision 1: **E1 = `thickSlice` first** (all 999 synthetic pancreas pairs, images re-sliced to the slice
thicknesses of the 43 MSD07 val cases, labels at the slab centres, plans copied from D102; `nnunet/build_thick_dataset.py`,
Dataset112).
Reasons: of the three image gaps found it is the only one we can close cleanly in the data without new generation
(fat and lobulation would need MAISI changes or hand-made intensity edits); thick slices are measured to matter
(0.16 vs 0.30 gland Dice); nnU-Net resamples real 2.5 / 5 mm scans with z order 0 (slices repeated), so writing the
synthetic scans on a thick grid and letting nnU-Net's own preprocessing resample them gives training images that
went through the same path as the test images. Data-only change: trainer, plans and evaluation are the baseline's.
Assumptions: thick reconstructions are close to a slab average with spacing = thickness; the val thickness mix is a
fair stand-in for test (val and test are random halves of the same MSD07 set, test never read); labels at the slab
centre are what an annotator of a thick scan would draw.
Alternatives: on-the-fly thickness augmentation in a custom trainer (changes the trainer, so the arm would differ
from `all` in more than data; more code to get right unattended); thickness mix 50/50 1 mm and thick (arbitrary;
the val distribution is the principled target).

Decision 2 (*withdrawn in s0b*): **E2 = pancreas `ccReject` (911)**, a D102 fold with the liver cc rules unchanged.
Reasons: it is the generalisation check the paper planned (and the supervisor meeting deferred until the baseline
was known); it needs no new code (ids file exists), which matters unattended; on liver ccReject gained +0.079.
Expected small (only 88 volumes differ), which is itself the result the paper needs: label filters help where the
generator makes label errors.
Assumptions: `ids_ccReject.json` for pancreas is on /work (cc_rules ran 2026-11-14); msd_025 keep/exclude does not
matter here (it only sets the AUC cut).
Alternatives: pancreas ccAucRepair (the best liver arm) needs a pancreas port of `build_repair_dataset.py`, and the
pancreas cc caches were measured on the uncropped pool labels while D102 labels are cropped: too risky to write
unattended; left for the user.

Decision 3 (*replaced in s0b*): **E3 = a second run only of an arm that beats the baseline on val by >= 0.02** (val = 43 real cases;
test never decides). Reason: liver taught that one run can be 0.02 lucky; repeating a null arm teaches nothing.
Decision 4 (*replaced in s0b; size test back to the backlog*): **then liver `sizeLarge` + `sizeSmall`** (ids ready since 2026-11-15) if >= 16 h remain, else the other
rep or CPU work. Reason: keeps the GPU on the next-most-useful open question with no generation cost; the user
proposed it; it only means something as a pair.
Decision 5: **mechanics**: one supervisor loop (bash, detached), a new headless session per event (job ended,
wake-up due, nothing running, final), sessions never wait; all autopilot commits on branch `autopilot` (pushed,
`main` untouched, deny rules for force-push/main/reset/rm of data); `--permission-mode bypassPermissions` by
default, because an unattended session that hits a permission prompt just stops and leaves the GPU idle; the blast
radius is the UCloud container, limited further by the deny list and MISSION's hard rules. Failed sessions retry
with backoff (10, 20, 40, 80, 120 min), 6 in a row stop the autopilot; heartbeat session at least every 8 h; final
session 1 h before the deadline writes the summary.
Assumptions: the UCloud job runs >= 48 h; Claude Code is logged in on UCloud and the subscription's usage limits
allow ~10-20 short sessions in 2 days (rate-limit failures are absorbed by the backoff); deny rules hold under
bypassPermissions (the MISSION rules are the second line).
Started: nothing (VM). Fake-tested here: `build_thick_dataset.py` on tiny volumes (geometry exact to < 0.1 mm, labels
aligned), the supervisor end to end with a fake `claude` (job end, wake-up, failed-session retry, heartbeat, final,
smoke mode, branch push to a scratch remote).
