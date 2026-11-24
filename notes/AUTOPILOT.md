# Autopilot journal (newest first)

Unattended sessions on UCloud (`autopilot/supervisor.sh`, handoff `autopilot/MISSION.md`). Each entry: what the
session saw, what it decided, reasons, assumptions, alternatives, what it started. The final session writes a
summary for the user at the top.

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
