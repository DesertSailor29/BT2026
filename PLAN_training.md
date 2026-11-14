# Plan: filtering + nnU-Net training on the MAISI pool (rewritten 2026-11-12, arms replaced 2026-11-14)

Question: does a synthetic training set with the faulty labels **removed (reject)** or **fixed (repair)** beat
training on all 999 synthetic liver volumes, for real LiTS liver/tumour segmentation? Liver first; pancreas (MSD07)
later with the same code. Decisions with dates and reasons are in NOTES.md "Decisions"; the candidate filters with
arguments in `PLAN_filtering_options.md`.

## Design (fixed)

- **Training data: synthetic only** (MAISI pool, 999 liver volumes, 512³ @ 0.8×0.8×1.0, cropped only at MAISI's constant
  padding to 480×480×384). Arms differ only in which synthetic volumes / labels they use.
- **Validation: 20 real LiTS cases** (thesis train split, seed 42), used by nnU-Net during training. **Test: the other
  111 LiTS cases** (incl. the 40 thesis test cases), never used for any choice. `checkpoint_final` for test (no
  selection on val), default `nnUNetTrainer`, 3d_fullres, 1000 epochs, `nnUNet_compile=f`, Dataset101's plans.
- **One preprocessed dataset per label set:** an arm that only selects volumes is a new fold of a dataset
  (`make_splits.py`, append-only `arms.json`); an arm that changes labels gets its own dataset with Dataset101's
  plans copied (same patch size, spacing, normalisation) and the same 20 val cases.
- **Noise floor** (arm `all` trained twice): per case |Δ Dice| ≈ 0.03, mean differences < 0.01 (liver p 0.50,
  tumour p 0.07). An arm's gain has to clear this.

## Filter rules (amended 2026-11-14, before any filter arm was trained)

**Principle: "the liver is one organ, a tumour lies in it", plus "a tumour label must be visible".** Both rules are
definitional or take their one number from real data; no hand-tuned threshold.

- **Main structure:** main liver = the largest 26-connected liver piece; the main structure = the 26-connected
  component of liver ∪ tumour that contains it.
- **cc** (`maisi/cc_rules.py`): every liver or tumour voxel outside the main structure → background. A tumour
  component is connected, so it lies wholly inside or wholly outside. No parameter.
- **ccAuc:** cc, then every tumour component inside the main structure with separability `sep` < **cut 0.5475**
  → its surroundings' label (liver if ≥ 50% of its outer 26-neighbour shell is liver, else background). Components
  with < 10 measurable voxels get no score and pass.
- **AUC cut = the minimum `sep` over the real val lesions** (lits_021 c3). **lits_049 is left out of the reference
  as a whole case** (unreliable labels: nothing visible under its least visible lesions; the only real case below
  0.547; 0/681 real test lesions below). Entry in `maisi/real_reference_exclusions.json` (an entry without
  `component` excludes the whole case). It stays a real val case for nnU-Net.
- Separability `sep` per component (`maisi/filter_metrics.py`): core = the component eroded by one voxel (MAISI label
  edges are 1.5 mm steps); shell = voxels 2–10 mm outside it, not tumour, restricted to the liver if the component sits
  in the liver (≥ 50% of its outer shell is liver) or the liver is ≥ 25% of the shell. AUC = P(core HU > shell HU),
  `sep` = max(AUC, 1 − AUC) ∈ [0.5, 1] (0.5 = invisible).

**Reject and repair use the same rule.** Reject keeps exactly the volumes the repair leaves unchanged, with identical
labels, so the two differ only in the action. A filtered arm keeps every volume that passes (no fixed K).

Counts on the pool (`$MAISI_RUNS/pool/filter_metrics/liver/cc_summary.md`, `cc_rules.json`):
- cc: unchanged 343; changed: liver only 58, tumour only 477, liver + tumour 121; tumour-free after repair 157.
- ccAuc: unchanged 300; 162 in-liver components relabelled (84 to liver) in 144 volumes; tumour-free 177.

**Superseded (history).** The 11-12 rules were fixed before looking at the pool metrics: *touch* (a component fails if
no voxel is 26-adjacent to the liver → background) and *AUC* (sep < real val minimum, then 0.504 from lits_049),
each with a reject and a repair arm. The measure session (2026-11-13/14) changed them as a documented amendment, with
no arm trained yet: touch v1 and v2 ("in the main structure") gave identical volume classes on all 999, so cc replaced
touch (it also removes detached liver pieces, 179 volumes, all < 2 ml); lits_049 was excluded (cut 0.504 → 0.5475);
AUC is kept only on top of cc, because most AUC-failing volumes also fail touch; the liver HU/border check was
dropped as a filter (blind review: 28/30 flagged livers ok, the one bad liver unflagged). The touch/AUC rules still
exist in code for follow-ups and comparison (`ids_touchReject.json` 401, `ids_aucReject.json` 730,
`ids_touchAucReject.json`; datasets 105–107).

**Sanity checks (done in the measure session):** real val lesions all touch the liver (199/199; test 707/709); the
least visible real lesions were viewed (lits_049 excluded); classes and cross-tabs per rule with contact sheets
(`reports/figures/filter_review/`); every arm keeps ≳ 100 volumes / changes ≳ 20 volumes.

## Arms

| fold / dataset | arm | training data | n |
|---|---|---|---|
| D101 f0 | `all` | 999 synthetic | done: liver 0.870, tumour 0.301 (111 test) |
| D101 f1 | `all_rep2` | same, second run (noise floor) | done: liver 0.876, tumour 0.305 |
| D101 fold | **`ccReject`** | volumes cc leaves unchanged (`ids_ccReject.json`) | 343 (1st) |
| D103 (`Dataset103_LiverSynthRepairCC`) | **`ccRepair`** | all, cc-repaired labels | 999 (2nd) |
| D103 fold | **`ccRepairDel`** | ccRepair without the volumes left tumour-free (`ids_ccRepairDel.json`) | 842 (3rd) |
| D104 (`Dataset104_LiverSynthRepairCCAUC`) | **`ccAucRepair`** | all, ccAuc-repaired labels | 999 (4th) |
| follow-ups | `aucReject` (D101, `ids_aucReject.json`, 730), `aucRepair` (`Dataset105_LiverSynthRepairAUC`), `ccAucReject` (D101, `ids_ccAucReject.json`, 300), `<best>_rep2` | see "Later" | after the first 4 |

The queue assigns fold numbers by arm name (it looks them up and never assumes them; ccReject got D101 fold 2).
The four main arms form a chain: clean 343 → + repaired with tumour (842) → + tumour-free (999), and ccAucRepair vs
ccRepair isolates AUC's own contribution (invisible tumour labels inside the liver). Names: technique first, action
second. Touch-only datasets, if ever needed: `--criterion touch` → Dataset106, `both` → Dataset107.

### Evaluation and analysis (decided 2026-11-12; `nnunet/predict_test.py eval`)

The study is **exploratory**: no predictions are written down in advance (synthetic data can behave unexpectedly);
mechanisms seen here become hypotheses for future work. We compare from as many sides as possible, but name one
main comparison so that a single win can't be called cherry-picked:

- **Main metric: per-case tumour Dice on all 111 test cases** (comparable with the thesis).
- **Baseline: the per-case mean of the two `all` runs** (f0, f1). It uses both runs, has less noise, and involves no
  choice made on test data (picking "the better run" would be one, and that run is partly lucky). Each arm is also
  reported against f0 and f1 separately.
- **Clear win** = paired Wilcoxon (arm vs baseline, per case) significant after **Holm over the 4 main arms**
  (ccReject, ccRepair, ccRepairDel, ccAucRepair: sort the p-values, the smallest must be < 0.05/4, the next < 0.05/3,
  …, stop at the first that fails) **and** a mean difference above the noise floor (> 0.01). Every difference also
  gets a paired bootstrap 95% CI, which shows how large the effect is.
- **Secondary, exploratory** (no correction, labelled as such): liver and liver+tumour Dice, global tumour Dice,
  lesion-wise recall (all, ≥ 1 ml), precision, FP components and FP volume per case, tumour-free cases with predicted
  tumour, the 40 thesis test cases, the chain steps (ccReject vs ccRepairDel vs ccRepair: do clean negatives cut FPs?),
  and every follow-up arm.

Implemented in `predict_test.py eval` (`--baseline 101:0 101:1`, `--family` = the 4 main arms; the queue passes both).
The summary also shows the noise row (`all_rep2` − `all` with its CI) and each arm against each `all` run separately.
Lesion-wise metrics: 26-connected tumour components; detected = any overlap. Output:
`$nnUNet_results/test_eval_liver/summary.md`.

## The GPU session (full B200, one session, ~25 h)

Everything runs in one GPU session; preprocessing overlaps training (the full-B200 session has a 48-core quota and
costs less than a big CPU session). `nnunet/run_queue.py` runs three lanes:

- **prep:** filter metrics, cc rules (`ids_cc*.json`), build + preprocess D103 (cc) and D104 (ccAuc), eval of `all`,
  figures.
- **train:** arms in order ccReject → ccRepair → ccRepairDel → ccAucRepair. An arm waits only for the prep step it
  needs, so ccReject (a D101 fold) trains while D103/D104 are built.
- **predict:** test inference (111 cases) + eval of each trained arm, while the next arm trains (`--no-pipeline`
  turns this off).

```bash
cd "$PROJ/BT2026" && git pull
source nnunet/env.sh && python nnunet/run_queue.py --dry-run      # check the plan
setsid nohup bash -c 'export QUEUE_CPU_WORKERS=20 && source "$PROJ/BT2026/nnunet/env.sh" && python "$PROJ/BT2026/nnunet/run_queue.py"' \
  > "$PROJ/logs/queue_liver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

Per arm ~4.9 h training; prediction (~2 h) overlaps the next training, so 4 arms ≈ 4 × 4.9 + 2 ≈ 22 h plus prep.
Watch the queue status file `$PROJ/QUEUE_liver_arms__…_CET.txt`, the training status files `NNUNET_D10x_…` and
`$PROJ/queue_liver/progress.md`, and copy the key numbers into NOTES.md. A failed step skips only what depends on
it. After a session end, run the same command again (every step is resumable).

Started 2026-11-14 12:36 (log `$PROJ/logs/queue_liver_cc_20261114_123637.log`).

## Later

- **Follow-up arms, after the first 4 results are in** (`run_queue.py --arms aucReject aucRepair ccAucReject`,
  `run_queue.py --second-run <arm>`):
  - **AUC alone** (aucReject 730, aucRepair D105): mostly repeats cc, so only if the main results call for it.
  - **ccAucReject** (300): the strict reject twin of ccAucRepair.
  - **a second run of the best arm** (is its result run-to-run luck?).
- Backlog:
  - **Where does `all` put its false-positive tumours?** On the 20 real validation predictions (no test data): FP
    components inside vs outside the liver, their size and contrast. Tells whether a filter that wins removed the
    kind of error the model makes. Cheap, CPU.
  - Random subset of the best reject arm's size (quantity vs quality); train on the rejects (control).
  - **Contrast phase (2026-11-14):** MAISI has no phase input (`modality` 1 = generic CT; 2/3 = without/with
    contrast exist in its mapping but are undocumented for rflow-ct). `liver_metrics.py` measures the pool's phase mix
    from vessel HU. Later: estimate the phase of the 111 LiTS test cases (TotalSegmentator) → tumour Dice per phase;
    phase-relative liver HU check (liver vs spleen/aorta); test whether MAISI reacts to modality 2/3 (same mask +
    seed, GPU, minutes).
- Pancreas: Dataset102 (MSD07 val/test split, real reference from its val cases), `all` + the best liver arms.
- Paper caveats: leakage (MAISI trained on MSD Task03 = LiTS, shared by every arm), 1.5 mm label steps in synthetic
  masks, thesis baseline trained on broken LiTS headers (28–47), synthetic-only training (defence in NOTES 2026-11-07).
