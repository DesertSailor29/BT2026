# Plan: filtering + nnU-Net training on the MAISI pool (rewritten 2026-11-12, arms redesigned the same day)

Question: does a synthetic training set with the faulty tumour labels **removed (reject)** or **fixed (repair)** beat
training on all 999 synthetic liver volumes, for real LiTS liver/tumour segmentation? Liver first; pancreas (MSD07)
later with the same code. Decisions with dates and reasons are in NOTES.md "Decisions"; the candidate filters with
arguments in `PLAN_filtering_options.md`.

## Design (fixed)

- **Training data: synthetic only** (MAISI pool, 999 liver volumes, 512³ @ 0.8×0.8×1.0, cropped only at MAISI's constant
  padding to 480×480×384). Arms differ only in which synthetic volumes / labels they use.
- **Validation: 20 real LiTS cases** (thesis train split, seed 42), used by nnU-Net during training. **Test: the other
  111 LiTS cases** (incl. the 40 thesis test cases), never used for any choice. `checkpoint_final` for test (no
  selection on val), default `nnUNetTrainer`, 3d_fullres, 1000 epochs, `nnUNet_compile=f`, Dataset101's plans.
- **One preprocessed dataset per label set:** an arm that only selects volumes is a new fold of Dataset101
  (`make_splits.py`, append-only `arms.json`); an arm that changes labels gets its own dataset with Dataset101's
  plans copied (same patch size, spacing, normalisation) and the same 20 val cases.
- **Noise floor** (arm `all` trained twice): per case |Δ Dice| ≈ 0.03, mean differences < 0.01 (liver p 0.50,
  tumour p 0.07). An arm's gain has to clear this.

## Filter rules (fixed 2026-11-12, *before* looking at the pool metrics)

**Unit = tumour component** (26-connected set of tumour voxels). A criterion C says pass / fail per component.

- **reject C:** keep the volumes in which every component passes (tumour-free volumes pass). Drop the rest.
- **repair C:** keep all 999 and relabel every failing component (the new label depends on C, below).

Both arms use the same C and the same cut. So the reject set is exactly the set of volumes that repair leaves
unchanged, with identical labels: **the two arms differ only in whether the repaired volumes are included.** A
filtered arm keeps every volume that passes (no fixed K); beating all 999 with fewer volumes is itself the argument.

**Principle: a component fails if it is unlike every real lesion.** The real reference = the lesions of the 20 real
validation cases (Dataset101 `labelsTr/lits_*`), measured by the same code. Test cases are never read.

| criterion | a component fails if | repair relabels it to | parameter |
|---|---|---|---|
| **Touch** (geometry) | none of its voxels is 26-adjacent to a liver voxel | background (LiTS labels only liver tumours, so outside the liver the true label is background) | none |
| **AUC** (label–image consistency) | separability `sep` < **cut = the minimum `sep` over the scored real val lesions** | the label of its surroundings: liver if ≥ 50% of its outer 26-neighbour shell is liver, else background | none (cut from real data) |

- Touch has no tolerance: MAISI's 1.5 mm mask is resampled to 0.8 mm with nearest neighbour, which cannot open a gap
  between labels, so a gap in the label is a real gap.
- Separability `sep` per component: core = the component eroded by one voxel (MAISI label edges are 1.5 mm steps);
  shell = voxels 2–10 mm outside it, not tumour, restricted to the liver if the component sits in the liver (≥ 50% of
  its outer shell is liver, the same "surroundings" as the repair) or the liver is ≥ 25% of the shell. AUC =
  P(core HU > shell HU), `sep` = max(AUC, 1 − AUC) ∈ [0.5, 1] (hypo- and hyperdense both count; 0.5 = invisible).
  Components with < 10 measurable voxels get no score and **pass** (nothing to judge).
- The two criteria are judged independently (the AUC arms ignore touch and vice versa); a combined arm comes later.
- Descriptive only (no decisions): contact share (part of the outer shell that is liver), min / median distance to the
  liver, the thesis 15 mm fraction (for comparison with the thesis rule, which is volume-level and so can't drive a
  repair), MAISI QC.

**Why these rules:** no hand-tuned threshold (a reviewer can't ask "why 15 mm / 90%?"). The rule is per component, so
it catches a small stray blob next to a large tumour (invisible to a voxel fraction) and doesn't throw away a good
volume because of one stray blob. A percentile cut (e.g. p5) would fail 5% of real-like lesions by construction, so
"any component fails → reject" would drop realistic multi-lesion volumes. A minimum-based cut and a definitional
rule avoid that.

**Sanity checks (CPU session).** A rule changes only if one of these fails, recorded in NOTES.md with the reason:
1. Real val lesions: how many touch the liver (expected: all). Any that don't are viewed (label error?).
2. The 5 least visible real lesions are viewed. If the minimum is a label error (tumour label on normal liver), that
   lesion is left out of the reference and the cut becomes the next one.
3. Synthetic: classes per criterion (no tumour / clean / repaired / tumour-free), touch × AUC cross-tab, near misses
   (fails touch but within 3 mm of the liver), contact sheets of failing components.
4. Not degenerate: each reject arm keeps ≳ 100 volumes and each repair changes ≳ 20 volumes; otherwise the arm is
   pointless, so we rethink before spending GPU time on it.

## Arms

| fold / dataset | arm | training data | status |
|---|---|---|---|
| D101 f0 | `all` | 999 synthetic | done: liver 0.870, tumour 0.301 (111 test) |
| D101 f1 | `all_rep2` | same, second run (noise floor) | done: liver 0.876, tumour 0.305 |
| D103 f0 | **`repairTouch`** | 999, components not touching the liver → background | queued (1st) |
| D101 f2 | **`rejectTouch`** | volumes in which every component touches the liver | queued (2nd) |
| D104 f0 | **`repairAUC`** | 999, invisible components → liver / background (surroundings) | queued (3rd) |
| D101 f3 | **`rejectAUC`** | volumes without an invisible component | queued (4th) |
| later | `repairBoth` (D105), `rejectBoth`; `<best>_rep2` | see "Later" | after the first 4 |

The queue assigns fold numbers by arm name (it looks them up and never assumes them). For the paper this is a 2×2:
criterion (geometry vs label–image consistency) × action (reject vs repair). Dropped for now: the thesis 15 mm reject
arm. Its ids are still written (`ids_thesis15mm.json`) for comparison.

### Evaluation and analysis (decided 2026-11-12; `nnunet/predict_test.py eval`)

The study is **exploratory**: no predictions are written down in advance (synthetic data can behave unexpectedly);
mechanisms seen here become hypotheses for future work. We compare from as many sides as possible, but name one
main comparison so that a single win can't be called cherry-picked:

- **Main metric: per-case tumour Dice on all 111 test cases** (comparable with the thesis).
- **Baseline: the per-case mean of the two `all` runs** (f0, f1). It uses both runs, has less noise, and involves no
  choice made on test data (picking "the better run" would be one, and that run is partly lucky). Each arm is also
  reported against f0 and f1 separately.
- **Clear win** = paired Wilcoxon (arm vs baseline, per case) significant after **Holm** over the 4 arms (sort the
  p-values, the smallest must be < 0.05/4, the next < 0.05/3, …, stop at the first that fails) **and** a mean
  difference above the noise floor (> 0.01; `all_rep2` vs `all` gave < 0.01). Every difference also gets a paired
  bootstrap 95% CI, which shows how large the effect is.
- **Secondary, exploratory** (no correction, labelled as such): liver and liver+tumour Dice, global tumour Dice,
  lesion-wise recall (all, ≥ 1 ml), precision, FP components and FP volume per case, tumour-free cases with predicted
  tumour, the 40 thesis test cases, and repair C vs reject C (the action at a fixed criterion).

Implemented in `predict_test.py eval` (`--baseline 101:0 101:1`, `--family` = the 4 main arms; the queue passes both).
The summary also shows the noise row (`all_rep2` − `all` with its CI) and each arm against each `all` run separately.
Lesion-wise metrics: 26-connected tumour components; detected = any overlap. Output:
`$nnUNet_results/test_eval_liver/summary.md`.

## Amendment candidates after the measure session (planned 2026-11-14, nothing decided on counts yet)

Prompted by the review figures (`reports/figures/filter_review/`). No filter arm has been trained, so no arm result
can have shaped these changes; they are recorded as amendments with the figure that prompted each.

1. **Touch v2 = "connected to the main liver" (decided).** Main liver = the largest 26-connected liver piece. A tumour
   component passes if it lies in the 26-connected component of liver ∪ tumour that contains the main liver. Reason:
   liver_0842 c4 (`maisi_stray_island.png`) touches only a detached speck of liver label, which v1 lets through.
   No parameter. Replaces v1 in every touch arm (reject, repair, both) once the counts are in.
2. **Tumour-free after repair (open, decide on counts):** volumes in which every tumour component fails. Keep as
   tumour-free training volumes (repairTouch as designed) or delete (user's proposal: "if no tumour remains, delete").
3. **Detached liver pieces (open, measure first):** liver label not connected to the main liver (e.g. the blob next to
   the colon in liver_0948). Leave / relabel to background / reject, decided after comparing with the real livers.
4. **Liver plausibility check (open, measure first + human review):** catches whole bad volumes whose liver label
   covers bowel, fat or several organs, not liver tissue. Volume level, so reject only. Candidate measures (inside the
   liver label, core eroded by 2 voxels so label-edge steps don't count): median HU, IQR, p5/p95, share of fat
   (< −30 HU), air (< −200 HU) and bone (> 300 HU) voxels; liver volume (ml), number of pieces, share of the largest,
   centroid side (patient right?), craniocaudal extent; boundary edge strength (median HU gradient on the label
   surface: a label that follows no organ edge has a weak one). Reference = the 20 real val livers (same principle as
   AUC: a volume is suspicious if it is unlike every real liver); all 131 LiTS descriptive only. **No rule until the
   user has looked at the flagged cases and judged whether the flags match visibly bad livers.**

**Measure script `maisi/liver_metrics.py`** (descriptive, no decisions; reads the pool + the filter_metrics cache,
cross-checks component numbering against it). Per volume: liver pieces (ml, distance to the main liver), per tumour
component touch v1 / in-main v2, the liver measures above. Output `$MAISI_RUNS/pool/filter_metrics/liver/liver_check/`
(cache JSON, CSV, `summary.md`):
- counts per variant (touch v1, v2; AUC at cut 0.504 and 0.547; both): unchanged / repaired (≥ 1 tumour left) /
  tumour-free after repair (would be deleted) / no tumour from the start, and the resulting arm sizes (reject,
  repair-keep, repair-delete-tumour-free);
- liver pieces: pool vs real val vs all 131 (how many volumes have pieces, their sizes);
- each liver measure: real range vs pool distribution, synthetic volumes outside the real range per measure and in
  total (counts only).

Also counted: the **"keep the main structure" repair** (detached liver pieces and tumour outside the main structure
→ background, i.e. largest-connected-component cleaning of liver + tumour) and its reject twin (keep volumes where
nothing lies outside), so detached liver pieces can be filtered or repaired like tumours.

**Review figures** (`reports/figures/liver_review/`, shared style `maisi/viz.py`, image only + image + label):
- **Blind review first** (`blind/`, decided 2026-11-14): 30 flagged (outside the real val range on ≥ 1 liver measure)
  + 30 unflagged pool volumes, shuffled, shown only by code (B01..), axial at the liver's largest cross-section +
  coronal through its centroid. The user marks each ok / bad / unsure in `blind/marks.csv` without knowing the flags;
  `liver_metrics.py --score marks.csv` then compares marks and flags (key only on UCloud, `<out>/blind_key.json`).
  This tests whether the measures find visibly bad livers, which is stronger than viewing flagged cases alone.
- Then the open figures (`open/`): largest detached liver pieces (pool and real), all tumours that pass v1 but fail v2,
  per liver measure the most extreme pool volume beyond the real range next to the most extreme real val liver.
- Caveat: a new real liver falls outside the min–max of 20 real livers with probability ≈ 2/21 ≈ 10% per measure, so
  "outside on any of 10 measures" will also flag many good livers. `summary.md` shows this directly (real test livers
  outside); the blind review shows how many flags are visibly bad. A liver rule, if any, is chosen after both.

**Session:** written + fake-tested on the VM (small volumes; VM RAM is tight), then one UCloud CPU session
(16 cores / 48 GB, ~30–45 min, detached, status file), measuring only. Building D103/D104 waits for the decisions
(D103 depends on the touch definition; lits_049 / AUC cut still open).

## The CPU session (first; no GPU)

Two CPU sessions, so no preprocessing is wasted if a sanity check changes a rule:
1. **Measure only** (16 cores / 48 GB, `QUEUE_CPU_WORKERS=14`, ~40 min):
   `run_queue.py --prep-only --arms rejectTouch rejectAUC` = re-evaluation of `all`, filter metrics (pool + real
   val), sanity-check figures. Then we read `filter_metrics/liver/summary.md` and `figures/` and go through the
   sanity checks above.
2. **Build** (32 cores / 96 GB, `QUEUE_CPU_WORKERS=24`, ~1 h), once the rules hold: `run_queue.py --prep-only` =
   build + preprocess Dataset103 and Dataset104 (repaired labels, hard-linked images, Dataset101 plans, +69 GB each).

## The GPU session (full B200, ~28 h, can be extended)

The same command without `--prep-only`. The prep steps are already done and get skipped, so the GPU trains from the
start:

```bash
cd "$PROJ/BT2026" && git pull
source nnunet/env.sh && python nnunet/run_queue.py --dry-run      # check the plan
setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && python "$PROJ/BT2026/nnunet/run_queue.py"' \
  > "$PROJ/logs/queue_liver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

Per arm: train (~4.9 h) → predict 111 (~2 h) → eval (minutes) ≈ 7 h, so 4 arms ≈ 28 h. Watch the queue status file
`$PROJ/QUEUE_liver_arms__…_CET.txt`, the training status files `NNUNET_D10x_…` and `$PROJ/queue_liver/progress.md`,
and copy the key numbers into NOTES.md. A failed step skips only what depends on it. After a session end, run the
same command again.

## Later

- **Follow-up arms (decided 2026-11-12), after the first 4 results are in** (in code: `run_queue.py --arms repairBoth
  rejectBoth`, `run_queue.py --second-run <arm>`):
  - **combined touch + AUC**, reject and repair. A component fails if it fails either rule; the label is background
    if it fails touch, otherwise its surroundings' label. This asks whether stacking filters costs more data than it
    gains in quality. The measure-only session already gives the counts (touch × AUC table).
  - **a second run of the best arm** (is its result run-to-run luck?).
- Backlog:
  - **Where does `all` put its false-positive tumours?** On the 20 real validation predictions (no test data): FP
    components inside vs outside the liver, their size and contrast. Tells whether a filter that wins removed the
    kind of error the model makes. Cheap, CPU; fits the measure-only session.
  - Random subset of the best reject arm's size (quantity vs quality); train on the rejects (control).
- Pancreas: Dataset102 (MSD07 val/test split, real reference from its val cases), `all` + the best liver arms.
- Paper caveats: leakage (MAISI trained on MSD Task03 = LiTS, shared by every arm), 1.5 mm label steps in synthetic
  masks, thesis baseline trained on broken LiTS headers (28–47), synthetic-only training (defence in NOTES 2026-11-07).
