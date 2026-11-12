# Plan: filtering + nnU-Net training on the MAISI pool (rewritten 2026-11-12)

Question: does a synthetic training set with the faulty volumes **removed (reject)** or **fixed (repair)** beat training
on all 999 synthetic liver volumes, for real LiTS liver/tumour segmentation? Liver first; pancreas (MSD07) later with
the same code. Decisions with dates and reasons are in NOTES.md "Decisions"; candidate filters with arguments in
`PLAN_filtering_options.md`.

## Design (fixed)

- **Training data: synthetic only** (MAISI pool, 999 liver volumes, 512³ @ 0.8×0.8×1.0, cropped only at MAISI's constant
  padding to 480×480×384). Arms differ only in which synthetic volumes / labels they use.
- **Validation: 20 real LiTS cases** (thesis train split, seed 42), used by nnU-Net during training. **Test: the other
  111 LiTS cases** (incl. the 40 thesis test cases), never used for any choice. `checkpoint_final` for test (no
  selection on val), default `nnUNetTrainer`, 3d_fullres, 1000 epochs, `nnUNet_compile=f`, Dataset101's plans.
- **One preprocessed dataset per label set:** an arm that only selects volumes is a new fold of Dataset101
  (`make_splits.py`, append-only `arms.json`); an arm that changes labels gets its own dataset with Dataset101's
  plans copied (same patch size, spacing, normalisation) and the same 20 val cases.
- **Filtered arm = every volume that passes** (no fixed K). Beating all 999 with fewer volumes is itself the argument;
  a size-matched random arm only if needed (backlog).
- **Noise floor** (arm `all` trained twice): per case |Δ Dice| ≈ 0.03, mean differences < 0.01 (liver p 0.50,
  tumour p 0.07). An arm's gain has to clear this.
- **Thresholds never come from the test set.** The 15 mm rule is the thesis' fixed rule; the AUC cut comes from the
  20 real validation cases.

## Arms

| fold / dataset | arm | training data | status |
|---|---|---|---|
| D101 f0 | `all` | 999 synthetic | done: liver 0.870, tumour 0.301 (111 test) |
| D101 f1 | `all_rep2` | same, second run (noise floor) | done: liver 0.876, tumour 0.305 |
| D103 f0 | **`repair`** | 999, tumour components with < 50% of voxels within 15 mm of the liver → background | queued (1st) |
| D101 f2 | **`reject15mm`** | volumes with ≥ 90% of tumour voxels within 15 mm of the liver (thesis rule, ~621) | queued (2nd) |
| D101 f3 | **`rejectAUC`** | volumes whose tumour is visible in the image: volume-weighted separability AUC ≥ 5th percentile of real lesions | queued (3rd) |

The fold numbers are assigned in this order by the queue (they are looked up by arm name, never assumed).
2×2 view for the paper: criterion (geometry vs label–image consistency) × action (reject vs repair). This session
runs three of the four cells; `repair by consistency` (invisible tumour → liver or nnU-Net `ignore`) and
`train on the rejects` (control) are the next candidates.

### Separability AUC (family 2, `maisi/filter_metrics.py`)

Per tumour component: core = component eroded by one voxel (MAISI label edges are 1.5 mm steps); shell = voxels
2–10 mm outside it, not tumour, restricted to the liver if the liver is ≥ 25% of the shell. AUC = P(core HU > shell
HU) on integer HU, sep = max(AUC, 1 − AUC) ∈ [0.5, 1] (hypo- and hyperdense both count; 0.5 = invisible). Components
with < 10 measurable voxels get no score. Volume score = voxel-weighted mean of the component seps. The same code on
each lesion of the 20 real val cases gives the reference; **cut = its 5th percentile** ("less visible than 95% of
real tumours"). Volumes with no scored tumour pass (nothing to judge).

### Evaluation (`nnunet/predict_test.py eval`)

Per case Dice liver / liver+tumour / tumour, global tumour Dice; **lesion-wise** (26-connected tumour components;
detected = any overlap): recall (all, ≥ 1 ml), precision, FP components and FP volume per case, tumour-free cases
with any predicted tumour. Every arm vs `all` f0, paired per case + Wilcoxon; `all_rep2` vs `all` is the noise row.
Both on all 111 and on the 40 thesis test cases. Output: `$nnUNet_results/test_eval_liver/summary.md`.

## The GPU session (full B200, ~23 h, can be extended)

One command, everything else is automatic and resumable (`nnunet/run_queue.py`):

```bash
cd "$PROJ/BT2026" && git pull
source nnunet/env.sh && python nnunet/run_queue.py --dry-run      # check the plan
setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && python "$PROJ/BT2026/nnunet/run_queue.py"' \
  > "$PROJ/logs/queue_liver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

| step | what | time |
|---|---|---|
| eval-all | lesion-wise re-evaluation of `all` f0/f1 | minutes |
| metrics | `filter_metrics.py` families 1a + 2 → `ids_thesis15mm.json`, `ids_auc.json` | ~20–30 min |
| build | Dataset103 raw (repaired labels, hard-linked images) + Dataset101 plans | ~10 min |
| preprocess | `nnUNetv2_preprocess -d 103` (+69 GB) | ~25 min |
| repair | train D103 f0 → predict 111 → eval | ~4.9 + 2 h |
| reject15mm | train D101 f2 → predict → eval | ~4.9 + 2 h |
| rejectAUC | train D101 f3 → predict → eval | ~4.9 + 2 h |

GPU idles for the first ~1 h (accepted, so repair stays first). Watch: the status file
`$PROJ/QUEUE_liver_arms__<k>of16_<step>__ETA_…_CET.txt`, the training status files `NNUNET_D10x_…`, and
`$PROJ/queue_liver/progress.md` (step times, pass counts, val Dice, test tables) → copy the key numbers into
NOTES.md. A failed step skips only what depends on it. After a session end: run the same command again.
While `repair` trains, check `filter_metrics/liver/summary.md` (repair split, AUC distributions, cut); if the AUC
arm looks wrong, fix it before the queue reaches it (`--arms` limits the queue).

## Later

- Pancreas: Dataset102 (MSD07 val/test split, real reference for the AUC cut), `all` + the best liver arms.
- More arms if the first three are informative: repair by consistency, train on the rejects, a second run of the best arm.
- Paper caveats: leakage (MAISI trained on MSD Task03 = LiTS, shared by every arm), 1.5 mm label steps in synthetic
  masks, thesis baseline trained on broken LiTS headers (28–47), synthetic-only training (defence in NOTES 2026-11-07).
