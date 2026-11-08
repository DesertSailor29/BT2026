# Plan: filtering + nnU-Net training on the MAISI pool (draft 2026-11-07)

Goal: does filtering bad synthetic volumes (Option B) beat using all of them, for LiTS liver/tumour segmentation?
Liver only for now (999 synthetic liver volumes); the pancreas half (MSD07) uses the same code later.

## Phase 1: CPU session (start now, while the pool finishes on the GPU session)

Nothing here needs a GPU. The GPU session's 48 cores are busy with the 4 MAISI workers, so heavy CPU work
(header fix, metrics, nnU-Net preprocessing) goes into a separate CPU session; ask for many cores and
≥ 256 GB RAM (nnU-Net preprocessing of 512³ volumes needs several GB per process).

1. **LiTS header fix + RAS** (old Next step 0, GUIDE 6b): `fix_lits_headers.py` (dry run → 40 fix + 5 ok),
   `orient.py --check`, convert to `$PROJ/raw_data/LiTS_RAS`. Everything below uses LiTS_RAS.
2. **venv-nnunet** for B200: rewrite `setup.sh` → `nnunet/setup_nnunet.sh` (uv, Py 3.11, torch 2.9.1+cu128,
   nnunetv2 2.6.x, paths in `$PROJ`, `env.sh`-style `nnunet/env.sh` that sets nnUNet_raw/preprocessed/results).
3. **Filter metrics** `maisi/filter_metrics.py` → `$MAISI_RUNS/pool/filter_metrics.csv`, one row per volume,
   resumable, parallel. Already in `results.jsonl`: MAISI QC, tumour present, components, largest share, organ/tumour
   median HU, 15 mm fraction. New (cheap, CPU):
   - tumour–liver contrast: ΔHU, CNR = |μ_liver − μ_tumour| / σ_liver; tumour HU in soft-tissue range;
   - topology: liver components/holes (Euler number), tumour fraction inside the liver mask, stray labels outside the body;
   - image sanity: HU percentiles in the body, air inside the liver, tile-seam score (intensity jumps at the
     VAE window period 64/64/48);
   - mask–image consistency: liver HU histogram vs real LiTS liver HU (from LiTS_RAS).
   The model-agreement metric (Phase 3) is added to the same CSV later.
4. **Dataset builder** `nnunet/build_datasets.py`:
   - real: LiTS_RAS train 91 → fixed 20 for validation (seed 42, same idea as the thesis), 71 train; test 40 untouched.
   - synthetic liver pairs (labels already 0/1/2, RAS): **crop in z** to the liver bbox ± margin before adding
     (MAISI makes a whole 512 mm torso, LiTS is mostly abdominal crops; cropping cuts disk + preprocessing ~2–3×
     and matches the real FOV). Decision needed (D3).
   - **one nnU-Net dataset with everything** (Dataset100_LiTSMaisiPool), preprocessed **once**; each experiment arm is
     a fold in `splits_final.json` (train = 71 real + the arm's synthetic ids, val = the same 20 real). Saves
     preprocessing and disk N-fold, and all arms share one plan.
   - plans from the **real data only** (Dataset099_LiTS: plan_and_preprocess), then transferred to Dataset100
     (`nnUNetv2_move_plans_between_datasets`), so spacing/patch size/normalisation don't depend on which synthetic
     volumes are in. Dataset099 is also the LiTS-only baseline.
5. **Preprocess** Dataset099 + Dataset100 (detached, logged). Estimate disk first on ~20 volumes.
6. **Eval script** `nnunet/evaluate.py`: predict the 40 LiTS test cases per arm, per-case Dice liver + tumour,
   tumour Dice on the case level and lesion-wise detection (F1), paired Wilcoxon vs baseline and vs "all",
   tables + figures in the shared style (`viz.py`). Reuse `compare_predictions.py` where it fits.

All of this is tested on a few volumes in the CPU session (nnU-Net can run a 1-epoch CPU smoke train).

## Phase 2: GPU, baselines

| arm | training data |
|---|---|
| R | 71 real (Dataset099) |
| A | 71 real + all 999 synthetic |
| A-K | 71 real + K random synthetic (size control, see D1) |

3d_fullres, 1000 epochs (as the thesis), `checkpoint_final` (no selection on val). Measure s/epoch on the B200
first; several trainings can share one B200 (180 GB), so check 2–4 in parallel the way `bench_workers.py` did.

## Phase 3: filtering arms (same K synthetic each, so only the selection differs)

| arm | rule | passes now (liver, of 999) |
|---|---|---|
| F-15mm | thesis rule: ≥ 90% tumour within 15 mm of the liver | 621 |
| F-rules | 15 mm + tumour present + ≤ 3 tumour components + organ 1 component | ~400 |
| F-contrast | 15 mm + tumour ≥ 10 HU darker than liver (or CNR threshold) | 492 |
| F-agree | model agreement: R predicts each synthetic volume, keep the top K by Dice(prediction, MAISI mask) | rank-based |
| F-combo | best of the above combined | — |

(MAISI QC alone removes 6/999: not worth an arm.) F-agree needs R trained first, then ~999 predictions
(~1–2 GPU h). Every arm is compared with the random arm of the same size (A-K), otherwise "filtering helps" can just
mean "fewer synthetic volumes help".

## Decisions needed

- **D1 size control.** Fixed budget K per arm (e.g. K = 300 so every filter has enough passes; filter = ranking +
  cut) vs "everything that passes" (N differs per arm). Recommend fixed K plus the "all" arm.
- **D2 real:synthetic ratio.** 999 synthetic vs 71 real means ~93% of patches are synthetic. Option: a small custom
  trainer that draws 50% of each batch from real cases. Recommend: first plain sampling with K = 300 (~1:4), add
  the trainer only if A is clearly worse than R.
- **D3 z-crop of synthetic volumes** to the liver ± margin (recommend yes, e.g. ± 50 mm).
- **D4 runs per arm.** 1 seed per arm is noisy; 2–3 seeds for the main arms (R, A-K, best filter) if GPU hours allow.
- **D5 epochs**: 1000 (thesis) or fewer for screening (e.g. 300 for all arms, 1000 for the final ones).

## Known caveats

- Leakage: MAISI was trained on MSD Task03 (= LiTS); some LiTS test cases may have been seen by MAISI.
- Synthetic labels are 1.5 mm steps (nearest upsampling), real labels are fine.
- The thesis baseline trained on broken headers for 28–47; R here is the corrected baseline.
