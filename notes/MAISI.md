# MAISI and environment facts (so they don't need re-researching)

Read when generating, patching or setting up. How-to for UCloud: `maisi/GUIDE.md`.

## MAISI
- MAISI = github.com/NVIDIA-Medtech/NV-Generate-CTMR, pinned at `dba166e`. Params: `docs/inference.md`,
  `configs/config_infer*.json`; memory/time table: `docs/performance.md`. Weights: huggingface.co/nvidia/NV-Generate-CT.
  The thesis used the hosted NIM v1 API (docs.nvidia.com/nim/medical/maisi/api-reference), no seeds.
- The v2 mask model accepts **only one** `controllable_anatomy_size` entry (the thesis used liver + tumour via NIM v1).
  Non-empty = Path A (mask generated; `body_region`/`anatomy_list` ignored). Empty = Path B (real mask from database).
- Bug at `dba166e`: `scripts/inference.py` calls `.to()` on the v2 mask checkpoint's `scale_factor`, a float →
  AttributeError in every Path A run. `setup_maisi.sh` patches the clone (`torch.as_tensor(...)`); keep the patch if
  the pin moves, unless upstream fixes it.
- Path A geometry bug at `dba166e`: the v2 mask model always makes a 256³ mask at native 1.5 mm, but `scripts/sample.py`
  conditions it on and labels it with the target spacing, then only pads to `output_size` → at 512³ @ 0.8 mm the anatomy
  is ~half scale (body ~180 mm). `maisi/patches/mask_native_spacing.patch` (applied by `setup_maisi.sh`) uses 1.5 mm for
  both, so MAISI resamples the mask (nearest) to the target spacing: 480×480×384 voxels of content, padded to 512³.
- **No contrast-phase control** (checked 2026-11-14): the image diffusion UNet only takes spacing + a `modality` class
  integer (`configs/modality_mapping.json`: `ct` 1, `ct_wo_contrast` 2, `ct_contrast` 3, MR 8+); docs list only 1 for
  CT; untested whether rflow-ct learned 2/3. We generate with 1 (`make_jobs.py`). Measured per pool volume by
  `maisi/liver_metrics.py` (vessel HU from the raw 132-class mask: aorta 6, IVC 7, portal + splenic vein 17, hepatic
  vessel 25, spleen 3, kidneys 5/14).
- rflow-ct needs `num_inference_steps` 30; the shipped 80g presets say 1000 (a DDPM value).
- `scripts.download_model_data` writes models to `./models` relative to the CWD; run MAISI from inside its clone or use
  absolute paths in `environment.json` (`run_jobs.py` does the latter).
- Path A with `num_output_samples` 1: a QC-failed volume is still saved (log line "quality check for label 'X'
  failed"). A missing target label makes MAISI retry the mask 5×, then it continues anyway.
- Our settings: 512³ @ 0.8×0.8×1.0, AE window 64×64×48 / overlap 0.4 / 8 splits, 4 workers per full B200 (102 s/vol),
  torch threads capped at cgroup quota / workers. 512³ does not fit a 23 GB MIG slice.

## Software
- B200 (sm_100) needs a CUDA 12.8+ torch build: venv-maisi = torch 2.9.1+cu128 + MONAI 1.6.1 (MONAI 1.6 needs torch
  ≥ 2.8). venv-nnunet = nnunetv2 2.6.4 with torch 2.15.0+cu132 (2.6.4 excludes torch 2.9). Default nnUNetPlans
  3d_fullres, `nnUNet_compile=F` (as the thesis).

## Thesis reference
- Thesis MAISI config: 512³ @ 0.75 mm, body_region abdomen, hepatic tumour ∈ {0.1..0.9} × liver ∈ {0.4, 0.5, 0.6},
  90 jobs, 82 usable. Thesis filter: keep a volume if ≥ 90 % of tumour voxels are within 15 mm of the liver.
  LM30-filtered vs LM0 tumour Dice 0.704 vs 0.653, p = 0.033 (1000 epochs).
- Progression options (Progression Ideas.docx): A real-vs-synthetic equivalence, **B automatic filtering** (chosen),
  C synthetic-only nnU-Net → SAM prompting, D nnU-Net uncertainty → SAM on uncertain regions.

## Data
- LiTS split at `$PROJ/raw_data/LiTS` (RAS copy `LiTS_RAS`), MSD07 at `$PROJ/raw_data/Task07_Pancreas`. All data RAS
  (axis flips only). nnU-Net: Dataset101 (999 synthetic + 20 real val), D103 RepairCC, D104 RepairCCAUC; real val =
  20 LiTS from the thesis train split (seed 42), test = the other 111. MSD07 split 43 val / 238 test
  (`maisi/msd07_split.json`).
- UCloud git push via a fine-grained PAT in `$PROJ/.git-credentials` (expires **2026-12-18**).
