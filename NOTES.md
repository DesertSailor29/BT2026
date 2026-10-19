# Project notes

## Current state (2026-10-19)
- Direction: **Option B**. Sweep: **Option 1** (Path A, tumor-size sweep), 512³ @ 0.8×0.8×1.0 mm. Targets: liver (hepatic tumor) + pancreas (pancreatic tumor, eval on MSD Task07).
- **Generation code: `maisi/`** (committed + pushed): `GUIDE.md` (step-by-step for a fresh UCloud Terminal session), `env.sh`, `setup_maisi.sh` (uv + Py3.11, torch 2.9.1+cu128, MONAI 1.6.1, NV-Generate-CTMR @ dba166e, weights), `make_jobs.py`, `run_jobs.py` (resumable, per-job subprocess, workers, `--drop-raw-image`), `postprocess.py` (132→0/1/2, nnU-Net naming, metrics incl. thesis 15 mm metric + recomputed MAISI QC), `summarize.py` (summary.md incl. measured disk/volume + 1000-vol projection, PNG previews), `orient.py` (RAS reorientation; pairs are saved RAS; CLI for LiTS/MSD07, verified bit-exact on 3 LiTS cases LPS/LAS/RAS). Tested end to end on the Mac with a fake MAISI (resume, retry, 2 workers, QC parsing, detached tumor). **Not yet run on real MAISI.**
- Setup (GUIDE steps 1–3) now runs on a CPU-only session (GPU checks optional in `setup_maisi.sh`; GPU check moved to step 4). UCloud fractions are **k separate 1g.23gb MIG slices** (not 3g/4g profiles); one process = one slice, so a volume must fit 23 GB. Changes: `make_jobs.py` defaults now window 64×64×48 + 8 splits (NVIDIA 24 GB preset with more splits); `run_jobs.py` pins worker i to MIG slice i (`--gpu` override, `expandable_segments`); GUIDE step 5 loop starts one worker per slice.
- Optional Claude Code on UCloud (GUIDE section; login kept in `$PROJ/.claude-config`). CLAUDE.md + NOTES.md are in the repo for both machines.

## Next steps
1. **User on UCloud:** follow `BT2026/maisi/GUIDE.md`: steps 1–3 on a CPU session, step 4 smoke test on 1/7 (does 512³ fit 23 GB? fallback `--ae-splits 16`, else full B200), steps 5–6 on 4/7 = 4 workers. Report what step 7 asks for: driver/CUDA, summary totals, time/vol, peak GB, axcodes, previews.
2. Based on the trial: fix the pool size/workers and decide whether 1000 is per target or total; decide on tumor `cfg_guidance_scale`.
3. Reorient LiTS (and later MSD07) to RAS on UCloud with `orient.py` (GUIDE step 6b). LiTS split is at `$PROJ/nifti_data` (91/40); `$PROJ/LiTS/` is empty.
4. venv-nnunet for B200 (rewrite `setup.sh`: torch cu128, nnunetv2 2.6.x).
5. Filtering metrics module (candidates in Progression Ideas; basic ones are already in postprocess metrics).

## Decisions
- 2026-10-19: **CLAUDE.md + NOTES.md live in the repo** (BT2026/) so Claude on the Mac and on UCloud share them via git (replaces "notes outside the repo"). Research/CLAUDE.md on the Mac only imports BT2026/CLAUDE.md. The notes are public with the repo.
- 2026-10-19: Setup on a CPU session (saves GPU hours). Generation on **23 GB MIG slices, one worker per slice** (full GPUs hard to get; 4/7 = 4 slices = 4 workers). AE tiling = NVIDIA 24 GB preset (window [64,64,48], overlap 0.4) with 8 splits instead of 2, used for ALL volumes (consistency across hardware). Trade-off: window 64×64×48 vs 80³ may give faint tile seams (NVIDIA docs); check previews. Fallback 16 splits, then full B200.
- 2026-10-18: **Sweep = Option 1** (Path A, hepatic tumor size sweep). Geometry 512×512×512 at **0.8×0.8×1.0 mm** (LiTS median in-plane 0.77 mm, median slice 1.0 mm, from LiTS/scan_stats_all_planes.csv). Trial: 10 pairs per organ, then ~1000. Second target: **pancreas + pancreatic tumor** (Path A, `pancreatic tumor` size), evaluated on MSD Task07. UCloud app: Terminal (Ubuntu).
- 2026-10-18: Option B; local rflow-ct (seeds, newest model, no rate limits).
- 2026-10-18: no Claude attribution in commits.
- 2026-10-18: **MAISI out of the box**: no extra QC retries or discarding on our side. (In Path A, MAISI itself saves QC-failed volumes and only logs them; its 2-retry rule exists only in Path B.) The runner only records pass/fail. User: test it as it comes, research what helps afterwards.
- 2026-10-18: **All data in RAS** (MAISI pairs + LiTS + MSD07), via axis flips only. LiTS was mixed: 63 LPS, 35 RAS, 33 LAS. The user notes nnU-Net may be robust to this anyway (mirroring augmentation); it's cheap insurance.
- 2026-10-18: Trial = 10 sizes 0.05..0.95 per target (probes the extremes), seeds 0–19; the pool will use the thesis grid 0.1..0.9. One MAISI subprocess per job (robust and resumable; the model-load overhead is acceptable). torch 2.9.1+cu128 + MONAI 1.6.1 (MONAI 1.6 needs torch ≥2.8). Python via uv, everything in /work (UCloud home is ephemeral).

## Open questions
- UCloud: which app/image, job time limit, persistent `/work` path, internet access (GitHub, HuggingFace, PyPI), driver/CUDA version from `nvidia-smi`?
- ~~Is LiTS on UCloud?~~ yes, `$PROJ/nifti_data`. Are the 90 thesis MAISI volumes there? Where are the old NIM scripts?
- How many synthetic volumes (target pool size)?
- **Leakage:** MAISI's diffusion model and ControlNet were trained on MSD Task03 Liver (88–105 vols), which is built from LiTS. AbdomenCT-1K, also in its training data, partly reuses LiTS as well. Our LiTS test cases may have been seen by MAISI.
- Is the ~1000 volume pool per target or in total?
- ~~Second target~~ decided: pancreas + pancreatic tumor (MSD Task07, https://kaiko-ai.github.io/eva/main/datasets/msd_task7_pancreas/).
- MIDL 2027 full paper due around 3 Dec 2026 (about 6.5 weeks away), short paper around 15 Apr 2027. Which one are we aiming for?

## Key facts
- Stable facts (MAISI gotchas, thesis numbers) are in `CLAUDE.md`.
- Run pattern on UCloud: `nohup bash -c 'source venv/bin/activate && export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=... && <cmd>' > <name>_$(date +%Y%m%d_%H%M%S).log 2>&1 &`

## Log
- 2026-10-19 (UCloud, CPU session): Install check. MAISI setup (GUIDE 1–3) complete: log setup_maisi_20261019_134830.log has no errors; venv-maisi = Py 3.11.16, torch 2.9.1+cu128 (CUDA 12.8), MONAI 1.6.1, nibabel/scipy/matplotlib import OK; NV-Generate-CTMR @ dba166e; 5 rflow-ct weights (4.7 GB). uv comes from the image (/usr/local/bin). Data: `$PROJ/nifti_data` = LiTS split (91 Tr / 40 Ts + split_mapping.csv); `$PROJ/LiTS/` has only empty folders. Not yet: GPU check (no GPU here), venv-nnunet, MSD07.
- 2026-10-19 (UCloud): First Claude Code session on UCloud. Login confirmed persisted in `$PROJ/.claude-config` (.credentials.json, settings, history). It only works if `env.sh` is sourced before `claude` (else CLAUDE_CONFIG_DIR is unset and it uses ephemeral ~/.claude → new login). Session was started in /work/Research, not BT2026, so CLAUDE.md was not auto-loaded.
- 2026-10-19 (Mac): Moved CLAUDE.md + NOTES.md into BT2026 (CLAUDE.md rewritten for two machines, Mac-only memory facts folded in); .gitignore ignores Claude per-machine files; GUIDE starts claude in $PROJ/BT2026. Committing + pushing per user; user then installs on UCloud.
- 2026-10-19: User asked about running Claude Code on UCloud. Added to env.sh: CLAUDE_CONFIG_DIR=$PROJ/.claude-config (login persists) + ~/.local/bin on PATH; GUIDE "Optional: Claude Code on UCloud" (reinstall binary per session, headless login via URL + pasted code). Note: a Claude on UCloud does not see NOTES.md/CLAUDE.md (they live on the Mac, outside the repo).
- 2026-10-19: Correction: user's screenshots show 3/7 = "3 MIGs (B200-mig.1g.23gb)", 18 vCPU, 108 GB RAM; 4/7 = 4 MIGs, 24 vCPU, 144 GB RAM (RAM is CPU RAM). NVIDIA presets at dba166e: 80g 512³ = [80,80,80] 4 splits (~45 GB), 32g = [80,80,48] 4 splits (28.4), 24g = [64,64,48] 2 splits (22.8 GB, torch max_memory_allocated). Implemented MIG pinning + new defaults; tested pinning with fake nvidia-smi (4 MIG → 4 workers on distinct UUIDs).
- 2026-10-19: User can't easily get a full B200, so the guide now recommends a 3/7 slice. Fetched MAISI performance table (512³ rflow on A100: DM 19 s + VAE 32 s, peak 45.3 GB with 2 splits; 24 GB preset 22.8 GB uses window [64,64,48] → stitching risk).
- 2026-10-19: User asked about CPU-only setup + B200 MIG options (DeiC Interactive HPC SDU/K8s: 1/7 = 1g.23gb, 6 vCPU, 36 GB RAM; slider up to 8 GPUs). Made setup GPU-optional, rewrote GUIDE step 0 (two sessions + machine table), GPU check in step 4.
- 2026-10-18: Per user: no QC retry/discard logic (MAISI as shipped), all data to RAS (added orient.py, pairs RAS), disk usage measured per job and projected in summary.md (2 TB allocated). GUIDE updated (QC wording, step 6b reorient, disk). Fake end-to-end test still passes.
- 2026-10-18: Wrote BT2026/maisi/ (guide + 6 scripts) for the 10 liver + 10 pancreas trial and tested it with a fake MAISI. MAISI code facts: models download to ./models relative to cwd; output files are sample_<timestamp>_{image,label}.nii.gz; QC = organ median HU vs configs/image_median_statistics_ct.json; Path A retries the mask 5× if the target label is missing. A100 reference: 512³ rflow image ≈ 51 s (+ mask DM, I/O). Found LiTS mixed orientations. Fixed .gitignore case clash.
- 2026-10-18: User picked Option 1 + pancreas secondary; spacing 0.8×0.8×1.0; Ubuntu Terminal app. Started writing BT2026/maisi/ (guide + scripts for a 10+10 trial).
- 2026-10-18: Read thesis, Progression Ideas, BT2026. Researched MAISI (NV-Generate-CTMR docs, NIM API, MAISI paper). Set up CLAUDE.md + NOTES.md. Cleaned the repo (commit 1d1f3eb, unpushed). Wrote MAISI_generation_options.pdf (3 pages, 4 sweep options + sources; generator: scratchpad make_pdf.py, not kept). No Claude co-author existed in BT2026 history/GitHub, so nothing to rewrite.
