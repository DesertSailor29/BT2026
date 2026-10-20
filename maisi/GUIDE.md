# MAISI-v2 generation on UCloud: trial run

This guide goes from a brand-new UCloud session to **10 liver + 10 pancreas synthetic CT/label pairs** with varying
tumour sizes. It uses NVIDIA's open MAISI-v2 (`rflow-ct`) running locally on B200 MIG slices (23 GB each, one worker per slice).

Every command below can be copied as is. The only thing you type yourself is your folder name in step 1.

## What gets generated

| | Liver set | Pancreas set |
|---|---|---|
| MAISI size control (Path A) | `hepatic tumor` | `pancreatic tumor` |
| Tumour sizes (percentile) | 0.05, 0.15, …, 0.95 (10 jobs) | same |
| Seeds | 0–9 | 10–19 |
| Final label | 0 background, 1 liver, 2 tumour (like LiTS) | 0 background, 1 pancreas, 2 tumour (like MSD Task07) |

These settings are the same for every job:

| Setting | Value | Why |
|---|---|---|
| Volume | 512 × 512 × 512 voxels | |
| Spacing | 0.8 × 0.8 × 1.0 mm (FOV 410 × 410 × 512 mm) | LiTS median: 0.77 mm in-plane, 1.0 mm slices |
| Image diffusion steps | 30 | rflow-ct value (the 1000 in some presets is for DDPM) |
| Mask diffusion steps / guidance | 100 / 2.0 | MAISI defaults |
| Tumour guidance (`cfg_guidance_scale`) | 0 (off) | MAISI default; we can sweep it later |
| Autoencoder tiling | window 64×64×48, overlap 0.4, 8 splits | NVIDIA's 24 GB preset for 512³ with more splits, to fit a 23 GB MIG slice (see step 0) |

**MAISI is used exactly as shipped.** After each image it runs its own check (the median HU of each organ must
fall within real-CT ranges). In Path A, which we use, MAISI's code does **not** regenerate a volume that fails;
it saves it and logs the failure. The "up to 2 retries" in MAISI's docs only applies to Path B (masks from its
database). So whatever MAISI saves is the dataset. The runner never retries or discards anything; it only
records MAISI's pass/fail per volume (`maisi_qc_pass`) so it can be reported.

**One orientation for everything.** The final pairs are stored in RAS voxel order (by axis flips/permutations
only, no resampling). Real data is converted the same way with `orient.py` (Step 6b). LiTS is currently mixed:
63 LPS, 35 RAS, 33 LAS.

## Files in `BT2026/maisi/`

| File | What it does |
|---|---|
| `env.sh` | Source this in every terminal. It sets `PROJ`, activates the venv and puts all caches in `/work`. |
| `setup_maisi.sh` | One-time setup: installs uv, Python 3.11, torch 2.9.1 (CUDA 12.8, Blackwell support) and MONAI 1.6.1, clones NV-Generate-CTMR at `dba166e` and downloads the weights. |
| `make_jobs.py` | Writes one JSON per volume (target, size, seed, the full MAISI config). |
| `run_jobs.py` | Runs the jobs one by one. Resumable. Keeps the raw output and logs time, peak VRAM, MAISI QC result and mask retries. |
| `postprocess.py` | Remaps the 132-class mask to 0/1/2 and writes nnU-Net-named pairs. Computes metrics: volumes, connected components, tumour fraction within 15 mm of the organ, recomputed MAISI QC. |
| `orient.py` | Converts NIfTI files to RAS orientation without changing any voxel value (used for the MAISI pairs, and as a CLI for LiTS / MSD Task07). |
| `summarize.py` | Writes `summary.md` (including measured time and disk per volume and the projection for 1000) and PNG previews. |

---

## Step 0: start the UCloud job

Steps 1–3 (clone, install, download weights) need **no GPU**. Run them in a cheap CPU session, then start a GPU
session for step 4 onwards. Both sessions must mount **the same folder**, because the venv points to the Python
in `$PROJ/tools` by its absolute path.

1. On UCloud, open **Apps → Terminal (Ubuntu)**.
2. **Machine type:**
   - Steps 1–3: any CPU-only machine type. If the Terminal app only offers GPU machines, take the smallest
     slice (`gpu-nvidia-b200-1-mig.1g`, 1/7 of a B200).
   - Step 4 (smoke test): **1/7** (`gpu-nvidia-b200-1-mig.1g`). It shows whether one volume fits in 23 GB.
   - Step 5 onwards: **4/7** (`gpu-nvidia-b200-4-mig.1g`), which runs 4 workers at once, one per slice.
     If the smoke test runs out of memory even with the fallback, use **1** full B200 instead.
3. **Time:** CPU session 1 hour (setup about 15–30 min). Smoke-test session 1 hour (one volume, roughly 5–10 min
   on a slice). Trial session 3 hours (roughly 30–60 min on 4 slices). When the job ends, background processes are killed too. Files in `/work` stay.
4. **Folders:** click *Add folder* and choose the drive folder where the project should live (for example `Research`).
   It is mounted at `/work/Research`. Everything (code, venv, weights, outputs) goes there, so the next session
   can start straight away.
5. Submit the job, then open the terminal when it is running.

B200 machine types: a full B200 has about 180 GB of GPU memory and 7 compute units. The fractions on the slider
are **groups of separate 1/7 slices** (MIG `1g.23gb`), not one bigger slice: "4/7" is four independent 23 GB GPUs.
One process can use only one slice, so a volume always has to fit in 23 GB, and more slices means more workers
in parallel. The RAM shown (36 GB per slice) is CPU memory.

| Slider | What you get | Use for |
|---|---|---|
| 1/7 | 1 slice: 23 GB, 6 vCPU, 36 GB RAM | setup (if no CPU machine), smoke test |
| 2/7–4/7 | 2–4 slices, 23 GB each | trial and pool: one worker per slice (4/7 = 4 volumes at a time) |
| 1 | full B200, ~180 GB | fallback if one volume doesn't fit in 23 GB |
| 2–8 | several full B200s | not needed |

**Memory.** NVIDIA measured MAISI at 512³ on an A100 (`docs/performance.md`): the peak is in the autoencoder
decoding, and it is 45.3 GB with their 80 GB preset (window 80³) and 22.8 GB with their 24 GB preset
(window 64×64×48, 2 splits). A 23 GB slice holds slightly less than 23 GB, so we use the 24 GB preset with
**8 splits**, which lowers the peak further. NVIDIA notes that more splits barely affect the image, while the
smaller window (compared with 80³) can add faint seams between tiles, so look for seams in the previews. Fallback
if it still runs out of memory: `--ae-splits 16`. All volumes of a study should use the same settings, so these
are the `make_jobs.py` defaults now, also on a full GPU.

**Speed.** Each slice has 1/7 of the B200's compute, so one volume takes several minutes; 4 slices make 4 at a
time. If UCloud bills proportionally (check the price shown when submitting), the cost per volume is about the
same as on a full GPU. For the big pool, several UCloud jobs can run in parallel on the same folder; see the end
of this guide.

## Step 1: set your folder and check the machine

Replace `Research` with the name of the folder you mounted. This is the only edit in the whole guide.

```bash
export PROJ=/work/Research
cd "$PROJ" && mkdir -p logs
nvidia-smi
df -h "$PROJ"
for u in https://github.com https://pypi.org https://download.pytorch.org/whl/cu128/ https://huggingface.co https://astral.sh; do printf "%-45s " "$u"; curl -s -o /dev/null -w "%{http_code}\n" --max-time 10 "$u"; done
```

Check:
- On a CPU session `nvidia-smi` says *command not found*; that is fine. On a GPU session it lists a **B200**, and
  the top-right says **CUDA Version 12.8 or higher**, which needs driver ≥ 570. If it is lower, stop and tell me,
  because the torch build has to change.
- `df -h` shows at least **~30 GB free**. The trial needs about 15 GB; the venv and weights about 10 GB.
- Every URL returns a code (200/301/302/404 all mean reachable). `000` means no internet access to that site.

## Step 2: get the BT2026 code

First session:

```bash
git clone https://github.com/DesertSailor29/BT2026.git "$PROJ/BT2026"
```

Later sessions (to get updates):

```bash
git -C "$PROJ/BT2026" pull
```

## Step 3: one-time setup (about 15–30 min)

```bash
setsid nohup bash "$PROJ/BT2026/maisi/setup_maisi.sh" > "$PROJ/logs/setup_maisi_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
tail -f "$(ls -t $PROJ/logs/setup_maisi_*.log | head -1)"
```

(`Ctrl+C` stops the `tail` only. Setup keeps running in the background.)

The setup does the following. You don't need to do any of it by hand:
- installs `uv` into `$PROJ/tools` and Python 3.11 into `$PROJ/tools/python`,
- clones **https://github.com/NVIDIA-Medtech/NV-Generate-CTMR** into `$PROJ/NV-Generate-CTMR` and checks out commit `dba166e`,
- creates `$PROJ/venv-maisi` with `torch==2.9.1` from https://download.pytorch.org/whl/cu128, `monai==1.6.1` and NV-Generate-CTMR's `requirements.txt`,
  and saves the exact package list to `$PROJ/venv-maisi-freeze.txt`,
- downloads the rflow-ct weights from **https://huggingface.co/nvidia/NV-Generate-CT** into `$PROJ/NV-Generate-CTMR/models/`.
  Path A doesn't need the ~4000-mask database, so it is not downloaded.

It worked if the log ends with `== setup done` and contains:
- on a CPU session: `no GPU visible: skipped` (the GPU check then happens in step 4); on a GPU session:
  `GPU: NVIDIA B200 | capability (10, 0)`, an arch list that includes `sm_100`, and `matmul ok: True`,
- an `ls` of `models/` listing 5 `.pt` files: `autoencoder_v1.pt`, `diff_unet_3d_rflow-ct.pt`, `controlnet_3d_rflow-ct.pt`,
  `mask_generation_autoencoder.pt`, `mask_generation_diffusion_unet_v2.pt`.

The script can be re-run safely if something fails halfway.

<details><summary>Doing the clone and download by hand (only if the script cannot)</summary>

```bash
git clone https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git "$PROJ/NV-Generate-CTMR"
git -C "$PROJ/NV-Generate-CTMR" checkout dba166e
source "$PROJ/BT2026/maisi/env.sh"     # after the venv exists
cd "$PROJ/NV-Generate-CTMR" && python -m scripts.download_model_data --version rflow-ct --root_dir ./ --model_only
```
</details>

## Step 4: smoke test, one liver volume (about 5 min)

This checks that everything works end to end, and gives the time per volume and the peak VRAM.
**GPU session from here on.** In a new session, first run the "Every later session" block at the bottom, then
`nvidia-smi` (B200, CUDA ≥ 12.8, see step 1) and this GPU check:

```bash
source "$PROJ/BT2026/maisi/env.sh"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.mem_get_info()[1]/1e9, 'GB', 'sm_100' in torch.cuda.get_arch_list()); x=torch.randn(4096,4096,device='cuda'); print('matmul ok:', float((x@x).abs().mean())>0)"
```

It should print `2.9.1+cu128 NVIDIA B200 …` (on a slice the name ends in `MIG 1g.23gb`), `(10, 0)`, the GPU
memory (about 23 on a slice, about 180 on a full B200), `True` and `matmul ok: True`. Then the smoke test:

```bash
source "$PROJ/BT2026/maisi/env.sh"
python "$PROJ/BT2026/maisi/make_jobs.py" --run-dir "$MAISI_RUNS/smoke" --targets liver --sizes 0.5 --maisi-dir "$MAISI_DIR"
setsid nohup bash -c 'source "$PROJ/BT2026/maisi/env.sh" && python "$PROJ/BT2026/maisi/run_jobs.py" --run-dir "$MAISI_RUNS/smoke"' > "$PROJ/logs/maisi_smoke_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
tail -f "$(ls -t $PROJ/logs/maisi_smoke_*.log | head -1)"
```

To follow MAISI's own progress in a second terminal:

```bash
tail -f "$MAISI_RUNS/smoke/raw/liver_0000/maisi.log"
```

When the run log shows `worker 0: ran 1 job(s)`:

```bash
python "$PROJ/BT2026/maisi/summarize.py" --run-dir "$MAISI_RUNS/smoke" --previews
```

Check:
- status `ok`, the wall time, and `peak GB` (on a slice it must stay below ~22; the closer, the riskier),
- the `MAISI axcodes` column (the orientation MAISI writes; the pairs are converted to RAS regardless),
- `$MAISI_RUNS/smoke/previews/liver_0000.png` (download it via the UCloud file browser). It should look like an
  abdominal CT with a green liver outline and a red tumour outline.

If status is `error`, the last lines of `$MAISI_RUNS/smoke/raw/liver_0000/maisi.log` show why. If they say
`OutOfMemoryError`, try the fallback with 16 splits:

```bash
source "$PROJ/BT2026/maisi/env.sh"
python "$PROJ/BT2026/maisi/make_jobs.py" --run-dir "$MAISI_RUNS/smoke16" --targets liver --sizes 0.5 --ae-splits 16 --maisi-dir "$MAISI_DIR"
setsid nohup bash -c 'source "$PROJ/BT2026/maisi/env.sh" && python "$PROJ/BT2026/maisi/run_jobs.py" --run-dir "$MAISI_RUNS/smoke16"' > "$PROJ/logs/maisi_smoke16_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

If that works, add `--ae-splits 16` to every later `make_jobs.py` command. If it also runs out of memory, 23 GB
is not enough: tell me, and we switch to a full B200. For other errors see Troubleshooting.

## Step 5: trial run, 10 liver + 10 pancreas

On the 4/7 machine (4 slices). The first line counts the slices and starts one worker per slice; each worker is
pinned to its own slice automatically. On a full B200 it starts a single worker.

```bash
source "$PROJ/BT2026/maisi/env.sh"
python "$PROJ/BT2026/maisi/make_jobs.py" --run-dir "$MAISI_RUNS/trial10" --maisi-dir "$MAISI_DIR"
N=$(nvidia-smi -L | grep -c MIG); [ "$N" -ge 1 ] || N=1; echo "$N worker(s)"
for i in $(seq 0 $((N-1))); do setsid nohup bash -c "source \"$PROJ/BT2026/maisi/env.sh\" && python \"$PROJ/BT2026/maisi/run_jobs.py\" --run-dir \"$MAISI_RUNS/trial10\" --num-workers $N --worker $i" > "$PROJ/logs/maisi_trial10_w${i}_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null & done
tail -f $(ls -t $PROJ/logs/maisi_trial10_w*.log | head -$N)
```

Each worker log starts with the slice it uses (`device MIG-…`), then prints one line per finished job: status,
seconds, peak GB, MAISI QC pass/FAIL, tumour and organ volume.

**If the session ends or something crashes, run the same `N=…` and `for …` lines again** on a machine with the
same number of slices (the job split depends on `N`). Finished jobs are skipped. Failed jobs are skipped too
unless you add `--retry-failed` right after `--worker $i` (inside the quotes).

## Step 6: summary and previews

```bash
source "$PROJ/BT2026/maisi/env.sh"
python "$PROJ/BT2026/maisi/summarize.py" --run-dir "$MAISI_RUNS/trial10" --previews
```

Download these through the UCloud file browser (*Files → your folder → maisi_runs/trial10*):
- `summary.md`: one row per volume, plus totals: time per volume, **disk per volume** (raw image, raw label,
  final pair) and the estimates for 1000 volumes, both keeping everything and with `--drop-raw-image`,
- `previews/overview_liver.png` and `previews/overview_pancreas.png`: one axial slice per volume,
- `previews/<job_id>.png`: 3 planes through the tumour, for any volume that looks odd.

## Step 6b: reorient the real data to RAS (once per dataset)

Writes a converted copy. The originals are never touched, file names stay the same, and running it again
skips files already done. LiTS (the 91/40 split) is in `$PROJ/nifti_data`. First see what orientations it has:

```bash
source "$PROJ/BT2026/maisi/env.sh"
python "$PROJ/BT2026/maisi/orient.py" --in "$PROJ/nifti_data" --check
```

Then convert (about 5–10 min for LiTS):

```bash
setsid nohup bash -c 'source "$PROJ/BT2026/maisi/env.sh" && python "$PROJ/BT2026/maisi/orient.py" --in "$PROJ/nifti_data" --out "$PROJ/nifti_data_RAS"' > "$PROJ/logs/orient_lits_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

The log ends with `done, now all RAS` and the counts of the original orientations. Use the same command for
MSD Task07 once it is downloaded. nnU-Net's mirroring augmentation already covers flips to some degree, so this
is cheap insurance rather than a fix, but it removes orientation as a variable from the comparison.

## Step 7: send me the results

Paste these so they go into NOTES.md:
1. the `nvidia-smi` header line (driver / CUDA version),
2. the top block of `summary.md` (jobs ok, QC pass, tumour present, median time, peak GB, disk per volume, 1000-volume estimates),
3. the `MAISI axcodes` value and the `orient.py --check` counts for LiTS,
4. the overview PNGs, or your impression of them (realistic? tumour inside the organ?),
5. any `error` rows, with the last ~20 lines of that job's `maisi.log`.

---

## Every later session

```bash
export PROJ=/work/Research
source "$PROJ/BT2026/maisi/env.sh"
git -C "$PROJ/BT2026" pull
```

Everything else (venv, MAISI clone, weights, runs) is already in `/work`.

## Optional: Claude Code on UCloud

The home folder is wiped when a session ends, so the program is reinstalled each session (a few seconds). Your
login and settings are kept in `$PROJ/.claude-config` (set by `env.sh`), so you log in only once.

```bash
source "$PROJ/BT2026/maisi/env.sh"
curl -fsSL https://claude.ai/install.sh | bash
git -C "$PROJ/BT2026" pull
cd "$PROJ/BT2026" && claude
```

Always source `env.sh` first: without it Claude Code doesn't find the saved login and asks you to log in again.
Start it inside `BT2026`, where it finds `CLAUDE.md` and `NOTES.md` (the project notes shared with the Mac). To make
starting from `$PROJ` work too, create a pointer once:

```bash
printf '@BT2026/CLAUDE.md\n' > "$PROJ/CLAUDE.md"
```

First time only: the machine has no browser. Claude Code prints a login URL: select it, open it in the browser
on your own computer and sign in. The browser then shows a code; paste it into the terminal at
`Paste code here if prompted`.

**Letting Claude run a step.** Claude can run the commands of a step itself (for example "run GUIDE step 4 and
evaluate it"): it starts them detached with `setsid nohup` and a log in `$PROJ/logs` (rule in `CLAUDE.md`), reads
the log, `result.json` and `summary.md`, looks at the preview PNGs, and writes the numbers into `NOTES.md`. You
approve each command. Because the job is detached, it keeps running if the terminal or Claude closes.

**If the terminal closes.** Open a new terminal in the same UCloud job; the generation is still running. Run the
three install lines above again, then `claude --continue` to resume the previous conversation (it is stored in
`$PROJ/.claude-config`). Claude finds the run through its log in `$PROJ/logs` and the path in `NOTES.md`.

## Output layout

```
$PROJ/
├── BT2026/                       this repo
├── NV-Generate-CTMR/             MAISI code @ dba166e, weights in models/
├── venv-maisi/                   Python env (venv-maisi-freeze.txt = exact versions)
├── tools/                        uv + Python 3.11
├── logs/                         nohup logs (setup, runs)
└── maisi_runs/<run>/
    ├── run_config.json           make_jobs arguments + MAISI commit
    ├── jobs/<job_id>.json        one job = one volume (seed, size, full MAISI config)
    ├── raw/<job_id>/             MAISI's own output, never modified:
    │   ├── sample_*_image.nii.gz   float HU
    │   ├── sample_*_label.nii.gz   132-class MAISI mask (kept for organ-level filter metrics)
    │   ├── maisi.log, config_infer.json, environment.json
    │   └── result.json             everything recorded about this job
    ├── pairs/<target>/imagesTr/<job_id>_0000.nii.gz   int16 HU, RAS
    ├── pairs/<target>/labelsTr/<job_id>.nii.gz        uint8 0/1/2, RAS
    ├── results.jsonl             one line per finished attempt (all jobs)
    ├── summary.md
    └── previews/
```

Fields in `result.json`:
- run: `wall_s`, `peak_gpu_gb`, `mask_time_s`, `returncode`, `status`, `raw_image_bytes`, `raw_label_bytes`;
- MAISI's QC from its log: `maisi_qc_pass`, `maisi_qc_failed`;
- mask retries: `mask_retries`, and `mask_target_missing`, which is true when MAISI gave up after 5 attempts to get a
  mask containing the requested tumour;
- `metrics`: `organ_ml`, `tumor_ml`, `tumor_present`, `organ_components`, `tumor_components`, `*_largest_share`,
  `organ_median_hu`, `tumor_median_hu`, `tumor_frac_within_15mm` (the thesis filter), `maisi_qc_recomputed`
  (per-organ QC detail), `raw_axcodes` (as MAISI wrote it), `axcodes` (always RAS), `spacing`, `shape`, `pair_bytes`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no kernel image is available for execution on the device` / `sm_100` missing | The wrong torch build got installed. Check `$PROJ/venv-maisi-freeze.txt` has `torch==2.9.1+cu128`, delete `$PROJ/venv-maisi` and rerun setup. |
| `CUDA driver version is insufficient` | The driver is older than 570. Tell me the `nvidia-smi` output so we can pick a matching torch build. |
| `CUDA out of memory` | On a 23 GB slice: recreate the run with `--ae-splits 16` (see step 4). Still failing: use a full B200. |
| Hugging Face download fails during setup | Rerun setup (finished files are skipped). Check `curl -I https://huggingface.co`. |
| `AttributeError: 'float' object has no attribute 'to'` at `inference.py` `mask_generation_scale_factor` | Upstream MAISI bug at `dba166e` (the v2 mask checkpoint stores `scale_factor` as a float). `setup_maisi.sh` patches it; an older clone lacks the patch: rerun `setup_maisi.sh` (CPU is fine, skips finished parts), then rerun the job with `--retry-failed`. |
| Job `error`, log ends in a traceback from `scripts/…` | Send me the last ~30 lines of that `maisi.log`. |
| Job `postprocess_error` | The raw output is fine. Send me the `error` field from `result.json`. After a fix: `python $PROJ/BT2026/maisi/postprocess.py --run-dir … --job <id> --maisi-dir $MAISI_DIR`. |
| UCloud job ended mid-run | Start a new job with the same folder and run the "every later session" block, then the same `nohup` line. |

## Next: the 1000-volume pool (after the trial)

Only once the trial results look sane. Option 1 uses the thesis grid 0.1–0.9 (9 sizes); 111 repeats gives 999
volumes per target, with seeds that don't overlap the trial:

```bash
python "$PROJ/BT2026/maisi/make_jobs.py" --run-dir "$MAISI_RUNS/pool_liver" --targets liver \
  --sizes 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --repeats 111 --seed-base 100000 --maisi-dir "$MAISI_DIR"
```

**Disk.** `summary.md` of the trial gives the measured size per volume and the projected total for 1000 volumes.
The ~0.3–0.5 GB per raw float image is an estimate until then. If the projection is too large for the 2 TB
allocation, add `--drop-raw-image` to `run_jobs.py`: it deletes MAISI's float image once the int16 copy in
`pairs/` is written. The raw 132-class label is always kept, because organ-level filter metrics need it.

**Several UCloud jobs at once.** Workers are numbered across all jobs, and worker `i` uses slice `i mod (slices
on its machine)`. Example with two 4/7 machines on the same folder: `--num-workers 8`, workers 0–3 on the first
machine and 4–7 on the second (`for i in 0 1 2 3` and `for i in 4 5 6 7` in the loop from step 5). Keep the same
`--num-workers` when resuming.
