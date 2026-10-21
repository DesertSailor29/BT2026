# MAISI-v2 generation on UCloud: trial run

This guide goes from a brand-new UCloud session to **10 liver + 10 pancreas synthetic CT/label pairs** with varying
tumour sizes. It uses NVIDIA's open MAISI-v2 (`rflow-ct`) running locally on **one full B200** (several workers share the GPU).

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
| Autoencoder tiling | window 64×64×48, overlap 0.4, 8 splits | NVIDIA's 24 GB preset for 512³ with more splits; chosen for 23 GB MIG slices and kept so all volumes match (see step 0) |

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
| `fix_lits_headers.py` + `lits_headers_28_47.json` | Repairs the identity headers in the LiTS release (volume-28..47 image + label, labels 48..52). Run once before `orient.py` (Step 6b). |
| `notify.py` | Emails you during a long run: milestones (e.g. 250/500/950) with the projected finish time, and alerts if it stalls, workers die or jobs fail. |
| `summarize.py` | Writes `summary.md` (including measured time and disk per volume and the projection for 1000) and PNG previews. |
| `viz.py` | Shared figure style (window, colours, radiological orientation) used by every script that draws CT slices. |
| `msd07_overview.py` | Checks MSD Task07 Pancreas, writes `stats.csv` and overview figures. |
| `bench_workers.py` | Optional: measures how many workers fit on one GPU and how fast they are (step 4b). |
| `patches/` | Our fixes to the MAISI clone, applied by `setup_maisi.sh`. |

---

## Step 0: start the UCloud job

Steps 1–3 (clone, install, download weights) need **no GPU**. Run them in a cheap CPU session, then start a GPU
session for step 4 onwards. Both sessions must mount **the same folder**, because the venv points to the Python
in `$PROJ/tools` by its absolute path.

1. On UCloud, open **Apps → Terminal (Ubuntu)**.
2. **Machine type:**
   - Steps 1–3: any CPU-only machine type. If the Terminal app only offers GPU machines, take the smallest
     slice (`gpu-nvidia-b200-1-mig.1g`, 1/7 of a B200).
   - Step 4 onwards: **1 full B200** (slider at 1). A 512³ volume does **not** fit in a 23 GB MIG slice (the image
     diffusion step runs out of memory, also with MAISI's `--low-vram`; tested 2026-10-20).
3. **Time:** CPU session 1 hour (setup about 15–30 min). GPU session: smoke test about 5 min, trial (20 volumes,
   4 workers) about 40 min, so 2 hours covers smoke test + trial. When the job ends, background processes are
   killed too. Files in `/work` stay.
4. **Folders:** click *Add folder* and choose the drive folder where the project should live (for example `Research`).
   It is mounted at `/work/Research`. Everything (code, venv, weights, outputs) goes there, so the next session
   can start straight away.
5. Submit the job, then open the terminal when it is running.

B200 machine types: a full B200 has about 180 GB of GPU memory. The fractions on the slider are **groups of
separate 1/7 slices** (MIG `1g.23gb`), not one bigger slice: "4/7" is four independent 23 GB GPUs, and one process
can use only one slice. The RAM shown is CPU memory.

| Slider | What you get | Use for |
|---|---|---|
| 1/7 | 1 slice: 23 GB, 6 vCPU, 36 GB RAM | setup (if no CPU machine). Too small for 512³ generation |
| 2/7–4/7 | 2–4 slices, 23 GB each | not usable for 512³ (each slice is too small) |
| 1 | full B200, ~180 GB, CPU quota 48 cores, 2.2 TB RAM | smoke test, trial, pool: 4 workers share the GPU |
| 2–8 | several full B200s | the pool, if one is too slow (see the end of this guide) |

**Memory.** One 512³ volume peaks at **33.5 GB** (measured on the B200 with our tiling: window 64×64×48, 8
splits). NVIDIA's 80 GB preset (window 80³) peaks at about 45 GB on an A100 and may be faster, but we keep the
small window so all volumes are made the same way. NVIDIA notes that the smaller window can add faint seams
between tiles, so look for seams in the previews.

**Speed (measured 2026-10-20, step 4b).** One worker alone makes a volume every ~160 s and leaves the GPU idle
more than half the time (mask preparation, file writing and the stitching of the decoded image run on the CPU).
Several workers on the same GPU overlap these phases:

| workers | s per volume | 1000 volumes | GPU memory |
|---|---|---|---|
| 1 | 162 | 45 h | 37 GB |
| 2 | 111 | 31 h | 67 GB |
| 3 | 106 | 29 h | 104 GB |
| **4** | **102** | **28 h** | 148 GB |
| 5 | 111 | 31 h | 161 GB |

So use **4 workers** on a full B200. More than ~2 give little because the processes take turns on the GPU.
`run_jobs.py` gives each worker its share of the CPU quota as torch threads (48 / workers); without that, every
process starts 384 threads (torch sees all host CPUs) and they slow each other down.

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

It should print `2.9.1+cu128 NVIDIA B200 …`, `(10, 0)`, the GPU memory (about 190 GB on a full B200; if the name
ends in `MIG 1g.23gb` you are on a slice, which is too small), `True` and `matmul ok: True`. Then the smoke test:

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
- status `ok`, the wall time (about 150–170 s) and `peak GB` (about 33.5),
- the `MAISI axcodes` column (the orientation MAISI writes; the pairs are converted to RAS regardless),
- `$MAISI_RUNS/smoke/previews/liver_0000.png` (download it via the UCloud file browser). It should look like an
  abdominal CT with a green liver outline and a red tumour outline.

- the anatomy is full size: in the preview the body fills most of the frame (not a small box in the middle).
  If it looks shrunk, the MAISI clone lacks our patch (see Troubleshooting).

If status is `error`, the last lines of `$MAISI_RUNS/smoke/raw/liver_0000/maisi.log` show why; see Troubleshooting.

## Step 4b (optional): how many workers fit and how fast they are

Only needed on a new kind of machine (another GPU type or CPU quota); the numbers for a full B200 are in step 0.
For each worker count it runs `workers × 2` liver volumes at once and writes a table to
`$MAISI_RUNS/bench/bench.md`. It stops at the first worker count with a failed job (e.g. out of memory). It takes
about 1 hour for 1–5 workers.

```bash
source "$PROJ/BT2026/maisi/env.sh"
setsid nohup bash -c 'source "$PROJ/BT2026/maisi/env.sh" && python "$PROJ/BT2026/maisi/bench_workers.py" --run-root "$MAISI_RUNS/bench" --workers 1 2 3 4 5 --maisi-dir "$MAISI_DIR"' > "$PROJ/logs/maisi_bench_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

Use the worker count with the lowest "s per volume" whose "GPU mem max" stays clearly below the GPU's total.

## Step 5: trial run, 10 liver + 10 pancreas

On a full B200, with **4 workers** sharing the GPU (see "Speed" in step 0). Each worker takes every 4th job and
gets 12 of the 48 CPU threads automatically.

```bash
source "$PROJ/BT2026/maisi/env.sh"
python "$PROJ/BT2026/maisi/make_jobs.py" --run-dir "$MAISI_RUNS/trial10" --maisi-dir "$MAISI_DIR"
N=4
for i in $(seq 0 $((N-1))); do setsid nohup bash -c "source \"$PROJ/BT2026/maisi/env.sh\" && python \"$PROJ/BT2026/maisi/run_jobs.py\" --run-dir \"$MAISI_RUNS/trial10\" --num-workers $N --worker $i" > "$PROJ/logs/maisi_trial10_w${i}_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null & done
tail -n 5 $(ls -t $PROJ/logs/maisi_trial10_w*.log | head -$N)
```

Each worker log starts with its device and CPU threads (`device default, 12 CPU threads`), then prints one line
per finished job: status, seconds, peak GB, MAISI QC pass/FAIL, tumour and organ volume. With 4 workers each job
takes about 6–7 min (they share the GPU), and the 20 volumes take about 35–40 min in total.

**If the session ends or something crashes, run the same `N=4` and `for …` lines again** (keep `N`: the job
split depends on it). Finished jobs are skipped. Failed jobs are skipped too unless you add `--retry-failed` right
after `--worker $i` (inside the quotes).

Optional, about 4% faster: NVIDIA MPS lets the workers' GPU work run at the same time instead of taking turns
(measured: 98 instead of 102 s per volume with 4 workers, and 124 instead of 148 GB GPU memory). Start it before
the loop and stop it when all workers are done:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY && nvidia-cuda-mps-control -d
# ... the for loop from above, in the same terminal (the workers inherit the two variables) ...
echo quit | nvidia-cuda-mps-control   # afterwards
```

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
skips files already done. Real data lives in `$PROJ/raw_data/`: `LiTS/` (the 91/40 split) and `Task07_Pancreas/`
(MSD, already all RAS, nothing to convert).

LiTS first needs its broken headers repaired: in the released files, volume-28..47 (the 20 3D-IRCADb-01 cases)
have an identity affine in image and label (1 mm isotropic, L/R mirrored), and the labels of volume-48..52 too.
`fix_lits_headers.py` writes the true affines (from `lits_headers_28_47.json`, derived from the IRCAD DICOMs) into
these 45 files in place. Only the header changes, the voxels are verified bit-exact, and rerunning skips fixed files:

```bash
source "$PROJ/BT2026/maisi/env.sh"
python "$PROJ/BT2026/maisi/fix_lits_headers.py" --lits "$PROJ/raw_data/LiTS" --dry-run   # expect 'would fix' or 'ok' for 45 files
python "$PROJ/BT2026/maisi/fix_lits_headers.py" --lits "$PROJ/raw_data/LiTS"
```

Then see what orientations LiTS has (after the fix: no identity headers left, 28–47 count as LAS):

```bash
python "$PROJ/BT2026/maisi/orient.py" --in "$PROJ/raw_data/LiTS" --check
```

Then convert (about 5–10 min for LiTS):

```bash
setsid nohup bash -c 'source "$PROJ/BT2026/maisi/env.sh" && python "$PROJ/BT2026/maisi/orient.py" --in "$PROJ/raw_data/LiTS" --out "$PROJ/raw_data/LiTS_RAS"' > "$PROJ/logs/orient_lits_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

The log ends with `done, now all RAS` and the counts of the original orientations. Use the same command for
any other dataset that `--check` shows is not all RAS. nnU-Net's mirroring augmentation already covers flips to some degree, so this
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

## Optional: email notifier for long runs

`notify.py` watches a run directory and mails you at milestones, each with progress, failures, the rate and the
projected finish time (Copenhagen time), so you know when to log in. It also mails when nothing has finished for
30 min, when no worker is running any more, when jobs fail, and (with `--session-end`) when the run won't finish
before the UCloud job ends. It only reads `result.json` files, so it can be started, stopped and restarted at any
time without affecting the workers; milestones already sent are not sent again.

**One-time: credentials** in `$PROJ/.notify.env` (outside the repo, the repo is public). With Gmail: turn on
2-step verification, create an app password (Google account → Security → App passwords), then:

```bash
cat > "$PROJ/.notify.env" <<'X'
NOTIFY_TO=you@example.com
NOTIFY_SMTP_USER=you@gmail.com
NOTIFY_SMTP_PASS=abcdabcdabcdabcd
# optional dead-man switch, see below
# NOTIFY_HC_URL=https://hc-ping.com/<uuid>
X
chmod 600 "$PROJ/.notify.env"
python "$PROJ/BT2026/maisi/notify.py" --run-dir "$MAISI_RUNS/trial10" --test
```

The `--test` line sends one progress mail right away. If it fails with a timeout, UCloud blocks outgoing SMTP: use
ntfy instead (plain HTTPS, no account): add `NOTIFY_NTFY_TOPIC=<long random name>` to the file and add
`--via ntfy` to the commands. ntfy.sh forwards each message to `NOTIFY_TO` by email, and you can also subscribe
to the topic in the ntfy phone app for push notifications. Anyone who knows the topic name can read the
messages (they only contain progress numbers), so make it hard to guess.

**Start it** (detached, like the workers), right after starting the workers:

```bash
setsid nohup bash -c "source \"$PROJ/BT2026/maisi/env.sh\" && python \"$PROJ/BT2026/maisi/notify.py\" --run-dir \"$MAISI_RUNS/pool1000\" --milestones 250 500 750 950 --session-end '2026-10-23 14:00'" > "$PROJ/logs/notify_pool1000_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

`--milestones` takes counts or percentages (default 25 % 50 % 75 % 95 %); a mail for "finished" is always sent.
`--session-end` is when the UCloud job ends (shown in the UCloud job page). The projection uses the rate of the
last 2 h (`--window-h`) and also gives the range against the overall rate.

**Limit: it dies with the UCloud job**, so it can't tell you the job itself ended. For that, create a free check
on healthchecks.io (period = 15 min, grace = 15 min, email integration), put its ping URL in `NOTIFY_HC_URL`. The
notifier pings it every poll; when the pings stop (job ended, machine gone), healthchecks.io emails you. It also
pings `/fail` on a stall or when no workers are running.

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
| `CUDA out of memory` | On a MIG slice: 512³ does not fit, use a full B200 (step 0). On a full B200 with several workers: use fewer workers (each needs ~35 GB). |
| Hugging Face download fails during setup | Rerun setup (finished files are skipped). Check `curl -I https://huggingface.co`. |
| `AttributeError: 'float' object has no attribute 'to'` at `inference.py` `mask_generation_scale_factor` | Upstream MAISI bug at `dba166e` (the v2 mask checkpoint stores `scale_factor` as a float). `setup_maisi.sh` patches it; an older clone lacks the patch: rerun `setup_maisi.sh` (CPU is fine, skips finished parts), then rerun the job with `--retry-failed`. |
| Anatomy looks shrunk: body only ~180 mm wide, everything inside a central 256³ box with black padding, liver ~400 ml | MAISI Path A at `dba166e` stamps the target spacing on the 256³ generated mask (native 1.5 mm) and only pads it. `setup_maisi.sh` applies `patches/mask_native_spacing.patch` (mask generated + labelled at 1.5 mm, then resampled to the target spacing); an older clone lacks it: rerun `setup_maisi.sh`. Check with `git -C "$MAISI_DIR" diff --stat` (should list `scripts/sample.py`). |
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

**Several UCloud jobs at once.** Workers are numbered across all jobs. Example with two full-B200 machines on the
same folder: `--num-workers 8`, workers 0–3 on the first machine and 4–7 on the second (`for i in 0 1 2 3` and
`for i in 4 5 6 7` in the loop from step 5), and add `--threads 12` (the automatic share is 48 CPU threads divided
by `--num-workers`, which would give only 6 here). Keep the same `--num-workers` when resuming.
