# BT2026: synthetic MAISI data for CT segmentation

Continuation of Sebestyén's BSc thesis (SDU 2026, "On the impact of synthetic data in medical image segmentation"):
MAISI synthetic CT/mask pairs added to LiTS for nnU-Net liver/tumour segmentation. Direction chosen: **Option B,
automatic filtering of bad synthetic scans**. Target venue MIDL 2027.

This repo is public (github.com/DesertSailor29/BT2026). Don't commit credentials, data or patient files.

## Two machines

Claude may run on either. Check which one before acting (`uname`, `pwd`):

| | Mac (`/Users/sebi/Documents/0Denmark/University/Research/`) | UCloud (`/work/<folder>/`, Ubuntu) |
|---|---|---|
| Role | write code, plan, read papers | run MAISI generation and nnU-Net training |
| GPU | none | B200 MIG slices (23 GB each) or a full B200, depending on the session |
| Only here | thesis PDF, `Progression Ideas.docx`, `MAISI_generation_options.pdf`, local `LiTS/` (split script, seed 42; 91 train / 40 test) | venvs, MAISI clone + weights, `maisi_runs/`, logs |
| Commands | hand UCloud commands to the user to copy across | may run them directly; source `maisi/env.sh` first |

UCloud specifics: the home folder is wiped when a session ends; everything persistent lives in `$PROJ` (the mounted
`/work/<folder>`). Background processes die with the session. GPU hours are limited: don't leave GPU work running
that the user didn't ask for, and say when a session can be stopped. How-to for everything there:
`maisi/GUIDE.md`.

## Session notes (mandatory)

Sessions can time out without warning, so notes are kept **as work happens**, not at the end. `NOTES.md` (this
folder) is shared by both machines through git.

1. At session start, `git pull` (the other machine may have pushed newer notes), then continue from "Next steps"
   in `NOTES.md` unless the user says otherwise. Note in the log which machine the session ran on.
2. After every meaningful step (decision made, file changed, command handed to the user, result received, problem found):
   - update **Current state** and **Next steps** in place (keep them short and true),
   - append a dated line to **Log** (`YYYY-MM-DD: what happened / outcome`),
   - record decisions with their reason under **Decisions**, open items under **Open questions**.
3. When UCloud output arrives (timings, errors, metrics), copy the key numbers into the log.
4. If the Log section grows past ~150 lines, move older entries to `NOTES_ARCHIVE.md`.
5. Notes only reach the other machine when committed and pushed. At the end of a block of work, remind the user
   to let Claude commit and push (see Rules), so the other machine is up to date.

## Rules

- Never add Claude as co-author or "Generated with Claude Code" to commits or PRs.
- Commit only when asked, and don't push without asking. **Exception on UCloud (standing permission from the
  user):** whenever NOTES.md gets a started job's log path, a job's results or a status change, commit **only
  NOTES.md** (`git add NOTES.md`, message like `notes: <what>`) and push it immediately, without asking. Pull first
  (`git pull --rebase --autostash`) so the push doesn't fail on newer commits from the Mac. Code changes still need asking.
- Long-running UCloud commands: see "Running jobs on UCloud" below. This holds for commands Claude runs itself and
  for commands handed to the user.

## Running jobs on UCloud (mandatory: terminals close without warning)

A UCloud terminal (and a Claude session in it) can close at any moment, and every process attached to it dies
with it. So **anything that runs longer than ~1 minute** (MAISI, postprocessing, reorienting, nnU-Net, installs,
downloads) is started detached and logged, never in the foreground and never with Claude's own background tasks
(`run_in_background`), which die together with Claude:

```bash
setsid nohup bash -c 'source "$PROJ/BT2026/maisi/env.sh" && <cmd>' \
  > "$PROJ/logs/<name>_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

- `setsid` + `nohup` detach the job from the terminal; `source env.sh` inside the quotes, because the job gets a
  fresh shell. Logs always go to a timestamped file in `$PROJ/logs` (persistent), never only to the screen.
- Right after starting, check it is alive (`pgrep -af <script>` or `tail` of the log), write the log path into
  the NOTES.md log, and **commit + push NOTES.md right away** (see Rules), so a new session or the Mac can pick it
  up if this one dies. Same again when the job finishes and its results are in NOTES.md.
- Follow progress by reading the log (`tail -n 30 <log>`), in short checks; don't block on `tail -f`.
- Jobs must be resumable (re-running the same command skips finished work, as `run_jobs.py` does).
- They still die when the UCloud **job** ends (time limit or stopped), only the terminal closing is survived. Check
  the remaining session time before starting something long.
- Claude on UCloud may run these itself (the user approves the commands). After a run it evaluates the output
  against the "Check" list of the GUIDE step, copies the key numbers into NOTES.md, and says whether the GPU
  session can be stopped.

## Key facts (so they don't need re-researching)

- MAISI = github.com/NVIDIA-Medtech/NV-Generate-CTMR, pinned at `dba166e`. Params: `docs/inference.md`,
  `configs/config_infer*.json`; memory/time table: `docs/performance.md`. Weights: huggingface.co/nvidia/NV-Generate-CT.
  The thesis used the hosted NIM v1 API (docs.nvidia.com/nim/medical/maisi/api-reference), no seeds.
- The v2 mask model accepts **only one** `controllable_anatomy_size` entry (the thesis used liver + tumour via NIM v1).
  Non-empty = Path A (mask generated; `body_region`/`anatomy_list` ignored). Empty = Path B (real mask from database).
- MAISI bug at `dba166e`: `scripts/inference.py` calls `.to()` on the v2 mask checkpoint's `scale_factor`, a float
  → AttributeError in every Path A run. `setup_maisi.sh` patches the clone (`torch.as_tensor(...)`); keep the patch
  if the pin moves, unless upstream fixes it.
- MAISI Path A geometry bug at `dba166e`: the v2 mask model always makes a 256³ mask at native 1.5 mm, but `scripts/sample.py`
  conditions it on and labels it with the target spacing, then only pads to `output_size` → at 512³ @ 0.8 mm the anatomy is
  ~half scale (body ~180 mm). `maisi/patches/mask_native_spacing.patch` (applied by `setup_maisi.sh`) uses 1.5 mm for both,
  so MAISI resamples the mask (nearest) to the target spacing: 480×480×384 voxels of content, padded to 512³.
- rflow-ct needs `num_inference_steps` 30; the shipped 80g presets say 1000 (a DDPM value).
- B200 (sm_100) needs a CUDA 12.8+ torch build: we use torch 2.9.1+cu128 + MONAI 1.6.1 (MONAI 1.6 needs torch ≥ 2.8).
  The old nnU-Net `setup.sh` pins torch 2.3.0+cu121, which has no Blackwell kernels.
- `scripts.download_model_data` writes models to `./models` relative to the CWD; run MAISI from inside its clone or
  use absolute paths in `environment.json` (`run_jobs.py` does the latter).
- Path A with `num_output_samples` 1: a QC-failed volume is still saved (log line "quality check for label 'X'
  failed"). A missing target label makes MAISI retry the mask 5×, then it continues anyway.
- Thesis MAISI config: 512³ @ 0.75 mm, body_region abdomen, hepatic tumour ∈ {0.1..0.9} × liver ∈ {0.4, 0.5, 0.6},
  90 jobs, 82 usable. Thesis filter: keep a volume if ≥ 90% of tumour voxels are within 15 mm of the liver.
  LM30-filtered vs LM0 tumour Dice 0.704 vs 0.653, p = 0.033 (1000 epochs).
- Progression options (Progression Ideas.docx): A real-vs-synthetic equivalence, **B automatic filtering** (HU
  stats, connected components, topology, CNR, model agreement), C synthetic-only nnU-Net → SAM prompting,
  D nnU-Net uncertainty → SAM on uncertain regions.

@NOTES.md
