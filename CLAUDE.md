# BT2026: synthetic MAISI data for CT segmentation

Continuation of Sebestyén's BSc thesis (SDU 2026, "On the impact of synthetic data in medical image segmentation"):
MAISI synthetic CT/mask pairs for nnU-Net liver/tumour segmentation on LiTS. Direction: **Option B, automatic
filtering of bad synthetic scans**. Target venue MIDL 2027.

This repo is public (github.com/DesertSailor29/BT2026). Don't commit credentials, data or patient files.

## Machines

Check which one before acting (`uname`, `pwd`):

| | Kali VM (`~/Documents/Research/`) / Mac (`/Users/sebi/Documents/0Denmark/University/Research/`) | UCloud (`/work/<folder>/`, Ubuntu) |
|---|---|---|
| Role | write code, fake-test, plan, reports | run MAISI generation, nnU-Net training, evaluations |
| GPU / data | none (fake tests in the scratchpad) | full B200 or MIG slices; all data |
| Only here | `Research/` PDFs (reports, thesis), Mac: `Progression Ideas.docx`, local `LiTS/` | venvs, MAISI clone + weights, `$MAISI_RUNS`, `$PROJ/logs` |
| Commands | hand UCloud commands to the user | may run them (source `maisi/env.sh` first) |

UCloud: the home folder is wiped when a session ends; everything persistent lives in `$PROJ`. GPU hours are limited:
don't leave GPU work running that the user didn't ask for, and say when a session can be stopped. How-to:
`maisi/GUIDE.md`.

## Notes (mandatory; sessions can time out without warning)

`NOTES.md` is loaded every session and shared through git; keep it short (~8 KB). Details live in `notes/`, read only
when needed: `RESULTS.md` (finished experiments + numbers), `DECISIONS.md` (all decisions with reasons),
`LOG.md` (full history), `BACKLOG.md`, `PAPER.md`, `MAISI.md` (MAISI gotchas, versions, thesis numbers, data paths).

1. At session start, `git pull`, then continue from "Next steps" in `NOTES.md` unless the user says otherwise.
2. After every meaningful step (decision, file changed, command handed over, result received, problem found), as it
   happens:
   - update **Current state** and **Next steps** in `NOTES.md` in place (short, true; drop what is done);
   - add a dated line at the top of the `NOTES.md` **Log** (`YYYY-MM-DD HH:MM (machine): what / outcome`); when it has
     more than ~6 entries, move the older ones to the top of `notes/LOG.md`;
   - decisions: full record with reason at the top of `notes/DECISIONS.md`, one-liner under "Decisions in force" if it
     changes them; finished results: key numbers into `notes/RESULTS.md`; parked ideas: `notes/BACKLOG.md`.
3. When UCloud output arrives (timings, errors, metrics), copy the key numbers into the Log (and RESULTS when final).
4. Notes only reach the other machine when pushed: at the end of a block of work, remind the user to let Claude
   commit and push.

## Rules

- Never add Claude as co-author or "Generated with Claude Code" to commits or PRs.
- Commit only when asked, and don't push without asking. **Exception on UCloud (standing permission):** whenever the
  notes get a started job's log path, a job's results or a status change, commit **only `NOTES.md` and `notes/`**
  (message `notes: <what>`) and push right away, after `git pull --rebase --autostash`. Code changes still need asking.
- **Figures:** every CT figure imports `maisi/viz.py`; every chart, diagram and report page imports `reports/style/`
  (see its README). Never restyle; real and synthetic figures must look the same.

## Running jobs on UCloud (mandatory: terminals close without warning)

Every command that does work (computes, writes files, installs, downloads, trains, makes figures), however short, is
started detached and logged; only instant read-only checks (`ls`, `cat`, `tail`, `git status`, `pgrep`, a quick
`python -c`) run directly. Never in the foreground, never with Claude's own background tasks (`run_in_background`,
Monitor), which die with Claude:

```bash
setsid nohup bash -c 'source "$PROJ/BT2026/maisi/env.sh" && <cmd>' \
  > "$PROJ/logs/<name>_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

Then: check it is alive (`pgrep -af`, `tail` of the log), start its **status file** (`job_status.py` or a specialised
one; conventions in `maisi/GUIDE.md` "Job conventions"), write the log path into the NOTES Log and push the notes.
Jobs must be resumable. They still die when the UCloud job ends: check the remaining session time first. After a run,
evaluate against the GUIDE step's checks, copy the key numbers into the notes, and say whether the session can stop.

@NOTES.md
