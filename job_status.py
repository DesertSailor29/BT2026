"""Generic status file for any detached job: one file at the top of $PROJ whose *name* says what runs and when it
is projected to finish, e.g.

    PREDICT_test_D101_f0f1__92of222__ETA_Sun-08-Nov_14-50_CET.txt
    PREDICT_test_D101_f0f1__222of222__DONE_Sun-08-Nov_14-52_CET.txt
    PREDICT_test_D101_f0f1__130of222__STOPPED_Sun-08-Nov_12-10_CET.txt   (process gone before all outputs existed)

so it can be read straight from the UCloud file browser (the MAISI pool and nnU-Net training have their own,
richer versions: maisi/status_file.py, nnunet/train_status.py).

Progress = number of output files matching --glob (several allowed) against --total. ETA = now + files left / rate,
rate from the modification times of the last --window outputs (so parallel workers are counted automatically),
plus --tail-min for a final step that writes no counted files (e.g. an eval). Rough, by design.
The job is followed by --pid (or --pgrep pattern): when it is gone the name gets DONE (all outputs there) or
STOPPED (not), and this script exits. Without --pid/--pgrep, DONE as soon as all outputs exist.

On start, finished status files (*__DONE_*, *__STOPPED_*) at the top are moved to $PROJ/status_done/, so only
running or just-finished jobs are visible. Restart-safe (renames its own file). Times are Copenhagen time.

    setsid nohup bash -c 'source "$PROJ/BT2026/maisi/env.sh" && python "$PROJ/BT2026/job_status.py" \
      --name PREDICT_test --glob "$OUT/fold_0/*.nii.gz" --total 111 --pid <job pid>' \
      > "$PROJ/logs/status_<name>_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
"""

import argparse
import glob
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Copenhagen")
FINISHED = ("__DONE_", "__STOPPED_")


def outputs(patterns):
    files = {f for p in patterns for f in glob.glob(p)}
    mtimes = []
    for f in files:
        try:
            mtimes.append(os.path.getmtime(f))
        except OSError:
            pass
    return sorted(mtimes)


def alive(args):
    if args.pid:
        try:
            os.kill(args.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    if args.pgrep:
        r = subprocess.run(["pgrep", "-f", args.pgrep], capture_output=True, text=True)
        me = str(os.getpid())
        return any(pid != me for pid in r.stdout.split())
    return None


def status(args):
    mt = outputs(args.glob)
    now = datetime.now(TZ)
    done, left = len(mt), max(args.total - len(mt), 0)
    recent = mt[-args.window:]
    rate = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) >= 3 and recent[-1] > recent[0] else None
    eta = now + timedelta(seconds=left / rate + args.tail_min * 60) if rate else None
    if left == 0 and not (args.pid or args.pgrep):
        eta = None
    a = alive(args)
    state = "running"
    if a is False:
        state = "DONE" if left == 0 else "STOPPED"
    elif a is None and left == 0:
        state = "DONE"
    return {"done": done, "left": left, "rate": rate, "eta": eta, "now": now, "state": state}


def stamp(t):
    return f"{t:%a-%d-%b_%H-%M}_{t.tzname()}"


def file_name(name, total, s):
    if s["state"] != "running":
        when = f"{s['state']}_{stamp(s['now'])}"
    elif s["left"] == 0:
        when = f"finishing_ETA_{stamp(s['eta'] or s['now'])}"  # outputs complete, final step still running
    elif s["eta"] is None:
        when = "ETA_unknown-yet"
    else:
        when = f"ETA_{stamp(s['eta'])}"
    return f"{name}__{s['done']}of{total}__{when}.txt"


def body(args, s):
    fmt = "%a %d %b %Y %H:%M %Z"
    lines = [
        f"job:             {args.name}",
        f"what:            {args.what}" if args.what else "",
        f"log:             {args.log}" if args.log else "",
        f"outputs:         {' '.join(args.glob)}",
        f"updated:         {s['now']:{fmt}}",
        f"done:            {s['done']} of {args.total}, {s['left']} left",
        f"rate:            {s['rate'] * 3600:.1f} outputs/h (last {args.window})" if s["rate"] else "rate:            not known yet",
        f"projected done:  {s['eta']:{fmt}}" + (f" (incl. {args.tail_min} min final step)" if args.tail_min else "") if s["eta"] and s["state"] == "running" else "",
        f"state:           {s['state']}",
    ]
    return "\n".join(l for l in lines if l) + "\n"


def retire(top):
    dest = top / "status_done"
    for f in top.glob("*.txt"):
        if any(k in f.name for k in FINISHED):
            dest.mkdir(exist_ok=True)
            f.rename(dest / f.name)
            print(f"retired {f.name} -> {dest}", flush=True)


def write(top, name, total, args, s):
    new = top / file_name(name, total, s)
    mine = [f for f in top.glob(f"{name}__*.txt")]
    for old in mine:
        if old != new:
            old.rename(new)
            break
    tmp = new.with_suffix(".tmp")
    tmp.write_text(body(args, s))
    os.replace(tmp, new)
    for extra in top.glob(f"{name}__*.txt"):
        if extra != new:
            extra.unlink()
    print(f"[{s['now']:%H:%M:%S}] {new.name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="what runs, first part of the file name (no '__')")
    p.add_argument("--glob", nargs="+", required=True, help="output file pattern(s) that count as progress")
    p.add_argument("--total", type=int, required=True, help="number of outputs when finished")
    p.add_argument("--pid", type=int, help="PID of the job (DONE/STOPPED when it is gone)")
    p.add_argument("--pgrep", help="pgrep -f pattern of the job, instead of --pid")
    p.add_argument("--tail-min", type=float, default=0, help="minutes added to the ETA for a final uncounted step")
    p.add_argument("--window", type=int, default=20, help="last N outputs used for the rate (default 20)")
    p.add_argument("--what", help="one line description written into the file")
    p.add_argument("--log", help="log path written into the file")
    p.add_argument("--top", type=Path, default=Path(os.environ.get("PROJ", "/work/Research")))
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    if "__" in args.name:
        p.error("--name must not contain '__'")

    retire(args.top)
    last = None
    while True:
        s = status(args)
        key = (s["done"], s["state"], s["eta"] and s["eta"].strftime("%H%M"))
        if key != last:
            write(args.top, args.name, args.total, args, s)
            last = key
        if args.once or s["state"] != "running":
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
