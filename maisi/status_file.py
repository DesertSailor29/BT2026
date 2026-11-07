"""Keep one status file at the top of $PROJ whose *name* shows the progress and projected finish of a MAISI run.

The name is updated at the start and every --every-jobs finished jobs (and when the run is done), e.g.

    MAISI_pool_liver__150of999__ETA_Fri-06-Nov_19-40_CET__in_maisi_runs-pool_liver.txt

so it can be read straight from the UCloud file browser without opening anything. The file itself holds the
full run path, counts, failures, average job time and the ETA.

ETA = now + (average job time x jobs left) / workers. Average job time = mean (finished - started) of the jobs
finished so far, i.e. including postprocessing; the jobs run --workers at a time, hence the division. Times are
Copenhagen time (CET in winter, CEST in summer; the label says which).

Reads only result.json files (never touches the workers); exits when every job has a result. Restart-safe: it
renames whatever MAISI_<run>__*.txt file is already there. Start it detached, like the workers:

    setsid nohup bash -c 'source "$PROJ/BT2026/maisi/env.sh" && python "$PROJ/BT2026/maisi/status_file.py" \
      --run-dir "$MAISI_RUNS/pool_liver" --workers 4' > "$PROJ/logs/status_pool_liver_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Copenhagen")


def collect(run_dir):
    total = len(list((run_dir / "jobs").glob("*.json")))
    results = []
    for p in (run_dir / "raw").glob("*/result.json"):
        try:
            results.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):  # being written right now
            pass
    return total, results


def job_seconds(r):
    try:
        return (datetime.fromisoformat(r["finished"]) - datetime.fromisoformat(r["started"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return r.get("wall_s")


def status(run_dir, workers):
    total, results = collect(run_dir)
    done = len(results)
    ok = [r for r in results if r.get("status") == "ok"]
    failed = sorted(r.get("job_id", "?") for r in results if r.get("status") != "ok")
    secs = [s for s in (job_seconds(r) for r in ok) if s]
    avg = sum(secs) / len(secs) if secs else None
    left = total - done
    now = datetime.now(TZ)
    eta = now + timedelta(seconds=avg * left / workers) if avg is not None else None
    return {"total": total, "done": done, "ok": len(ok), "failed": failed, "left": left, "avg": avg, "now": now, "eta": eta}


def file_name(run_dir, s):
    run = run_dir.name
    where = "-".join(run_dir.parts[-2:])  # e.g. maisi_runs-pool_liver
    if s["left"] == 0:
        when = f"DONE_{s['now']:%a-%d-%b_%H-%M}_{s['now'].tzname()}"
    elif s["eta"] is None:
        when = "ETA_unknown-yet"
    else:
        when = f"ETA_{s['eta']:%a-%d-%b_%H-%M}_{s['eta'].tzname()}"
    fails = f"__{len(s['failed'])}failed" if s["failed"] else ""
    return f"MAISI_{run}__{s['done']}of{s['total']}{fails}__{when}__in_{where}.txt"


def body(run_dir, s, workers):
    fmt = "%a %d %b %Y %H:%M %Z"
    lines = [
        f"MAISI run:       {run_dir}",
        f"pairs (nnU-Net): {run_dir / 'pairs'}",
        f"updated:         {s['now']:{fmt}}",
        f"finished:        {s['done']} of {s['total']} ({s['ok']} ok, {len(s['failed'])} failed), {s['left']} left",
        f"avg job time:    {s['avg']:.0f} s per job, {workers} workers in parallel" if s["avg"] else "avg job time:    not known yet",
        f"projected done:  {s['eta']:{fmt}}  (= now + avg x left / workers)" if s["eta"] and s["left"] else "",
        f"failed jobs:     {', '.join(s['failed'])}" if s["failed"] else "",
    ]
    return "\n".join(l for l in lines if l) + "\n"


def write(top, run_dir, s, workers):
    new = top / file_name(run_dir, s)
    for old in top.glob(f"MAISI_{run_dir.name}__*.txt"):
        if old != new:
            old.rename(new)  # keeps a single file; the content is rewritten below
            break
    tmp = new.with_suffix(".tmp")
    tmp.write_text(body(run_dir, s, workers))
    os.replace(tmp, new)
    for extra in top.glob(f"MAISI_{run_dir.name}__*.txt"):  # leftovers from a crash between rename and replace
        if extra != new:
            extra.unlink()
    print(f"[{s['now']:%H:%M:%S}] {new.name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--top", type=Path, default=Path(os.environ.get("PROJ", "/work/Research")), help="where the status file lives (default $PROJ)")
    p.add_argument("--workers", type=int, default=4, help="jobs running in parallel (default 4)")
    p.add_argument("--every-jobs", type=int, default=50, help="rename every this many finished jobs (default 50)")
    p.add_argument("--poll", type=int, default=60, help="seconds between checks (default 60)")
    p.add_argument("--once", action="store_true", help="write the file once and exit")
    args = p.parse_args()
    run_dir = args.run_dir.resolve()

    last_bucket = None
    while True:
        s = status(run_dir, args.workers)
        bucket = s["done"] // args.every_jobs
        if args.once or bucket != last_bucket or s["left"] == 0:
            write(args.top, run_dir, s, args.workers)
            last_bucket = bucket
        if args.once or (s["left"] == 0 and s["total"] > 0):
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()