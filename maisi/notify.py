"""Email progress of a MAISI run: milestones with an ETA, plus alerts when it stalls, workers die or jobs fail.

Polls <run-dir> every --every seconds (reads only result.json / results.jsonl, never touches the workers) and
mails at each milestone (default 25/50/75/95 % and done), e.g. `--milestones 250 500 950`. Every mail has the
progress, failures, recent + overall rate and the projected finish time (Copenhagen time). Sent milestones are
remembered in <run-dir>/notify_state.json, so restarting the notifier doesn't resend them. Exits when all jobs
have a result.

Alerts (once per episode): no job finished for --stall-min minutes, no run_jobs.py process for this run dir
(two polls in a row), new failed jobs, projected finish after --session-end.

Mail goes via SMTP (credentials in an env file outside the repo, see GUIDE "Optional: email notifier") or, if
UCloud blocks SMTP, via ntfy.sh over HTTPS (--via ntfy; ntfy forwards it to your email and/or phone app).
The notifier dies with the UCloud job, so it can't tell you the job ended: pass --hc-url (a healthchecks.io
ping URL) and healthchecks.io mails you when the pings stop.

    python notify.py --run-dir $MAISI_RUNS/pool1000 --milestones 250 500 950 --session-end "2026-10-23 14:00"
    python notify.py --run-dir $MAISI_RUNS/pool1000 --test     # send one mail now and exit
"""

import argparse
import json
import os
import shutil
import smtplib
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

try:
    from zoneinfo import ZoneInfo

    TZ = ZoneInfo("Europe/Copenhagen")
except Exception:  # no tz database: fall back to the machine's local time
    TZ = None


def now():
    return datetime.now(TZ) if TZ else datetime.now().astimezone()


def parse_time(s):
    """ISO time written by run_jobs.py (machine-local, naive) -> aware datetime."""
    return datetime.fromisoformat(s).astimezone(TZ) if TZ else datetime.fromisoformat(s).astimezone()


def fmt(t):
    return t.strftime("%a %d %b %H:%M")


def fmt_dur(seconds):
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h // 24}d {h % 24}h {m:02d}m" if h >= 24 else f"{h}h {m:02d}m"


def collect(run_dir, window_h):
    """Progress numbers of a run directory."""
    total = len(list((run_dir / "jobs").glob("*.json")))
    results = {}
    for p in (run_dir / "raw").glob("*/result.json"):
        try:
            results[p.parent.name] = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):  # being written right now
            pass
    ok = [r for r in results.values() if r.get("status") == "ok"]
    failed = sorted(k for k, r in results.items() if r.get("status") != "ok")
    qc_fail = sum(1 for r in ok if r.get("maisi_qc_pass") is False)

    t_now = now()
    starts = [parse_time(r["started"]) for r in results.values() if r.get("started")]
    ends = sorted(parse_time(r["finished"]) for r in ok if r.get("finished"))
    s = {"total": total, "done": len(ok), "failed": failed, "qc_fail": qc_fail, "remaining": total - len(results),
         "last_finish": ends[-1] if ends else None, "rate_recent": None, "rate_all": None, "eta": None}  # fmt: skip
    if starts and ends:
        first = min(starts)
        elapsed_h = (t_now - first).total_seconds() / 3600
        s["rate_all"] = len(ends) / elapsed_h if elapsed_h > 0 else None
        window_start = max(first, t_now - timedelta(hours=window_h))
        recent = [t for t in ends if t >= window_start]
        span_h = (t_now - window_start).total_seconds() / 3600
        if len(recent) >= 3 and span_h > 0:
            s["rate_recent"] = len(recent) / span_h
        rate = s["rate_recent"] or s["rate_all"]
        if rate and s["remaining"] > 0:
            s["eta"] = t_now + timedelta(hours=s["remaining"] / rate)
        s["elapsed_h"] = elapsed_h
    return s


def workers_alive(run_dir):
    out = subprocess.run(["pgrep", "-af", "run_jobs.py"], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if str(run_dir) in line or f"/{run_dir.name}" in line)


def run_size_gb(run_dir):
    total = 0
    for root, _, files in os.walk(run_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def report(run_dir, s, args):
    lines = [f"Run: {run_dir}", f"Done: {s['done']} / {s['total']}  ({100 * s['done'] / max(s['total'], 1):.1f} %)"]
    lines.append(f"Failed: {len(s['failed'])}" + (f"  ({', '.join(s['failed'][:10])}{' ...' if len(s['failed']) > 10 else ''})" if s["failed"] else ""))
    lines.append(f"MAISI QC fail (saved anyway): {s['qc_fail']}")
    lines.append(f"Workers running: {workers_alive(run_dir)}")
    if s["rate_all"]:
        recent = f"{s['rate_recent']:.1f}/h over the last {args.window_h:g} h, " if s["rate_recent"] else ""
        lines.append(f"Rate: {recent}{s['rate_all']:.1f}/h overall ({fmt_dur(s['elapsed_h'] * 3600)} since start)")
    if s["eta"]:
        left = (s["eta"] - now()).total_seconds()
        lines.append(f"Projected finish: {fmt(s['eta'])} (in {fmt_dur(left)})")
        if s["rate_recent"] and s["rate_all"]:  # the two rates bracket the estimate
            etas = sorted(now() + timedelta(hours=s["remaining"] / r) for r in (s["rate_recent"], s["rate_all"]))
            if abs((etas[1] - etas[0]).total_seconds()) >= 600:
                lines.append(f"  range from recent vs overall rate: {fmt(etas[0])} .. {fmt(etas[1])}")
    if args.session_end:
        lines.append(f"UCloud session ends: {fmt(args.session_end)}")
        if s["eta"] and s["eta"] > args.session_end:
            lines.append("  !! projected finish is AFTER the session end: extend the job or plan a restart (runs resume)")
    if s["last_finish"]:
        lines.append(f"Last volume finished: {fmt(s['last_finish'])}")
    disk = shutil.disk_usage(run_dir)
    lines.append(f"Disk: run dir {run_size_gb(run_dir):.0f} GB, {disk.free / 1e12:.2f} TB free on the volume")
    lines.append(f"\nTime zone: {'Europe/Copenhagen' if TZ else 'machine local'}. Sent by BT2026/maisi/notify.py.")
    return "\n".join(lines)


def send(subject, body, args):
    print(f"[{now():%Y-%m-%d %H:%M}] send: {subject}", flush=True)
    if args.via == "print":
        print(body, flush=True)
        return
    try:
        if args.via == "smtp":
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = subject, os.environ["NOTIFY_SMTP_USER"], os.environ["NOTIFY_TO"]
            msg.set_content(body)
            with smtplib.SMTP(os.environ.get("NOTIFY_SMTP_HOST", "smtp.gmail.com"), int(os.environ.get("NOTIFY_SMTP_PORT", "587")), timeout=30) as smtp:
                smtp.starttls()
                smtp.login(os.environ["NOTIFY_SMTP_USER"], os.environ["NOTIFY_SMTP_PASS"])
                smtp.send_message(msg)
        else:  # ntfy: plain HTTPS POST; with NOTIFY_TO set, ntfy.sh also forwards it as an email
            headers = {"Title": subject}
            if os.environ.get("NOTIFY_TO"):
                headers["Email"] = os.environ["NOTIFY_TO"]
            url = os.environ.get("NOTIFY_NTFY_URL", "https://ntfy.sh") + "/" + os.environ["NOTIFY_NTFY_TOPIC"]
            urllib.request.urlopen(urllib.request.Request(url, data=body.encode(), headers=headers), timeout=30)
    except Exception as e:  # never let a mail problem kill the notifier
        print(f"  send failed: {e!r}", file=sys.stderr, flush=True)


def ping(args, fail=False):
    if args.hc_url:
        try:
            urllib.request.urlopen(args.hc_url + ("/fail" if fail else ""), timeout=10)
        except Exception as e:
            print(f"  healthcheck ping failed: {e!r}", file=sys.stderr, flush=True)


def load_env(path):
    if path and path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))


def milestone_counts(specs, total):
    out = set()
    for m in specs:
        n = round(total * float(m[:-1]) / 100) if m.endswith("%") else int(m)
        if 0 < n < total:
            out.add(n)
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--milestones", nargs="+", default=["25%", "50%", "75%", "95%"], help="counts or percentages; 'done' is always sent")
    p.add_argument("--every", type=int, default=300, help="poll interval in seconds (default 300)")
    p.add_argument("--window-h", type=float, default=2.0, help="hours of recent finishes for the recent rate (default 2)")
    p.add_argument("--stall-min", type=float, default=30, help="alert if no volume finished for this long (default 30)")
    p.add_argument("--session-end", default=None, help='UCloud job end, local Copenhagen time, e.g. "2026-10-23 14:00"')
    p.add_argument("--via", choices=["smtp", "ntfy", "print"], default="smtp")
    p.add_argument("--env-file", type=Path, default=Path(os.environ.get("PROJ", ".")) / ".notify.env", help="credentials (default $PROJ/.notify.env)")
    p.add_argument("--hc-url", default=os.environ.get("NOTIFY_HC_URL"), help="healthchecks.io ping URL (dead-man switch); default $NOTIFY_HC_URL")
    p.add_argument("--test", action="store_true", help="send one progress mail now and exit")
    args = p.parse_args()

    load_env(args.env_file)
    args.hc_url = args.hc_url or os.environ.get("NOTIFY_HC_URL")
    if args.session_end:
        t = datetime.fromisoformat(args.session_end)
        args.session_end = t.replace(tzinfo=TZ) if TZ else t.astimezone()
    run_dir = args.run_dir.resolve()
    name = run_dir.name
    s = collect(run_dir, args.window_h)
    if s["total"] == 0:
        sys.exit(f"no jobs in {run_dir / 'jobs'}")
    if args.test:
        send(f"[MAISI {name}] test: {s['done']}/{s['total']} done", report(run_dir, s, args), args)
        ping(args)
        return

    state_path = run_dir / "notify_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"sent": [], "failed_seen": 0}
    milestones = milestone_counts(args.milestones, s["total"])
    stalled = no_workers = late = False
    absent_polls = 0
    print(f"notifier for {run_dir}: {s['total']} jobs, milestones {milestones}, via {args.via}, every {args.every}s", flush=True)
    send(f"[MAISI {name}] notifier started: {s['done']}/{s['total']} done", report(run_dir, s, args), args)

    while True:
        s = collect(run_dir, args.window_h)
        alive = workers_alive(run_dir)
        eta = f", ETA {fmt(s['eta'])}" if s["eta"] else ""

        if s["remaining"] <= 0:
            send(f"[MAISI {name}] finished: {s['done']}/{s['total']} ok, {len(s['failed'])} failed", report(run_dir, s, args), args)
            ping(args)
            break

        due = [m for m in milestones if s["done"] >= m and m not in state["sent"]]
        if due:  # several at once (e.g. after a restart) -> one mail for the highest
            send(f"[MAISI {name}] {s['done']}/{s['total']} done{eta}", report(run_dir, s, args), args)
            state["sent"] += due

        if len(s["failed"]) > state["failed_seen"]:
            new = len(s["failed"]) - state["failed_seen"]
            send(f"[MAISI {name}] {new} new failed job(s), {len(s['failed'])} in total", report(run_dir, s, args), args)
            state["failed_seen"] = len(s["failed"])

        since = s["last_finish"]
        if since and (now() - since).total_seconds() > args.stall_min * 60:
            if not stalled:
                send(f"[MAISI {name}] STALLED: nothing finished since {fmt(since)}", report(run_dir, s, args), args)
                ping(args, fail=True)
            stalled = True
        else:
            stalled = False

        absent_polls = absent_polls + 1 if alive == 0 else 0
        if absent_polls >= 2 and not no_workers:
            send(f"[MAISI {name}] no workers running, {s['remaining']} job(s) left", report(run_dir, s, args), args)
            ping(args, fail=True)
        no_workers = absent_polls >= 2

        if args.session_end and s["eta"] and s["eta"] > args.session_end and s["done"] >= 5:
            if not late:
                send(f"[MAISI {name}] won't finish before the session ends ({fmt(args.session_end)}){eta}", report(run_dir, s, args), args)
            late = True

        state_path.write_text(json.dumps(state))
        if not (stalled or no_workers):
            ping(args)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
