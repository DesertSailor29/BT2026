"""Keep one status file at the top of $PROJ whose *name* shows the progress and projected finish of an nnU-Net
training run (the training counterpart of maisi/status_file.py), e.g.

    NNUNET_D101_3d_fullres_f0__ep120of1000__dice0.712__ETA_Tue-10-Nov_04-30_CET.txt

so it can be read straight from the UCloud file browser. The file itself holds the fold folder, the last epoch's
losses / pseudo Dice, the best EMA pseudo Dice and the ETA.

Reads only nnU-Net's own training_log_*.txt files in the fold folder (all of them, so a run continued with --c
keeps its history); never touches the trainer. ETA = now + median of the last 20 epoch times x epochs left.
The name gets "STALLED" when no log line was written for --stall-min minutes (trainer dead or hung), and "DONE"
once checkpoint_final.pth exists, after which the script exits. Times are Copenhagen time.

    setsid nohup bash -c 'source "$PROJ/BT2026/nnunet/env.sh" && python "$PROJ/BT2026/nnunet/train_status.py" \
      --fold-dir "$nnUNet_results/Dataset101_LiverSynth/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"' \
      > "$PROJ/logs/status_train_<name>_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
"""

import argparse
import os
import re
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Copenhagen")
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+): (.*)$")


def parse(fold_dir):
    """Epoch records from all training logs (oldest first); later logs override epochs repeated after a resume."""
    epochs, cur, last_line = {}, None, None
    for log in sorted(fold_dir.glob("training_log_*.txt")):
        for line in log.read_text(errors="replace").splitlines():
            m = STAMP.match(line)
            if not m:
                continue
            last_line = datetime.fromisoformat(m.group(1)).replace(tzinfo=TZ)
            msg = m.group(2).strip()
            if msg.startswith("Epoch ") and msg[6:].isdigit():
                cur = int(msg[6:])
                epochs[cur] = {"epoch": cur}
            elif cur is None:
                continue
            elif msg.startswith("Epoch time:"):
                epochs[cur]["time"] = float(msg.split(":")[1].split()[0])
            elif msg.startswith("train_loss"):
                epochs[cur]["train_loss"] = float(msg.split()[-1])
            elif msg.startswith("val_loss"):
                epochs[cur]["val_loss"] = float(msg.split()[-1])
            elif msg.startswith("Pseudo dice"):
                epochs[cur]["dice"] = "[" + ", ".join(f"{float(x):.3f}" for x in re.findall(r"\(([-\d.eE]+)\)|(?<![\w(])([-\d.]+)(?=[,\]])", msg) for x in x if x) + "]"
            elif "New best EMA pseudo Dice" in msg:
                epochs[cur]["best"] = float(msg.split(":")[-1])
    return [epochs[k] for k in sorted(epochs)], last_line


def status(fold_dir, total, stall_min):
    eps, last_line = parse(fold_dir)
    now = datetime.now(TZ)
    done_eps = [e for e in eps if "time" in e]
    n = done_eps[-1]["epoch"] + 1 if done_eps else 0
    times = [e["time"] for e in done_eps[-20:]]
    per = statistics.median(times) if times else None
    left = max(total - n, 0)
    best = max((e["best"] for e in eps if "best" in e), default=None)
    final = (fold_dir / "checkpoint_final.pth").exists()
    stalled = (not final and last_line is not None and now - last_line > timedelta(minutes=stall_min))
    return {"n": n, "total": total, "per": per, "left": left, "best": best, "final": final, "stalled": stalled,
            "last": done_eps[-1] if done_eps else None, "last_line": last_line, "now": now,
            "eta": now + timedelta(seconds=per * left) if per else None}


def file_name(name, s):
    if s["final"]:
        when = f"DONE_{s['now']:%a-%d-%b_%H-%M}_{s['now'].tzname()}"
    elif s["eta"] is None:
        when = "ETA_unknown-yet"
    else:
        when = f"ETA_{s['eta']:%a-%d-%b_%H-%M}_{s['eta'].tzname()}"
    stall = "__STALLED" if s["stalled"] else ""
    dice = f"__dice{s['best']:.3f}" if s["best"] is not None else ""
    return f"NNUNET_{name}__ep{s['n']}of{s['total']}{dice}{stall}__{when}.txt"


def body(fold_dir, s):
    fmt = "%a %d %b %Y %H:%M %Z"
    last = s["last"] or {}
    lines = [
        f"nnU-Net fold:    {fold_dir}",
        f"updated:         {s['now']:{fmt}}",
        f"epochs done:     {s['n']} of {s['total']}, {s['left']} left",
        f"epoch time:      {s['per']:.0f} s (median of the last 20)" if s["per"] else "epoch time:      not known yet",
        f"projected done:  {s['eta']:{fmt}}" if s["eta"] and not s["final"] else "",
        f"last epoch:      train_loss {last.get('train_loss')}  val_loss {last.get('val_loss')}  pseudo Dice {last.get('dice')}" if last else "",
        f"best EMA pseudo Dice (val, patches): {s['best']:.4f}" if s["best"] is not None else "",
        f"STALLED: no log line since {s['last_line']:{fmt}} (trainer dead or hung?)" if s["stalled"] else "",
        "checkpoint_final.pth written: training finished" if s["final"] else "",
    ]
    return "\n".join(l for l in lines if l) + "\n"


def write(top, name, fold_dir, s):
    new = top / file_name(name, s)
    for old in top.glob(f"NNUNET_{name}__*.txt"):
        if old != new:
            old.rename(new)
            break
    tmp = new.with_suffix(".tmp")
    tmp.write_text(body(fold_dir, s))
    os.replace(tmp, new)
    for extra in top.glob(f"NNUNET_{name}__*.txt"):
        if extra != new:
            extra.unlink()
    print(f"[{s['now']:%H:%M:%S}] {new.name}", flush=True)


def default_name(fold_dir):
    # .../Dataset101_LiverSynth/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0 -> D101_3d_fullres_f0
    ds = fold_dir.parent.parent.name.split("_")[0].replace("Dataset", "D")
    cfg = fold_dir.parent.name.split("__")[-1]
    trainer = fold_dir.parent.name.split("__")[0]
    tr = "" if trainer == "nnUNetTrainer" else f"_{trainer}"
    return f"{ds}_{cfg}{tr}_f{fold_dir.name.split('_')[-1]}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold-dir", type=Path, required=True)
    p.add_argument("--name", help="label in the file name (default e.g. D101_3d_fullres_f0)")
    p.add_argument("--epochs", type=int, default=1000, help="total epochs of the trainer (default 1000)")
    p.add_argument("--top", type=Path, default=Path(os.environ.get("PROJ", "/work/Research")))
    p.add_argument("--every-epochs", type=int, default=5, help="rename every this many epochs (default 5)")
    p.add_argument("--stall-min", type=int, default=30)
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    fold_dir = args.fold_dir.resolve()
    name = args.name or default_name(fold_dir)

    last_key = None
    while True:
        s = status(fold_dir, args.epochs, args.stall_min) if fold_dir.exists() else None
        if s is None:
            time.sleep(args.poll)  # trainer has not created the fold folder yet
            continue
        key = (s["n"] // args.every_epochs, s["stalled"], s["final"], s["per"] is None)
        if args.once or key != last_key:
            write(args.top, name, fold_dir, s)
            last_key = key
        if args.once or s["final"]:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
