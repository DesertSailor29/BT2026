#!/usr/bin/env python
"""One chart comparing the training curves of every liver run (all arms on the same axes), to see where each
arm stops improving on the real validation cases and how far it drifts from its own training data.

Reads the newest nnU-Net training_log_*.txt of every fold in Dataset1xx (also a run still training) and writes, next to
the test results (run_queue.py's publish() commits that folder):
  reports/data/test_eval_liver/progress/training_curves.png   4 panels, shared epoch axis
  reports/data/test_eval_liver/progress/training_curves.md    table view of the same numbers per run
  reports/data/test_eval_liver/progress/training_curves.csv   per-epoch raw values (train/val loss, pseudo Dice)

Validation = the 20 real LiTS val cases, so "val" curves measure how well training on synthetic data transfers to
real CT. Curves are smoothed with a centred rolling mean over 51 epochs (raw per-epoch values are very noisy: one
validation pass = 50 random patches). "Stops improving" = epoch where the smoothed val loss is lowest.

  python nnunet/training_curves.py          # called by predict_test.py eval after every test evaluation
  python nnunet/training_curves.py pancreas # Dataset102 etc. -> reports/data/test_eval_pancreas/progress/
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

TRAINER = "nnUNetTrainer__nnUNetPlans__3d_fullres"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{REPO}/reports/data/test_eval_liver/progress"  # main(target) switches it
VAL_NAME = {"liver": "20 real LiTS cases", "pancreas": "43 real MSD07 cases"}
WIN = 51
# fixed colour per arm (validated categorical palette, light mode); the repeat run reuses its arm's colour, dashed
COLOR = {"all": "#2a78d6", "ccReject": "#eb6834", "ccRepair": "#1baf7a", "ccRepairDel": "#eda100",
         "ccAucRepair": "#e87ba4"}
EXTRA = ["#008300", "#6250d6", "#e34948"]  # follow-up arms, in order of appearance
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e58", "#e6e5e0", "#fcfcfb"

RX = {"epoch": re.compile(r": Epoch (\d+)\s*$"), "train": re.compile(r": train_loss (-?[\d.]+)"),
      "val": re.compile(r": val_loss (-?[\d.]+)"), "dice": re.compile(r": Pseudo dice \[(.*)\]")}


def arm_name(ds_dir, fold):
    f = ds_dir.replace(os.environ["nnUNet_results"], os.environ["nnUNet_preprocessed"]) + "/arms.json"
    arms = json.load(open(f)) if os.path.exists(f) else []
    return arms[fold]["arm"] if fold < len(arms) else f"fold{fold}"


def parse(log):
    rows, cur = [], None
    for line in open(log, errors="replace"):
        if m := RX["epoch"].search(line):
            cur = {"epoch": int(m.group(1))}
            continue
        if cur is None:
            continue
        if m := RX["train"].search(line):
            cur["train"] = float(m.group(1))
        elif m := RX["val"].search(line):
            cur["val"] = float(m.group(1))
        elif m := RX["dice"].search(line):
            v = [float(x) for x in re.findall(r"\(([-\d.e]+)\)", m.group(1))] or \
                [float(x) for x in m.group(1).split(",") if x.strip()]
            if len(v) >= 2:
                cur["dice_liver"], cur["dice_tumour"] = v[0], v[1]
                rows.append(cur)  # epoch complete
                cur = None
    return rows


def smooth(y):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)
    h = WIN // 2
    for i in range(len(y)):
        lo, hi = max(0, i - h), min(len(y), i + h + 1)
        if hi - lo >= WIN // 2:  # skip the ragged first/last ~25 epochs
            out[i] = y[lo:hi].mean()
    return out


def runs(target="liver"):
    found = []
    for ds in sorted(glob.glob(f"{os.environ['nnUNet_results']}/Dataset1*/{TRAINER}")):
        if ("Pancreas" in ds.split("/Dataset")[1].split("/")[0]) != (target == "pancreas"):
            continue  # one target per chart
        did = int(ds.split("/Dataset")[1][:3])
        for fd in sorted(glob.glob(f"{ds}/fold_*")):
            fold = int(fd.rsplit("_", 1)[1])
            logs = sorted(glob.glob(f"{fd}/training_log_*.txt"), key=os.path.getmtime)
            if not logs:
                continue
            rows = []
            for lg in logs:  # a resumed training continues in a new log; later logs win per epoch
                for r in parse(lg):
                    rows = [x for x in rows if x["epoch"] != r["epoch"]] + [r]
            rows.sort(key=lambda r: r["epoch"])
            if len(rows) >= 10:
                found.append({"did": did, "fold": fold, "arm": arm_name(ds.rsplit("/", 1)[0], fold), "rows": rows})
    return found


def main(target="liver"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global OUT
    OUT = f"{REPO}/reports/data/test_eval_{target}/progress"
    os.makedirs(OUT, exist_ok=True)
    rs = runs(target)
    if not rs:
        print("training curves: no logs found")
        return
    extra = iter(EXTRA)
    base_arm = lambda a: re.sub(r"_rep\d+$", "", a)  # noqa: E731
    for r in rs:
        b = base_arm(r["arm"])
        if b not in COLOR:
            COLOR[b] = next(extra, "#5f5e58")
        r["color"], r["dash"] = COLOR[b], (b != r["arm"])
        e = np.array([x["epoch"] for x in r["rows"]])
        r["e"] = e
        for k in ("train", "val", "dice_tumour", "dice_liver"):
            r[k] = smooth([x[k] for x in r["rows"]])
        r["gap"] = r["val"] - r["train"]
        ok = ~np.isnan(r["val"])
        r["best_i"] = int(np.nanargmin(np.where(ok, r["val"], np.nan))) if ok.any() else None

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True, facecolor=SURFACE)
    panels = [("val", f"Validation loss ({VAL_NAME[target]}), lower is better", "loss"),
              ("gap", "Generalisation gap: validation loss minus training loss", "val − train loss"),
              ("dice_tumour", "Tumour pseudo Dice on the validation cases", "Dice"),
              ("dice_liver", f"{target.capitalize()} pseudo Dice on the validation cases", "Dice")]
    for ax, (k, title, ylab) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, loc="left", fontsize=11, pad=8)
        ax.set_ylabel(ylab)
        for r in rs:
            ax.plot(r["e"], r[k], color=r["color"], linewidth=2, linestyle=(0, (5, 3)) if r["dash"] else "-",
                    label=f"{r['arm']} (D{r['did']} f{r['fold']}, {len(r['rows'])} ep)", solid_capstyle="round")
            if k == "val" and r["best_i"] is not None:
                i = r["best_i"]
                ax.plot(r["e"][i], r[k][i], "o", markersize=8, color=r["color"], markeredgecolor=SURFACE,
                        markeredgewidth=2, zorder=5)
            last = np.where(~np.isnan(r[k]))[0]
            if len(last) and k in ("val", "dice_tumour"):  # selective direct labels at the line ends
                j = last[-1]
                ax.annotate(r["arm"], (r["e"][j], r[k][j]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=INK2)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    axes[-1].set_xlabel("epoch")
    axes[-1].set_xlim(0, 1080)
    fig.text(0.01, 0.988, f"nnU-Net training curves per {target} arm", ha="left", va="top", fontsize=14, color=INK)
    fig.text(0.01, 0.962, f"Centred {WIN}-epoch rolling mean of the per-epoch values. Dot = epoch of the lowest smoothed "
             "validation loss (where the arm stops improving on real CT).\nDashed = repeat run of the same arm. "
             "Runs still training stop at their current epoch.", ha="left", va="top", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/training_curves.png", dpi=130, facecolor=SURFACE)

    lines = [f"# Training curves per {target} arm (table view of training_curves.png)", "",
             f"Smoothed = centred {WIN}-epoch rolling mean. Validation = {VAL_NAME[target]}. Written by "
             "`nnunet/training_curves.py`.", "",
             "| run | epochs | lowest smoothed val loss (epoch) | val loss at last smoothed epoch | gap at last smoothed epoch | best smoothed tumour pseudo Dice (epoch) | "
             f"best smoothed {target} pseudo Dice |",
             "|---|---|---|---|---|---|---|"]
    with open(f"{OUT}/training_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "fold", "arm", "epoch", "train_loss", "val_loss", "dice_liver", "dice_tumour"])
        for r in rs:
            for x in r["rows"]:
                w.writerow([r["did"], r["fold"], r["arm"], x["epoch"], x["train"], x["val"], x["dice_liver"], x["dice_tumour"]])
            last = np.where(~np.isnan(r["val"]))[0]
            j, i = last[-1], r["best_i"]
            t = int(np.nanargmax(r["dice_tumour"]))
            lines.append(f"| D{r['did']} f{r['fold']} {r['arm']} | {len(r['rows'])} | {r['val'][i]:.3f} ({r['e'][i]}) | "
                         f"{r['val'][j]:.3f} | {r['gap'][j]:.3f} | {r['dice_tumour'][t]:.3f} ({r['e'][t]}) | "
                         f"{np.nanmax(r['dice_liver']):.3f} |")
    open(f"{OUT}/training_curves.md", "w").write("\n".join(lines) + "\n")
    print(f"training curves: {len(rs)} runs -> {OUT}/training_curves.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "liver")
