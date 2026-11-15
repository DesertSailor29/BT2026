#!/usr/bin/env python
"""Charts for the liver test-results report (reports/liver_results/report.html), in the shared style.

Reads reports/data/test_eval_liver/ (per-case CSVs, training_curves.csv) and writes SVGs into <out>:
  composition.svg  what each arm trains on (unchanged / repaired / tumour-free volumes)
  dice_gain.svg    tumour Dice per case, arm - baseline, mean with paired bootstrap 95% CI
  per_case.svg     the per-case differences behind each mean
  burden.svg       tumour Dice by tumour burden (terciles of reference tumour voxels) per run
  detection.svg    voxel sensitivity / precision, lesion recall, false-positive pieces per case
  curves.svg       smoothed validation loss + tumour pseudo Dice over training
  fp_where.svg     false-positive tumour pieces / ml per case, inside vs outside the predicted main liver structure
  ccpost.svg       tumour Dice gain over baseline, as saved vs both post-processed with the cc rule (ccPost)
and prints the numbers it plotted (to check against summary.md).

  python reports/liver_results/make_charts.py <out_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "style"))
import style  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "..", "nnunet"))
from training_curves import smooth  # noqa: E402  (same smoothing as training_curves.md)

DATA = os.path.join(HERE, "..", "data", "test_eval_liver")
FPL = os.path.join(DATA, "fp_location")
RUN_ID = {"all": "101:0", "all_rep2": "101:1", "ccReject": "101:2", "ccRepair": "103:0", "ccRepairDel": "103:1",
          "ccAucRepair": "104:0"}
RUNS = {"all": "D101_fold_0", "all_rep2": "D101_fold_1", "ccReject": "D101_fold_2", "ccRepair": "D103_fold_0",
        "ccRepairDel": "D103_fold_1", "ccAucRepair": "D104_fold_0"}
ARMS = ["ccReject", "ccRepair", "ccRepairDel", "ccAucRepair"]
N_TRAIN = {"all": 999, "all_rep2": 999, "ccReject": 343, "ccRepair": 999, "ccRepairDel": 842, "ccAucRepair": 999}
B = 10000


def load():
    d = {k: pd.read_csv(f"{DATA}/{v}_per_case.csv").set_index("case") for k, v in RUNS.items()}
    base = (d["all"].tumour_dice + d["all_rep2"].tumour_dice) / 2
    return d, base


def boot_ci(x, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def label_rows(ax, names, colors, sub=None):
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([""] * len(names))
    for i, (n, c) in enumerate(zip(names, colors)):
        ax.annotate("●", (0, i), xycoords=("axes fraction", "data"), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", color=c, fontsize=9)
        ax.annotate(n, (0, i), xycoords=("axes fraction", "data"), xytext=(-20, 0), textcoords="offset points",
                    ha="right", va="center", color=style.INK, fontsize=8.5)
        if sub:
            ax.annotate(sub[i], (0, i), xycoords=("axes fraction", "data"), xytext=(-20, -9),
                        textcoords="offset points", ha="right", va="center", color=style.MUTED, fontsize=7)
    ax.set_ylim(len(names) - 0.5, -0.6)
    ax.grid(axis="y", visible=False)


def composition(out, plt):
    rows = [("all", [0, 0, 0, 999]), ("ccReject", [343, 0, 0, 0]), ("ccRepair", [343, 499, 157, 0]),
            ("ccRepairDel", [343, 499, 0, 0]), ("ccAucRepair", [300, 522, 177, 0])]
    seg = [("unchanged by the rule", "#4d4c48"), ("repaired, tumour left", "#9c9b95"),
           ("repaired, now tumour-free", "#d6d5cf"), ("as generated (no rule)", "#c6dcf5")]
    fig, ax = plt.subplots(figsize=style.size(1.0, 0.30))
    for i, (arm, vals) in enumerate(rows):
        left = 0
        for (name, c), v in zip(seg, vals):
            if v:
                ax.barh(i, v, left=left, height=0.62, color=c, edgecolor=style.SURFACE, linewidth=1.5)
                dark = c in ("#4d4c48",)
                ax.text(left + v / 2, i, f"{v}", ha="center", va="center", fontsize=7.8,
                        color="#ffffff" if dark else style.INK)
                left += v
        ax.text(left + 12, i, f"{left}", va="center", fontsize=8, color=style.INK2, fontweight="semibold")
    label_rows(ax, [r[0] for r in rows], [style.arm_color(r[0]) for r in rows])
    ax.set_xlim(0, 1060)
    ax.set_xlabel("synthetic training volumes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in seg]
    ax.legend(handles, [s for s, _ in seg], ncol=4, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
              handlelength=1.0, columnspacing=1.4, fontsize=7.6)
    style.save_svg(fig, f"{out}/composition.svg")


def dice_gain(d, base, out, plt):
    rows = []
    for a in ARMS:
        rows.append((a, *boot_ci(d[a].tumour_dice - base)))
    noise = boot_ci(d["all_rep2"].tumour_dice - d["all"].tumour_dice)
    names = ARMS + ["noise: all_rep2 − all"]
    fig, ax = plt.subplots(figsize=style.size(1.0, 0.33))
    ax.axvspan(-0.01, 0.01, color=style.PANEL, zorder=0)
    ax.axvline(0, color=style.INK2, linewidth=0.8, zorder=1)
    for i, (a, m, lo, hi) in enumerate(rows + [("noise", *noise)]):
        c = style.arm_color(a) if a != "noise" else style.MUTED
        ax.plot([lo, hi], [i, i], color=c, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.plot(m, i, "o", color=c, markersize=8, markeredgecolor=style.SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]", (hi, i), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=7.8, color=style.INK)
    label_rows(ax, names, [style.arm_color(a) for a in ARMS] + [style.MUTED])
    ax.set_xlim(-0.03, 0.20)
    ax.set_xlabel("tumour Dice per case, difference to baseline (mean of the two all runs)")
    ax.text(0.0, -0.75, "±0.01 noise floor", fontsize=7, color=style.MUTED, ha="center", va="bottom")
    style.save_svg(fig, f"{out}/dice_gain.svg")
    return rows, noise


def per_case(d, base, out, plt):
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=style.size(1.0, 0.34))
    ax.axvline(0, color=style.INK2, linewidth=0.8)
    stats = {}
    for i, a in enumerate(ARMS):
        x = (d[a].tumour_dice - base).to_numpy()
        y = i + rng.uniform(-0.22, 0.22, len(x))
        ax.scatter(x, y, s=11, color=style.arm_color(a), alpha=0.75, linewidths=0.4, edgecolors=style.SURFACE)
        med = np.median(x)
        ax.plot([med, med], [i - 0.32, i + 0.32], color=style.INK, linewidth=1.6)
        w, t, l = (x > 0.01).sum(), (abs(x) <= 0.01).sum(), (x < -0.01).sum()
        stats[a] = (w, t, l, med)
        ax.annotate(f"better {w} · same {t} · worse {l}", (1, i), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.6, color=style.INK2)
    label_rows(ax, ARMS, [style.arm_color(a) for a in ARMS])
    ax.set_xlim(-0.75, 0.85)
    ax.set_xlabel("tumour Dice difference to baseline, one dot per test case (black tick = median)")
    style.save_svg(fig, f"{out}/per_case.svg")
    return stats


def burden(d, base, out, plt):
    ref = d["all"].tumour_ref
    pos = ref[ref > 0]
    q = pd.qcut(pos, 3, labels=["small", "medium", "large"]).reindex(ref.index)
    ranges = pos.groupby(q[ref > 0], observed=True).agg(["min", "max", "count"])
    runs = ["baseline"] + ARMS
    fig, axes = plt.subplots(1, 3, figsize=style.size(1.0, 0.36), sharey=True)
    res = {}
    for j, (k, ax) in enumerate(zip(["small", "medium", "large"], axes)):
        m = q == k
        for i, r in enumerate(runs):
            v = (base if r == "baseline" else d[r].tumour_dice)[m]
            mean, lo, hi = boot_ci(v)
            res[(k, r)] = mean
            c = style.arm_color("all") if r == "baseline" else style.arm_color(r)
            ax.plot([lo, hi], [i, i], color=c, linewidth=2, alpha=0.45)
            ax.plot(mean, i, "o", color=c, markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=1.2)
            ax.annotate(f"{mean:.2f}", (mean, i), xytext=(0, 6.5), textcoords="offset points", ha="center",
                        fontsize=7, color=style.INK2)
        n, lo_v, hi_v = int(ranges.loc[k, "count"]), ranges.loc[k, "min"], ranges.loc[k, "max"]
        ax.set_title(f"{k} tumours ({n} cases)", fontsize=8.5, pad=14)
        ax.text(0, 1.02, f"{lo_v/1000:.1f}k–{hi_v/1000:,.0f}k tumour voxels", transform=ax.transAxes, fontsize=7,
                color=style.MUTED, va="bottom")
        ax.set_xlim(0, 0.8)
        ax.set_xticks([0, 0.4, 0.8])
    label_rows(axes[0], ["baseline (all)"] + ARMS, [style.arm_color("all")] + [style.arm_color(a) for a in ARMS])
    for ax in axes[1:]:
        ax.grid(axis="y", visible=False)
    fig.supxlabel("mean tumour Dice per case (bar = 95% CI)", fontsize=8.5, color=style.INK2, y=-0.06)
    fig.subplots_adjust(wspace=0.16)
    style.save_svg(fig, f"{out}/burden.svg")
    return res, ranges


def detection(d, out, plt):
    runs = ["all", "all_rep2"] + ARMS
    m = {}
    for r in runs:
        x = d[r]
        m[r] = {"sens": x.tumour_tp.sum() / x.tumour_ref.sum(), "prec": x.tumour_tp.sum() / x.tumour_pred.sum(),
                "recall": x.les_det.sum() / x.les_ref.sum(), "fp": x.les_fp.mean()}
    panels = [("sens", "tumour voxels found (sensitivity)", (0, 1)),
              ("prec", "predicted tumour voxels that are tumour (precision)", (0, 1)),
              ("recall", "reference lesions found (lesion recall)", (0, 1)),
              ("fp", "false-positive tumour pieces per case", (0, 75))]
    fig, axes = plt.subplots(2, 2, figsize=style.size(1.0, 0.56))
    for ax, (k, title, lim) in zip(axes.flat, panels):
        for i, r in enumerate(runs):
            v = m[r][k]
            c = style.arm_color(r)
            ax.plot([lim[0], v], [i, i], color=style.GRID, linewidth=1, zorder=1)
            rep = "_rep" in r
            ax.plot(v, i, "o", color=c, markersize=7 if not rep else 6, markeredgecolor=c if rep else style.SURFACE,
                    markeredgewidth=1.4, markerfacecolor=style.SURFACE if rep else c, zorder=2)
            ax.annotate(f"{v:.0f}" if k == "fp" else f"{v:.2f}", (v, i), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=7.4, color=style.INK2)
        ax.set_title(title, fontsize=8.3)
        ax.set_xlim(*lim)
        ax.set_xticks([0, 0.5, 1.0] if k != "fp" else [0, 25, 50, 75])
    for ax in axes[:, 0]:
        label_rows(ax, runs, [style.arm_color(r) for r in runs])
    for ax in axes[:, 1]:
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([])
        ax.set_ylim(len(runs) - 0.5, -0.6)
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.14)
    style.save_svg(fig, f"{out}/detection.svg")
    return m


def curves(out, plt):
    c = pd.read_csv(f"{DATA}/progress/training_curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=style.size(1.0, 0.36))
    lows = {}
    for arm, g in c.groupby("arm", sort=False):
        g = g.sort_values("epoch")
        col = style.arm_color(arm)
        ls = (0, (4, 2.5)) if "_rep" in arm else "-"
        for ax, key in zip(axes, ["val_loss", "dice_tumour"]):
            s = pd.Series(smooth(g[key].to_numpy()), index=g.index)
            ax.plot(g.epoch, s, color=col, linewidth=1.5, linestyle=ls, label=arm)
            if key == "val_loss":
                i = s.idxmin()
                lows[arm] = (int(g.epoch[i]), float(s[i]), float(s.dropna().iloc[-1]))
                ax.plot(g.epoch[i], s[i], "o", color=col, markersize=6, markeredgecolor=style.SURFACE,
                        markeredgewidth=1.2)
    axes[0].set_title("validation loss (lower = better)")
    axes[1].set_title("tumour pseudo Dice (validation)")
    axes[0].set_ylim(-0.5, -0.2)
    for ax in axes:
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("epoch")
        ax.set_xlim(0, 1000)
    axes[0].legend(loc="upper right", fontsize=7.2, handlelength=1.8, ncol=2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.22)
    style.save_svg(fig, f"{out}/curves.svg")
    return lows


def fp_where(out, plt):
    """FP pieces and ml per case, split by inside / outside the predicted main structure (fp_location.py)."""
    c = pd.read_csv(f"{FPL}/components.csv")
    n = pd.read_csv(f"{FPL}/per_case.csv").query("variant == 'none'").groupby("run").case.nunique()
    fp = c[~c.hit]
    runs = ["all", "all_rep2"] + ARMS
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=style.size(1.0, 0.34), sharey=True)
    for r in runs:
        g = fp[fp.run == RUN_ID[r]]
        k = n[RUN_ID[r]]
        res[r] = {"out_n": (~g.in_pred_main).sum() / k, "in_n": g.in_pred_main.sum() / k,
                  "out_ml": g.ml[~g.in_pred_main].sum() / k, "in_ml": g.ml[g.in_pred_main].sum() / k}
    for ax, unit, lim, title in [(axes[0], "n", 88, "false-positive tumour pieces per case"),
                                 (axes[1], "ml", 52, "false-positive tumour volume per case (ml)")]:
        for i, r in enumerate(runs):
            col = style.arm_color(r)
            o, n_in = res[r][f"out_{unit}"], res[r][f"in_{unit}"]
            ax.barh(i, n_in, height=0.62, color=col, edgecolor=style.SURFACE, linewidth=1)
            ax.barh(i, o, left=n_in, height=0.62, color=col, alpha=0.32, edgecolor=style.SURFACE, linewidth=1)
            ax.annotate(f"{n_in:.1f} + {o:.1f}", (n_in + o, i),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.3, color=style.INK2)
        ax.set_xlim(0, lim)
        ax.set_title(title, fontsize=8.3)
        ax.grid(axis="y", visible=False)
    label_rows(axes[0], runs, [style.arm_color(r) for r in runs])
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.INK2), plt.Rectangle((0, 0), 1, 1, color=style.INK2, alpha=0.32)]
    axes[0].legend(handles, ["inside the predicted liver (solid)", "outside it (light): removed by ccPost"], ncol=2,
                   loc="upper left", bbox_to_anchor=(-0.02, -0.1), handlelength=1.0, fontsize=7.6)
    fig.subplots_adjust(wspace=0.12)
    style.save_svg(fig, f"{out}/fp_where.svg")
    return res


def ccpost(out, plt):
    """Arm - baseline tumour Dice per case, as saved (hollow) and with both post-processed by ccPost (filled)."""
    p = pd.read_csv(f"{FPL}/per_case.csv")
    t = {(r, v): p[(p.run == RUN_ID[r]) & (p.variant == v)].set_index("case").tumour_dice
         for r in RUN_ID for v in ("none", "ccPost")}
    res = {}
    fig, ax = plt.subplots(figsize=style.size(1.0, 0.36))
    ax.axvspan(-0.01, 0.01, color=style.PANEL, zorder=0)
    ax.axvline(0, color=style.INK2, linewidth=0.8, zorder=1)
    rows = ARMS + ["ccPost on all: all − all"]
    for i, a in enumerate(rows):
        if i < len(ARMS):
            pairs = [("none", -0.17), ("ccPost", 0.17)]
            col = style.arm_color(a)
        else:
            pairs = [("gain", 0.0)]
            col = style.arm_color("all")
        for v, dy in pairs:
            if v == "gain":  # what ccPost alone does to the baseline
                b0 = (t[("all", "none")] + t[("all_rep2", "none")]) / 2
                b1 = (t[("all", "ccPost")] + t[("all_rep2", "ccPost")]) / 2
                d = b1 - b0
            else:
                b = (t[("all", v)] + t[("all_rep2", v)]) / 2
                d = t[(a, v)] - b
            m, lo, hi = boot_ci(d.dropna())
            res[(a, v)] = (m, lo, hi)
            filled = v != "none"
            ax.plot([lo, hi], [i + dy, i + dy], color=col, linewidth=2 if filled else 1.4,
                    alpha=1 if filled else 0.6, solid_capstyle="round", zorder=2)
            ax.plot(m, i + dy, "o", color=col, markersize=7, markeredgecolor=col if not filled else style.SURFACE,
                    markerfacecolor=col if filled else style.SURFACE, markeredgewidth=1.4, zorder=3)
            ax.annotate(f"{m:+.3f}", (hi, i + dy), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=7.4, color=style.INK if filled else style.MUTED)
    label_rows(ax, rows, [style.arm_color(a) for a in ARMS] + [style.arm_color("all")])
    ax.set_xlim(-0.05, 0.20)
    ax.set_xlabel("tumour Dice per case, difference (mean, paired bootstrap 95% CI)")
    h = [plt.Line2D([], [], marker="o", color=style.INK2, markerfacecolor=style.SURFACE, linestyle="none"),
         plt.Line2D([], [], marker="o", color=style.INK2, linestyle="none")]
    ax.legend(h, ["arm − baseline, as saved", "arm − baseline, both after ccPost"], ncol=2, loc="lower left",
              bbox_to_anchor=(-0.02, 1.0), fontsize=7.6)
    style.save_svg(fig, f"{out}/ccpost.svg")
    return res


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build")
    os.makedirs(out, exist_ok=True)
    style.use()
    import matplotlib.pyplot as plt
    d, base = load()
    composition(out, plt)
    rows, noise = dice_gain(d, base, out, plt)
    print("dice gain (mean, lo, hi):", [(a, round(m, 4), round(lo, 4), round(hi, 4)) for a, m, lo, hi in rows],
          "noise", [round(v, 4) for v in noise])
    print("per case (better, same, worse, median):", per_case(d, base, out, plt))
    res, ranges = burden(d, base, out, plt)
    print("burden:", {k: round(v, 3) for k, v in res.items()})
    print(ranges)
    print("detection:", {r: {k: round(v, 3) for k, v in x.items()} for r, x in detection(d, out, plt).items()})
    print("val loss low (epoch, low, last):", curves(out, plt))
    print("fp where (per case):", {r: {k: round(v, 2) for k, v in x.items()} for r, x in fp_where(out, plt).items()})
    print("ccpost:", {k: tuple(round(x, 4) for x in v) for k, v in ccpost(out, plt).items()})


if __name__ == "__main__":
    main()
