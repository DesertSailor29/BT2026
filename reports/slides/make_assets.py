#!/usr/bin/env python
"""Charts and CT crops for the supervisor slide deck (reports/slides/build_deck.js), as PNGs.

Same data, colours and marks as report 4 (reports/liver_results/make_charts.py, reports/style/style.py), redrawn
for slides: larger text, fewer labels, and every repeat run as its own solid, labelled row ("ccAucRepair (run 2)")
instead of hollow/dashed markers. Each chart is drawn at the size it takes on the slide (inches), so a font size here
is its point size on the slide.

  python reports/slides/make_assets.py <out_dir>
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "reports", "style"))
sys.path.insert(0, os.path.join(REPO, "reports", "liver_results"))
sys.path.insert(0, os.path.join(REPO, "nnunet"))
import style  # noqa: E402
import make_charts as mc  # noqa: E402  (data paths, RUN_ID, load, boot_ci, smoothing)
import build as rb  # noqa: E402  (CT panel grid of the ccpost_review figures, per-run case stats)

BEST = os.path.join(REPO, "reports", "data", "test_eval_liver_best")
FIG = os.path.join(REPO, "reports", "figures")
NAME = {"all": "all", "all_rep2": "all (run 2)", "ccReject": "ccReject", "ccRepair": "ccRepair",
        "ccRepairDel": "ccRepairDel", "ccAucRepair": "ccAucRepair", "ccAucRepair_rep2": "ccAucRepair (run 2)",
        "baseline": "baseline (mean of all)"}
RUNS = mc.RUNS_ALL  # all, all_rep2, ccReject, ccRepair, ccRepairDel, ccAucRepair, ccAucRepair_rep2
ARMS_X = mc.ARMS_X
FS, FS_S, FS_T = 13, 11.5, 14  # labels, small labels / ticks, panel titles (pt on the slide)
DPI = 220


def setup():
    style.use()
    import matplotlib.pyplot as plt
    # slide sizes; the deck's text is Arial, so the charts use its metric twin where Arial itself is missing
    plt.rcParams.update({"font.sans-serif": ["Arial", "Liberation Sans"] + style.FONT, "font.size": FS,
                         "axes.titlesize": FS_T, "axes.labelsize": FS_S, "xtick.labelsize": FS_S,
                         "ytick.labelsize": FS, "legend.fontsize": FS_S, "lines.markersize": 9,
                         "axes.titlepad": 10})
    return plt


def save(fig, out, name):
    fig.savefig(f"{out}/{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    Image.open(f"{out}/{name}.png").convert("RGB").save(f"{out}/{name}.png")  # no alpha: renders white everywhere
    import matplotlib.pyplot as plt
    plt.close(fig)


def rows(ax, names, colors):
    """Arm rows: a coloured dot + the run's name left of the axis, top to bottom."""
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([""] * len(names))
    for i, (n, c) in enumerate(zip(names, colors)):
        ax.annotate("●", (0, i), xycoords=("axes fraction", "data"), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", color=c, fontsize=FS)
        ax.annotate(n, (0, i), xycoords=("axes fraction", "data"), xytext=(-24, 0), textcoords="offset points",
                    ha="right", va="center", color=style.INK, fontsize=FS)
    ax.set_ylim(len(names) - 0.5, -0.6)
    ax.grid(axis="y", visible=False)


def no_rows(ax, n):
    ax.set_yticks(range(n))
    ax.set_yticklabels([])
    ax.set_ylim(n - 0.5, -0.6)
    ax.grid(axis="y", visible=False)


def color(r):
    return style.arm_color("all" if r == "baseline" else r)


def sgn(v, nd=3):
    """Signed number with a real minus sign."""
    return f"{v:+.{nd}f}".replace("-", "−")


def forest(ax, items, xlim):
    """items: (label, colour, mean, lo, hi). Mean dot + CI bar, value right of the bar."""
    ax.axvspan(-0.01, 0.01, color=style.PANEL, zorder=0)
    ax.axvline(0, color=style.INK2, linewidth=0.9, zorder=1)
    for i, (_, c, m, lo, hi) in enumerate(items):
        ax.plot([lo, hi], [i, i], color=c, linewidth=3, solid_capstyle="round", zorder=2)
        ax.plot(m, i, "o", color=c, markersize=11, markeredgecolor=style.SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(sgn(m), (hi, i), xytext=(9, 0), textcoords="offset points", va="center", fontsize=FS,
                    color=style.INK, fontweight="bold")
    rows(ax, [x[0] for x in items], [x[1] for x in items])
    ax.set_xlim(*xlim)


# ---------------------------------------------------------------- charts

def gain(d, base, out, plt):
    """Slide 4: tumour Dice gain over the baseline, every run incl. run 2; noise rows below."""
    items = [(NAME[a], color(a), *mc.boot_ci(d[a].tumour_dice - base)) for a in ARMS_X]
    items.append(("all: run 2 − run 1", style.MUTED, *mc.boot_ci(d["all_rep2"].tumour_dice - d["all"].tumour_dice)))
    items.append(("ccAucRepair: run 2 − run 1", style.MUTED,
                  *mc.boot_ci(d["ccAucRepair_rep2"].tumour_dice - d["ccAucRepair"].tumour_dice)))
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    forest(ax, items, (-0.05, 0.17))
    ax.axhline(4.5, color=style.GRID, linewidth=1)
    ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.15])
    ax.set_xlabel("tumour Dice per case, difference to baseline (dot = mean, bar = 95% CI)")
    ax.text(0, -0.85, "±0.01 noise floor", fontsize=FS_S - 1, color=style.MUTED, ha="center", va="bottom")
    save(fig, out, "gain")
    return items


def detection(d, out, plt):
    """Slide 5: false-positive pieces per case, precision, lesion recall per run."""
    m = {r: {"fp": d[r].les_fp.mean(), "prec": d[r].tumour_tp.sum() / d[r].tumour_pred.sum(),
             "recall": d[r].les_det.sum() / d[r].les_ref.sum()} for r in RUNS}
    panels = [("fp", "false-positive pieces / case", (0, 80), "{:.0f}"),
              ("prec", "precision", (0, 1), "{:.2f}"),
              ("recall", "lesion recall", (0, 1), "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for ax, (k, title, lim, f) in zip(axes, panels):
        for i, r in enumerate(RUNS):
            v = m[r][k]
            ax.plot([lim[0], v], [i, i], color=color(r), linewidth=3, alpha=0.35, solid_capstyle="butt")
            ax.plot(v, i, "o", color=color(r), markersize=10, markeredgecolor=style.SURFACE, markeredgewidth=1.3)
            ax.annotate(f.format(v), (v, i), xytext=(9, 0), textcoords="offset points", va="center", fontsize=FS_S,
                        color=style.INK)
        ax.set_title(title, fontsize=FS)
        ax.set_xlim(lim[0], lim[1] * 1.12)
        ax.set_xticks([0, 40, 80] if k == "fp" else [0, 0.5, 1])
    rows(axes[0], [NAME[r] for r in RUNS], [color(r) for r in RUNS])
    for ax in axes[1:]:
        no_rows(ax, len(RUNS))
    fig.subplots_adjust(wspace=0.12)
    save(fig, out, "detection")
    return m


def fp_where(out, plt):
    """Slide 6: false-positive pieces per case, inside vs outside the predicted liver."""
    c = pd.read_csv(f"{mc.FPL}/components.csv")
    n = pd.read_csv(f"{mc.FPL}/per_case.csv").query("variant == 'none'").groupby("run").case.nunique()
    fp = c[~c.hit]
    res = {}
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    for i, r in enumerate(RUNS):
        g = fp[fp.run == mc.RUN_ID[r]]
        k = n[mc.RUN_ID[r]]
        n_in, o = g.in_pred_main.sum() / k, (~g.in_pred_main).sum() / k
        res[r] = (n_in, o)
        ax.barh(i, n_in, height=0.66, color=color(r), edgecolor=style.SURFACE, linewidth=1)
        ax.barh(i, o, left=n_in, height=0.66, color=color(r), alpha=0.3, edgecolor=style.SURFACE, linewidth=1)
        ax.annotate(f"{n_in:.0f} + {o:.0f}", (n_in + o, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=FS_S, color=style.INK)
    rows(ax, [NAME[r] for r in RUNS], [color(r) for r in RUNS])
    ax.set_xlim(0, 85)
    ax.set_xlabel("false-positive tumour pieces per test case")
    h = [plt.Rectangle((0, 0), 1, 1, color=style.INK2), plt.Rectangle((0, 0), 1, 1, color=style.INK2, alpha=0.3)]
    ax.legend(h, ["inside the predicted liver", "outside it"], ncol=1, loc="lower right", handlelength=1.0,
              fontsize=FS_S)
    save(fig, out, "fp_where")
    return res


def mean_dice(p, r, v):
    g = p[(p.run == mc.RUN_ID[r]) & (p.variant == v)]
    return g.tumour_dice.mean()


def ccpost(out, plt):
    """Slide 7: tumour Dice per run, as saved → after ccPost."""
    p = pd.read_csv(f"{mc.FPL}/per_case.csv")
    res = {r: (mean_dice(p, r, "none"), mean_dice(p, r, "ccPost")) for r in RUNS}
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for i, r in enumerate(RUNS):
        a, b = res[r]
        ax.plot([a, b], [i, i], color=color(r), linewidth=3, alpha=0.45, solid_capstyle="butt")
        ax.plot(a, i, "o", color=style.MUTED, markersize=8, markeredgecolor=style.SURFACE, markeredgewidth=1.2)
        ax.plot(b, i, "o", color=color(r), markersize=12, markeredgecolor=style.SURFACE, markeredgewidth=1.4)
        ax.annotate(f"{a:.3f} → {b:.3f}", (max(a, b), i), xytext=(12, 0), textcoords="offset points", va="center",
                    fontsize=FS, color=style.INK)
    rows(ax, [NAME[r] for r in RUNS], [color(r) for r in RUNS])
    ax.set_xlim(0.28, 0.48)
    ax.set_xticks([0.30, 0.35, 0.40, 0.45])
    ax.set_xlabel("mean tumour Dice per case (111 test cases)")
    h = [plt.Line2D([], [], marker="o", color=style.MUTED, linestyle="none", markersize=8),
         plt.Line2D([], [], marker="o", color=style.INK2, linestyle="none", markersize=11)]
    ax.legend(h, ["as saved", "after ccPost"], ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.0))
    save(fig, out, "ccpost")
    return res


def aucpost(out, plt):
    """Slide 8 left: tumour Dice per run after ccPost → after ccAucPost (cut 0.5475)."""
    p = pd.read_csv(f"{mc.AUP}/per_case.csv")
    res = {r: (mean_dice(p, r, "ccPost"), mean_dice(p, r, "ccAucPost0.5475")) for r in RUNS}
    fig, ax = plt.subplots(figsize=(6.1, 4.3))
    for i, r in enumerate(RUNS):
        a, b = res[r]
        ax.plot([a, b], [i, i], color=color(r), linewidth=3, alpha=0.45, solid_capstyle="butt")
        ax.plot(a, i, "o", color=style.MUTED, markersize=8, markeredgecolor=style.SURFACE, markeredgewidth=1.2)
        ax.plot(b, i, "o", color=color(r), markersize=12, markeredgecolor=style.SURFACE, markeredgewidth=1.4)
        ax.annotate(f"{b:.3f}", (max(a, b), i), xytext=(12, 0), textcoords="offset points", va="center",
                    fontsize=FS, color=style.INK, fontweight="bold")
    rows(ax, [NAME[r] for r in RUNS], [color(r) for r in RUNS])
    ax.set_xlim(0.37, 0.445)
    ax.set_xticks([0.38, 0.40, 0.42, 0.44])
    ax.set_xlabel("mean tumour Dice per case (111 test cases)")
    h = [plt.Line2D([], [], marker="o", color=style.MUTED, linestyle="none", markersize=8),
         plt.Line2D([], [], marker="o", color=style.INK2, linestyle="none", markersize=11)]
    ax.legend(h, ["after ccPost", "after ccAucPost (cut 0.5475)"], ncol=2, loc="lower left",
              bbox_to_anchor=(-0.02, 1.0))
    save(fig, out, "aucpost")
    return res


def sweep(out, plt):
    """Slide 8 right: stricter AUC cuts on the predictions, test Dice and real lesions lost, mean over the 7 runs
    with the range (min-max) of the runs as a band."""
    t = pd.read_csv(f"{mc.AUP}/per_case.csv")
    v = pd.read_csv(f"{mc.AUP}_val/per_case.csv")
    dice_t = np.array([[mean_dice(t, r, c) for c in mc.CUTS] for r in RUNS])
    dice_v = np.array([[mean_dice(v, r, c) for c in mc.CUTS] for r in RUNS])
    lost = []
    for r in RUNS:
        g = t[t.run == mc.RUN_ID[r]]
        det0 = g[g.variant == "ccPost"].les_det.sum()
        lost.append([det0 - g[g.variant == c].les_det.sum() for c in mc.CUTS])
    lost = np.array(lost)
    x = mc.CUT_X
    fig, axes = plt.subplots(2, 1, figsize=(4.9, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axes[0]
    for arr, c, lab in ((dice_t, style.INK, "test (111)"), (dice_v, style.MUTED, "val (20)")):
        ax.fill_between(x, arr.min(0), arr.max(0), color=c, alpha=0.12, linewidth=0)
        ax.plot(x, arr.mean(0), color=c, marker="o", markersize=6, linewidth=2, label=lab)
    ax.set_title("tumour Dice (mean of 7 runs, band = range)", fontsize=FS)
    ax.legend(loc="lower right", ncol=2)
    ax.set_ylim(0.32, 0.47)
    ax = axes[1]
    ax.fill_between(x, lost.min(0), lost.max(0), color=style.TUMOUR_DARK, alpha=0.12, linewidth=0)
    ax.plot(x, lost.mean(0), color=style.TUMOUR_DARK, marker="o", markersize=6, linewidth=2)
    ax.set_title("real test lesions lost per run", fontsize=FS)
    for a in axes:
        a.axvline(0.5475, color=style.MUTED, linewidth=1, linestyle=":")
        a.axvspan(0.585, 0.615, color=style.ORGAN_TINT, zorder=0)
        a.grid(axis="y", visible=True)
        a.grid(axis="x", visible=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["cc only", ".5475", ".60", ".637", ".70"])
    axes[1].set_xlabel("AUC cut on the predictions")
    fig.subplots_adjust(hspace=0.42)
    save(fig, out, "sweep")
    return dice_t, dice_v, lost


def burden(d, base, out, plt):
    """Slide 9: tumour Dice by tumour burden thirds."""
    ref = d["all"].tumour_ref
    pos = ref[ref > 0]
    q = pd.qcut(pos, 3, labels=["small", "medium", "large"]).reindex(ref.index)
    runs = ["baseline"] + ARMS_X
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), sharey=True)
    res = {}
    for k, ax in zip(["small", "medium", "large"], axes):
        m = q == k
        for i, r in enumerate(runs):
            v = (base if r == "baseline" else d[r].tumour_dice)[m]
            mean, lo, hi = mc.boot_ci(v)
            res[(k, r)] = mean
            ax.plot([lo, hi], [i, i], color=color(r), linewidth=3, alpha=0.4, solid_capstyle="round")
            ax.plot(mean, i, "o", color=color(r), markersize=10, markeredgecolor=style.SURFACE, markeredgewidth=1.3)
            ax.annotate(f"{mean:.2f}", (hi, i), xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=FS_S, color=style.INK)
        ax.set_title(f"{k} tumours ({int(m.sum())} cases)", fontsize=FS)
        ax.set_xlim(0, 0.9)
        ax.set_xticks([0, 0.4, 0.8])
    rows(axes[0], [NAME[r] for r in runs], [color(r) for r in runs])
    for ax in axes[1:]:
        ax.grid(axis="y", visible=False)
    fig.supxlabel("mean tumour Dice per case (bar = 95% CI); tumour-free cases left out", fontsize=FS_S,
                  color=style.INK2, y=-0.04)
    fig.subplots_adjust(wspace=0.12)
    save(fig, out, "burden")
    return res


def noise(d, out, plt):
    """Slide 10: two identical runs vs the differences between filter arms."""
    t = {r: d[r].tumour_dice for r in RUNS}
    pairs = [("all: run 2 − run 1", style.MUTED, t["all_rep2"] - t["all"]),
             ("ccAucRepair: run 2 − run 1", style.MUTED, t["ccAucRepair_rep2"] - t["ccAucRepair"]),
             ("ccAucRepair − ccRepair", color("ccAucRepair"), t["ccAucRepair"] - t["ccRepair"]),
             ("ccAucRepair − ccReject", color("ccAucRepair"), t["ccAucRepair"] - t["ccReject"]),
             ("ccRepair − ccRepairDel", color("ccRepair"), t["ccRepair"] - t["ccRepairDel"]),
             ("ccRepair − ccReject", color("ccRepair"), t["ccRepair"] - t["ccReject"])]
    items = [(n, c, *mc.boot_ci(x.dropna())) for n, c, x in pairs]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    forest(ax, items, (-0.06, 0.08))
    ax.axhline(1.5, color=style.GRID, linewidth=1)
    ax.set_xticks([-0.05, -0.025, 0, 0.025, 0.05, 0.075])
    ax.set_xlabel("tumour Dice per case, difference (dot = mean, bar = 95% CI)")
    save(fig, out, "noise")
    return items


def best(out, plt):
    """Slide 11: checkpoint_final vs checkpoint_best, as saved and after ccPost, gain over the baseline."""
    pf = pd.read_csv(f"{mc.FPL}/per_case.csv")
    pb = pd.read_csv(f"{BEST}/fp_location/per_case.csv")

    def series(p, r, v):
        return p[(p.run == mc.RUN_ID[r]) & (p.variant == v)].set_index("case").tumour_dice

    runs = ["baseline"] + ARMS_X
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), sharey=True)
    for ax, v, title in ((axes[0], "none", "as saved"), (axes[1], "ccPost", "both sides after ccPost")):
        for i, r in enumerate(runs):
            for p, ck, mk, dy in ((pf, "final", "o", -0.14), (pb, "best", "D", 0.14)):
                b = (series(p, "all", v) + series(p, "all_rep2", v)) / 2
                if r == "baseline":  # the reference itself: its absolute Dice is written in the row
                    res[(r, v, ck)] = b.mean()
                    continue
                dd = (series(p, r, v) - b).dropna()
                m, lo, hi = mc.boot_ci(dd)
                res[(r, v, ck)] = m
                ax.plot([lo, hi], [i + dy, i + dy], color=color(r), linewidth=2.6,
                        alpha=1 if ck == "best" else 0.45, solid_capstyle="round")
                ax.plot(m, i + dy, mk, color=color(r), markersize=9 if mk == "o" else 8,
                        markeredgecolor=style.SURFACE, alpha=1 if ck == "best" else 0.45)
            if r != "baseline":  # one label per row in a fixed column: final → best
                ax.annotate(f"{sgn(res[(r, v, 'final')])} → {sgn(res[(r, v, 'best')])}", (0.135, i),
                            va="center", fontsize=FS_S, color=style.INK)
        ax.axvspan(-0.01, 0.01, color=style.PANEL, zorder=0)
        ax.axvline(0, color=style.INK2, linewidth=0.9)
        ax.set_title(title, fontsize=FS)
        ax.set_xlim(-0.09, 0.25)
        ax.set_xticks([-0.05, 0, 0.05, 0.10])
    axes[1].grid(axis="y", visible=False)
    for ax, v in zip(axes, ("none", "ccPost")):  # baseline row: absolute Dice, written in the row
        ax.annotate(f"Dice {res[('baseline', v, 'final')]:.3f} (final) · "
                    f"{res[('baseline', v, 'best')]:.3f} (best)", (0.02, 0), va="center", fontsize=FS_S - 1,
                    color=style.INK2)
    rows(axes[0], ["baseline"] + [NAME[a] for a in ARMS_X], [color(r) for r in runs])
    h = [plt.Line2D([], [], marker="o", color=style.INK2, alpha=0.45, linestyle="none", markersize=9),
         plt.Line2D([], [], marker="D", color=style.INK2, linestyle="none", markersize=8)]
    axes[0].legend(h, ["checkpoint_final (main)", "checkpoint_best"], ncol=2, loc="lower left",
                   bbox_to_anchor=(-0.02, 1.08))
    fig.supxlabel("tumour Dice per case, arm − baseline (bar = 95% CI)", fontsize=FS_S, color=style.INK2, y=-0.04)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, "best")
    return res


def composition(out, plt):
    """Backup: what each arm trains on."""
    data = [("all", [0, 0, 0, 999]), ("ccReject", [343, 0, 0, 0]), ("ccRepair", [343, 499, 157, 0]),
            ("ccRepairDel", [343, 499, 0, 0]), ("ccAucRepair", [300, 522, 177, 0])]
    seg = [("unchanged by the rule", "#4d4c48"), ("repaired, tumour left", "#9c9b95"),
           ("repaired, now tumour-free", "#d6d5cf"), ("as generated (no rule)", "#c6dcf5")]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, (arm, vals) in enumerate(data):
        left = 0
        for (name, c), v in zip(seg, vals):
            if v:
                ax.barh(i, v, left=left, height=0.64, color=c, edgecolor=style.SURFACE, linewidth=1.5)
                ax.text(left + v / 2, i, f"{v}", ha="center", va="center", fontsize=FS_S,
                        color="#ffffff" if c == "#4d4c48" else style.INK)
                left += v
        ax.text(left + 14, i, f"{left}", va="center", fontsize=FS_S, color=style.INK, fontweight="bold")
    rows(ax, [r[0] for r in data], [color(r[0]) for r in data])
    ax.set_xlim(0, 1090)
    ax.set_xlabel("synthetic training volumes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in seg]
    ax.legend(handles, [s for s, _ in seg], ncol=2, loc="lower left", bbox_to_anchor=(-0.3, 1.0),
              handlelength=1.0, fontsize=FS_S - 1)
    save(fig, out, "composition")


def per_case(d, base, out, plt):
    """Backup: per-case differences behind the means."""
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.axvline(0, color=style.INK2, linewidth=0.9)
    for i, a in enumerate(ARMS_X):
        x = (d[a].tumour_dice - base).to_numpy()
        y = i + rng.uniform(-0.23, 0.23, len(x))
        ax.scatter(x, y, s=16, color=color(a), alpha=0.7, linewidths=0.4, edgecolors=style.SURFACE)
        med = np.median(x)
        ax.plot([med, med], [i - 0.34, i + 0.34], color=style.INK, linewidth=2)
        w, t, l = (x > 0.01).sum(), (abs(x) <= 0.01).sum(), (x < -0.01).sum()
        ax.annotate(f"{w} better · {l} worse", (1, i), xycoords=("axes fraction", "data"), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=FS_S, color=style.INK2)
    rows(ax, [NAME[a] for a in ARMS_X], [color(a) for a in ARMS_X])
    ax.set_xlim(-0.75, 0.85)
    ax.set_xlabel("tumour Dice, arm − baseline, one dot per test case (tick = median)")
    save(fig, out, "per_case")


def curves(out, plt):
    """Backup: smoothed validation loss over training."""
    c = pd.read_csv(f"{mc.DATA}/progress/training_curves.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for arm, g in c.groupby("arm", sort=False):
        g = g.sort_values("epoch")
        s = pd.Series(mc.smooth(g["val_loss"].to_numpy()), index=g.index)
        ax.plot(g.epoch, s, color=color(arm), linewidth=1.8, alpha=0.55 if "_rep" in arm else 1)
        i = s.idxmin()
        ax.plot(g.epoch[i], s[i], "o", color=color(arm), markersize=7, markeredgecolor=style.SURFACE)
    ax.annotate("ccReject", (640, -0.265), ha="center",
                color=color("ccReject"), fontsize=FS_S, fontweight="bold")
    ax.set_title("validation loss (lower = better)", fontsize=FS)
    ax.set_ylim(-0.5, -0.2)
    ax.set_xlim(0, 1000)
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("epoch (dot = lowest)")
    save(fig, out, "curves")


# ---------------------------------------------------------------- CT crops

def review_panels(src, row):
    """The 4 panels (image only, image + label, axial, coronal) of one row of a review_figures.py sheet."""
    im = Image.open(src).convert("RGB")
    k = im.width / 1534  # sheet geometry measured at 1534 px width (as in reports/liver_results/build.py)
    y0, y1 = int((190 + 367.6 * row) * k), int((510 + 367.6 * row) * k)
    band = im.crop((0, y0, im.width, y1))
    a = np.asarray(band.convert("L")) < 200
    cols, s0 = [], None
    for i, v in enumerate(list(a.sum(0) > a.shape[0] * 0.25) + [False]):
        if v and s0 is None:
            s0 = i
        if not v and s0 is not None:
            if i - s0 >= 100:
                cols.append((s0, i))
            s0 = None
    assert len(cols) == 4, (src, cols)
    out = []
    for c0, c1 in cols:
        p = band.crop((c0, 0, c1, band.height))
        g = np.asarray(p.convert("L")) < 200  # trim white margins above/below the panel
        rr = np.where(g.mean(1) > 0.25)[0]
        out.append(p.crop((2, rr[0] + 2, p.width - 2, rr[-1] - 1)))
    return out


def ct(out):
    off = review_panels(f"{FIG}/filter_review/maisi_off_liver.png", 0)        # liver_0048, pelvis, 95 ml
    inv = review_panels(f"{FIG}/filter_review/maisi_invisible_label.png", 0)   # liver_0735, invisible, 49 ml
    for name, ps in (("off", off), ("inv", inv)):
        for j, p in enumerate(ps):
            p.save(f"{out}/ct_{name}_{j}.jpg", quality=92)
    # ccpost_review: (row, col) panels; row 0 = full slice, row 1 = zoom; col 0 CT, 1 ref, 2 as saved, 3 ccPost
    picks = {"A1": ("A1_caught_lits_115_all.png", [(0, 2), (1, 0), (1, 2), (1, 3)]),
             "A5": ("A5_caught_lits_091_all.png", [(0, 2), (1, 0), (1, 2), (1, 3)]),
             "A3": ("A3_caught_lits_043_all_rep2.png", [(0, 0), (0, 1), (0, 2), (0, 3)]),
             "B1": ("B1_harm_lits_130_all_rep2.png", [(0, 1), (1, 1), (1, 2), (1, 3)])}
    for name, (fn, ps) in picks.items():
        g = rb.grid(f"{rb.CCP}/{fn}")
        im = Image.open(f"{rb.CCP}/{fn}").convert("RGB")
        for j, (r, c) in enumerate(ps):
            x0, y0, x1, y1 = g[(r, c)]
            im.crop((x0 + 1, y0 + 1, x1 - 1, y1 - 1)).save(f"{out}/ct_{name}_{j}.jpg", quality=92)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build")
    os.makedirs(out, exist_ok=True)
    plt = setup()
    d, base = mc.load()
    for n, c, m, lo, hi in gain(d, base, out, plt):
        print(f"gain {n}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("detection:", {r: {k: round(v, 3) for k, v in x.items()} for r, x in detection(d, out, plt).items()})
    print("fp where (in, out):", {r: tuple(round(v, 1) for v in x) for r, x in fp_where(out, plt).items()})
    print("ccpost (saved, ccPost):", {r: tuple(round(v, 3) for v in x) for r, x in ccpost(out, plt).items()})
    print("aucpost (ccPost, ccAucPost):", {r: tuple(round(v, 3) for v in x) for r, x in aucpost(out, plt).items()})
    dt, dv, lost = sweep(out, plt)
    print("sweep test mean/min/max:", dt.mean(0).round(3), dt.min(0).round(3), dt.max(0).round(3))
    print("sweep val mean:", dv.mean(0).round(3), "lost mean/min/max:", lost.mean(0), lost.min(0), lost.max(0))
    print("burden:", {k: round(v, 3) for k, v in burden(d, base, out, plt).items()})
    for n, c, m, lo, hi in noise(d, out, plt):
        print(f"noise {n}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("best:", {k: round(v, 3) for k, v in best(out, plt).items()})
    composition(out, plt)
    per_case(d, base, out, plt)
    curves(out, plt)
    ct(out)
    import json  # pixel sizes, so the deck can place every image at its own aspect ratio
    json.dump({f: Image.open(f"{out}/{f}").size for f in sorted(os.listdir(out)) if f.endswith((".png", ".jpg"))},
              open(f"{out}/sizes.json", "w"), indent=0)
    print("wrote", out)


if __name__ == "__main__":
    main()
