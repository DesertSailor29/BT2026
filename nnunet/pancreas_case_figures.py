#!/usr/bin/env python
"""Pancreas baseline on the MSD07 test set: which cases succeed, which fail, and why the misses happen.

Exploratory, after the pancreas baseline test (2026-11-18). Reads the checkpoint_final test predictions (`fold_X/
test_final`, as saved) of D102 f0 `all` and f1 `all_rep2`, the references, the per-case CSVs of `predict_test.py eval`
and, for comparison, the 999 synthetic training pairs and the 43 real val cases of Dataset102 (every `--syn-step`-th
synthetic case).

1 Measure (one row per case; test cases also per run):
    spacing; reference gland (pancreas + tumour), pancreas and tumour volume (ml); median HU of the pancreas (label 1,
    eroded 1 voxel in-plane), of the tumour, and of a 3-voxel in-plane shell around the gland (surroundings);
    per run: the share of reference gland voxels predicted as background / pancreas / tumour, gland recall in the
    right, middle and left third of the gland along x (patient right = head, left = tail; approximate), the share of
    reference gland slices with any predicted gland (cranio-caudal coverage), predicted gland volume, and the share of
    predicted gland voxels within 10 mm of the reference gland.
2 Charts (reports/style): synthetic vs real distributions; where the reference gland voxels go; recall by gland third;
  per-case gland Dice against the case properties (Spearman).
3 CT figures (maisi/viz.py), cases picked by fixed rules, not by eye: the best cases (mean over the two runs of
  (pancreas Dice + tumour Dice) / 2), two median cases, and the worst cases of each failure type:
    gland_missed      gland Dice < 0.05 in both runs
    under_segmented   gland found (Dice >= 0.05), but recall < 0.35 with precision >= 0.5
    tumour_missed     gland Dice >= 0.4, tumour Dice < 0.05
    pancreas_as_tumour  >= 30 % of the reference pancreas (label 1) predicted as tumour
  Each figure: CT | reference | all | all_rep2; rows = axial full slice, axial zoom on the gland, coronal zoom.

Output: reports/figures/pancreas_cases/ (PNGs + README.md; READING.md, written by hand, is appended) and
reports/data/test_eval_pancreas/case_diagnostics/ (per-case CSVs, charts as SVG, summary.md).

  python nnunet/pancreas_case_figures.py --workers 32
"""
import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import ndimage, stats  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{REPO}/nnunet", f"{REPO}/maisi", f"{REPO}/reports/style"]
import predict_test as pt  # noqa: E402
import style  # noqa: E402
from viz import CAPTION, draw, legend_handles, plane, to_ras  # noqa: E402

pt.set_target("pancreas")
RUNS = {"all": "102:0", "all_rep2": "102:1"}
EVAL = f"{os.environ['nnUNet_results']}/test_eval_pancreas"
FIG_OUT = f"{REPO}/reports/figures/pancreas_cases"
DATA_OUT = f"{REPO}/reports/data/test_eval_pancreas/case_diagnostics"
TRAIN_IMG, TRAIN_LAB = f"{pt.RAW}/imagesTr", f"{pt.RAW}/labelsTr"
THIRDS = ["right (head)", "middle (body)", "left (tail)"]
ZOOM_MM = 50.0


def fnum(x):
    return float("nan") if x in ("", "nan", None) else float(x)


def load(img_f, lab_f):
    n = nib.load(img_f)
    img, lab, sp = to_ras(np.asarray(n.dataobj).astype(np.int16),
                          np.asarray(nib.load(lab_f).dataobj).astype(np.uint8), n.affine)
    return img, lab, sp, n.affine


def med(img, m):
    return float(np.median(img[m])) if m.any() else float("nan")


def props(img, lab, sp):
    """Case properties that do not depend on a prediction."""
    ml = float(np.prod(sp)) / 1000.0
    gland = lab > 0
    st2 = np.zeros((3, 3, 3), bool)
    st2[:, :, 1] = ndimage.generate_binary_structure(2, 1)  # in-plane (x, y) only: slices may be 2.5 mm apart
    panc = ndimage.binary_erosion(lab == 1, st2)
    if not panc.any():
        panc = lab == 1
    shell = ndimage.binary_dilation(gland, st2, iterations=3) & ~gland
    zs = np.nonzero(gland.any(axis=(0, 1)))[0]
    return {"sx": sp[0], "sy": sp[1], "sz": sp[2], "gland_ml": gland.sum() * ml, "pancreas_ml": (lab == 1).sum() * ml,
            "tumour_ml": (lab == 2).sum() * ml, "tumour_frac": (lab == 2).sum() / max(gland.sum(), 1),
            "hu_pancreas": med(img, panc), "hu_tumour": med(img, lab == 2), "hu_shell": med(img, shell),
            "gland_z_mm": (zs.max() - zs.min() + 1) * sp[2] if len(zs) else 0.0}


def run_metrics(lab, pred, sp):
    gland, pg = lab > 0, pred > 0
    n = max(gland.sum(), 1)
    r = {"to_bg": ((pred == 0) & gland).sum() / n, "to_pancreas": ((pred == 1) & gland).sum() / n,
         "to_tumour": ((pred == 2) & gland).sum() / n,
         "pancreas_as_tumour": ((pred == 2) & (lab == 1)).sum() / max((lab == 1).sum(), 1),
         "tumour_as_pancreas": ((pred == 1) & (lab == 2)).sum() / max((lab == 2).sum(), 1),
         "gland_recall": (pg & gland).sum() / n, "gland_precision": (pg & gland).sum() / max(pg.sum(), 1),
         "pred_gland_ml": pg.sum() * float(np.prod(sp)) / 1000.0}
    gx = np.nonzero(gland)[0]
    cuts = np.quantile(gx, [1 / 3, 2 / 3]) if len(gx) else [0, 0]
    xi = np.arange(lab.shape[0])[:, None, None]
    # RAS: x grows towards patient right, so the low-x third is the patient-left third (tail)
    for name, m in zip(("rec_left", "rec_middle", "rec_right"),
                       (xi < cuts[0], (xi >= cuts[0]) & (xi < cuts[1]), xi >= cuts[1])):
        g = gland & m
        r[name] = (g & pg).sum() / g.sum() if g.sum() else float("nan")
    zr = gland.any(axis=(0, 1))
    r["z_coverage"] = (zr & pg.any(axis=(0, 1))).sum() / max(zr.sum(), 1)
    if pg.any() and gland.any():
        dist = ndimage.distance_transform_edt(~gland, sampling=sp)
        r["pred_near_ref"] = float((dist[pg] <= 10.0).mean())
    else:
        r["pred_near_ref"] = float("nan")
    return r


def measure_test(case):
    img, lab, sp, _ = load(f"{pt.IMG}/{case}_0000.nii.gz", f"{pt.LAB}/{case}.nii.gz")
    row = {"case": case, "set": "real test", **props(img, lab, sp)}
    for arm, run in RUNS.items():
        did, fold = map(int, run.split(":"))
        pn = nib.load(f"{pt.res_dir(did)}/fold_{fold}/test_final/{case}.nii.gz")
        pred, _, _ = to_ras(np.asarray(pn.dataobj).astype(np.uint8), None, pn.affine)
        row.update({f"{arm}:{k}": v for k, v in run_metrics(lab, pred, sp).items()})
    return row


def measure_train(case):
    img, lab, sp, _ = load(f"{TRAIN_IMG}/{case}_0000.nii.gz", f"{TRAIN_LAB}/{case}.nii.gz")
    return {"case": case, "set": "real val" if case.startswith("msd") else "synthetic", **props(img, lab, sp)}


def write_csv(rows, f):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- selection

def classify(r, d):
    """Failure type of one test case from both runs (None = no failure type)."""
    g = [d[a]["pancreas_tumour_dice"] for a in RUNS]
    t = [d[a]["tumour_dice"] for a in RUNS]
    if max(g) < 0.05:
        return "gland_missed"
    if np.mean([r[f"{a}:pancreas_as_tumour"] for a in RUNS]) >= 0.3:
        return "pancreas_as_tumour"
    if (np.mean([r[f"{a}:gland_recall"] for a in RUNS]) < 0.35
            and np.mean([r[f"{a}:gland_precision"] for a in RUNS]) >= 0.5):
        return "under_segmented"
    if min(g) >= 0.4 and max(t) < 0.05:
        return "tumour_missed"
    return None


def select(rows, dice, n_best, n_type):
    score = {c: np.mean([(dice[c][a]["pancreas_dice"] + dice[c][a]["tumour_dice"]) / 2 for a in RUNS]) for c in rows}
    order = sorted(rows, key=lambda c: (-score[c], c))
    picks, used = [], set()

    def add(c, rule, why):
        used.add(c)
        picks.append({"case": c, "rule": rule, "why": why, "score": score[c]})

    for c in order[:n_best]:
        add(c, "success", f"rank {order.index(c) + 1}/{len(order)} by (pancreas + tumour Dice) / 2 = {score[c]:.2f}")
    mid = len(order) // 2
    for c in order[mid - 1: mid + 1]:
        add(c, "median", f"rank {order.index(c) + 1}/{len(order)} (median case), score {score[c]:.2f}")
    types = {c: rows[c]["type"] for c in rows}
    for t in ("gland_missed", "under_segmented", "pancreas_as_tumour", "tumour_missed"):
        # worst of the type first; ties (many zeros) broken by the largest reference gland, then name
        cs = sorted([c for c in rows if types[c] == t and c not in used],
                    key=lambda c: (score[c], -rows[c]["gland_ml"], c))
        for c in cs[:n_type]:
            add(c, t, f"{t.replace('_', ' ')}; score {score[c]:.2f}, gland recall "
                      f"{np.mean([rows[c][f'{a}:gland_recall'] for a in RUNS]):.2f}")
    return picks


# ---------------------------------------------------------------- CT figures

def zoom_box(m2d, sp_rc, half_mm):
    ys, xs = np.nonzero(m2d)
    if not len(ys):
        ys, xs = np.array([m2d.shape[0] // 2]), np.array([m2d.shape[1] // 2])
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    hy = max(half_mm, (ys.max() - ys.min() + 1) * sp_rc[0] / 2 + 10) / sp_rc[0]
    hx = max(half_mm, (xs.max() - xs.min() + 1) * sp_rc[1] / 2 + 10) / sp_rc[1]
    return (int(max(cy - hy, 0)), int(min(cy + hy, m2d.shape[0] - 1)) + 1,
            int(max(cx - hx, 0)), int(min(cx + hx, m2d.shape[1] - 1)) + 1)


def render(job):
    i, e, r, d, fname = job
    case = e["case"]
    img, lab, sp, _ = load(f"{pt.IMG}/{case}_0000.nii.gz", f"{pt.LAB}/{case}.nii.gz")
    preds = {}
    for arm, run in RUNS.items():
        did, fold = map(int, run.split(":"))
        pn = nib.load(f"{pt.res_dir(did)}/fold_{fold}/test_final/{case}.nii.gz")
        preds[arm] = to_ras(np.asarray(pn.dataobj).astype(np.uint8), None, pn.affine)[0]
    # axial slice: largest reference gland area, tumour counted double (so a lesion is shown)
    k = int(np.argmax((lab > 0).sum(axis=(0, 1)) + (lab == 2).sum(axis=(0, 1))))
    # coronal slice (y in RAS): through the reference gland's centre of mass
    j = int(round(ndimage.center_of_mass(lab > 0)[1]))
    labs = [None, lab] + [preds[a] for a in RUNS]
    ax_p = [plane(img, l_, sp, "axial", k) for l_ in labs]
    co_p = [plane(img, l_, sp, "coronal", j) for l_ in labs]
    a_box = zoom_box(ax_p[1][1] > 0, (sp[1], sp[0]), ZOOM_MM)
    c_box = zoom_box(co_p[1][1] > 0, (sp[2], sp[0]), ZOOM_MM)

    n = len(labs)
    fig, axs = plt.subplots(3, n, figsize=(2.5 * n, 8.4), gridspec_kw={"height_ratios": [1.15, 1, 1]})
    titles = ["CT", f"reference\ngland {r['gland_ml']:.0f} ml, tumour {r['tumour_ml']:.1f} ml"]
    for a in RUNS:
        titles.append(f"{a}\npancreas {d[a]['pancreas_dice']:.2f} · tumour {d[a]['tumour_dice']:.2f} · "
                      f"gland {d[a]['pancreas_tumour_dice']:.2f}\ngland recall {r[f'{a}:gland_recall']:.2f}, "
                      f"lesions {int(d[a]['les_det'])}/{int(d[a]['les_ref'])}")
    ya, yb, xa, xb = a_box
    za, zb, xc, xd = c_box
    for col in range(n):
        i2, l2, asp = ax_p[col]
        draw(axs[0, col], i2, l2, asp)
        axs[0, col].add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, ec="white", ls="--", lw=0.8))
        axs[0, col].set_title(titles[col], fontsize=7.5)
        draw(axs[1, col], i2[ya:yb, xa:xb], None if l2 is None else l2[ya:yb, xa:xb], asp)
        i3, l3, asp3 = co_p[col]
        draw(axs[2, col], i3[za:zb, xc:xd], None if l3 is None else l3[za:zb, xc:xd], asp3)
    axs[0, 0].set_ylabel(f"axial {k}", fontsize=8)
    axs[1, 0].set_ylabel("axial zoom", fontsize=8)
    axs[2, 0].set_ylabel(f"coronal {j} zoom", fontsize=8)
    lid = f"P{i:02d}"
    fig.suptitle(f"{lid} · {case} · {e['rule'].replace('_', ' ')} · {e['why']}\n"
                 f"spacing {r['sx']:.2f}×{r['sy']:.2f}×{r['sz']:.1f} mm · HU pancreas {r['hu_pancreas']:.0f}, "
                 f"tumour {r['hu_tumour']:.0f}, surroundings {r['hu_shell']:.0f}", fontsize=9.5)
    fig.legend(handles=legend_handles("pancreas", "tumour")
               + [Patch(fc="none", ec="black", ls="--", label="zoom region (reference gland)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=8, frameon=False)
    fig.text(0.5, 0.005, CAPTION + " Predictions as saved (checkpoint_final, no post-processing); Dice values are "
             "whole-case; gland = pancreas + tumour.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94), w_pad=0.2, h_pad=0.5)
    fig.savefig(fname, dpi=105)
    plt.close(fig)
    return i, k, j


def synth_figure(train, fname, n=4):
    """What the model trained on vs real: n synthetic training cases and n real val cases nearest their set's
    median gland volume, axial slice with the largest gland area, zoomed on the gland, without and with labels."""
    sets = []
    for name, rows in (("synthetic (train)", [r for r in train if r["set"] == "synthetic"]),
                       ("real MSD07 val", [r for r in train if r["set"] == "real val"])):
        m = np.median([r["gland_ml"] for r in rows])
        sets.append((name, sorted(rows, key=lambda r: (abs(r["gland_ml"] - m), r["case"]))[:n]))
    fig, axs = plt.subplots(4, n, figsize=(2.6 * n, 10.4))
    for si, (name, rows) in enumerate(sets):
        for ci, r in enumerate(rows):
            img, lab, sp, _ = load(f"{TRAIN_IMG}/{r['case']}_0000.nii.gz", f"{TRAIN_LAB}/{r['case']}.nii.gz")
            k = int(np.argmax((lab > 0).sum(axis=(0, 1))))
            i2, l2, asp = plane(img, lab, sp, "axial", k)
            ya, yb, xa, xb = zoom_box(l2 > 0, (sp[1], sp[0]), ZOOM_MM)
            draw(axs[2 * si, ci], i2[ya:yb, xa:xb], None, asp)
            draw(axs[2 * si + 1, ci], i2[ya:yb, xa:xb], l2[ya:yb, xa:xb], asp)
            axs[2 * si, ci].set_title(f"{name}\n{r['case']}, slices {r['sz']:.1f} mm\npancreas {r['hu_pancreas']:.0f} "
                                      f"HU, surroundings {r['hu_shell']:.0f} HU", fontsize=7)
    fig.suptitle("What the model trained on (synthetic) vs real scans, axial zoom on the gland (cases nearest the "
                 "median gland volume)", fontsize=10)
    fig.legend(handles=legend_handles("pancreas", "tumour"), loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=2, fontsize=8, frameon=False)
    fig.text(0.5, 0.003, CAPTION, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), w_pad=0.2, h_pad=0.6)
    fig.savefig(fname, dpi=105)
    plt.close(fig)


# ---------------------------------------------------------------- charts

def charts(test, train, dice):
    style.use()
    syn = [r for r in train if r["set"] == "synthetic"]
    val = [r for r in train if r["set"] == "real val"]
    sets = [("synthetic (train)", syn, style.ARM_EXTRA[1]), ("real val", val, style.MUTED),
            ("real test", test, style.ARM_COLOR["all"])]
    feats = [("gland_ml", "gland volume (ml)", True), ("tumour_ml", "tumour volume (ml)", True),
             ("tumour_frac", "tumour share of gland", False), ("hu_pancreas", "pancreas median HU", False),
             ("hu_tumour", "tumour median HU", False), ("hu_shell", "surroundings median HU (3-voxel shell)", False),
             ("pancreas_minus_shell", "pancreas − surroundings HU", False),
             ("pancreas_minus_tumour", "pancreas − tumour HU", False), ("sz", "slice spacing (mm)", False),
             ("gland_z_mm", "gland cranio-caudal extent (mm)", False)]
    fig, axs = plt.subplots(5, 2, figsize=style.size(1.0, 1.15))
    rng = np.random.default_rng(0)
    for ax, (k, name, logx) in zip(axs.flat, feats):
        for yi, (lab, rows, col) in enumerate(sets):
            v = np.array([r[k] for r in rows], float)
            v = v[np.isfinite(v)]
            if logx:
                v = np.maximum(v, 0.01)
            ax.scatter(v, yi + rng.uniform(-0.18, 0.18, len(v)), s=4, color=col, alpha=0.45, lw=0)
            if len(v):
                ax.plot([np.median(v)] * 2, [yi - 0.32, yi + 0.32], color=style.INK, lw=1.4)
        ax.set_yticks(range(len(sets)), [s[0] for s in sets])
        ax.set_title(name)
        if logx:
            ax.set_xscale("log")
    fig.suptitle("Synthetic training scans vs real MSD07 (black bar = median)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    style.save_svg(fig, f"{DATA_OUT}/synthetic_vs_real.svg")

    # where the reference gland voxels go, and recall per third
    fig, (a1, a2) = plt.subplots(1, 2, figsize=style.size(1.0, 0.36))
    arms = list(RUNS)
    for yi, a in enumerate(arms):
        left = 0.0
        for k, name, col in (("to_pancreas", "pancreas", style.ORGAN), ("to_tumour", "tumour", style.TUMOUR),
                             ("to_bg", "background", style.GRID)):
            v = float(np.mean([r[f"{a}:{k}"] for r in test]))
            a1.barh(yi, v, left=left, color=col, height=0.55)
            a1.text(left + v / 2, yi, f"{name}\n{v:.0%}", ha="center", va="center", fontsize=7, color=style.INK)
            left += v
    a1.set_yticks(range(len(arms)), arms)
    a1.set_xlim(0, 1)
    a1.set_title("Reference gland voxels predicted as (mean over cases)")
    for yi, a in enumerate(arms):
        for xi, k in enumerate(("rec_right", "rec_middle", "rec_left")):
            v = np.array([r[f"{a}:{k}"] for r in test], float)
            a2.scatter(np.nanmean(v), yi + (xi - 1) * 0.22, color=[style.ORGAN_DARK, style.INK2, style.TUMOUR_DARK][xi],
                       s=26, label=THIRDS[xi] if yi == 0 else None)
    a2.set_yticks(range(len(arms)), arms)
    a2.set_xlim(0, 1)
    a2.set_title("Gland recall by third along x (mean)")
    a2.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    style.save_svg(fig, f"{DATA_OUT}/where_gland_goes.svg")

    # gland Dice vs case properties
    gd = np.array([np.mean([dice[r["case"]][a]["pancreas_tumour_dice"] for a in RUNS]) for r in test])
    fk = [("gland_ml", "gland volume (ml)", True), ("hu_shell", "surroundings median HU", False),
          ("hu_pancreas", "pancreas median HU", False), ("pancreas_minus_shell", "pancreas − surroundings HU", False),
          ("sz", "slice spacing (mm)", False), ("gland_z_mm", "gland cranio-caudal extent (mm)", False)]
    fig, axs = plt.subplots(2, 3, figsize=style.size(1.0, 0.62))
    corr = []
    for ax, (k, name, logx) in zip(axs.flat, fk):
        v = np.array([r[k] for r in test], float)
        ok = np.isfinite(v)
        rho, p = stats.spearmanr(v[ok], gd[ok])
        corr.append((name, rho, p))
        ax.scatter(v, gd, s=5, color=style.ARM_COLOR["all"], alpha=0.5, lw=0)
        ax.set_title(f"{name}\nSpearman {rho:+.2f} (p {p:.1g})", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y")
        if logx:
            ax.set_xscale("log")
    axs[0, 0].set_ylabel("gland Dice (mean of runs)")
    axs[1, 0].set_ylabel("gland Dice (mean of runs)")
    fig.tight_layout()
    style.save_svg(fig, f"{DATA_OUT}/dice_vs_properties.svg")
    return corr


def q(v):
    v = np.array(v, float)
    v = v[np.isfinite(v)]
    return f"{np.median(v):.1f} [{np.percentile(v, 25):.1f}, {np.percentile(v, 75):.1f}]" if len(v) else "–"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--syn-step", type=int, default=5, help="measure every n-th synthetic training case")
    ap.add_argument("--n-best", type=int, default=4)
    ap.add_argument("--n-type", type=int, default=2)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    os.makedirs(FIG_OUT, exist_ok=True)
    os.makedirs(DATA_OUT, exist_ok=True)

    dice = {}
    for arm, run in RUNS.items():
        did, fold = map(int, run.split(":"))
        for r in csv.DictReader(open(f"{EVAL}/D{did}_fold_{fold}_per_case.csv")):
            dice.setdefault(r["case"], {})[arm] = {k: fnum(v) for k, v in r.items() if k not in ("case", "version")}
    cases = sorted(c for c in dice if len(dice[c]) == len(RUNS))
    syn = sorted(os.path.basename(f)[:-len("_0000.nii.gz")] for f in glob.glob(f"{TRAIN_IMG}/pancreas_*_0000.nii.gz"))
    val = sorted(os.path.basename(f)[:-len("_0000.nii.gz")] for f in glob.glob(f"{TRAIN_IMG}/msd_*_0000.nii.gz"))
    train_cases = syn[:: a.syn_step] + val
    print(f"measure: {len(cases)} test cases, {len(syn[:: a.syn_step])}/{len(syn)} synthetic, {len(val)} real val",
          flush=True)
    with Pool(a.workers) as p:
        test = p.map(measure_test, cases, chunksize=1)
        print("  test measured", flush=True)
        train = p.map(measure_train, train_cases, chunksize=1)
        print("  train measured", flush=True)
    for r in test:
        c = r["case"]
        for arm in RUNS:
            for k in ("pancreas_dice", "tumour_dice", "pancreas_tumour_dice"):
                r[f"{arm}:{k.replace('pancreas_tumour', 'gland')}"] = dice[c][arm][k]
        r["type"] = classify(r, dice[c]) or ""
    for r in test + train:
        r["pancreas_minus_shell"] = r["hu_pancreas"] - r["hu_shell"]
        r["pancreas_minus_tumour"] = r["hu_pancreas"] - r["hu_tumour"]
    write_csv(test, f"{DATA_OUT}/test_cases.csv")
    write_csv(train, f"{DATA_OUT}/train_cases.csv")
    corr = charts(test, train, dice)
    print("  charts written", flush=True)

    rows = {r["case"]: r for r in test}
    picks = select(rows, dice, a.n_best, a.n_type)
    jobs = []
    for i, e in enumerate(picks, 1):
        e["file"] = f"P{i:02d}_{e['rule']}_{e['case']}.png"
        jobs.append((i, e, rows[e["case"]], dice[e["case"]], f"{FIG_OUT}/{e['file']}"))
    with Pool(min(a.workers, len(jobs))) as p:
        sl = {i: (k, j) for i, k, j in p.map(render, jobs, chunksize=1)}
    print(f"  {len(jobs)} CT figures written", flush=True)
    synth_figure(train, f"{FIG_OUT}/P00_synthetic_vs_real.png")
    print("  synthetic vs real figure written", flush=True)

    # ---- summary.md (numbers) and README.md (figure index)
    syn_r = [r for r in train if r["set"] == "synthetic"]
    val_r = [r for r in train if r["set"] == "real val"]
    types = {}
    for r in test:
        types[r["type"] or "none"] = types.get(r["type"] or "none", 0) + 1
    S = ["# Pancreas test cases: why the baseline misses", "",
         f"Written by `nnunet/pancreas_case_figures.py`. {len(test)} MSD07 test cases, D102 `all` (f0) and "
         f"`all_rep2` (f1), checkpoint_final predictions as saved. Synthetic = every {a.syn_step}th of the "
         f"{len(syn)} training pairs ({len(syn_r)}); real val = the {len(val_r)} MSD07 val cases. Gland = pancreas + "
         "tumour. HU medians: pancreas eroded 1 voxel in-plane; surroundings = 3-voxel in-plane shell around the gland.",
         "", "## Synthetic vs real (median [IQR])", "", "| property | synthetic | real val | real test |", "|---|---|---|---|"]
    for k, name in (("gland_ml", "gland ml"), ("pancreas_ml", "pancreas ml"), ("tumour_ml", "tumour ml"),
                    ("tumour_frac", "tumour share of gland"), ("hu_pancreas", "pancreas HU"),
                    ("hu_tumour", "tumour HU"), ("hu_shell", "surroundings HU"),
                    ("pancreas_minus_shell", "pancreas − surroundings HU"),
                    ("pancreas_minus_tumour", "pancreas − tumour HU"), ("sx", "in-plane spacing mm"),
                    ("sz", "slice spacing mm"), ("gland_z_mm", "gland z extent mm")):
        S.append(f"| {name} | {q([r[k] for r in syn_r])} | {q([r[k] for r in val_r])} | {q([r[k] for r in test])} |")
    S += ["", "## Where the reference gland goes (mean over test cases)", "",
          "| run | → pancreas | → tumour | → background | gland recall | gland precision | z coverage | recall right "
          "(head) | middle | left (tail) | pred / ref gland ml (median) | pred gland within 10 mm (median) | ref "
          "pancreas → tumour | ref tumour → pancreas |", "|" + "---|" * 14]
    for arm in RUNS:
        m = lambda k: float(np.nanmean([r[f"{arm}:{k}"] for r in test]))  # noqa: E731
        ratio = np.median([r[f"{arm}:pred_gland_ml"] / max(r["gland_ml"], 1e-6) for r in test])
        near = np.nanmedian([r[f"{arm}:pred_near_ref"] for r in test])
        S.append(f"| {arm} | {m('to_pancreas'):.2f} | {m('to_tumour'):.2f} | {m('to_bg'):.2f} | {m('gland_recall'):.2f}"
                 f" | {m('gland_precision'):.2f} | {m('z_coverage'):.2f} | {m('rec_right'):.2f} | {m('rec_middle'):.2f}"
                 f" | {m('rec_left'):.2f} | {ratio:.2f} | {near:.2f} | {m('pancreas_as_tumour'):.2f} | "
                 f"{m('tumour_as_pancreas'):.2f} |")
    S += ["", "## Failure types (both runs, rules in the script header)", "",
          "| type | cases |", "|---|---|"] + [f"| {t} | {n} |" for t, n in sorted(types.items())]
    gd = {r["case"]: np.mean([r[f"{a}:gland_dice"] for a in RUNS]) for r in test}
    S += ["", "## Gland Dice (mean of runs) by slice spacing", "", "| slice spacing | cases | gland Dice mean |",
          "|---|---|---|"]
    for lab_, lo, hi in (("< 2.5 mm", 0, 2.49), ("2.5 mm", 2.49, 2.51), ("> 2.5 to < 5 mm", 2.51, 4.99),
                         (">= 5 mm", 4.99, 99)):
        v = [gd[r["case"]] for r in test if lo <= r["sz"] < hi]
        S.append(f"| {lab_} | {len(v)} | {np.mean(v):.3f} |" if v else f"| {lab_} | 0 | – |")
    S += ["", "## Gland Dice (mean of runs) by surroundings median HU (fat around the gland = negative)", "",
          "| surroundings HU | cases | gland Dice mean | of these 2.5 mm slices | gland Dice mean (2.5 mm) |",
          "|---|---|---|---|---|"]
    for lo, hi in ((-999, 0), (0, 40), (40, 999)):
        sel = [r for r in test if lo <= r["hu_shell"] < hi]
        s25 = [r for r in sel if 2.49 <= r["sz"] < 2.51]
        S.append(f"| {lo}–{hi} | {len(sel)} | {np.mean([gd[r['case']] for r in sel]):.3f} | {len(s25)} | "
                 f"{np.mean([gd[r['case']] for r in s25]):.3f} |")
    syn_shell = np.array([r["hu_shell"] for r in syn_r])
    S.append(f"\nSynthetic cases with surroundings < 0 HU: {(syn_shell < 0).mean():.0%} "
             f"(real test {np.mean([r['hu_shell'] < 0 for r in test]):.0%}).")
    S += ["", "## Gland Dice (mean of runs) by pancreas median HU", "", "| pancreas HU | cases | gland Dice mean |",
          "|---|---|---|"]
    for lo, hi in ((-999, 40), (40, 70), (70, 90), (90, 110), (110, 130), (130, 999)):
        v = [gd[r["case"]] for r in test if lo <= r["hu_pancreas"] < hi]
        S.append(f"| {lo}–{hi} | {len(v)} | {np.mean(v):.3f} |" if v else f"| {lo}–{hi} | 0 | – |")
    S += ["", "## Gland Dice (mean of runs) vs case properties, Spearman", "", "| property | rho | p |", "|---|---|---|"]
    S += [f"| {n} | {rho:+.2f} | {p:.2g} |" for n, rho, p in corr]
    S += ["", "Charts: [synthetic_vs_real.svg](synthetic_vs_real.svg), [where_gland_goes.svg](where_gland_goes.svg), "
          "[dice_vs_properties.svg](dice_vs_properties.svg). CT figures: `reports/figures/pancreas_cases/`."]
    open(f"{DATA_OUT}/summary.md", "w").write("\n".join(S) + "\n")

    L = ["# Pancreas baseline: test cases that succeed and fail", "",
         "Written by `nnunet/pancreas_case_figures.py` from the checkpoint_final MSD07 test predictions as saved of "
         "D102 `all` (f0) and `all_rep2` (f1). Cases picked by fixed rules, not by eye: the best by (pancreas + "
         "tumour Dice) / 2 (mean of runs), two median cases, then the worst cases of each failure type (gland "
         "missed, under-segmented, pancreas labelled as tumour, tumour missed). Rows: axial slice with the largest "
         "reference gland area, zoom on the gland, coronal zoom through the gland centre. Numbers behind the "
         "diagnosis: `reports/data/test_eval_pancreas/case_diagnostics/summary.md`. Style: " + CAPTION, "",
         "[P00](P00_synthetic_vs_real.png): synthetic training scans next to real val scans (no predictions).", "",
         "| fig | case | type | axial | coronal | why | pancreas Dice all / rep2 | tumour Dice all / rep2 |",
         "|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(picks, 1):
        d = dice[e["case"]]
        L.append(f"| [P{i:02d}]({e['file']}) | {e['case']} | {e['rule']} | {sl[i][0]} | {sl[i][1]} | {e['why']} | "
                 f"{d['all']['pancreas_dice']:.2f} / {d['all_rep2']['pancreas_dice']:.2f} | "
                 f"{d['all']['tumour_dice']:.2f} / {d['all_rep2']['tumour_dice']:.2f} |")
    rd = f"{FIG_OUT}/READING.md"
    if os.path.exists(rd):
        L += ["", open(rd).read()]
    open(f"{FIG_OUT}/README.md", "w").write("\n".join(L) + "\n")
    print(open(f"{DATA_OUT}/summary.md").read(), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
