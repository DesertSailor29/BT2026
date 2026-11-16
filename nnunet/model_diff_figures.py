#!/usr/bin/env python
"""Where the liver runs differ: one figure per test case, CT | reference | every run's prediction side by side.

Exploratory, for the report / supervisor meeting. Reads the checkpoint_final test predictions (`fold_X/test_final`,
as saved, no post-processing) of the 7 liver runs and `reports/data/test_eval_liver/fp_location/per_case.csv`
(variant `none`) for the per-case numbers.

Cases are picked by fixed rules (deterministic, not by eye), one new case per rule where possible:
  spread   the cases with the largest tumour Dice range (max − min) over the 7 runs (`--n-spread`)
  contrast for each pair below, the case with the largest difference in each direction:
           ccAucRepair − baseline (mean of all, all_rep2), ccReject − ccRepair, ccRepair − ccRepairDel,
           ccAucRepair − ccRepair, ccAucRepair_rep2 − ccAucRepair (run-to-run noise), all_rep2 − all (noise)
  fp       the tumour-free case with the largest range of false-positive ml over the runs
Slice = the axial slice where the runs disagree most: voxels where some but not all runs predict tumour, plus
reference tumour voxels missed by at least one run, plus 2 x reference tumour voxels (so a real lesion is shown).
Zoom = the largest connected blob of disagreement + reference tumour on that slice.

Each figure: top row full slice, bottom row zoomed on the disagreement (dashed box in the top row); panel titles give
whole-case tumour Dice, FP pieces and lesions found / reference. Shared CT style (maisi/viz.py).

Output: reports/figures/model_diff/ (PNGs + README.md).

  python nnunet/model_diff_figures.py --workers 8
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{REPO}/nnunet", f"{REPO}/maisi"]
import predict_test as pt  # noqa: E402
from orient import reorient_array, spacing_of  # noqa: E402
from viz import CAPTION, draw, legend_handles, plane  # noqa: E402

RUNS = ["101:0", "101:1", "101:2", "103:0", "103:1", "104:0", "104:1"]
PER_CASE = f"{REPO}/reports/data/test_eval_liver/fp_location/per_case.csv"
OUT = f"{REPO}/reports/figures/model_diff"
MIN_HALF_MM = 40.0


def short(run):  # "D104 f0 ccAucRepair" -> "ccAucRepair"
    return pt.arm_name(*map(int, run.split(":"))).split()[-1]


def read_rows():
    rows = defaultdict(dict)  # case -> run -> row
    for r in csv.DictReader(open(PER_CASE)):
        if r["variant"] == "none" and r["run"] in RUNS:
            rows[r["case"]][r["run"]] = r
    return {c: v for c, v in rows.items() if len(v) == len(RUNS)}


def fnum(x):
    return float("nan") if x in ("", "nan") else float(x)


def select(rows, n_spread):
    tumour = [c for c in sorted(rows) if int(rows[c][RUNS[0]]["les_ref"]) > 0]
    d = {c: {run: fnum(rows[c][run]["tumour_dice"]) for run in RUNS} for c in tumour}
    used, picks = set(), []

    def add(case, rule, why):
        used.add(case)
        picks.append({"case": case, "rule": rule, "why": why})

    spread = sorted(tumour, key=lambda c: -(max(d[c].values()) - min(d[c].values())))
    for c in spread[:n_spread]:
        lo, hi = min(d[c], key=d[c].get), max(d[c], key=d[c].get)
        add(c, "spread", f"tumour Dice range {d[c][hi] - d[c][lo]:.2f} ({short(lo)} {d[c][lo]:.2f} → "
                         f"{short(hi)} {d[c][hi]:.2f})")

    base = {c: (d[c]["101:0"] + d[c]["101:1"]) / 2 for c in tumour}
    pairs = [("ccAucRepair", lambda c: d[c]["104:0"], "baseline", lambda c: base[c]),
             ("ccReject", lambda c: d[c]["101:2"], "ccRepair", lambda c: d[c]["103:0"]),
             ("ccRepair", lambda c: d[c]["103:0"], "ccRepairDel", lambda c: d[c]["103:1"]),
             ("ccAucRepair", lambda c: d[c]["104:0"], "ccRepair", lambda c: d[c]["103:0"]),
             ("ccAucRepair_rep2", lambda c: d[c]["104:1"], "ccAucRepair", lambda c: d[c]["104:0"]),
             ("all_rep2", lambda c: d[c]["101:1"], "all", lambda c: d[c]["101:0"])]
    for an, fa, bn, fb in pairs:
        diff = {c: fa(c) - fb(c) for c in tumour if np.isfinite(fa(c) - fb(c))}
        for sign, word in ((1, "better"), (-1, "worse")):
            for c in sorted(diff, key=lambda c: -sign * diff[c]):
                if c not in used:
                    add(c, "contrast", f"{an} − {bn} = {diff[c]:+.2f} (largest where {an} is {word}, "
                                       f"case not shown yet)")
                    break

    free = [c for c in sorted(rows) if int(rows[c][RUNS[0]]["les_ref"]) == 0]
    fp = {c: [fnum(rows[c][run]["fp_ml"]) for run in RUNS] for c in free}
    if fp:
        c = max(fp, key=lambda c: max(fp[c]) - min(fp[c]))
        add(c, "fp", f"tumour-free case, FP ml range {min(fp[c]):.1f}–{max(fp[c]):.1f} over the runs")
    return picks


def title_txt(row):
    dc = fnum(row["tumour_dice"])
    dc = "–" if not np.isfinite(dc) else f"{dc:.2f}"
    return f"Dice {dc} · FP {row['les_fp']}\nlesions {row['les_det']}/{row['les_ref']}"


def render(job):
    i, e, rws, fname = job
    case = e["case"]
    img_n = nib.load(f"{pt.IMG}/{case}_0000.nii.gz")
    aff = img_n.affine
    ras = lambda a: reorient_array(a, aff)[0]  # noqa: E731
    img = ras(np.asarray(img_n.dataobj).astype(np.int16))
    sp = spacing_of(reorient_array(np.zeros((1, 1, 1)), aff)[1])
    ref = ras(np.asarray(nib.load(f"{pt.LAB}/{case}.nii.gz").dataobj).astype(np.uint8))
    preds = {}
    for run in RUNS:
        did, fold = map(int, run.split(":"))
        preds[run] = ras(np.asarray(nib.load(f"{pt.res_dir(did)}/fold_{fold}/test_final/{case}.nii.gz").dataobj)
                         .astype(np.uint8))
    cnt = sum((p == 2).astype(np.uint8) for p in preds.values())
    dis = ((cnt > 0) & (cnt < len(RUNS))) | ((ref == 2) & (cnt < len(RUNS)))
    # reference tumour counts double, so the slice shows a real lesion when the runs differ on one
    k = int(np.argmax(dis.sum(axis=(0, 1)) + 2 * (ref == 2).sum(axis=(0, 1))))
    if not dis[:, :, k].any():  # no disagreement at all: slice with most reference / predicted tumour
        k = int(np.argmax(((ref == 2) | (cnt > 0)).sum(axis=(0, 1))))

    labs = [None, ref] + [preds[r] for r in RUNS]
    panels = [plane(img, lab, sp, "axial", k) for lab in labs]
    asp = panels[0][2]
    _, d2, _ = plane(img, (dis | (ref == 2)).astype(np.uint8), sp, "axial", k)
    if not d2.any():
        _, d2, _ = plane(img, (ref > 0).astype(np.uint8), sp, "axial", k)
    # zoom box: around the largest disagreement blob on the slice (closed by 3 px so speckle merges), >= ±40 mm
    lab2, n2 = ndimage.label(ndimage.binary_closing(d2 > 0, iterations=3) | (d2 > 0))
    if n2:
        big = 1 + int(np.argmax(ndimage.sum(np.ones_like(lab2), lab2, range(1, n2 + 1))))
        ys, xs = np.nonzero(lab2 == big)
    else:
        ys, xs = np.array([d2.shape[0] // 2]), np.array([d2.shape[1] // 2])
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    half = max(MIN_HALF_MM, (y1 - y0 + 1) * sp[1] / 2 + 10, (x1 - x0 + 1) * sp[0] / 2 + 10)
    hy, hx = half / sp[1], half / sp[0]
    ya, yb = int(max(cy - hy, 0)), int(min(cy + hy, d2.shape[0] - 1)) + 1
    xa, xb = int(max(cx - hx, 0)), int(min(cx + hx, d2.shape[1] - 1)) + 1

    n = len(labs)
    fig, axs = plt.subplots(2, n, figsize=(2.15 * n, 5.6))
    titles = ["CT", "reference"] + [f"{short(r)}\n{title_txt(rws[r])}" for r in RUNS]
    for j, (i2, l2, _) in enumerate(panels):
        draw(axs[0, j], i2, l2, asp)
        axs[0, j].add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, ec="white", ls="--", lw=0.8))
        axs[0, j].set_title(titles[j], fontsize=7.5)
        draw(axs[1, j], i2[ya:yb, xa:xb], None if l2 is None else l2[ya:yb, xa:xb], asp)
    axs[1, 0].set_ylabel(f"zoom ±{half:.0f} mm", fontsize=8)
    lid = f"M{i:02d}"
    fig.suptitle(f"{lid} · {case} · axial slice {k} (RAS) · {e['why']}", fontsize=10)
    fig.legend(handles=legend_handles("liver", "tumour")
               + [Patch(fc="none", ec="black", ls="--", label="zoom region (largest disagreement blob)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=3, fontsize=8, frameon=False)
    fig.text(0.5, 0.008, CAPTION + " Predictions as saved (checkpoint_final, no post-processing); Dice / FP pieces / "
             "lesions are whole-case numbers.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), w_pad=0.2, h_pad=0.4)
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    return i, k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-spread", type=int, default=6)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    rows = read_rows()
    print(f"{len(rows)} cases with all {len(RUNS)} runs in {PER_CASE}", flush=True)
    picks = select(rows, a.n_spread)
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for i, e in enumerate(picks, 1):
        e["file"] = f"M{i:02d}_{e['rule']}_{e['case']}.png"
        jobs.append((i, e, rows[e["case"]], f"{OUT}/{e['file']}"))
    print(f"draw: {len(jobs)} figures", flush=True)
    with Pool(min(a.workers, len(jobs))) as pool:
        slices = dict(pool.map(render, jobs, chunksize=1))
    L = ["# Where the liver runs differ: CT | reference | every run", "",
         "Written by `nnunet/model_diff_figures.py` from the checkpoint_final test predictions as saved (no "
         "post-processing) of the 7 liver runs and `reports/data/test_eval_liver/fp_location/per_case.csv`. Cases "
         "picked by fixed rules (largest tumour Dice range over the runs; largest difference in each direction for "
         "fixed arm pairs incl. the two noise pairs; the tumour-free case with the largest FP range), not by eye. "
         "Slice = where the runs disagree most (some but not all runs predict tumour, or a reference lesion voxel "
         "missed by at least one run). Top row full slice, bottom row zoomed on the disagreement. Style: " + CAPTION,
         "", "| fig | case | slice | rule | why | " + " | ".join(short(r) for r in RUNS) + " |",
         "|---|---|---|---|---|" + "---|" * len(RUNS)]
    for i, e in enumerate(picks, 1):
        ds = [fnum(rows[e["case"]][r]["tumour_dice"]) for r in RUNS]
        L.append(f"| [M{i:02d}]({e['file']}) | {e['case']} | {slices[i]} | {e['rule']} | {e['why']} | "
                 + " | ".join("–" if not np.isfinite(x) else f"{x:.2f}" for x in ds) + " |")
        print(f"  {e['file']}: {e['why']}", flush=True)
    rd = f"{OUT}/READING.md"
    if os.path.exists(rd):
        L += ["", open(rd).read()]
    open(f"{OUT}/README.md", "w").write("\n".join(L) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
