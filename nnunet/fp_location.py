#!/usr/bin/env python
"""Where do the false-positive tumour pieces sit, and what would the cc rule do if applied to the predictions?

For every run (dataset:fold) and every real LiTS test case, on the saved checkpoint_final predictions
(fold_X/test_final, as predict_test.py writes them):

  * every predicted tumour piece (26-connected): size, hit (overlaps a reference lesion) or false positive,
    inside / outside the *predicted* main liver structure, share of its voxels inside the *reference* liver
    (label > 0 in the reference, so a piece on a real lesion or in the real liver counts as inside)
  * the same per-case metrics as predict_test.py (Dice liver / liver+tumour / tumour, lesion-wise) for
      none    the prediction as saved
      ccPost  the cc rule of the training labels (maisi/cc_rules.py main_structure) applied to the prediction:
              main liver = largest 26-connected predicted liver piece; keep the 26-connected piece of
              predicted liver ∪ tumour that holds it; every other liver and tumour voxel -> background.
              A prediction without liver loses all its tumour.

Writes to $nnUNet_results/test_eval_liver/fp_location/ (cache/<run>/<case>.json, components.csv, per_case.csv,
summary.md) and copies summary.md + the CSVs to reports/data/test_eval_liver/fp_location/ (not committed).
Descriptive and exploratory: the main analysis stays predict_test.py on the saved predictions (decided before results).

  python nnunet/fp_location.py                       # the 6 liver runs, 8 workers
  python nnunet/fp_location.py --runs 101:0 104:0 --workers 4
  python nnunet/fp_location.py --tables-only         # summary from the cache
  python nnunet/fp_location.py --checkpoint best     # checkpoint_best predictions (fold_X/test_best)
                                                     # -> $nnUNet_results/test_eval_liver_best/fp_location/
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "maisi"))
import predict_test as pt  # noqa: E402  (paths, Dice, lesion metrics, paired stats: same definitions as the eval)
from cc_rules import main_structure  # noqa: E402  (the training-label cc rule, unchanged)

VERSION = "fploc1"
RUNS = ["101:0", "101:1", "101:2", "103:0", "103:1", "104:0"]
BASELINE = ["101:0", "101:1"]
VARIANTS = ("none", "ccPost")
SIZE_BINS = [(0, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, float("inf"))]
OUT = f"{os.environ['nnUNet_results']}/test_eval_liver/fp_location"
PUB = f"{REPO}/reports/data/test_eval_liver/fp_location"
CKPT = "final"  # set by --checkpoint


def use_checkpoint(ckpt):
    """Point OUT/PUB at test_eval_liver_best/... for checkpoint_best (as predict_test.py eval does)."""
    global OUT, PUB, CKPT
    CKPT = ckpt
    if ckpt == "best":
        OUT = f"{os.environ['nnUNet_results']}/test_eval_liver_best/fp_location"
        PUB = f"{REPO}/reports/data/test_eval_liver_best/fp_location"


def metrics(r, p, voxel_ml):
    row = {}
    for name, rm, pm in (("liver", r == 1, p == 1), ("liver_tumour", r > 0, p > 0), ("tumour", r == 2, p == 2)):
        d, tp, a, b = pt.dice(rm, pm)
        row.update({f"{name}_dice": d, f"{name}_tp": tp, f"{name}_ref": a, f"{name}_pred": b})
    row.update(pt.lesions(r == 2, p == 2, voxel_ml))
    return row


def measure(args):
    run, case, ref_f, pred_f, cache_f = args
    ref = nib.load(ref_f)
    r = np.asarray(ref.dataobj).astype(np.uint8)
    p = np.asarray(nib.load(pred_f).dataobj).astype(np.uint8)
    assert r.shape == p.shape, (ref_f, r.shape, p.shape)
    voxel_ml = float(np.prod(np.sqrt((ref.affine[:3, :3] ** 2).sum(axis=0)))) / 1000.0

    keep = main_structure(p)
    comp, n = ndimage.label(p == 2, pt.STRUCT26)
    comps = []
    if n:
        idx = range(1, n + 1)
        vox = np.bincount(comp.ravel(), minlength=n + 1)[1:]
        hit = ndimage.maximum(r == 2, comp, idx)
        in_main = ndimage.maximum(keep, comp, idx)
        in_ref = ndimage.mean(r > 0, comp, idx)
        for i in range(n):
            comps.append({"run": run, "case": case, "component": i + 1, "voxels": int(vox[i]),
                          "ml": round(float(vox[i]) * voxel_ml, 4), "hit": bool(hit[i]),
                          "in_pred_main": bool(in_main[i]), "ref_organ_share": round(float(in_ref[i]), 4)})
    post = np.where(keep, p, 0).astype(np.uint8)
    rows = {"none": metrics(r, p, voxel_ml), "ccPost": metrics(r, post, voxel_ml)}
    extra = {"pred_liver": bool((p == 1).any()), "liver_removed_ml": round(float(((p == 1) & ~keep).sum()) * voxel_ml, 3),
             "ref_tumour_free": not bool((r == 2).any())}
    out = {"version": VERSION, "run": run, "case": case, "voxel_ml": voxel_ml, "components": comps,
           "metrics": rows, **extra}
    tmp = cache_f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, cache_f)
    return run, case


def load_cache(runs, cases):
    data = {}
    for run in runs:
        for c in cases:
            f = f"{OUT}/cache/{run.replace(':', '_')}/{c}.json"
            d = json.load(open(f))
            assert d["version"] == VERSION, f
            data[(run, c)] = d
    return data


def pct(a, b):
    return f"{100 * a / b:.0f} %" if b else "–"


def tables(runs, cases, names):
    data = load_cache(runs, cases)
    comps = [x for k in data.values() for x in k["components"]]
    with open(f"{OUT}/components.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comps[0]) + ["arm"])
        w.writeheader()
        for x in comps:
            w.writerow({**x, "arm": names[x["run"]]})
    with open(f"{OUT}/per_case.csv", "w", newline="") as fh:
        keys = list(next(iter(data.values()))["metrics"]["none"])
        w = csv.DictWriter(fh, fieldnames=["run", "arm", "case", "variant", "pred_liver", "liver_removed_ml"] + keys)
        w.writeheader()
        for (run, c), d in data.items():
            for v in VARIANTS:
                w.writerow({"run": run, "arm": names[run], "case": c, "variant": v, "pred_liver": d["pred_liver"],
                            "liver_removed_ml": d["liver_removed_ml"], **d["metrics"][v]})

    nc = len(cases)
    L = ["# False-positive tumour pieces: where they sit, and the cc rule applied to the predictions", "",
         f"{nc} real LiTS test cases, checkpoint_{CKPT} predictions. Piece = 26-connected predicted tumour component; "
         "FP = overlaps no reference lesion. *Predicted main structure* = the piece of predicted liver ∪ tumour that "
         "holds the largest predicted liver piece (the training cc rule). *Reference liver share* = fraction of the "
         "piece's voxels with reference label > 0. ccPost = everything outside the predicted main structure set to "
         "background. Written by `nnunet/fp_location.py` (exploratory; main analysis = predict_test.py).", "",
         "## 1. Where the false positives sit", "",
         "| run | FP pieces / case | outside pred. main | FP ml / case | ml outside pred. main | FP outside ref. liver (share < 0.5) | "
         "hit pieces outside pred. main | cases without predicted liver |", "|---|---|---|---|---|---|---|---|"]
    for run in runs:
        cs = [x for x in comps if x["run"] == run]
        fp = [x for x in cs if not x["hit"]]
        out_ = [x for x in fp if not x["in_pred_main"]]
        ml = sum(x["ml"] for x in fp)
        ml_out = sum(x["ml"] for x in out_)
        out_ref = sum(x["ref_organ_share"] < 0.5 for x in fp)
        hit_out = sum(1 for x in cs if x["hit"] and not x["in_pred_main"])
        noliv = sum(not data[(run, c)]["pred_liver"] for c in cases)
        L.append(f"| {names[run]} | {len(fp) / nc:.1f} | {len(out_) / nc:.1f} ({pct(len(out_), len(fp))}) | "
                 f"{ml / nc:.2f} | {ml_out / nc:.2f} ({pct(ml_out, ml)}) | {pct(out_ref, len(fp))} | {hit_out} | {noliv} |")

    L += ["", "## 2. Size of the false-positive pieces (count, share of the run's FP pieces)", "",
          "| run | where | " + " | ".join(f"{a:g}–{b:g} ml" if b != float("inf") else f"≥ {a:g} ml" for a, b in SIZE_BINS)
          + " | median ml |", "|---|---|" + "---|" * (len(SIZE_BINS) + 1)]
    for run in runs:
        fp = [x for x in comps if x["run"] == run and not x["hit"]]
        for where, sub in (("inside pred. main", [x for x in fp if x["in_pred_main"]]),
                           ("outside pred. main", [x for x in fp if not x["in_pred_main"]])):
            cells = []
            for a, b in SIZE_BINS:
                k = sum(a <= x["ml"] < b for x in sub)
                cells.append(f"{k} ({pct(k, len(fp))})")
            med = np.median([x["ml"] for x in sub]) if sub else float("nan")
            L.append(f"| {names[run]} | {where} | " + " | ".join(cells) + f" | {med:.3f} |")

    def col(run, variant, key):
        return np.array([data[(run, c)]["metrics"][variant][key] for c in cases], float)

    def tot(run, variant, key):
        return col(run, variant, key).sum()

    L += ["", "## 3. ccPost vs the saved prediction, per run", "",
          "Means as predict_test.py (NaN = both empty, left out); diff/CI/p over the cases scored in both variants. "
          "A tumour-free case that becomes clean turns from Dice 0 into NaN and drops out (counted in the last column).", "",
          "| run | tumour Dice none → ccPost | diff [95% CI] | Wilcoxon p | liver Dice none → ccPost | lesion recall | "
          "FP pieces / case | FP ml / case | ref. lesions lost | tumour-free cases clean |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    free = [c for c in cases if data[(runs[0], c)]["ref_tumour_free"]]
    for run in runs:
        t = pt.paired(col(run, "none", "tumour_dice"), col(run, "ccPost", "tumour_dice"))
        lv = pt.paired(col(run, "none", "liver_dice"), col(run, "ccPost", "liver_dice"))
        rec = [tot(run, v, "les_det") / tot(run, v, "les_ref") for v in VARIANTS]
        fpc = [tot(run, v, "les_fp") / nc for v in VARIANTS]
        fpm = [tot(run, v, "fp_ml") / nc for v in VARIANTS]
        lost = int(tot(run, "none", "les_det") - tot(run, "ccPost", "les_det"))
        clean = sum(data[(run, c)]["metrics"]["ccPost"]["les_pred"] == 0 for c in free)
        tm = [np.nanmean(col(run, v, "tumour_dice")) for v in VARIANTS]
        lm = [np.nanmean(col(run, v, "liver_dice")) for v in VARIANTS]
        L.append(f"| {names[run]} | {tm[0]:.3f} → {tm[1]:.3f} | {pt.fmt_diff(t, 3)} | {t['p']:.2g} | "
                 f"{lm[0]:.3f} → {lm[1]:.3f} | {rec[0]:.3f} → {rec[1]:.3f} | {fpc[0]:.1f} → {fpc[1]:.1f} | "
                 f"{fpm[0]:.2f} → {fpm[1]:.2f} | {lost} | {clean}/{len(free)} |")

    base = [r for r in BASELINE if r in runs]
    arms = [r for r in runs if r not in base]
    if base and arms:
        L += ["", "## 4. Arms vs baseline when both are post-processed (exploratory)", "",
              f"Baseline = per-case mean of {' + '.join(names[b] for b in base)} (NaN only if all are NaN). "
              "Holm over the arms listed.", "",
              "| arm | variant | arm tumour Dice | baseline | diff [95% CI] | Wilcoxon p | Holm p |", "|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            b = np.array([col(x, v, "tumour_dice") for x in base])
            nb = (~np.isnan(b)).sum(axis=0)
            bv = np.where(nb > 0, np.nansum(b, axis=0) / np.maximum(nb, 1), np.nan)
            res = [pt.paired(bv, col(a, v, "tumour_dice")) for a in arms]
            hp = pt.holm(np.array([r["p"] for r in res]))
            for a, r_, h in zip(arms, res, hp):
                L.append(f"| {names[a]} | {v} | {r_['y']:.3f} | {r_['x']:.3f} | {pt.fmt_diff(r_, 3)} | {r_['p']:.2g} | {h:.2g} |")
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    os.makedirs(PUB, exist_ok=True)
    for f in ("summary.md", "per_case.csv", "components.csv"):
        shutil.copy(f"{OUT}/{f}", f"{PUB}/{f}")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=RUNS, help="dataset:fold (default: the 6 liver runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint", choices=("final", "best"), default="final",
                    help="which predictions: fold_X/test_final (default) or fold_X/test_best")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    use_checkpoint(a.checkpoint)
    cases = [f"lits_{i:03d}" for i in pt.split()["test"]]
    names = {r: f"D{r.split(':')[0]} f{r.split(':')[1]} {pt.arm_name(*map(int, r.split(':')))}" for r in a.runs}
    if not a.tables_only:
        jobs = []
        for run in a.runs:
            did, fold = map(int, run.split(":"))
            pdir = f"{pt.res_dir(did)}/fold_{fold}/test_{a.checkpoint}"
            cdir = f"{OUT}/cache/{run.replace(':', '_')}"
            os.makedirs(cdir, exist_ok=True)
            for c in cases:
                pred, cache = f"{pdir}/{c}.nii.gz", f"{cdir}/{c}.json"
                if not os.path.exists(pred):
                    sys.exit(f"missing prediction {pred}")
                if os.path.exists(cache) and json.load(open(cache)).get("version") == VERSION:
                    continue
                jobs.append((run, c, f"{pt.LAB}/{c}.nii.gz", pred, cache))
        print(f"{len(jobs)} of {len(a.runs) * len(cases)} (run, case) pairs to measure, {a.workers} workers", flush=True)
        with Pool(a.workers) as pool:
            for k, (run, c) in enumerate(pool.imap_unordered(measure, jobs, chunksize=1), 1):
                if k % 25 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} done (last {run} {c})", flush=True)
    tables(a.runs, cases, names)


if __name__ == "__main__":
    main()
