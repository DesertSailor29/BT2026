#!/usr/bin/env python
"""Predict the real LiTS test cases with trained folds of Dataset101 and score them.

Test cases = real_split.json "test" (111 LiTS cases: thesis train split minus the 20 val cases + the 40 thesis test
cases), read from the RAS copy (raw_data/LiTS_RAS). Each fold = one experiment arm (make_splits.py).

  python predict_test.py prepare                      # copy test images/labels into the dataset (int16/uint8, RAS)
  python predict_test.py predict --fold 0             # nnUNetv2_predict with checkpoint_final -> fold_0/test_final
  python predict_test.py eval --folds 0 1             # per-case Dice CSV per fold + test_eval/summary.md

Everything is resumable: prepare skips existing files, predict uses --continue_prediction, eval recomputes (cheap).

Metrics per case: liver (label 1 only, as nnU-Net reports it), liver+tumour (label > 0, LiTS convention), tumour
(label 2). Dice is NaN when reference and prediction are both empty (excluded from means; counted). Global tumour Dice
= 2*sum(TP) / (sum(|ref|) + sum(|pred|)) over all cases (LiTS "Dice global"). Comparison of two folds: paired
per-case differences + Wilcoxon signed-rank test.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import copy_real  # noqa: E402  (same int16/uint8 + qform/sform conversion as the val cases)

PROJ = os.environ.get("PROJ", "/work/Research")
DS = "Dataset101_LiverSynth"
RAW = f"{os.environ['nnUNet_raw']}/{DS}"
RES = f"{os.environ['nnUNet_results']}/{DS}/nnUNetTrainer__nnUNetPlans__3d_fullres"
IMG, LAB = f"{RAW}/imagesTs_real", f"{RAW}/labelsTs_real"
CHK = "checkpoint_final.pth"


def split():
    return json.load(open(f"{RAW}/real_split.json"))


def prepare(workers):
    s = split()
    src = s["source"]
    thesis_test = set(s["thesis_test"])
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(LAB, exist_ok=True)
    jobs, missing = [], []
    for i in s["test"]:
        sub = "Ts" if i in thesis_test else "Tr"
        si = glob.glob(f"{src}/images{sub}/volume-{i}_0000.nii*")
        sl = glob.glob(f"{src}/labels{sub}/volume-{i}.nii*")
        if len(si) == 1 and len(sl) == 1:
            jobs.append((si[0], f"{IMG}/lits_{i:03d}_0000.nii.gz", sl[0], f"{LAB}/lits_{i:03d}.nii.gz"))
        else:
            missing.append(i)
    if missing:
        raise SystemExit(f"missing in {src}: {missing}")
    with Pool(workers) as p:
        list(p.imap_unordered(copy_real, jobs))
    n = len(glob.glob(f"{IMG}/*_0000.nii.gz"))
    print(f"prepare: {n} test images in {IMG}", flush=True)
    assert n == len(s["test"]), n


def predict(fold, npp, nps):
    prepare(16)
    out = f"{RES}/fold_{fold}/test_final"
    cmd = ["nnUNetv2_predict", "-i", IMG, "-o", out, "-d", "101", "-c", "3d_fullres", "-f", str(fold),
           "-chk", CHK, "-npp", str(npp), "-nps", str(nps), "--continue_prediction"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    n = len(glob.glob(f"{out}/lits_*.nii.gz"))
    print(f"fold {fold}: {n} predictions in {out}", flush=True)


def dice(r, p):
    a, b = int(r.sum()), int(p.sum())
    tp = int(np.logical_and(r, p).sum())
    return (2 * tp / (a + b) if a + b else float("nan")), tp, a, b


def score(args):
    ref_f, pred_f = args
    r = np.asarray(nib.load(ref_f).dataobj)
    p = np.asarray(nib.load(pred_f).dataobj)
    assert r.shape == p.shape, (ref_f, r.shape, p.shape)
    row = {"case": os.path.basename(ref_f)[: -len(".nii.gz")]}
    for name, rm, pm in (("liver", r == 1, p == 1), ("liver_tumour", r > 0, p > 0), ("tumour", r == 2, p == 2)):
        d, tp, a, b = dice(rm, pm)
        row.update({f"{name}_dice": d, f"{name}_tp": tp, f"{name}_ref": a, f"{name}_pred": b})
    return row


def eval_folds(folds, workers):
    import csv
    from scipy.stats import wilcoxon

    s = split()
    thesis_test = {f"lits_{i:03d}" for i in s["thesis_test"]}
    cases = [f"lits_{i:03d}" for i in s["test"]]
    out_dir = f"{RES}/test_eval"
    os.makedirs(out_dir, exist_ok=True)
    rows = {}
    for f in folds:
        pdir = f"{RES}/fold_{f}/test_final"
        missing = [c for c in cases if not os.path.exists(f"{pdir}/{c}.nii.gz")]
        if missing:
            raise SystemExit(f"fold {f}: {len(missing)} predictions missing, e.g. {missing[:5]}")
        with Pool(workers) as p:
            rs = sorted(p.map(score, [(f"{LAB}/{c}.nii.gz", f"{pdir}/{c}.nii.gz") for c in cases]), key=lambda x: x["case"])
        for r in rs:
            r["thesis_test"] = int(r["case"] in thesis_test)
        with open(f"{out_dir}/fold_{f}_per_case.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
        rows[f] = rs

    def stats(rs, key):
        v = np.array([r[f"{key}_dice"] for r in rs], float)
        ok = v[~np.isnan(v)]
        return ok.mean(), np.median(ok), len(ok), int(np.isnan(v).sum())

    def glob_dice(rs, key):
        tp = sum(r[f"{key}_tp"] for r in rs)
        return 2 * tp / (sum(r[f"{key}_ref"] for r in rs) + sum(r[f"{key}_pred"] for r in rs))

    lines = [f"# Test set evaluation ({DS}, {CHK})", "",
             f"Test cases: {len(cases)} real LiTS (RAS), of which {len(thesis_test)} are the thesis test split.",
             "Dice NaN (both empty) excluded from mean/median; n = cases counted.", ""]
    for subset, filt in (("all 111", lambda r: True), ("thesis test 40", lambda r: r["thesis_test"])):
        lines += [f"## {subset}", "", "| fold | liver mean | liver median | liver+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |",
                  "|---|---|---|---|---|---|---|---|"]
        for f in folds:
            rs = [r for r in rows[f] if filt(r)]
            lm, lmed, _, _ = stats(rs, "liver")
            ltm, _, _, _ = stats(rs, "liver_tumour")
            tm, tmed, tn, tnan = stats(rs, "tumour")
            lines.append(f"| {f} | {lm:.3f} | {lmed:.3f} | {ltm:.3f} | {tm:.3f} | {tmed:.3f} | {tn} ({tnan}) | {glob_dice(rs, 'tumour'):.3f} |")
        lines.append("")
    if len(folds) >= 2:
        a, b = folds[0], folds[1]
        lines += [f"## Paired comparison fold {b} − fold {a} (all 111)", "", "| metric | mean diff | median diff | |diff| mean | n | Wilcoxon p |", "|---|---|---|---|---|---|"]
        for key in ("liver", "liver_tumour", "tumour"):
            x = np.array([r[f"{key}_dice"] for r in rows[a]], float)
            y = np.array([r[f"{key}_dice"] for r in rows[b]], float)
            m = ~np.isnan(x) & ~np.isnan(y)
            d = y[m] - x[m]
            p = wilcoxon(y[m], x[m]).pvalue if np.any(d != 0) else float("nan")
            lines.append(f"| {key} | {d.mean():+.4f} | {np.median(d):+.4f} | {np.abs(d).mean():.4f} | {m.sum()} | {p:.3g} |")
        lines.append("")
    worst = sorted(rows[folds[0]], key=lambda r: r["liver_dice"])[:5]
    lines += [f"Lowest liver Dice (fold {folds[0]}): " + ", ".join(f"{r['case']} {r['liver_dice']:.2f}" for r in worst), ""]
    txt = "\n".join(lines)
    open(f"{out_dir}/summary.md", "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    p = sp.add_parser("predict")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--npp", type=int, default=3, help="nnU-Net preprocessing workers")
    p.add_argument("--nps", type=int, default=3, help="nnU-Net export workers")
    e = sp.add_parser("eval")
    e.add_argument("--folds", type=int, nargs="+", required=True)
    e.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(16)
    elif a.cmd == "predict":
        predict(a.fold, a.npp, a.nps)
    else:
        eval_folds(a.folds, a.workers)


if __name__ == "__main__":
    main()
