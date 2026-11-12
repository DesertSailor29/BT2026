#!/usr/bin/env python
"""Predict the real LiTS test cases with trained arms (folds of Dataset101, or of the repair datasets 103 / 104) and score them.

Test cases = real_split.json "test" (111 LiTS cases: thesis train split minus the 20 val cases + the 40 thesis test
cases), read from the RAS copy (raw_data/LiTS_RAS) into Dataset101's imagesTs_real / labelsTs_real; every dataset of
the liver target uses these same files. A run = dataset id + fold (= one experiment arm, make_splits.py).

  python predict_test.py prepare                        # copy test images/labels into Dataset101 (int16/uint8, RAS)
  python predict_test.py predict --fold 0               # Dataset101 fold 0, checkpoint_final -> fold_0/test_final
  python predict_test.py predict --dataset 103 --fold 0
  python predict_test.py eval --runs 101:0 101:1 103:0  # per-case CSVs + summary.md; first run = reference
  python predict_test.py eval --folds 0 1               # = --runs 101:0 101:1

Everything is resumable: prepare skips existing files, predict uses --continue_prediction, eval reuses a run's
per-case CSV when it is newer than all its predictions (delete it to force a recompute).

Metrics per case: liver (label 1 only, as nnU-Net reports it), liver+tumour (label > 0, LiTS convention), tumour
(label 2). Dice is NaN when reference and prediction are both empty (excluded from means; counted). Global tumour Dice
= 2*sum(TP) / (sum(|ref|) + sum(|pred|)) over all cases (LiTS "Dice global").
Lesion-wise (tumour label, 26-connected components): a reference lesion is *detected* if any predicted tumour voxel
overlaps it; a predicted component is a *false positive* if it overlaps no reference tumour. Recall = detected /
reference lesions (all, and lesions >= 1 ml), precision = hit predicted components / predicted components, FP count
and FP volume per case, and how many tumour-free cases get any predicted tumour.
Comparisons: every run vs the first run, paired per case + Wilcoxon signed-rank test.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import copy_real  # noqa: E402  (same int16/uint8 + qform/sform conversion as the val cases)

PROJ = os.environ.get("PROJ", "/work/Research")
DS = "Dataset101_LiverSynth"
RAW = f"{os.environ['nnUNet_raw']}/{DS}"
IMG, LAB = f"{RAW}/imagesTs_real", f"{RAW}/labelsTs_real"
CHK = "checkpoint_final.pth"
TRAINER = "nnUNetTrainer__nnUNetPlans__3d_fullres"
STRUCT26 = np.ones((3, 3, 3), bool)
BIG_ML = 1.0
CSV_VERSION = "lesion1"


def dataset_name(did):
    found = glob.glob(f"{os.environ['nnUNet_raw']}/Dataset{did:03d}_*")
    assert len(found) == 1, f"Dataset{did:03d}_* in nnUNet_raw: {found}"
    return os.path.basename(found[0])


def res_dir(did):
    return f"{os.environ['nnUNet_results']}/{dataset_name(did)}/{TRAINER}"


def arm_name(did, fold):
    f = f"{os.environ['nnUNet_preprocessed']}/{dataset_name(did)}/arms.json"
    arms = json.load(open(f)) if os.path.exists(f) else []
    return arms[fold]["arm"] if fold < len(arms) else "?"


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
        elif os.path.exists(f"{IMG}/lits_{i:03d}_0000.nii.gz") and os.path.exists(f"{LAB}/lits_{i:03d}.nii.gz"):
            continue
        else:
            missing.append(i)
    if missing:
        raise SystemExit(f"missing in {src}: {missing}")
    with Pool(workers) as p:
        list(p.imap_unordered(copy_real, jobs))
    n = len(glob.glob(f"{IMG}/*_0000.nii.gz"))
    print(f"prepare: {n} test images in {IMG}", flush=True)
    assert n == len(s["test"]), n


def predict(did, fold, npp, nps):
    prepare(16)
    out = f"{res_dir(did)}/fold_{fold}/test_final"
    cmd = ["nnUNetv2_predict", "-i", IMG, "-o", out, "-d", str(did), "-c", "3d_fullres", "-f", str(fold),
           "-chk", CHK, "-npp", str(npp), "-nps", str(nps), "--continue_prediction"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    n = len(glob.glob(f"{out}/lits_*.nii.gz"))
    print(f"D{did} fold {fold}: {n} predictions in {out}", flush=True)


def dice(r, p):
    a, b = int(r.sum()), int(p.sum())
    tp = int(np.logical_and(r, p).sum())
    return (2 * tp / (a + b) if a + b else float("nan")), tp, a, b


def lesions(rt, pt, voxel_ml):
    rl, nr = ndimage.label(rt, structure=STRUCT26)
    pl, npred = ndimage.label(pt, structure=STRUCT26)
    rsize = np.bincount(rl.ravel(), minlength=nr + 1)[1:]
    psize = np.bincount(pl.ravel(), minlength=npred + 1)[1:]
    hit_r = np.zeros(nr, bool)
    hit_r[np.unique(rl[(rl > 0) & pt]) - 1] = True
    hit_p = np.zeros(npred, bool)
    hit_p[np.unique(pl[(pl > 0) & rt]) - 1] = True
    big = rsize * voxel_ml >= BIG_ML
    return {"les_ref": int(nr), "les_det": int(hit_r.sum()), "les_ref_big": int(big.sum()),
            "les_det_big": int((hit_r & big).sum()), "les_pred": int(npred), "les_pred_hit": int(hit_p.sum()),
            "les_fp": int((~hit_p).sum()), "fp_ml": round(float(psize[~hit_p].sum()) * voxel_ml, 3)}


def score(args):
    ref_f, pred_f = args
    ref = nib.load(ref_f)
    r = np.asarray(ref.dataobj)
    p = np.asarray(nib.load(pred_f).dataobj)
    assert r.shape == p.shape, (ref_f, r.shape, p.shape)
    voxel_ml = float(np.prod(np.sqrt((ref.affine[:3, :3] ** 2).sum(axis=0)))) / 1000.0
    row = {"case": os.path.basename(ref_f)[: -len(".nii.gz")]}
    for name, rm, pm in (("liver", r == 1, p == 1), ("liver_tumour", r > 0, p > 0), ("tumour", r == 2, p == 2)):
        d, tp, a, b = dice(rm, pm)
        row.update({f"{name}_dice": d, f"{name}_tp": tp, f"{name}_ref": a, f"{name}_pred": b})
    row.update(lesions(r == 2, p == 2, voxel_ml))
    return row


def run_rows(did, fold, cases, thesis_test, out_dir, workers):
    pdir = f"{res_dir(did)}/fold_{fold}/test_final"
    preds = [f"{pdir}/{c}.nii.gz" for c in cases]
    missing = [c for c, f in zip(cases, preds) if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"D{did} fold {fold}: {len(missing)} predictions missing, e.g. {missing[:5]}")
    csv_f = f"{out_dir}/D{did}_fold_{fold}_per_case.csv"
    if os.path.exists(csv_f) and os.path.getmtime(csv_f) > max(os.path.getmtime(f) for f in preds):
        rows = list(csv.DictReader(open(csv_f)))
        if rows and rows[0].get("version") == CSV_VERSION and len(rows) == len(cases):
            return [{k: (v if k in ("case", "version") else float(v)) for k, v in r.items()} for r in rows]
    with Pool(workers) as p:
        rows = sorted(p.map(score, [(f"{LAB}/{c}.nii.gz", f) for c, f in zip(cases, preds)]), key=lambda x: x["case"])
    for r in rows:
        r["thesis_test"] = int(r["case"] in thesis_test)
        r["version"] = CSV_VERSION
    with open(csv_f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def eval_runs(runs, workers, out_dir):
    from scipy.stats import wilcoxon

    s = split()
    thesis_test = {f"lits_{i:03d}" for i in s["thesis_test"]}
    cases = [f"lits_{i:03d}" for i in s["test"]]
    os.makedirs(out_dir, exist_ok=True)
    rows, names = {}, {}
    for did, fold in runs:
        rows[(did, fold)] = run_rows(did, fold, cases, thesis_test, out_dir, workers)
        names[(did, fold)] = f"D{did} f{fold} {arm_name(did, fold)}"

    def mean_med(rs, key):
        v = np.array([r[f"{key}_dice"] for r in rs], float)
        ok = v[~np.isnan(v)]
        return ok.mean(), np.median(ok), len(ok), int(np.isnan(v).sum())

    def tot(rs, k):
        return sum(r[k] for r in rs)

    def glob_dice(rs, key):
        return 2 * tot(rs, f"{key}_tp") / (tot(rs, f"{key}_ref") + tot(rs, f"{key}_pred"))

    lines = [f"# Test set evaluation ({CHK})", "",
             f"Test cases: {len(cases)} real LiTS (RAS), of which {len(thesis_test)} are the thesis test split.",
             "Dice NaN (both empty) excluded from mean/median; n = cases counted. Lesion: 26-connected tumour "
             f"components, detected = any overlap; 'big' = reference lesion >= {BIG_ML:g} ml.", ""]
    for subset, filt in ((f"all {len(cases)}", lambda r: True), (f"thesis test {len(thesis_test)}", lambda r: r["thesis_test"])):
        lines += [f"## {subset}: Dice", "",
                  "| run | liver mean | liver median | liver+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |",
                  "|---|---|---|---|---|---|---|---|"]
        for k in runs:
            rs = [r for r in rows[k] if filt(r)]
            lm, lmed, _, _ = mean_med(rs, "liver")
            ltm, _, _, _ = mean_med(rs, "liver_tumour")
            tm, tmed, tn, tnan = mean_med(rs, "tumour")
            lines.append(f"| {names[k]} | {lm:.3f} | {lmed:.3f} | {ltm:.3f} | {tm:.3f} | {tmed:.3f} | {tn} ({tnan}) | {glob_dice(rs, 'tumour'):.3f} |")
        lines += ["", f"## {subset}: lesions", "",
                  "| run | ref lesions | recall | recall >= 1 ml | pred comps | precision | FP comps / case | FP ml / case | tumour-free cases with FP |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k in runs:
            rs = [r for r in rows[k] if filt(r)]
            free = [r for r in rs if r["les_ref"] == 0]
            ratio = lambda a, b: f"{tot(rs, a) / tot(rs, b):.3f}" if tot(rs, b) else "-"  # noqa: E731
            lines.append(f"| {names[k]} | {tot(rs, 'les_ref'):.0f} | {ratio('les_det', 'les_ref')} | "
                         f"{ratio('les_det_big', 'les_ref_big')} | {tot(rs, 'les_pred'):.0f} | {ratio('les_pred_hit', 'les_pred')} | {tot(rs, 'les_fp') / len(rs):.2f} | {tot(rs, 'fp_ml') / len(rs):.2f} | "
                         f"{sum(r['les_pred'] > 0 for r in free)}/{len(free)} |")
        lines.append("")
    if len(runs) >= 2:
        ref = runs[0]
        lines += [f"## Paired vs {names[ref]} (all {len(cases)})", "",
                  "| run | metric | mean diff | median diff | mean abs diff | n | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
        for k in runs[1:]:
            for key in ("liver_dice", "liver_tumour_dice", "tumour_dice", "les_fp", "fp_ml"):
                x = np.array([r[key] for r in rows[ref]], float)
                y = np.array([r[key] for r in rows[k]], float)
                m = ~np.isnan(x) & ~np.isnan(y)
                d = y[m] - x[m]
                p = wilcoxon(y[m], x[m]).pvalue if np.any(d != 0) else float("nan")
                lines.append(f"| {names[k]} | {key} | {d.mean():+.4f} | {np.median(d):+.4f} | {np.abs(d).mean():.4f} | {m.sum()} | {p:.3g} |")
        lines.append("")
    for k in runs:
        worst = sorted(rows[k], key=lambda r: r["liver_dice"])[:5]
        lines.append(f"Lowest liver Dice ({names[k]}): " + ", ".join(f"{r['case']} {r['liver_dice']:.2f}" for r in worst))
    txt = "\n".join(lines) + "\n"
    open(f"{out_dir}/summary.md", "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    p = sp.add_parser("predict")
    p.add_argument("--dataset", type=int, default=101)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--npp", type=int, default=3, help="nnU-Net preprocessing workers")
    p.add_argument("--nps", type=int, default=3, help="nnU-Net export workers")
    e = sp.add_parser("eval")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--runs", nargs="+", help="dataset:fold, e.g. 101:0 103:0 (first = reference)")
    g.add_argument("--folds", type=int, nargs="+", help="folds of Dataset101")
    e.add_argument("--workers", type=int, default=12)
    e.add_argument("--out", default=f"{os.environ['nnUNet_results']}/test_eval_liver")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(16)
    elif a.cmd == "predict":
        predict(a.dataset, a.fold, a.npp, a.nps)
    else:
        runs = [tuple(int(x) for x in r.split(":")) for r in a.runs] if a.runs else [(101, f) for f in a.folds]
        eval_runs(runs, a.workers, a.out)


if __name__ == "__main__":
    main()
