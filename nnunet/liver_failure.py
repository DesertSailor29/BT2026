#!/usr/bin/env python
"""Liver L1 (autopilot/MISSION.md "Liver extension"): where do the liver models fail, and does a 3-run vote help?
CPU only, reads saved test predictions (checkpoint_final, fold_X/test_final) of the 111 LiTS test cases.

Part 1, missed tumours: every real lesion (26-connected piece of label 2 in the reference) with its volume, and per
run + variant (saved / ccPost) the share of its voxels predicted as tumour (detected = share > 0, as the eval's
lesion recall). Summary: lesion recall and mean coverage by lesion size bin, tumour Dice by the case's real tumour
burden, so the tumour-size arms (L3) can be read against it.
Part 2, vote: per voxel, tumour where >= 2 of the 3 baseline runs (101:0/1/3) say tumour, else liver where >= 2 say
liver or tumour, else background; scored as saved and with ccPost (maisi/cc_rules.py main_structure), compared per
case (paired, bootstrap CI, Wilcoxon: predict_test.py definitions) with each single run and the per-case 3-run mean.
Declared before any number (planning session 2026-11-26, MISSION.md L1): the vote "helps" if vote − 3-run mean > +0.01 with CI > 0.

  python nnunet/liver_failure.py --workers 12          (resumable: per-case caches; ~20-40 min)
  python nnunet/liver_failure.py --cases lits_000 ...  (fake test: a few cases, output to --out)
-> $nnUNet_results/test_eval_liver/failure/{cache/, lesions.csv, cases.csv, summary.md}, summary + CSVs copied to
   reports/data/test_eval_liver/failure/.
"""
import argparse
import csv
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
import predict_test as pt  # noqa: E402  (paths, Dice, lesion metrics, paired stats)
from cc_rules import main_structure  # noqa: E402  (the training-label cc rule = ccPost)

VERSION = "fail2"  # fail1 wrongly used 101:2 (= ccReject, see arms.json) as all_rep3
BASE3 = ["101:0", "101:1", "101:3"]           # all, all_rep2, all_rep3 (D101 arms.json folds 0/1/3)
OTHER = ["101:2", "103:0", "104:0", "104:1"]  # ccReject, ccRepair, ccAucRepair, ccAucRepair_rep2
RUNS = BASE3 + OTHER
SIZE_BINS = [(0, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, float("inf"))]  # ml
BURDEN_BINS = [(0, 1e-9), (1e-9, 10.0), (10.0, 100.0), (100.0, float("inf"))]          # ml per case
OUT = f"{os.environ['nnUNet_results']}/test_eval_liver/failure"
PUB = f"{REPO}/reports/data/test_eval_liver/failure"


def vote(preds):
    tum = sum((p == 2).astype(np.uint8) for p in preds) >= 2
    org = sum((p > 0).astype(np.uint8) for p in preds) >= 2
    return np.where(tum, 2, np.where(org, 1, 0)).astype(np.uint8)


def case_scores(r, p, voxel_ml):
    out = {}
    for name, rm, pm in (("liver", r == 1, p == 1), ("tumour", r == 2, p == 2)):
        out[f"{name}_dice"] = pt.dice(rm, pm)[0]
    out.update(pt.lesions(r == 2, p == 2, voxel_ml))
    return out


def measure(args):
    case, cache_f = args
    ref = nib.load(f"{pt.LAB}/{case}.nii.gz")
    r = np.asarray(ref.dataobj).astype(np.uint8)
    voxel_ml = float(np.prod(np.sqrt((ref.affine[:3, :3] ** 2).sum(axis=0)))) / 1000.0
    rl, nr = ndimage.label(r == 2, pt.STRUCT26)
    rsize = np.bincount(rl.ravel(), minlength=nr + 1)[1:]
    preds = {}
    for run in RUNS:
        did, fold = map(int, run.split(":"))
        p = np.asarray(nib.load(f"{pt.res_dir(did)}/fold_{fold}/{pt.PRED_SUB}/{case}.nii.gz").dataobj).astype(np.uint8)
        assert p.shape == r.shape, (case, run, p.shape, r.shape)
        preds[run] = p
    preds["vote3"] = vote([preds[x] for x in BASE3])
    cov, scores = {}, {}
    for run, p in preds.items():
        for var in ("none", "ccPost"):
            q = p if var == "none" else np.where(main_structure(p), p, 0).astype(np.uint8)
            key = f"{run}|{var}"
            scores[key] = case_scores(r, q, voxel_ml)
            hit = np.bincount(rl[(rl > 0) & (q == 2)], minlength=nr + 1)[1:] if nr else np.zeros(0)
            cov[key] = [round(float(h) / float(s), 4) for h, s in zip(hit, rsize)]
    out = {"version": VERSION, "case": case, "voxel_ml": voxel_ml, "tumour_ml": round(float((r == 2).sum()) * voxel_ml, 3),
           "lesion_ml": [round(float(s) * voxel_ml, 4) for s in rsize], "coverage": cov, "scores": scores}
    tmp = cache_f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, cache_f)
    return case


def nanmean(x):
    x = [v for v in x if v == v]
    return float(np.mean(x)) if x else float("nan")


def summarise(data, cases, out_dir):
    keys = [f"{r}|{v}" for r in RUNS + ["vote3"] for v in ("none", "ccPost")]
    les_rows, case_rows = [], []
    for c in cases:
        d = data[c]
        for i, ml in enumerate(d["lesion_ml"]):
            les_rows.append({"case": c, "lesion": i + 1, "ml": ml, **{k: d["coverage"][k][i] for k in keys}})
        for k in keys:
            case_rows.append({"case": c, "run": k, "case_tumour_ml": d["tumour_ml"], "n_lesions": len(d["lesion_ml"]),
                              **{m: d["scores"][k][m] for m in ("tumour_dice", "liver_dice", "les_ref", "les_det", "les_fp")}})
    for name, rows in (("lesions.csv", les_rows), ("cases.csv", case_rows)):
        with open(f"{out_dir}/{name}", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    L = [f"# Liver failure analysis + 3-run vote (L1, `nnunet/liver_failure.py`, {len(cases)} LiTS test cases)", "",
         "Exploratory. Runs: 101:0/1/3 = all/all_rep2/all_rep3, 101:2 ccReject, 103:0 ccRepair, 104:0/1 ccAucRepair(_rep2); "
         "vote3 = per-voxel majority of the three baseline runs. Variant ccPost = one-structure rule on the prediction.", "",
         f"Real lesions: {len(les_rows)}; cases with tumour: {sum(1 for c in cases if data[c]['lesion_ml'])}.", "",
         "## 1. Lesion recall (share of real lesions with any predicted tumour voxel) / mean coverage, by lesion size", ""]
    hdr = "| run | " + " | ".join(f"{a:g}-{b:g} ml" for a, b in SIZE_BINS) + " |"
    L += [hdr.replace("-inf ml", "+ ml"), "|---" * (len(SIZE_BINS) + 1) + "|"]
    counts = [sum(1 for x in les_rows if a <= x["ml"] < b) for a, b in SIZE_BINS]
    L.append("| n lesions | " + " | ".join(str(n) for n in counts) + " |")
    for k in keys:
        cells = []
        for a, b in SIZE_BINS:
            v = [x[k] for x in les_rows if a <= x["ml"] < b]
            cells.append(f"{np.mean([c > 0 for c in v]):.2f} / {np.mean(v):.2f}" if v else "-")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "## 2. Mean per-case tumour Dice by the case's real tumour burden", ""]
    L += ["| run | " + " | ".join(("tumour-free" if b < 1e-6 else ("0" if a < 1e-6 else f"{a:g}") + f"-{b:g} ml") for a, b in BURDEN_BINS).replace("-inf ml", "+ ml") + " | all |",
          "|---" * (len(BURDEN_BINS) + 2) + "|"]
    L.append("| n cases | " + " | ".join(str(sum(1 for c in cases if (data[c]['tumour_ml'] == 0 if b < 1e-6 else a <= data[c]['tumour_ml'] < b and data[c]['tumour_ml'] > 0)))
                                         for a, b in BURDEN_BINS) + f" | {len(cases)} |")
    for k in keys:
        cells = []
        for a, b in BURDEN_BINS:
            sel = [c for c in cases if (data[c]["tumour_ml"] == 0 if b < 1e-6 else a <= data[c]["tumour_ml"] < b and data[c]["tumour_ml"] > 0)]
            cells.append(f"{nanmean([data[c]['scores'][k]['tumour_dice'] for c in sel]):.3f}" if sel else "-")
        L.append(f"| {k} | " + " | ".join(cells) + f" | {nanmean([data[c]['scores'][k]['tumour_dice'] for c in cases]):.3f} |")
    L += ["", "(tumour-free cases: Dice is 1 if nothing predicted, else 0, as in predict_test.py, or NaN: see cases.csv)", "",
          "## 3. 3-run vote vs single runs (per-case tumour Dice, vote − other, 95 % bootstrap CI, Wilcoxon p)", "",
          "| variant | vs | n | other | vote | diff [CI] | p |", "|---|---|---|---|---|---|---|"]
    for var in ("none", "ccPost"):
        v = [data[c]["scores"][f"vote3|{var}"]["tumour_dice"] for c in cases]
        singles = {r: [data[c]["scores"][f"{r}|{var}"]["tumour_dice"] for c in cases] for r in BASE3}
        mean3 = [nanmean([singles[r][i] for r in BASE3]) for i in range(len(cases))]
        for name, x in list(singles.items()) + [("3-run mean", mean3)]:
            s = pt.paired(x, v)
            L.append(f"| {var} | {name} | {s['n']} | {s['x']:.4f} | {s['y']:.4f} | {pt.fmt_diff(s)} | {s['p']:.2g} |")
    L += ["", "Declared (planning session 2026-11-26, before the numbers): the vote helps if vote − 3-run mean > +0.01 with the CI above 0."]
    open(f"{out_dir}/summary.md", "w").write("\n".join(L) + "\n")
    return "\n".join(L)


def main():
    global OUT, PUB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", help="only these cases (fake test); default: the 111 test cases")
    ap.add_argument("--out", help="output dir (default $nnUNet_results/test_eval_liver/failure); no copy to reports/")
    a = ap.parse_args()
    cases = a.cases or [f"lits_{i:03d}" for i in pt.split()["test"]]
    if a.out:
        OUT, PUB = a.out, None
    os.makedirs(f"{OUT}/cache", exist_ok=True)
    jobs = []
    for c in cases:
        f = f"{OUT}/cache/{c}.json"
        if os.path.exists(f):
            try:
                if json.load(open(f))["version"] == VERSION:
                    continue
            except (ValueError, KeyError):
                pass
        jobs.append((c, f))
    print(f"{len(cases)} cases, {len(jobs)} to measure, runs {RUNS}", flush=True)
    with Pool(a.workers) as p:
        for i, c in enumerate(p.imap_unordered(measure, jobs), 1):
            print(f"[{i}/{len(jobs)}] {c}", flush=True)
    data = {c: json.load(open(f"{OUT}/cache/{c}.json")) for c in cases}
    print(summarise(data, cases, OUT))
    if PUB:
        os.makedirs(PUB, exist_ok=True)
        for n in ("summary.md", "lesions.csv", "cases.csv"):
            shutil.copy(f"{OUT}/{n}", f"{PUB}/{n}")
        print(f"copied to {PUB}")


if __name__ == "__main__":
    main()
