#!/usr/bin/env python
"""Predict the real LiTS test cases with trained arms (folds of Dataset101, or of the repair datasets 103 / 104) and score them.

Test cases = real_split.json "test" (111 LiTS cases: thesis train split minus the 20 val cases + the 40 thesis test
cases), read from the RAS copy (raw_data/LiTS_RAS) into Dataset101's imagesTs_real / labelsTs_real; every dataset of
the liver target uses these same files. A run = dataset id + fold (= one experiment arm, make_splits.py).

  python predict_test.py prepare                        # copy test images/labels into Dataset101 (int16/uint8, RAS)
  python predict_test.py predict --fold 0               # Dataset101 fold 0, checkpoint_final -> fold_0/test_final
  python predict_test.py predict --dataset 103 --fold 0
  python predict_test.py eval --runs 103:0 101:2         # arms vs the baseline (mean of 101:0 + 101:1)
  python predict_test.py eval                           # baseline runs only (+ the noise row)
  python predict_test.py predict --fold 2 --checkpoint best   # checkpoint_best -> fold_2/test_best
  python predict_test.py eval --checkpoint best --runs ...    # same eval on test_best -> test_eval_liver_best

Everything is resumable: prepare skips existing files, predict uses --continue_prediction, eval reuses a run's
per-case CSV when it is newer than all its predictions (delete it to force a recompute).

Metrics per case: liver (label 1 only, as nnU-Net reports it), liver+tumour (label > 0, LiTS convention), tumour
(label 2). Dice is NaN when reference and prediction are both empty (excluded from means; counted). Global tumour Dice
= 2*sum(TP) / (sum(|ref|) + sum(|pred|)) over all cases (LiTS "Dice global").
Lesion-wise (tumour label, 26-connected components): a reference lesion is *detected* if any predicted tumour voxel
overlaps it; a predicted component is a *false positive* if it overlaps no reference tumour. Recall = detected /
reference lesions (all, and lesions >= 1 ml), precision = hit predicted components / predicted components, FP count
and FP volume per case, and how many tumour-free cases get any predicted tumour.
Comparisons (PLAN_training.md "Evaluation and analysis"): baseline = the per-case mean of the `--baseline` runs (the
two `all` runs). Main: tumour Dice per case on all test cases, arm - baseline, paired Wilcoxon, Holm over the arms of
`--family` (the 4 main arms); clear win = Holm p < 0.05 and mean difference > 0.01 (noise floor). Every difference
gets a paired bootstrap 95% CI. Secondary (exploratory, uncorrected): liver / liver+tumour Dice, FP components and
volume, the thesis test subset, repair vs reject at the same criterion, each arm against each baseline run.
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
CHK = "checkpoint_final.pth"  # set by --checkpoint (final | best)
PRED_SUB = "test_final"
TRAINER = "nnUNetTrainer__nnUNetPlans__3d_fullres"
STRUCT26 = np.ones((3, 3, 3), bool)
BIG_ML = 1.0
CSV_VERSION = "lesion1"
MAIN_KEY = "tumour_dice"  # main metric (PLAN_training.md "Evaluation and analysis")
SECONDARY = ("liver_dice", "liver_tumour_dice", "les_fp", "fp_ml")
NOISE_FLOOR = 0.01  # |mean diff| between the two `all` runs was < 0.01
BOOT = 10000
FAMILY = ("ccReject", "ccRepair", "ccRepairDel", "ccAucRepair")  # Decisions 2026-11-14


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
    out = f"{res_dir(did)}/fold_{fold}/{PRED_SUB}"
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
    pdir = f"{res_dir(did)}/fold_{fold}/{PRED_SUB}"
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


def holm(pvals):
    """Holm-adjusted p-values (same order as given): sort ascending, p_(i) * (m - i + 1), running maximum, cap at 1."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj, run = np.empty(m), 0.0
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[k]))
        adj[k] = run
    return adj


def boot_ci(d, b=BOOT, seed=0):
    """Paired bootstrap 95% CI of the mean difference (cases resampled with replacement, percentile method)."""
    if len(d) < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(d), (b, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def paired(x, y):
    """y - x per case (NaN pairs dropped): dict with n, means, mean diff, CI, median diff, Wilcoxon p."""
    from scipy.stats import wilcoxon

    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y)
    d = y[m] - x[m]
    p = float(wilcoxon(y[m], x[m]).pvalue) if np.any(d != 0) else float("nan")
    lo, hi = boot_ci(d)
    return {"n": int(m.sum()), "x": float(x[m].mean()), "y": float(y[m].mean()), "diff": float(d.mean()),
            "lo": lo, "hi": hi, "median": float(np.median(d)), "p": p}


def fmt_diff(r, digits=4):
    return f"{r['diff']:+.{digits}f} [{r['lo']:+.{digits}f}, {r['hi']:+.{digits}f}]"


def eval_runs(baseline, runs, family, workers, out_dir):
    s = split()
    thesis_test = {f"lits_{i:03d}" for i in s["thesis_test"]}
    cases = [f"lits_{i:03d}" for i in s["test"]]
    os.makedirs(out_dir, exist_ok=True)
    runs = [k for k in runs if k not in baseline]
    every = baseline + runs
    rows, names = {}, {}
    for did, fold in every:
        rows[(did, fold)] = run_rows(did, fold, cases, thesis_test, out_dir, workers)
        names[(did, fold)] = f"D{did} f{fold} {arm_name(did, fold)}"
    arm_of = {k: arm_name(*k) for k in every}

    def vals(k, key, subset=None):
        return [r[key] for r in rows[k] if subset is None or r["case"] in subset]

    def base_vals(key, subset=None):
        """Per-case mean over the baseline runs (NaN only if every baseline run is NaN for that case)."""
        a = np.array([vals(k, key, subset) for k in baseline], float)
        n = (~np.isnan(a)).sum(axis=0)
        return np.where(n > 0, np.nansum(a, axis=0) / np.maximum(n, 1), np.nan)

    def mean_med(rs, key):
        v = np.array([r[f"{key}_dice"] for r in rs], float)
        ok = v[~np.isnan(v)]
        return ok.mean(), np.median(ok), len(ok), int(np.isnan(v).sum())

    def tot(rs, k):
        return sum(r[k] for r in rs)

    def glob_dice(rs, key):
        return 2 * tot(rs, f"{key}_tp") / (tot(rs, f"{key}_ref") + tot(rs, f"{key}_pred"))

    base_name = " + ".join(names[k] for k in baseline)
    lines = [f"# Test set evaluation ({CHK})", "",
             f"Test cases: {len(cases)} real LiTS (RAS), of which {len(thesis_test)} are the thesis test split. "
             f"Baseline = per-case mean of {base_name}.",
             "Dice NaN (both empty) excluded from mean/median and from paired tests; n = cases counted. Lesion: "
             f"26-connected tumour components, detected = any overlap; 'big' = reference lesion >= {BIG_ML:g} ml.", ""]

    # --- main comparison: tumour Dice per case on all test cases, vs the baseline mean, Holm over the family -----
    main = {k: paired(base_vals(MAIN_KEY), vals(k, MAIN_KEY)) for k in runs}
    fam = [k for k in runs if arm_of[k] in family]
    adj = dict(zip(fam, holm([main[k]["p"] if not np.isnan(main[k]["p"]) else 1.0 for k in fam]))) if fam else {}
    noise = paired(vals(baseline[0], MAIN_KEY), vals(baseline[1], MAIN_KEY)) if len(baseline) >= 2 else None
    lines += [f"## Main comparison: tumour Dice per case, all {len(cases)}, vs the baseline", "",
              f"Clear win = Holm-adjusted Wilcoxon p < 0.05 over the {len(fam)} arms of the family "
              f"({', '.join(family)}) and mean difference > {NOISE_FLOOR}. Difference = arm - baseline, with a paired "
              f"bootstrap 95% CI ({BOOT} resamples). Arms outside the family: exploratory (no Holm).", "",
              "| run | n | arm mean | baseline mean | mean diff [95% CI] | median diff | Wilcoxon p | Holm p | clear win |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k in runs:
        r = main[k]
        h = adj.get(k)
        win = "-" if h is None else ("yes" if h < 0.05 and r["diff"] > NOISE_FLOOR else "no")
        lines.append(f"| {names[k]} | {r['n']} | {r['y']:.3f} | {r['x']:.3f} | {fmt_diff(r)} | {r['median']:+.4f} | "
                     f"{r['p']:.3g} | {'-' if h is None else f'{h:.3g}'} | {win} |")
    if noise:
        lines.append(f"| noise: {names[baseline[1]]} - {names[baseline[0]]} | {noise['n']} | {noise['y']:.3f} | "
                     f"{noise['x']:.3f} | {fmt_diff(noise)} | {noise['median']:+.4f} | {noise['p']:.3g} | - | - |")
    lines.append("")

    if runs and len(baseline) >= 2:
        lines += ["Each arm against each baseline run separately (tumour Dice, mean diff [95% CI], Wilcoxon p):", "",
                  "| run | " + " | ".join(f"vs {names[b]}" for b in baseline) + " |", "|---|" + "---|" * len(baseline)]
        for k in runs:
            cells = [paired(vals(b, MAIN_KEY), vals(k, MAIN_KEY)) for b in baseline]
            lines.append(f"| {names[k]} | " + " | ".join(f"{fmt_diff(c)}, p {c['p']:.3g}" for c in cells) + " |")
        lines.append("")

    # --- secondary, exploratory -------------------------------------------------------------------------------
    if runs:
        lines += [f"## Secondary (exploratory, no correction): vs the baseline, all {len(cases)}", "",
                  "| run | metric | mean diff [95% CI] | median diff | Wilcoxon p |", "|---|---|---|---|---|"]
        for k in runs:
            for key in SECONDARY:
                r = paired(base_vals(key), vals(k, key))
                lines.append(f"| {names[k]} | {key} | {fmt_diff(r)} | {r['median']:+.4f} | {r['p']:.3g} |")
        lines += ["", f"Tumour Dice on the {len(thesis_test)} thesis test cases (exploratory):", "",
                  "| run | n | mean diff [95% CI] | Wilcoxon p |", "|---|---|---|---|"]
        for k in runs:
            r = paired(base_vals(MAIN_KEY, thesis_test), vals(k, MAIN_KEY, thesis_test))
            lines.append(f"| {names[k]} | {r['n']} | {fmt_diff(r)} | {r['p']:.3g} |")
        pairs = [(kr, kj) for kr in runs for kj in runs
                 if arm_of[kr].startswith("repair") and arm_of[kj] == "reject" + arm_of[kr][len("repair"):]]
        if pairs:
            lines += ["", "Repair vs reject at the same criterion (tumour Dice, repair - reject, exploratory):", "",
                      "| repair | reject | n | mean diff [95% CI] | Wilcoxon p |", "|---|---|---|---|---|"]
            for kr, kj in pairs:
                r = paired(vals(kj, MAIN_KEY), vals(kr, MAIN_KEY))
                lines.append(f"| {names[kr]} | {names[kj]} | {r['n']} | {fmt_diff(r)} | {r['p']:.3g} |")
        lines.append("")

    # --- descriptive tables per run ------------------------------------------------------------------------------
    for subset, filt in ((f"all {len(cases)}", lambda r: True), (f"thesis test {len(thesis_test)}", lambda r: r["thesis_test"])):
        lines += [f"## {subset}: Dice", "",
                  "| run | liver mean | liver median | liver+tumour mean | tumour mean | tumour median | tumour n (NaN) | tumour global |",
                  "|---|---|---|---|---|---|---|---|"]
        for k in every:
            rs = [r for r in rows[k] if filt(r)]
            lm, lmed, _, _ = mean_med(rs, "liver")
            ltm, _, _, _ = mean_med(rs, "liver_tumour")
            tm, tmed, tn, tnan = mean_med(rs, "tumour")
            lines.append(f"| {names[k]} | {lm:.3f} | {lmed:.3f} | {ltm:.3f} | {tm:.3f} | {tmed:.3f} | {tn} ({tnan}) | {glob_dice(rs, 'tumour'):.3f} |")
        lines += ["", f"## {subset}: lesions", "",
                  "| run | ref lesions | recall | recall >= 1 ml | pred comps | precision | FP comps / case | FP ml / case | tumour-free cases with FP |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k in every:
            rs = [r for r in rows[k] if filt(r)]
            free = [r for r in rs if r["les_ref"] == 0]
            ratio = lambda a, b: f"{tot(rs, a) / tot(rs, b):.3f}" if tot(rs, b) else "-"  # noqa: E731
            lines.append(f"| {names[k]} | {tot(rs, 'les_ref'):.0f} | {ratio('les_det', 'les_ref')} | "
                         f"{ratio('les_det_big', 'les_ref_big')} | {tot(rs, 'les_pred'):.0f} | {ratio('les_pred_hit', 'les_pred')} | {tot(rs, 'les_fp') / len(rs):.2f} | {tot(rs, 'fp_ml') / len(rs):.2f} | "
                         f"{sum(r['les_pred'] > 0 for r in free)}/{len(free)} |")
        lines.append("")
    for k in every:
        worst = sorted(rows[k], key=lambda r: r["liver_dice"])[:5]
        lines.append(f"Lowest liver Dice ({names[k]}): " + ", ".join(f"{r['case']} {r['liver_dice']:.2f}" for r in worst))
    txt = "\n".join(lines) + "\n"
    open(f"{out_dir}/summary.md", "w").write(txt)
    print(txt)
    try:  # extra report; must never break the eval
        confusion_tables(rows, names, every, cases, out_dir)
        print(f"confusion tables: {out_dir}/confusion.md", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"confusion tables failed: {ex!r}", flush=True)
    try:
        copy_progress_plots()
    except Exception as ex:  # noqa: BLE001
        print(f"progress plots failed: {ex!r}", flush=True)
    try:
        import training_curves
        training_curves.main()
    except Exception as ex:  # noqa: BLE001
        print(f"training curves failed: {ex!r}", flush=True)


def confusion_tables(rows, names, every, cases, out_dir):
    """confusion.md: TP / FP / FN / TN per run at voxel (tumour, liver), lesion and case level, summed over all test cases.

    Voxel TN = all voxels of the case minus TP, FP, FN (shapes from the reference label headers). Liver = label 1 only,
    as in the Dice tables. Lesion level has no TN. Case level: tumour cases count as TP if >= 1 lesion is detected,
    tumour-free cases as FP if any tumour voxel is predicted, else TN.
    """
    nvox = {c: int(np.prod(nib.load(f"{LAB}/{c}.nii.gz").shape)) for c in cases}
    tot = lambda rs, k: sum(float(r[k]) for r in rs)  # noqa: E731
    lines = ["# Test set: true / false positives and negatives per run", "",
             f"{len(cases)} real LiTS test cases, {CHK[:-4]}. Sums over all cases. Written by `predict_test.py eval`.", ""]
    for struct, label in (("tumour", "label 2"), ("liver", "label 1 only, without tumour")):
        lines += [f"## {struct.capitalize()}, voxel-level ({label}; millions of voxels)", "",
                  "| run | TP | FP | FN | TN | sensitivity | precision | FP / TP |", "|---|---|---|---|---|---|---|---|"]
        for k in every:
            rs = rows[k]
            tp, ref, pred = tot(rs, f"{struct}_tp"), tot(rs, f"{struct}_ref"), tot(rs, f"{struct}_pred")
            fp, fn = pred - tp, ref - tp
            tn = sum(nvox[r["case"]] for r in rs) - tp - fp - fn
            lines.append(f"| {names[k]} | {tp / 1e6:.2f} | {fp / 1e6:.2f} | {fn / 1e6:.2f} | {tn / 1e6:,.1f} | "
                         f"{tp / ref:.3f} | {tp / pred if pred else float('nan'):.3f} | {fp / tp if tp else float('nan'):.2f} |")
        lines.append("")
    lines += ["## Tumour, lesion-level (26-connected components; no TN at this level)", "",
              "| run | reference lesions | TP (found) | FN (missed) | FP (predicted pieces on no lesion) | FP per case |",
              "|---|---|---|---|---|---|"]
    for k in every:
        rs = rows[k]
        ref, det, fp = tot(rs, "les_ref"), tot(rs, "les_det"), tot(rs, "les_fp")
        lines.append(f"| {names[k]} | {ref:.0f} | {det:.0f} | {ref - det:.0f} | {fp:.0f} | {fp / len(rs):.1f} |")
    lines += ["", "## Tumour, case-level", "",
              "| run | TP (tumour case, >= 1 lesion found) | FN (tumour case, none found) | FP (tumour-free case, tumour predicted) | TN (tumour-free case, clean) |",
              "|---|---|---|---|---|"]
    for k in every:
        rs = rows[k]
        tc = [r for r in rs if float(r["les_ref"]) > 0]
        tf = [r for r in rs if float(r["les_ref"]) == 0]
        lines.append(f"| {names[k]} | {sum(float(r['les_det']) > 0 for r in tc)} | {sum(float(r['les_det']) == 0 for r in tc)} | "
                     f"{sum(float(r['tumour_pred']) > 0 for r in tf)} | {sum(float(r['tumour_pred']) == 0 for r in tf)} |")
    lines += ["", "Voxel TN outnumbers everything else ~1000x (specificity > 0.999 for every run), so compare runs on "
              "sensitivity, precision and FP / TP."]
    open(f"{out_dir}/confusion.md", "w").write("\n".join(lines) + "\n")


def copy_progress_plots():
    """Copy every run's nnU-Net progress.png (losses + pseudo Dice per epoch) into the repo next to the test results,
    as reports/data/test_eval_liver/progress/D<id>_f<fold>_<arm>.png; run_queue.py's publish() commits that folder.
    A run still training gets its current plot (refreshed at the next eval)."""
    import shutil
    dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports/data/test_eval_liver/progress")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(f"{os.environ['nnUNet_results']}/Dataset1*/{TRAINER}/fold_*/progress.png")):
        did = int(f.split("/Dataset")[1][:3])
        fold = int(f.split("/fold_")[1].split("/")[0])
        shutil.copy(f, f"{dst}/D{did}_f{fold}_{arm_name(did, fold)}.png")
        n += 1
    print(f"progress plots: {n} copied to {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    p = sp.add_parser("predict")
    p.add_argument("--dataset", type=int, default=101)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--npp", type=int, default=3, help="nnU-Net preprocessing workers")
    p.add_argument("--nps", type=int, default=3, help="nnU-Net export workers")
    p.add_argument("--checkpoint", choices=("final", "best"), default="final")
    e = sp.add_parser("eval")
    e.add_argument("--baseline", nargs="+", default=["101:0", "101:1"],
                   help="dataset:fold runs whose per-case mean is the baseline (default: `all` + `all_rep2`)")
    e.add_argument("--runs", nargs="*", default=[], help="arms to compare, dataset:fold, e.g. 103:0 101:2")
    e.add_argument("--family", nargs="+", default=list(FAMILY), help="arm names in the Holm family (main comparison)")
    e.add_argument("--workers", type=int, default=12)
    e.add_argument("--checkpoint", choices=("final", "best"), default="final",
                   help="which checkpoint's predictions (fold_X/test_<checkpoint>) to score")
    e.add_argument("--out", help="default $nnUNet_results/test_eval_liver (final) or test_eval_liver_best (best)")
    ap.set_defaults(checkpoint="final")
    a = ap.parse_args()
    global CHK, PRED_SUB
    CHK, PRED_SUB = f"checkpoint_{a.checkpoint}.pth", f"test_{a.checkpoint}"
    if a.cmd == "prepare":
        prepare(16)
    elif a.cmd == "predict":
        predict(a.dataset, a.fold, a.npp, a.nps)
    else:
        parse = lambda xs: [tuple(int(v) for v in x.split(":")) for x in xs]  # noqa: E731
        out = a.out or f"{os.environ['nnUNet_results']}/test_eval_liver" + ("" if a.checkpoint == "final" else "_best")
        eval_runs(parse(a.baseline), parse(a.runs), a.family, a.workers, out)


if __name__ == "__main__":
    main()
