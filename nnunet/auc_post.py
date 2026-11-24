#!/usr/bin/env python
"""The AUC (visibility) rule applied to the predictions: does a test-time ccAucPost do what training on the ccAuc
labels (ccAucRepair) did? Plus a sweep of stricter cuts. CPU, exploratory (plans/PLAN_followups.md, parts 1-2).

For every run (dataset:fold) and every real LiTS test case, on the saved checkpoint_final predictions (fold_X/test_final):

    none            the prediction as saved
    ccPost          the training cc rule on the prediction (as fp_location.py): keep the 26-connected piece of predicted
                    liver ∪ tumour that holds the largest predicted liver piece, everything else -> background
    ccAucPost<cut>  ccPost, then every predicted tumour piece inside the main structure whose separability sep < cut ->
                    its surroundings' label (liver if >= 50% of its outer 26-neighbour shell is predicted liver, else
                    background). sep, shell, contact and the relabel value are filter_metrics' functions, unchanged, with
                    the *predicted* liver as the organ and the real CT image. Unscored pieces (< 10 measurable voxels) pass.

Decisions use only the image and the prediction; the reference label is read afterwards, for scoring only.
The main cut is the training cut (rules.json auc_cut = 0.5475, checked, fixed 2026-11-14 before any result); 0.60, 0.637
and 0.70 are an exploratory dose-response sweep ("stricter filter"), not a choice of cut on test data.

Writes $nnUNet_results/test_eval_liver/auc_post/ (cache/<run>/<case>.json, components.csv, per_case.csv, summary.md)
and copies summary.md + the CSVs to reports/data/test_eval_liver/auc_post/ (not committed).
--split val runs the same on the 20 real val cases (nnU-Net's fold_X/validation predictions, checkpoint_final) into
.../auc_post_val/: the place to look if a stricter cut is ever to be *chosen* (never from the test results).

  python nnunet/auc_post.py --workers 8              # the 6 liver runs (resumable)
  python nnunet/auc_post.py --runs 101:0 101:1 101:2 103:0 103:1 104:0 104:1   # + ccAucRepair_rep2 once tested
  python nnunet/auc_post.py --tables-only
  python nnunet/auc_post.py --split val --workers 8   # the 20 val cases
  python nnunet/auc_post.py --checkpoint best         # checkpoint_best test predictions (fold_X/test_best)
                                                      # -> $nnUNet_results/test_eval_liver_best/auc_post/
(--split val exists for checkpoint_final only: nnU-Net writes fold_X/validation with the final weights.)

--target pancreas (autopilot A1, autopilot/MISSION.md): the same rules on the pancreas baseline runs 102:0 102:1
(organ = predicted pancreas, label 1), MSD07 val (43) or test (238), main cut = the pancreas training cut
(filter_metrics/pancreas/rules.json auc_cut = 0.509, set by msd_025 before any result). The stricter-cut sweep
(0.55, 0.60, 0.637, 0.70) runs on val only: test gets the fixed cut alone, so no cut can be picked from test.
-> $nnUNet_results/test_eval_pancreas/auc_post[_val]/, reports/data/test_eval_pancreas/auc_post[_val]/.
  python nnunet/auc_post.py --target pancreas --split val --workers 8
  python nnunet/auc_post.py --target pancreas --split test --workers 8
Memory: up to ~4 GB per worker (image float32 + label + crops).
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
import fp_location as fl  # noqa: E402  (RUNS, BASELINE, per-case metrics; same definitions as the eval)
import filter_metrics as fm  # noqa: E402  (separability, contact, relabel value of the training AUC rule)
import predict_test as pt  # noqa: E402
from cc_rules import main_structure  # noqa: E402
from orient import spacing_of  # noqa: E402

CUTS = (0.5475, 0.60, 0.637, 0.70)
MAIN_CUT = CUTS[0]
VERSION = f"aucpost1 {fm.VERSION} cuts={','.join(map(str, CUTS))}"
VARIANTS = ("none", "ccPost") + tuple(f"ccAucPost{c:g}" for c in CUTS)
MAIN = f"ccAucPost{MAIN_CUT:g}"
SWEEP_TXT = "0.60 / 0.637 / 0.70"
TARGET, ORG, REAL, PREFIX = "liver", "liver", "LiTS", "lits"
RUNS, BASELINE = fl.RUNS, fl.BASELINE


def set_target(target, split):
    """Switch paths, runs, cuts and names to the pancreas baseline (liver stays exactly as before)."""
    global CUTS, MAIN_CUT, VERSION, VARIANTS, MAIN, SWEEP_TXT, TARGET, ORG, REAL, PREFIX, RUNS, BASELINE, OUT, PUB, RULES
    if target == "liver":
        return
    pt.set_target(target)
    fl.ORG = target
    TARGET, ORG, REAL, PREFIX = target, target, "MSD07", "msd"
    RUNS = BASELINE = pt.BASELINE[target]
    CUTS = (0.509, 0.55, 0.60, 0.637, 0.70) if split == "val" else (0.509,)
    MAIN_CUT = CUTS[0]
    VERSION = f"aucpost1 {fm.VERSION} cuts={','.join(map(str, CUTS))}"
    VARIANTS = ("none", "ccPost") + tuple(f"ccAucPost{c:g}" for c in CUTS)
    MAIN = f"ccAucPost{MAIN_CUT:g}"
    SWEEP_TXT = " / ".join(f"{c:g}" for c in CUTS[1:]) or "none (test: fixed cut only)"
    OUT = OUT.replace("/test_eval_liver/", f"/test_eval_{target}/")
    PUB = PUB.replace("/test_eval_liver/", f"/test_eval_{target}/")
    RULES = RULES.replace("/filter_metrics/liver/", f"/filter_metrics/{target}/")
CKPT = "final"  # set by --checkpoint
OUT = f"{os.environ['nnUNet_results']}/test_eval_liver/auc_post"
PUB = f"{REPO}/reports/data/test_eval_liver/auc_post"
RULES = (os.environ.get("MAISI_RUNS", f"{os.environ.get('PROJ', '/work/Research')}/maisi_runs")
         + "/pool/filter_metrics/liver/rules.json")


def check_rules(path):
    """The main cut and the separability definition must be the ones the ccAuc training labels used."""
    r = json.load(open(path))
    if r["auc_cut"] != MAIN_CUT or r["version"] != fm.VERSION:
        sys.exit(f"{path}: cut {r['auc_cut']} / version {r['version']!r} differ from {MAIN_CUT} / {fm.VERSION!r}")


def relabel(post, comp, rows, cut):
    """post with every scored piece of sep < cut set to repair_value(row, 'auc'); (new label, n relabelled)."""
    lut = np.full(len(rows) + 1, 255, np.uint8)
    for row in rows:
        if row.get("sep") is not None and row["sep"] < cut:
            lut[row["component"]] = fm.repair_value(row, "auc")
    if (lut == 255).all():
        return post, 0
    new = lut[comp]
    return np.where(new == 255, post, new).astype(np.uint8), int((lut != 255).sum())


def measure(args):
    run, case, img_f, ref_f, pred_f, cache_f = args
    pnii = nib.load(pred_f)
    p = np.asarray(pnii.dataobj).astype(np.uint8)
    img = fm.load_image(img_f, p.shape)
    spacing = spacing_of(pnii.affine)
    voxel_ml = float(np.prod(spacing)) / 1000.0

    # decisions: image + prediction only
    keep = main_structure(p)
    post = np.where(keep, p, 0).astype(np.uint8)
    rows, _ = fm.component_rows(post, spacing, img)
    del img
    comp, n = ndimage.label(post == fm.TUMOR, fm.STRUCT26)
    assert n == len(rows), (pred_f, n, len(rows))
    variants = {"none": p, "ccPost": post}
    relabelled = {}
    for c in CUTS:
        variants[f"ccAucPost{c:g}"], relabelled[f"{c:g}"] = relabel(post, comp, rows, c)

    # scoring: reference read only now
    ref = nib.load(ref_f)
    r = np.asarray(ref.dataobj).astype(np.uint8)
    assert r.shape == p.shape, (ref_f, r.shape, p.shape)
    assert np.allclose(ref.affine, pnii.affine, atol=1e-3), (ref_f, "affine differs from the prediction")
    comps = []
    if n:
        idx = range(1, n + 1)
        hit = ndimage.maximum(r == 2, comp, idx)
        ref_tum = ndimage.mean(r == 2, comp, idx)
        for row, h, rt in zip(rows, hit, ref_tum):
            comps.append({"run": run, "case": case, "component": row["component"], "voxels": row["voxels"],
                          "ml": row["ml"], "hit": bool(h), "ref_tumour_share": round(float(rt), 4),
                          "contact": row["contact"], "sep": row.get("sep"), "shell": row.get("shell"),
                          "core_hu": row.get("core_hu"), "shell_hu": row.get("shell_hu"),
                          "to": fm.repair_value(row, "auc")})
    out = {"version": VERSION, "run": run, "case": case, "voxel_ml": voxel_ml, "components": comps,
           "relabelled": relabelled, "ref_tumour_free": not bool((r == 2).any()),
           "metrics": {v: fl.metrics(r, x, voxel_ml) for v, x in variants.items()}}
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


def auroc(neg, pos):
    """P(a random FP piece has lower sep than a random hit piece) + half the ties (1 = sep separates them perfectly)."""
    if not neg or not pos:
        return float("nan")
    a, b = np.asarray(neg), np.asarray(pos)
    return float(((a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (a.size * b.size))


def tables(runs, cases, names):
    data = load_cache(runs, cases)
    comps = [x for d in data.values() for x in d["components"]]
    os.makedirs(OUT, exist_ok=True)
    if comps:
        with open(f"{OUT}/components.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comps[0]) + ["arm"])
            w.writeheader()
            for x in comps:
                w.writerow({**x, "arm": names[x["run"]]})
    with open(f"{OUT}/per_case.csv", "w", newline="") as fh:
        keys = list(next(iter(data.values()))["metrics"]["none"])
        w = csv.DictWriter(fh, fieldnames=["run", "arm", "case", "variant"] + keys)
        w.writeheader()
        for (run, c), d in data.items():
            for v in VARIANTS:
                w.writerow({"run": run, "arm": names[run], "case": c, "variant": v, **d["metrics"][v]})

    nc = len(cases)

    def col(run, v, key):
        return np.array([data[(run, c)]["metrics"][v][key] for c in cases], float)

    def tot(run, v, key):
        return col(run, v, key).sum()

    base = [b for b in BASELINE if b in runs]
    arms = [r for r in runs if r not in base]

    def base_col(v, key="tumour_dice"):
        b = np.array([col(x, v, key) for x in base])
        nb = (~np.isnan(b)).sum(axis=0)
        return np.where(nb > 0, np.nansum(b, axis=0) / np.maximum(nb, 1), np.nan)

    L = ["# The AUC rule applied to the predictions (ccAucPost) + a sweep of stricter cuts", "",
         f"{nc} real {REAL} {'val' if OUT.endswith('_val') else 'test'} cases, checkpoint_{CKPT} predictions. ccPost = training cc rule on the prediction; "
         f"ccAucPost<cut> = ccPost, then predicted tumour pieces inside the predicted main structure with sep < cut -> "
         f"their surroundings' label (filter_metrics' separability with the predicted {ORG} as organ). Decisions use "
         f"image + prediction only. **Main cut {MAIN_CUT}** (= the training cut, fixed before results); {SWEEP_TXT} "
         "= exploratory sweep. Written by `nnunet/auc_post.py`; exploratory, the main analysis stays predict_test.py.",
         "", "## 1. Does sep tell false-positive pieces from hits? (pieces inside the predicted main structure)", "",
         "| run | pieces / case (hit, FP) | unscored | sep median hit / FP | AUROC sep (FP lower) | "
         + " | ".join(f"< {c:g}: FP / hit pieces (hit ml)" for c in CUTS) + " |",
         "|---|---|---|---|---|" + "---|" * len(CUTS)]
    for run in runs:
        cs = [x for x in comps if x["run"] == run]
        sc = [x for x in cs if x["sep"] is not None]
        h = [x["sep"] for x in sc if x["hit"]]
        f = [x["sep"] for x in sc if not x["hit"]]
        cells = []
        for c in CUTS:
            fb = sum(s < c for s in f)
            hb = [x for x in sc if x["hit"] and x["sep"] < c]
            cells.append(f"{fb} / {len(hb)} ({sum(x['ml'] for x in hb):.1f})")
        med = lambda a: f"{np.median(a):.3f}" if a else "–"  # noqa: E731
        L.append(f"| {names[run]} | {len(cs) / nc:.1f} ({sum(x['hit'] for x in cs) / nc:.1f}, "
                 f"{sum(not x['hit'] for x in cs) / nc:.1f}) | {len(cs) - len(sc)} | {med(h)} / {med(f)} | "
                 f"{auroc(f, h):.2f} | " + " | ".join(cells) + " |")

    L += ["", "## 2. Each variant vs ccPost, per run", "",
          "Tumour Dice mean (NaN = both empty, left out) and the paired difference to ccPost; lesion recall, FP pieces "
          "and reference lesions lost are totals over the cases.", "",
          f"| run | variant | tumour Dice | vs ccPost [95% CI] | p | {ORG} Dice | lesion recall | FP pieces / case | "
          "lesions lost vs ccPost | pieces relabelled |", "|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        for v in VARIANTS:
            t = pt.paired(col(run, "ccPost", "tumour_dice"), col(run, v, "tumour_dice"))
            vs = "–" if v == "ccPost" else f"{pt.fmt_diff(t, 3)} | {t['p']:.2g}"
            if v == "ccPost":
                vs += " | –"
            lost = int(tot(run, "ccPost", "les_det") - tot(run, v, "les_det"))
            rel = sum(data[(run, c)]["relabelled"].get(v[len("ccAucPost"):], 0) for c in cases) \
                if v.startswith("ccAucPost") else "–"
            L.append(f"| {names[run]} | {v} | {np.nanmean(col(run, v, 'tumour_dice')):.3f} | {vs} | "
                     f"{np.nanmean(col(run, v, f'{ORG}_dice')):.3f} | {tot(run, v, 'les_det') / tot(run, v, 'les_ref'):.3f} | "
                     f"{tot(run, v, 'les_fp') / nc:.1f} | {lost if v != 'none' else '–'} | {rel} |")

    if base:
        bname = " + ".join(names[b] for b in base)
        L += ["", "## 3. Post-processing vs training (the question of this script)", "",
              f"Baseline = per-case mean of {bname}. y − x per case, paired bootstrap CI, Wilcoxon p (unadjusted).", "",
              "| comparison | x | y | y − x [95% CI] | p |", "|---|---|---|---|---|"]

        def row(label, x, y):
            r_ = pt.paired(x, y)
            L.append(f"| {label} | {r_['x']:.3f} | {r_['y']:.3f} | {pt.fmt_diff(r_, 3)} | {r_['p']:.2g} |")

        row(f"baseline: {MAIN} vs ccPost (what AUC post-processing adds)", base_col("ccPost"), base_col(MAIN))
        for a in arms:
            if pt.arm_name(*map(int, a.split(":"))).startswith("ccAucRepair"):
                row(f"{names[a]} ccPost − baseline {MAIN} (training vs post-processing; ≈ 0 = post-processing does it)",
                    base_col(MAIN), col(a, "ccPost", "tumour_dice"))
                row(f"{names[a]}: {MAIN} vs ccPost (AUC post on top of AUC training)",
                    col(a, "ccPost", "tumour_dice"), col(a, MAIN, "tumour_dice"))
        if arms:
            L += ["", f"Arms vs baseline when both get the same post-processing (Holm over the {len(arms)} arms listed):", "",
                  "| arm | variant | arm | baseline | diff [95% CI] | p | Holm p |", "|---|---|---|---|---|---|---|"]
            for v in ("ccPost", MAIN):
                res = [pt.paired(base_col(v), col(a, v, "tumour_dice")) for a in arms]
                hp = pt.holm(np.array([r_["p"] for r_ in res]))
                for a, r_, h_ in zip(arms, res, hp):
                    L.append(f"| {names[a]} | {v} | {r_['y']:.3f} | {r_['x']:.3f} | {pt.fmt_diff(r_, 3)} | "
                             f"{r_['p']:.2g} | {h_:.2g} |")

    free = [c for c in cases if data[(runs[0], c)]["ref_tumour_free"]]
    L += ["", f"Tumour-free test cases with no predicted tumour left ({len(free)} cases): " + "; ".join(
        f"{names[run]} " + " / ".join(str(sum(data[(run, c)]["metrics"][v]["les_pred"] == 0 for c in free))
                                      for v in VARIANTS) for run in runs) + f" (variants in the order {', '.join(VARIANTS)})."]
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    os.makedirs(PUB, exist_ok=True)
    for f in ("summary.md", "per_case.csv", "components.csv"):
        if os.path.exists(f"{OUT}/{f}"):
            shutil.copy(f"{OUT}/{f}", f"{PUB}/{f}")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("liver", "pancreas"), default="liver")
    ap.add_argument("--runs", nargs="+", help="dataset:fold (default: the 6 liver runs / the 2 pancreas baseline runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rules", default=None, help="filter_metrics rules.json (cut + version check)")
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--checkpoint", choices=("final", "best"), default="final",
                    help="which test predictions: fold_X/test_final (default) or fold_X/test_best")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    global OUT, PUB, CKPT
    set_target(a.target, a.split)
    a.runs = a.runs or RUNS
    if a.rules is None:
        a.rules = RULES
    check_rules(a.rules)
    CKPT = a.checkpoint
    if CKPT == "best":
        if a.target != "liver":
            sys.exit("--checkpoint best: liver only")
        if a.split == "val":
            sys.exit("--split val needs --checkpoint final (fold_X/validation holds checkpoint_final predictions)")
        OUT = OUT.replace("/test_eval_liver/", "/test_eval_liver_best/")
        PUB = PUB.replace("/test_eval_liver/", "/test_eval_liver_best/")
    if a.split == "val":
        OUT, PUB = OUT + "_val", PUB + "_val"
    sub, img_dir, lab_dir = ((f"test_{CKPT}", pt.IMG, pt.LAB) if a.split == "test"
                             else ("validation", f"{pt.RAW}/imagesTr", f"{pt.RAW}/labelsTr"))
    cases = [f"{PREFIX}_{i:03d}" for i in pt.split()[a.split]]
    names = {r: f"D{r.split(':')[0]} f{r.split(':')[1]} {pt.arm_name(*map(int, r.split(':')))}" for r in a.runs}
    if not a.tables_only:
        jobs = []
        for run in a.runs:
            did, fold = map(int, run.split(":"))
            pdir = f"{pt.res_dir(did)}/fold_{fold}/{sub}"
            cdir = f"{OUT}/cache/{run.replace(':', '_')}"
            os.makedirs(cdir, exist_ok=True)
            for c in cases:
                pred, cache = f"{pdir}/{c}.nii.gz", f"{cdir}/{c}.json"
                if not os.path.exists(pred):
                    sys.exit(f"missing prediction {pred}")
                if os.path.exists(cache) and json.load(open(cache)).get("version") == VERSION:
                    continue
                jobs.append((run, c, f"{img_dir}/{c}_0000.nii.gz", f"{lab_dir}/{c}.nii.gz", pred, cache))
        print(f"{a.split}: {len(jobs)} of {len(a.runs) * len(cases)} (run, case) pairs to measure, {a.workers} workers", flush=True)
        with Pool(a.workers) as pool:
            for k, (run, c) in enumerate(pool.imap_unordered(measure, jobs, chunksize=1), 1):
                if k % 25 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} done (last {run} {c})", flush=True)
    tables(a.runs, cases, names)


if __name__ == "__main__":
    main()
