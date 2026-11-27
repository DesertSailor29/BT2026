#!/usr/bin/env python
"""Liver L3 eval (autopilot/MISSION.md "Liver extension" L3): tumour-size arms (Dataset104 subsets of the ccAucRepair
labels, nnunet/size_split.py) on the 111 LiTS test cases, checkpoint_final, saved predictions and with ccPost.
CPU only. Written in s33 (2026-11-27), before any L3 result; the comparisons below are the ones declared in
MISSION.md L3 on 2026-11-26 and are fixed here, nothing is chosen from the output.

Groups (runs found by arm name in the dataset's arms.json; a run counts only once its 111 test predictions exist,
so the script can be re-run after every arm):
  baseline    D101 all, all_rep2                      ccAucRepair  D104 ccAucRepair, ccAucRepair_rep2
  sizeLarge   D104 ccAucRepair_sizeLarge(_rep2)       sizeRand     D104 ccAucRepair_sizeRand(_rep2)
  sizeSmall   D104 ccAucRepair_sizeSmall(_rep2)
Per case, a group's score = mean over its available runs (as the baseline in predict_test.py).
Comparisons (per-case tumour Dice, paired bootstrap 95 % CI + Wilcoxon, predict_test.paired), saved and ccPost:
  primary    sizeLarge vs sizeRand: "large tumours help" if diff > +0.01, CI > 0 and both groups have 2 runs
  secondary  sizeSmall vs sizeRand; each size arm vs ccAucRepair and vs baseline; every single size run vs baseline
Lesion recall / mean coverage by real lesion size bin (bins of liver_failure.py L1), per group (per lesion: share of
the group's runs that detect it, then averaged) and per run: do large-tumour arms find the large lesions?

  python nnunet/size_eval.py --workers 12          (resumable: per-run per-case caches)
  python nnunet/size_eval.py --cases lits_000 ... --out /tmp/x   (fake test)
-> $nnUNet_results/test_eval_liver/size/{cache/, cases.csv, summary.md}, summary + CSV copied to
   reports/data/test_eval_liver/size/.
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
from cc_rules import main_structure  # noqa: E402  (ccPost, as liver_failure.py)
from liver_failure import SIZE_BINS, case_scores, nanmean  # noqa: E402

VERSION = "size1"
GROUPS = {
    "baseline": [(101, "all"), (101, "all_rep2")],
    "ccAucRepair": [(104, "ccAucRepair"), (104, "ccAucRepair_rep2")],
    "sizeLarge": [(104, "ccAucRepair_sizeLarge"), (104, "ccAucRepair_sizeLarge_rep2")],
    "sizeRand": [(104, "ccAucRepair_sizeRand"), (104, "ccAucRepair_sizeRand_rep2")],
    "sizeSmall": [(104, "ccAucRepair_sizeSmall"), (104, "ccAucRepair_sizeSmall_rep2")],
}
SIZE_ARMS = ["sizeLarge", "sizeRand", "sizeSmall"]
VARIANTS = ("none", "ccPost")
OUT = f"{os.environ['nnUNet_results']}/test_eval_liver/size"
PUB = f"{REPO}/reports/data/test_eval_liver/size"


def find_runs(n_cases, cases):
    """{group: [(run key 'did:fold', arm name)]} for runs whose test predictions are all there."""
    found = {}
    for g, members in GROUPS.items():
        found[g] = []
        for did, arm in members:
            arms_f = f"{os.environ['nnUNet_preprocessed']}/{pt.dataset_name(did)}/arms.json"
            folds = [i for i, a in enumerate(json.load(open(arms_f))) if a["arm"] == arm]
            if not folds:
                continue
            pdir = f"{pt.res_dir(did)}/fold_{folds[0]}/{pt.PRED_SUB}"
            if all(os.path.exists(f"{pdir}/{c}.nii.gz") for c in cases):
                found[g].append((f"{did}:{folds[0]}", arm))
            else:
                print(f"{g}: {arm} (D{did} f{folds[0]}) has no complete test predictions yet, skipped", flush=True)
    return found


def measure(args):
    case, run, cache_f = args
    ref = nib.load(f"{pt.LAB}/{case}.nii.gz")
    r = np.asarray(ref.dataobj).astype(np.uint8)
    voxel_ml = float(np.prod(np.sqrt((ref.affine[:3, :3] ** 2).sum(axis=0)))) / 1000.0
    rl, nr = ndimage.label(r == 2, pt.STRUCT26)
    rsize = np.bincount(rl.ravel(), minlength=nr + 1)[1:]
    did, fold = map(int, run.split(":"))
    p = np.asarray(nib.load(f"{pt.res_dir(did)}/fold_{fold}/{pt.PRED_SUB}/{case}.nii.gz").dataobj).astype(np.uint8)
    assert p.shape == r.shape, (case, run, p.shape, r.shape)
    out = {"version": VERSION, "case": case, "run": run, "tumour_ml": round(float((r == 2).sum()) * voxel_ml, 3),
           "lesion_ml": [round(float(s) * voxel_ml, 4) for s in rsize], "scores": {}, "coverage": {}}
    for var in VARIANTS:
        q = p if var == "none" else np.where(main_structure(p), p, 0).astype(np.uint8)
        out["scores"][var] = case_scores(r, q, voxel_ml)
        hit = np.bincount(rl[(rl > 0) & (q == 2)], minlength=nr + 1)[1:] if nr else np.zeros(0)
        out["coverage"][var] = [round(float(h) / float(s), 4) for h, s in zip(hit, rsize)]
    tmp = cache_f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, cache_f)
    return case, run


def summarise(data, runs, cases, out_dir):
    """data[run][case] = cache dict; runs = find_runs output."""
    groups = {g: [k for k, _ in v] for g, v in runs.items() if v}
    name = {k: a for v in runs.values() for k, a in v}

    def per_case(g, var, metric="tumour_dice"):
        return [nanmean([data[k][c]["scores"][var][metric] for k in groups[g]]) for c in cases]

    rows = []
    for g, ks in groups.items():
        for k in ks:
            for c in cases:
                d = data[k][c]
                rows.append({"group": g, "arm": name[k], "run": k, "case": c, "case_tumour_ml": d["tumour_ml"],
                             **{f"{m}_{v}": d["scores"][v][m] for v in VARIANTS
                                for m in ("tumour_dice", "liver_dice", "les_ref", "les_det", "les_fp")}})
    with open(f"{out_dir}/cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    L = [f"# Liver tumour-size arms (L3, `nnunet/size_eval.py`, {len(cases)} LiTS test cases, checkpoint_final)", "",
         "Exploratory. Comparisons declared in autopilot/MISSION.md L3 (2026-11-26) before any result. Group score per "
         "case = mean of its available runs. Variant ccPost = one-structure rule on the prediction.", "",
         "| group | runs |", "|---|---|"]
    L += [f"| {g} | " + (", ".join(f"{name[k]} ({k})" for k in groups[g]) if g in groups else "none yet") + " |"
          for g in GROUPS]
    L += ["", "## 1. Mean per-case tumour Dice (liver Dice, lesion recall, FP pieces / case)", "",
          "| group / run | variant | tumour Dice | liver Dice | lesion recall | FP pieces / case |", "|---|---|---|---|---|---|"]

    def line(label, ks, var):
        def m(metric):
            return np.mean([nanmean([data[k][c]["scores"][var][metric] for k in ks]) for c in cases])
        rec = sum(data[k][c]["scores"][var]["les_det"] for k in ks for c in cases) / max(1, sum(
            data[k][c]["scores"][var]["les_ref"] for k in ks for c in cases))
        return f"| {label} | {var} | {m('tumour_dice'):.4f} | {m('liver_dice'):.4f} | {rec:.3f} | {m('les_fp'):.1f} |"

    for g, ks in groups.items():
        for var in VARIANTS:
            L.append(line(f"**{g}** (mean of {len(ks)})", ks, var))
        if len(ks) > 1:
            for k in ks:
                for var in VARIANTS:
                    L.append(line(f"{name[k]}", [k], var))

    L += ["", "## 2. Declared comparisons (per-case tumour Dice, B − A, 95 % bootstrap CI, Wilcoxon p)", "",
          "| role | B vs A | runs B / A | variant | A | B | diff [CI] | p | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    pairs = [("primary", "sizeLarge", "sizeRand"), ("secondary", "sizeSmall", "sizeRand")]
    pairs += [("secondary", s, ref) for s in SIZE_ARMS for ref in ("ccAucRepair", "baseline")]
    for role, b, a in pairs:
        if b not in groups or a not in groups:
            L.append(f"| {role} | {b} vs {a} | {len(groups.get(b, []))} / {len(groups.get(a, []))} | – | – | – | not yet | – | – |")
            continue
        for var in VARIANTS:
            s = pt.paired(per_case(a, var), per_case(b, var))
            verdict = ""
            if role == "primary":
                ok = s["diff"] > 0.01 and s["lo"] > 0
                verdict = ("**large tumours help**" if ok and len(groups[a]) >= 2 and len(groups[b]) >= 2 else
                           ("meets size of effect, but < 2 runs per group" if ok else "not met"))
            L.append(f"| {role} | {b} vs {a} | {len(groups[b])} / {len(groups[a])} | {var} | {s['x']:.4f} | {s['y']:.4f} "
                     f"| {pt.fmt_diff(s)} | {s['p']:.2g} | {verdict} |")
    L += ["", "Single size runs vs the baseline (descriptive; runs of the same arm varied 0.02-0.04 before):", "",
          "| run | variant | baseline | run | diff [CI] | p |", "|---|---|---|---|---|---|"]
    if "baseline" in groups:
        for g in SIZE_ARMS:
            for k in groups.get(g, []):
                for var in VARIANTS:
                    x = per_case("baseline", var)
                    y = [data[k][c]["scores"][var]["tumour_dice"] for c in cases]
                    s = pt.paired(x, y)
                    L.append(f"| {name[k]} ({k}) | {var} | {s['x']:.4f} | {s['y']:.4f} | {pt.fmt_diff(s)} | {s['p']:.2g} |")

    lesion_ml = [ml for c in cases for ml in data[next(iter(data))][c]["lesion_ml"]]
    L += ["", "## 3. Lesion recall / mean coverage by real lesion size (per group: per-lesion share of its runs)", "",
          ("| group / run | variant | " + " | ".join(f"{a:g}-{b:g} ml" for a, b in SIZE_BINS) + " |").replace("-inf ml", "+ ml"),
          "|---|---" + "|---" * len(SIZE_BINS) + "|",
          "| n lesions | | " + " | ".join(str(sum(1 for x in lesion_ml if a <= x < b)) for a, b in SIZE_BINS) + " |"]

    def bins(ks, var):
        det = np.mean([[x > 0 for c in cases for x in data[k][c]["coverage"][var]] for k in ks], axis=0)
        cov = np.mean([[x for c in cases for x in data[k][c]["coverage"][var]] for k in ks], axis=0)
        ml = np.asarray(lesion_ml)
        return " | ".join((f"{det[(ml >= a) & (ml < b)].mean():.2f} / {cov[(ml >= a) & (ml < b)].mean():.2f}"
                           if ((ml >= a) & (ml < b)).any() else "-") for a, b in SIZE_BINS)

    for g, ks in groups.items():
        for var in VARIANTS:
            L.append(f"| **{g}** | {var} | {bins(ks, var)} |")
        if len(ks) > 1 and g in SIZE_ARMS:
            for k in ks:
                for var in VARIANTS:
                    L.append(f"| {name[k]} | {var} | {bins([k], var)} |")
    L += ["", "Declared (MISSION.md L3, 2026-11-26): \"large tumours help\" if sizeLarge − sizeRand > +0.01 with the CI "
          "above 0, on 2 runs each; everything else is secondary / descriptive."]
    open(f"{out_dir}/summary.md", "w").write("\n".join(L) + "\n")
    return "\n".join(L)


def main():
    global OUT, PUB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", help="only these cases (fake test); default: the 111 test cases")
    ap.add_argument("--out", help="output dir (default $nnUNet_results/test_eval_liver/size); no copy to reports/")
    a = ap.parse_args()
    cases = a.cases or [f"lits_{i:03d}" for i in pt.split()["test"]]
    if a.out:
        OUT, PUB = a.out, None
    runs = find_runs(len(cases), cases)
    keys = sorted({k for v in runs.values() for k, _ in v})
    jobs = []
    for k in keys:
        os.makedirs(f"{OUT}/cache/{k.replace(':', '_')}", exist_ok=True)
        for c in cases:
            f = f"{OUT}/cache/{k.replace(':', '_')}/{c}.json"
            try:
                if json.load(open(f))["version"] == VERSION:
                    continue
            except (OSError, ValueError, KeyError):
                pass
            jobs.append((c, k, f))
    print(f"{len(cases)} cases, runs {keys}, {len(jobs)} case-runs to measure", flush=True)
    with Pool(a.workers) as p:
        for i, (c, k) in enumerate(p.imap_unordered(measure, jobs), 1):
            print(f"[{i}/{len(jobs)}] {k} {c}", flush=True)
    data = {k: {c: json.load(open(f"{OUT}/cache/{k.replace(':', '_')}/{c}.json")) for c in cases} for k in keys}
    print(summarise(data, runs, cases, OUT))
    if PUB:
        os.makedirs(PUB, exist_ok=True)
        for n in ("summary.md", "cases.csv"):
            shutil.copy(f"{OUT}/{n}", f"{PUB}/{n}")
        print(f"copied to {PUB}")


if __name__ == "__main__":
    main()
