#!/usr/bin/env python
"""MISSION B comparisons for the confirmed pancreas rule minLes, using its pre-training twin minLesRepair.

minLesRepair = Dataset114 (build_rule_repair_dataset.py --rule minLes: tumour pieces < 1.19 ml, the smallest real
MSD07 val lesion, relabelled to background in the 999 synthetic labels; 131 volumes change, all become tumour-free).
Declared in notes/AUTOPILOT.md s14 before any count or training, MSD07 test (238), checkpoint_final, baseline =
per-case mean of 102:0 / 102:1, per-case tumour Dice (bootstrap 95 % CI, Wilcoxon p), pancreas Dice reported:
  M1  minLesRepair as saved            vs baseline + minLes            (pre vs post; equivalent if CI within ±0.02)
  M2  minLesRepair + grow0.1+minLes    vs baseline + grow0.1+minLes    (= F2 on both: does the pre arm add?)
  M3  minLesRepair as saved            vs baseline
References (not declared): the same with minLes alone (arm + minLes vs baseline + minLes) and thr0.1+minLes.
The frozen variants of post_rules_test.py (unchanged, imported) are computed for the arm into its own cache dir
(same VERSION); the baseline caches are those of the A3b run (s14). M2 needs the arm's test probabilities
(pancreas_arm.sh with PANCREAS_TEST_PROBS=1 writes fold_X/test_probs/); without them --no-probs gives M1/M3 only.

  python nnunet/pre_vs_post_pancreas.py --pre 114:0 --workers 12
  python nnunet/pre_vs_post_pancreas.py --pre 102:0 --arm all --cases msd_001 msd_004 --workers 2   # fake test
-> reports/data/post_rules/pancreas_test_pre_vs_post_minLes/summary.md (_subset for --cases). Exploratory.
Second run / 2-run mean (s21 declaration: the claim pre != post stands only if M1 on the mean excludes 0 and both
runs have the same sign); --pre other than 114:0 writes to ..._minLes_<runs>/:
  python nnunet/pre_vs_post_pancreas.py --pre 114:1 --workers 12
  python nnunet/pre_vs_post_pancreas.py --pre 114:0,114:1 --workers 12
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import post_rules_test as prt  # noqa: E402
from post_rules_test import fl, pr, pt  # noqa: E402

BASE = ["102:0", "102:1"]


def compare(arm, probs):
    c = [("M1", f"{arm} as saved vs baseline + minLes", ("pre", "none"), ("base", "minLes")),
         ("M2", f"{arm} + grow0.1+minLes vs baseline + grow0.1+minLes", ("pre", "grow0.1+minLes"),
          ("base", "grow0.1+minLes")),
         ("M3", f"{arm} as saved vs baseline", ("pre", "none"), ("base", "none")),
         ("R1", f"{arm} + minLes vs baseline + minLes (reference)", ("pre", "minLes"), ("base", "minLes")),
         ("R2", f"{arm} + thr0.1+minLes vs baseline + thr0.1+minLes (reference)", ("pre", "thr0.1+minLes"),
          ("base", "thr0.1+minLes"))]
    return [x for x in c if probs or not (x[2][1] in prt.PROB or x[3][1] in prt.PROB)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", required=True, help="dataset:fold of the pre arm, e.g. 114:0; several comma-separated "
                    "= per-case mean of the runs")
    ap.add_argument("--arm", default="minLesRepair", help="expected arm name of --pre (arms.json)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="+", help="subset (fake test)")
    ap.add_argument("--no-probs", action="store_true", help="M1/M3 only (no grow/thr variants)")
    a = ap.parse_args()
    pt.set_target("pancreas")
    fl.ORG = "pancreas"
    pres = a.pre.split(",")
    did, fold = map(int, pres[0].split(":"))
    if a.arm != "all":
        for p in pres:  # rep2 runs are registered as <arm>_rep2
            n = pt.arm_name(*map(int, p.split(":")))
            assert n in (a.arm, a.arm + "_rep2"), (p, n)
    cases = a.cases or sorted(f[: -len("_0000.nii.gz")] for f in os.listdir(pt.IMG) if f.endswith("_0000.nii.gz"))
    if not a.cases:
        assert len(cases) == 238, len(cases)
    rules = [r for r in prt.FROZEN + prt.CONTROLS if not (a.no_probs and r in prt.PROB)]
    params = json.load(open(f"{pr.PUB}/val_label_params.json"))
    bins = np.load(f"{pr.OUT}/pos_bins.npz")
    assert len(bins.files) == 43 and abs(params["min_lesion_ml"] - 1.19) < 0.01, (len(bins.files), params)
    support = pr.prior_support([bins[c] for c in bins.files])
    jobs = []
    for run in pres + BASE:
        cdir = f"{prt.OUT}/test/{run.replace(':', '_')}"
        os.makedirs(cdir, exist_ok=True)
        for c in cases:
            cache = f"{cdir}/{c}.json"
            if os.path.exists(cache):
                d = json.load(open(cache))
                if d.get("version") == prt.VERSION and all(r in d["metrics"] for r in rules):
                    continue
            assert run in pres, f"baseline cache missing or old: {cache} (run post_rules_test.py --split test)"
            assert c not in bins.files, c
            img_f, ref_f, pred_f, prob_f = prt.paths("test", run, c)
            prob_f = None if a.no_probs else prob_f
            if prob_f and not os.path.exists(prob_f):
                sys.exit(f"missing {prob_f} (PANCREAS_TEST_PROBS=1 in pancreas_arm.sh, or --no-probs)")
            jobs.append((run, c, img_f, ref_f, pred_f, prob_f, cache, params["min_lesion_ml"], support))
    print(f"{len(jobs)} (run, case) pairs to measure, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for k, (run, c) in enumerate(pool.imap_unordered(prt.measure, jobs, chunksize=1), 1):
            print(f"  {k}/{len(jobs)} {run} {c}", flush=True)

    data = {(run, c): json.load(open(f"{prt.OUT}/test/{run.replace(':', '_')}/{c}.json"))["metrics"]
            for run in pres + BASE for c in cases}

    def col(src, rule, key="tumour_dice"):
        runs = BASE if src == "base" else pres
        return np.nanmean([[data[(r, c)][rule][key] for c in cases] for r in runs], axis=0)

    what = f"D{did} f{fold}" if len(pres) == 1 else "per-case mean of " + " / ".join(pres)
    lines = [f"# Pre vs post for minLes: {a.arm} ({what}) vs the baseline, MSD07 test ({len(cases)} cases)",
             "", "Comparisons declared in `notes/AUTOPILOT.md` s14 before any count or training "
             "(`nnunet/pre_vs_post_pancreas.py`). Baseline = per-case mean of 102:0 / 102:1. b − a per case, "
             "bootstrap 95 % CI, Wilcoxon p. Equivalence (M1): CI within ±0.02. R rows are references, not "
             "declared. Exploratory.", "",
             "| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | pancreas Dice b − a [95% CI] "
             "| lesion recall a → b |",
             "|---|---|---|---|---|---|---|---|---|"]

    def recall(src, rule):
        return sum(np.mean([data[(r, c)][rule]["les_det"] for r in (BASE if src == "base" else pres)])
                   for c in cases) / max(1, sum(data[(BASE[0], c)]["none"]["les_ref"] for c in cases))

    for cid, what, b, a_ in compare(a.arm, not a.no_probs):
        a2 = a_
        st = pt.paired(col(*a2), col(*b))
        sp = pt.paired(col(*a2, key="pancreas_dice"), col(*b, key="pancreas_dice"))
        eq = "yes" if -0.02 < st["lo"] and st["hi"] < 0.02 else "no"
        lines.append(f"| {cid} | {what} | {st['x']:.3f} | {st['y']:.3f} | {pt.fmt_diff(st, 3)} | {st['p']:.2g} | "
                     f"{eq} | {pt.fmt_diff(sp, 3)} | {recall(*a2):.3f} → {recall(*b):.3f} |")
    tag = "" if pres == ["114:0"] else "_" + "_".join(p.replace(":", "f") for p in pres)
    pub = f"{pr.REPO}/reports/data/post_rules/pancreas_test_pre_vs_post_minLes{tag}" + ("_subset" if a.cases else "")
    os.makedirs(pub, exist_ok=True)
    open(f"{pub}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
