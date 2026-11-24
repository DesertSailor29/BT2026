#!/usr/bin/env python
"""MISSION B comparisons for the confirmed liver rule tumTouch, using its already trained pre-training twin.

tumTouch on the 999 synthetic labels (nnunet/label_rule_counts.py, s10) removes the same tumour pieces as the
one-organ repair (598 volumes, 1659281 vs 1659548 tumour voxels); ccRepair (Dataset103 fold 0) also removes liver
pieces outside the main structure (58 more volumes). So ccRepair is the trained pre twin of tumTouch (tumour part
identical) and of ccPost. Declared in notes/AUTOPILOT.md s10 before this run, LiTS test (111), checkpoint_final,
baseline = per-case mean of 101:0 / 101:1, per-case tumour Dice (bootstrap 95 % CI, Wilcoxon p):
  P1  ccRepair as saved      vs baseline + tumTouch   (pre vs post; equivalent if the CI lies within ±0.02)
  P2  ccRepair + tumTouch    vs baseline + tumTouch   (does the pre arm add to the post rule?)
  P3  ccRepair as saved      vs baseline              (pre arm alone; known from the liver study)
  R1-R2 the same as P1/P2 with ccPost (reference; known from the liver study)
The frozen variants of post_rules_test_liver.py (unchanged, imported) are computed for 103:0 into its own cache dir;
the baseline caches are those of the A3L run (s9).

  python nnunet/pre_vs_post_liver.py --workers 12
  python nnunet/pre_vs_post_liver.py --cases lits_000 lits_001     # fake test (-> _subset)
-> reports/data/post_rules/liver_test_pre_vs_post/summary.md. Exploratory.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import post_rules_test_liver as ptl  # noqa: E402  (sets post_rules target liver)
from post_rules_test_liver import fl, pr, pt  # noqa: E402

PRE = "103:0"
BASE = ["101:0", "101:1"]
COMPARE = (("P1", "ccRepair as saved vs baseline + tumTouch", (PRE, "none"), ("base", "tumTouch")),
           ("P2", "ccRepair + tumTouch vs baseline + tumTouch", (PRE, "tumTouch"), ("base", "tumTouch")),
           ("P3", "ccRepair as saved vs baseline", (PRE, "none"), ("base", "none")),
           ("R1", "ccRepair as saved vs baseline + ccPost", (PRE, "none"), ("base", "ccPost")),
           ("R2", "ccRepair + ccPost vs baseline + ccPost", (PRE, "ccPost"), ("base", "ccPost")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="+", help="subset (fake test)")
    a = ap.parse_args()
    pt.set_target("liver")
    fl.ORG = "liver"
    assert pt.arm_name(103, 0) == "ccRepair", pt.arm_name(103, 0)
    cases = a.cases or sorted(f[: -len("_0000.nii.gz")] for f in os.listdir(pt.IMG) if f.endswith("_0000.nii.gz"))
    if not a.cases:
        assert len(cases) == 111, len(cases)
    params = json.load(open(f"{pr.PUB}/val_label_params.json"))
    hu = (params["hu_lo"], params["hu_hi"])
    bins = np.load(f"{pr.OUT}/pos_bins.npz")
    assert len(bins.files) == 20 and hu == (1.0, 210.0)
    support = pr.prior_support([bins[c] for c in bins.files])
    jobs = []
    for run in [PRE] + BASE:
        cdir = f"{ptl.OUT}/test/{run.replace(':', '_')}"
        os.makedirs(cdir, exist_ok=True)
        for c in cases:
            cache = f"{cdir}/{c}.json"
            if os.path.exists(cache) and json.load(open(cache)).get("version") == ptl.VERSION:
                continue
            assert c not in bins.files, c
            jobs.append((run, c, *ptl.paths("test", run, c), cache, hu, support))
    print(f"{len(jobs)} (run, case) pairs to measure, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for k, (run, c) in enumerate(pool.imap_unordered(ptl.measure, jobs, chunksize=1), 1):
            print(f"  {k}/{len(jobs)} {run} {c}", flush=True)

    data = {(run, c): json.load(open(f"{ptl.OUT}/test/{run.replace(':', '_')}/{c}.json"))["metrics"]
            for run in [PRE] + BASE for c in cases}

    def col(src, rule, key="tumour_dice"):
        if src == "base":
            return np.nanmean([[data[(r, c)][rule][key] for c in cases] for r in BASE], axis=0)
        return np.array([data[(src, c)][rule][key] for c in cases], float)

    lines = [f"# Pre vs post for tumTouch / ccPost: ccRepair (D103 f0) vs the baseline, liver test ({len(cases)} cases)",
             "", "Comparisons declared in `notes/AUTOPILOT.md` s10 before this run (`nnunet/pre_vs_post_liver.py`). "
             "Baseline = per-case mean of 101:0 / 101:1. b − a per case, bootstrap 95 % CI, Wilcoxon p. "
             "Equivalence (P1, R1): CI within ±0.02. Exploratory.", "",
             "| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | liver Dice b − a [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for cid, what, b, a_ in COMPARE:
        st = pt.paired(col(*a_), col(*b))
        sl = pt.paired(col(*a_, key="liver_dice"), col(*b, key="liver_dice"))
        eq = "yes" if -0.02 < st["lo"] and st["hi"] < 0.02 else "no"
        lines.append(f"| {cid} | {what} | {st['x']:.3f} | {st['y']:.3f} | {pt.fmt_diff(st, 3)} | {st['p']:.2g} | "
                     f"{eq} | {pt.fmt_diff(sl, 3)} |")
    pub = f"{pr.REPO}/reports/data/post_rules/liver_test_pre_vs_post" + ("_subset" if a.cases else "")
    os.makedirs(pub, exist_ok=True)
    open(f"{pub}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
