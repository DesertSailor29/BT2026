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

posRepair (Dataset113, pos repair of the 999 labels; comparisons declared in s10 before training):
  Q1  posRepair as saved     vs baseline + pos        (pre vs post; equivalent if the CI lies within ±0.02)
  Q2  posRepair + pos        vs baseline + pos        (does the pre arm add to the post rule?)
  Q3  posRepair as saved     vs baseline
  R1-R2 the same with ccPost (reference, not declared)
  python nnunet/pre_vs_post_liver.py --pre 113:0 --rule pos --workers 12
-> reports/data/post_rules/liver_test_pre_vs_post_pos/summary.md.
Second run / 2-run mean (s20 declaration: the claim pre != post stands only if Q1 on the mean excludes 0 and both
runs have the same sign):
  python nnunet/pre_vs_post_liver.py --rule pos --pre 113:1 --workers 12        # -> ..._pos_113f1/
  python nnunet/pre_vs_post_liver.py --rule pos --pre 113:0,113:1 --workers 12  # -> ..._pos_113f0_113f1/
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

BASE = ["101:0", "101:1"]
ARMS = {"tumTouch": ("103:0", "ccRepair", "P", "s10"), "pos": ("113:0", "posRepair", "Q", "s10")}


def compare(rule, pre=None):
    """The declared comparisons for one rule's pre arm (tumTouch: the original P1-P3/R1-R2, unchanged).
    pre: the pre-arm column key (default the rule's run in ARMS; "pre" = per-case mean over --pre runs)."""
    if rule == "tumTouch" and pre is None:
        return COMPARE
    _, arm, k, _ = ARMS[rule]
    pre = pre or ARMS[rule][0]
    return ((f"{k}1", f"{arm} as saved vs baseline + {rule}", (pre, "none"), ("base", rule)),
            (f"{k}2", f"{arm} + {rule} vs baseline + {rule}", (pre, rule), ("base", rule)),
            (f"{k}3", f"{arm} as saved vs baseline", (pre, "none"), ("base", "none")),
            ("R1", f"{arm} as saved vs baseline + ccPost (reference)", (pre, "none"), ("base", "ccPost")),
            ("R2", f"{arm} + ccPost vs baseline + ccPost (reference)", (pre, "ccPost"), ("base", "ccPost")))


PRE = "103:0"
COMPARE = (("P1", "ccRepair as saved vs baseline + tumTouch", (PRE, "none"), ("base", "tumTouch")),
           ("P2", "ccRepair + tumTouch vs baseline + tumTouch", (PRE, "tumTouch"), ("base", "tumTouch")),
           ("P3", "ccRepair as saved vs baseline", (PRE, "none"), ("base", "none")),
           ("R1", "ccRepair as saved vs baseline + ccPost", (PRE, "none"), ("base", "ccPost")),
           ("R2", "ccRepair + ccPost vs baseline + ccPost", (PRE, "ccPost"), ("base", "ccPost")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="+", help="subset (fake test)")
    ap.add_argument("--rule", choices=sorted(ARMS), default="tumTouch", help="post rule whose pre arm is compared")
    ap.add_argument("--pre", help="dataset:fold of the pre arm (default: the rule's arm in ARMS); several runs "
                    "comma-separated (e.g. 113:0,113:1 = per-case mean of both, s20 declaration) -> own output dir")
    a = ap.parse_args()
    pre0, arm, _, decl = ARMS[a.rule]
    pres = (a.pre or pre0).split(",")
    pt.set_target("liver")
    fl.ORG = "liver"
    for p in pres:  # rep2 runs are registered as <arm>_rep2
        n = pt.arm_name(*map(int, p.split(":")))
        assert n in (arm, arm + "_rep2"), (p, n)
    single = pres == [pre0]
    pre = pres[0]
    cases = a.cases or sorted(f[: -len("_0000.nii.gz")] for f in os.listdir(pt.IMG) if f.endswith("_0000.nii.gz"))
    if not a.cases:
        assert len(cases) == 111, len(cases)
    params = json.load(open(f"{pr.PUB}/val_label_params.json"))
    hu = (params["hu_lo"], params["hu_hi"])
    bins = np.load(f"{pr.OUT}/pos_bins.npz")
    assert len(bins.files) == 20 and hu == (1.0, 210.0)
    support = pr.prior_support([bins[c] for c in bins.files])
    jobs = []
    for run in pres + BASE:
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
            for run in pres + BASE for c in cases}

    def col(src, rule, key="tumour_dice"):
        if src in ("base", "pre"):
            runs = BASE if src == "base" else pres
            return np.nanmean([[data[(r, c)][rule][key] for c in cases] for r in runs], axis=0)
        return np.array([data[(src, c)][rule][key] for c in cases], float)

    cmp = compare(a.rule) if single else compare(a.rule, "pre" if len(pres) > 1 else pre)
    what = f"D{pre.replace(':', ' f')}" if len(pres) == 1 else "per-case mean of " + " / ".join(pres)
    lines = [f"# Pre vs post for {a.rule} / ccPost: {arm} ({what}) vs the baseline, liver test "
             f"({len(cases)} cases)",
             "", f"Comparisons declared in `notes/AUTOPILOT.md` {decl} before this run (`nnunet/pre_vs_post_liver.py`). "
             "Baseline = per-case mean of 101:0 / 101:1. b − a per case, bootstrap 95 % CI, Wilcoxon p. "
             f"Equivalence ({cmp[0][0]}, R1): CI within ±0.02. Exploratory.", "",
             "| id | comparison | a | b | tumour Dice b − a [95% CI] | p | within ±0.02 | liver Dice b − a [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for cid, what, b, a_ in cmp:
        st = pt.paired(col(*a_), col(*b))
        sl = pt.paired(col(*a_, key="liver_dice"), col(*b, key="liver_dice"))
        eq = "yes" if -0.02 < st["lo"] and st["hi"] < 0.02 else "no"
        lines.append(f"| {cid} | {what} | {st['x']:.3f} | {st['y']:.3f} | {pt.fmt_diff(st, 3)} | {st['p']:.2g} | "
                     f"{eq} | {pt.fmt_diff(sl, 3)} |")
    pub = (f"{pr.REPO}/reports/data/post_rules/liver_test_pre_vs_post" + ("" if a.rule == "tumTouch" else f"_{a.rule}")
           + ("" if single else "_" + "_".join(p.replace(":", "f") for p in pres))
           + ("_subset" if a.cases else ""))
    os.makedirs(pub, exist_ok=True)
    open(f"{pub}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
