#!/usr/bin/env python
"""A3 (autopilot/MISSION.md): the frozen post-hoc rules on the liver predictions, one evaluation.

Rules, controls and parameters frozen in notes/AUTOPILOT.md (s8, 2026-11-24 21:25) after the liver val screen of
post_rules.py --target liver (whose functions this script reuses, unchanged) and before any test run of them. The
test result is evaluation, not selection: nothing here may be changed after reading it. (The pancreas twin is
post_rules_test.py, frozen in s6; kept separate so that frozen file stays untouched.)

Frozen rules (Holm over these 3, main metric per-case tumour Dice; definitions in post_rules.py):
  tumTouch    liver tumours lie in the liver: tumour pieces with no 26-neighbour predicted liver voxel -> bg
  pos         spine-relative 2D position prior of the real val liver labels; pieces < 50 % inside -> bg
  tissueVal   predicted liver / tumour voxels outside the real val labels' HU window (1..210 HU) -> bg
References and controls (reported, not in Holm):
  ccPost, ccTol10              the one-organ rule (its test effect is known from the liver study: a reference)
  ccPost+pos, ccPost+tissueTxt does anatomy add to the one-organ rule?
  tumNear10, tissueTxt         tolerant / textbook variants
  open1, rand_tumTouch, rand_pos   generic opening; random tumour-piece removal matched in count (seeded per case)
Declared contrasts (secondary): tumTouch − ccPost, ccPost+pos − ccPost, ccPost+tissueTxt − ccPost.
Parameters from the 20 real LiTS val labels only (reports/data/post_rules/liver_val/val_label_params.json, pos bins
cache of post_rules.py). On val the prior is leave-one-out (reproduces the screen); on test the union of all 20.

  python nnunet/post_rules_test_liver.py --split val  --workers 12    # sanity: reproduces the screen numbers
  python nnunet/post_rules_test_liver.py --split test --workers 12
-> reports/data/post_rules/liver_<split>_frozen/{summary.md, per_case.csv}. Exploratory.
"""
import argparse
import csv
import json
import os
import sys
import zlib
from multiprocessing import Pool

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import post_rules as pr  # noqa: E402
import predict_test as pt  # noqa: E402
import fp_location as fl  # noqa: E402
from cc_rules import main_structure  # noqa: E402
from build_dataset import real_case  # noqa: E402

VERSION = "liverfrozen1"
FROZEN = ("tumTouch", "pos", "tissueVal")
CONTROLS = ("ccPost", "ccTol10", "ccPost+pos", "ccPost+tissueTxt", "tumNear10", "tissueTxt", "open1",
            "rand_tumTouch", "rand_pos")
CONTRASTS = (("tumTouch", "ccPost"), ("ccPost+pos", "ccPost"), ("ccPost+tissueTxt", "ccPost"))
pr.set_target("liver")
OUT = f"{os.environ['nnUNet_results']}/test_eval_liver/post_rules_frozen"


def apply_frozen(p, img, spacing, aff, support, hu, seed):
    c = pr.spine_centre(img, spacing, aff)
    v = {"none": p, "ccPost": np.where(main_structure(p), p, 0).astype(np.uint8)}
    v["ccTol10"] = pr.cc_tol(p, spacing, 10.0)
    v["tumTouch"] = pr.tum_touch(p)[0]
    v["tumNear10"] = pr.tum_near(p, spacing, 10.0)[0]
    v["pos"] = pr.position(p, c, spacing, aff, support)
    v["tissueVal"] = pr.tissue(p, img, *hu)
    v["tissueTxt"] = pr.tissue(p, img, *pr.TXT_HU)
    v["ccPost+pos"] = pr.position(v["ccPost"], c, spacing, aff, support)
    v["ccPost+tissueTxt"] = pr.tissue(v["ccPost"], img, *pr.TXT_HU)
    n0 = pr.n_tumour_pieces(p)
    for r in ("tumTouch", "pos"):
        v[f"rand_{r}"] = pr.rand_remove(p, n0 - pr.n_tumour_pieces(v[r]), seed)
    v["open1"] = pr.opening(p)
    return v


def measure(args):
    run, case, img_f, ref_f, pred_f, cache_f, hu, support = args
    pnii = nib.load(pred_f)
    p = np.asarray(pnii.dataobj).astype(np.uint8)
    spacing = pr.spacing_of(pnii.affine)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    img = nib.load(img_f).get_fdata(dtype=np.float32)
    assert img.shape == p.shape, (img_f, img.shape, p.shape)
    seed = zlib.crc32(f"{run}/{case}".encode())
    variants = apply_frozen(p, img, spacing, pnii.affine, support, hu, seed)
    del img
    ref = nib.load(ref_f)  # the reference is read only after every rule has been applied
    r = np.asarray(ref.dataobj).astype(np.uint8)
    assert r.shape == p.shape and np.allclose(ref.affine, pnii.affine, atol=1e-3), ref_f
    out = {"version": VERSION, "run": run, "case": case, "voxel_ml": voxel_ml,
           "metrics": {k: {**fl.metrics(r, x, voxel_ml), "changed_vox": int((x != p).sum())}
                       for k, x in variants.items()}}
    tmp = cache_f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, cache_f)
    return run, case


def paths(split, run, case):
    did, fold = map(int, run.split(":"))
    fdir = f"{pt.res_dir(did)}/fold_{fold}"
    if split == "val":
        return (f"{pt.RAW}/imagesTr/{case}_0000.nii.gz", f"{pt.RAW}/labelsTr/{case}.nii.gz",
                f"{fdir}/validation/{case}.nii.gz")
    return f"{pt.IMG}/{case}_0000.nii.gz", f"{pt.LAB}/{case}.nii.gz", f"{fdir}/{pt.PRED_SUB}/{case}.nii.gz"


def tables(split, runs, cases, rules, pub):
    data = {(run, c): json.load(open(f"{OUT}/{split}/{run.replace(':', '_')}/{c}.json")) for run in runs for c in cases}
    assert all(d["version"] == VERSION for d in data.values())
    os.makedirs(pub, exist_ok=True)
    keys = list(data[(runs[0], cases[0])]["metrics"]["none"].keys())
    with open(f"{pub}/per_case.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "case", "rule"] + keys)
        for (run, c), d in data.items():
            for r in ["none"] + rules:
                w.writerow([run, c, r] + [d["metrics"][r][k] for k in keys])
    names = {r: f"D{r.split(':')[0]} f{r.split(':')[1]} {pt.arm_name(*map(int, r.split(':')))}" for r in runs}
    names["mean"] = "mean of the 2 runs"

    def col(rule, key, run):
        if run == "mean":
            return np.nanmean([[data[(r, c)]["metrics"][rule][key] for c in cases] for r in runs], axis=0)
        return np.array([data[(run, c)]["metrics"][rule][key] for c in cases], float)

    allr = list(runs) + ["mean"]
    res = {(rule, run): (pt.paired(col("none", "tumour_dice", run), col(rule, "tumour_dice", run)),
                         pt.paired(col("none", "liver_dice", run), col(rule, "liver_dice", run)))
           for rule in rules for run in allr}
    holm = {}
    for run in allr:
        for r, a in zip(FROZEN, pt.holm(np.nan_to_num([res[(r, run)][0]["p"] for r in FROZEN], nan=1.0))):
            holm[(r, run)] = a
    lines = [f"# A3: frozen post-hoc rules, liver {split} ({len(cases)} cases)", "",
             "Rules and controls frozen in `notes/AUTOPILOT.md` (s8) before this run; definitions `nnunet/post_rules.py`,"
             " `nnunet/post_rules_test_liver.py`. Each rule vs the saved prediction (`none`), per-case paired difference "
             "(bootstrap 95 % CI, Wilcoxon p, Holm over the 3 frozen rules, per run and for the per-case mean of the two "
             "runs). Exploratory.", "",
             "| rule | kind | run | tumour Dice | vs none [95% CI] | p | Holm p | liver Dice | vs none [95% CI] | "
             "lesion recall | FP pieces / case |", "|---|---|---|---|---|---|---|---|---|---|---|"]

    def recfp(rule, run):
        if run == "mean":
            return "–", "–"
        m = [data[(run, c)]["metrics"][rule] for c in cases]
        ref = sum(x["les_ref"] for x in m)
        return (f"{sum(x['les_det'] for x in m) / ref:.3f}" if ref else "–"), f"{np.mean([x['les_fp'] for x in m]):.1f}"

    for run in allr:
        st, sg = res[(rules[0], run)]
        rec, fp = recfp("none", run)
        lines.append(f"| none | – | {names[run]} | {st['x']:.3f} | – | – | – | {sg['x']:.3f} | – | {rec} | {fp} |")
    for rule in rules:
        kind = "frozen" if rule in FROZEN else "control"
        for run in allr:
            st, sg = res[(rule, run)]
            hp = f"{holm[(rule, run)]:.2g}" if (rule, run) in holm else ""
            rec, fp = recfp(rule, run)
            lines.append(f"| {rule} | {kind} | {names[run]} | {st['y']:.3f} | {pt.fmt_diff(st, 3)} | {st['p']:.2g} | "
                         f"{hp} | {sg['y']:.3f} | {pt.fmt_diff(sg, 3)} | {rec} | {fp} |")
    lines += ["", "Declared contrasts (a − b, per case).", "",
              "| contrast | run | tumour Dice diff [95% CI] | p | liver Dice diff [95% CI] | p |",
              "|---|---|---|---|---|---|"]
    for a, b in CONTRASTS:
        for run in allr:
            st = pt.paired(col(b, "tumour_dice", run), col(a, "tumour_dice", run))
            sg = pt.paired(col(b, "liver_dice", run), col(a, "liver_dice", run))
            lines.append(f"| {a} − {b} | {names[run]} | {pt.fmt_diff(st, 3)} | {st['p']:.2g} | "
                         f"{pt.fmt_diff(sg, 3)} | {sg['p']:.2g} |")
    open(f"{pub}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--runs", nargs="+", default=pr.RUNS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="+", help="subset (fake test)")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    pt.set_target("liver")
    fl.ORG = "liver"
    val = [real_case("liver", i) for i in pt.split()["val"]]
    if a.cases:
        cases = a.cases
    elif a.split == "val":
        cases = val
    else:
        cases = sorted(f[: -len("_0000.nii.gz")] for f in os.listdir(pt.IMG) if f.endswith("_0000.nii.gz"))
    rules = list(FROZEN + CONTROLS)
    pub = f"{pr.REPO}/reports/data/post_rules/liver_{a.split}_frozen" + ("_subset" if a.cases else "")
    if not a.tables_only:
        params = json.load(open(f"{pr.PUB}/val_label_params.json"))
        hu = (params["hu_lo"], params["hu_hi"])
        bins = np.load(f"{pr.OUT}/pos_bins.npz")
        bin_sets = {c: bins[c] for c in bins.files}
        assert len(bin_sets) == 20 and params["cases"] == 20 and hu == (1.0, 210.0), (len(bin_sets), params)
        full = pr.prior_support(list(bin_sets.values()))
        jobs = []
        for run in a.runs:
            cdir = f"{OUT}/{a.split}/{run.replace(':', '_')}"
            os.makedirs(cdir, exist_ok=True)
            for c in cases:
                img_f, ref_f, pred_f = paths(a.split, run, c)
                cache = f"{cdir}/{c}.json"
                if os.path.exists(cache) and json.load(open(cache)).get("version") == VERSION:
                    continue
                if a.split == "test":
                    assert c not in bin_sets, c
                support = pr.prior_support([b for k, b in bin_sets.items() if k != c]) if c in bin_sets else full
                jobs.append((run, c, img_f, ref_f, pred_f, cache, hu, support))
        print(f"{len(jobs)} (run, case) pairs to measure, {a.workers} workers", flush=True)
        with Pool(a.workers) as pool:
            for k, (run, c) in enumerate(pool.imap_unordered(measure, jobs, chunksize=1), 1):
                print(f"  {k}/{len(jobs)} {run} {c}", flush=True)
    tables(a.split, a.runs, cases, rules, pub)


if __name__ == "__main__":
    main()
