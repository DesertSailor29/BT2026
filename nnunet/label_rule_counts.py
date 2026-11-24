#!/usr/bin/env python
"""B-prep (autopilot/MISSION.md B): how many synthetic training labels would a confirmed post-hoc rule change?

A rule confirmed on the predictions (A3) gets a pre-training twin only if it is violated in the training labels:
MISSION B says that if it changes < 50 of the 999 volumes, no arm is trained ("the generator already obeys this
rule"). This script applies the rule functions of post_rules.py, unchanged and with the frozen parameters, to the
synthetic training labels of the baseline dataset (D101 liver: liver_0000..liver_0998, labels 0 bg, 1 liver,
2 tumour; images in imagesTr for the image-based spine centre) and records, per volume, the changed voxels and
tumour pieces removed. No prediction and no test data are read.

Rules (s9: those confirmed on liver test, frozen in s8):
  tumTouch   tumour pieces with no 26-neighbour liver voxel -> bg
  pos        spine-relative 2D position prior = union of the 20 real LiTS val liver labels (the test-time prior),
             pieces of liver ∪ tumour < 50 % inside -> bg
  ccPost     reference: the one-organ rule, whose training-label twin (ccRepair) is the liver study's arm

  python nnunet/label_rule_counts.py --target liver --workers 12
  python nnunet/label_rule_counts.py --target liver --cases liver_0000 liver_0001   # fake test
-> reports/data/post_rules/liver_label_counts/{per_volume.csv, summary.md}; cache under
   $nnUNet_results/test_eval_liver/label_rule_counts/. Exploratory.
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import post_rules as pr  # noqa: E402
from cc_rules import main_structure  # noqa: E402

VERSION = "labelcounts1"
RULES = ("tumTouch", "pos", "ccPost")
MIN_CHANGED = 50  # MISSION B threshold (volumes)


def raw_dir(did):
    r = os.environ["nnUNet_raw"]
    return f"{r}/" + next(d for d in os.listdir(r) if d.startswith(f"Dataset{did}_"))


def measure(args):
    case, raw, cache_f, support = args
    lnii = nib.load(f"{raw}/labelsTr/{case}.nii.gz")
    lab = np.asarray(lnii.dataobj).astype(np.uint8)
    img = nib.load(f"{raw}/imagesTr/{case}_0000.nii.gz").get_fdata(dtype=np.float32)
    assert img.shape == lab.shape, case
    sp = pr.spacing_of(lnii.affine)
    centre = pr.spine_centre(img, sp, lnii.affine)
    del img
    v = {"tumTouch": pr.tum_touch(lab)[0],
         "pos": pr.position(lab, centre, sp, lnii.affine, support),
         "ccPost": np.where(main_structure(lab), lab, 0).astype(np.uint8)}
    n0 = pr.n_tumour_pieces(lab)
    out = {"version": VERSION, "case": case, "spine_found": centre is not None,
           "voxel_ml": float(np.prod(sp)) / 1000.0, "fg_vox": int((lab > 0).sum()),
           "tum_vox": int((lab == 2).sum()), "tum_pieces": n0,
           "rules": {r: {"changed_vox": int((x != lab).sum()), "tum_vox_removed": int(((lab == 2) & (x != 2)).sum()),
                         "tum_pieces_removed": n0 - pr.n_tumour_pieces(x)} for r, x in v.items()}}
    tmp = cache_f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, cache_f)
    return case


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("liver",), default="liver")
    ap.add_argument("--dataset", type=int, default=101)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="+", help="subset (fake test)")
    a = ap.parse_args()
    pr.set_target(a.target)
    raw = raw_dir(a.dataset)
    cases = a.cases or sorted(f[:-len(".nii.gz")] for f in os.listdir(f"{raw}/labelsTr")
                              if f.startswith(f"{a.target}_"))
    if not a.cases:
        assert len(cases) == 999, len(cases)
    bins = np.load(f"{pr.OUT}/pos_bins.npz")
    assert len(bins.files) == 20, bins.files
    support = pr.prior_support([bins[c] for c in bins.files])
    cdir = f"{os.environ['nnUNet_results']}/test_eval_{a.target}/label_rule_counts"
    os.makedirs(cdir, exist_ok=True)
    jobs = []
    for c in cases:
        f = f"{cdir}/{c}.json"
        if os.path.exists(f) and json.load(open(f)).get("version") == VERSION:
            continue
        jobs.append((c, raw, f, support))
    print(f"{len(jobs)} of {len(cases)} volumes to measure, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for k, c in enumerate(pool.imap_unordered(measure, jobs, chunksize=1), 1):
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} {c}", flush=True)
    data = [json.load(open(f"{cdir}/{c}.json")) for c in cases]
    pub = f"{pr.REPO}/reports/data/post_rules/{a.target}_label_counts" + ("_subset" if a.cases else "")
    os.makedirs(pub, exist_ok=True)
    with open(f"{pub}/per_volume.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "spine_found", "fg_vox", "tum_vox", "tum_pieces"] +
                   [f"{r}_{k}" for r in RULES for k in ("changed_vox", "tum_vox_removed", "tum_pieces_removed")])
        for d in data:
            w.writerow([d["case"], d["spine_found"], d["fg_vox"], d["tum_vox"], d["tum_pieces"]] +
                       [d["rules"][r][k] for r in RULES
                        for k in ("changed_vox", "tum_vox_removed", "tum_pieces_removed")])
    n = len(data)
    lines = [f"# B-prep: confirmed liver rules applied to the {n} synthetic training labels (D{a.dataset})", "",
             "Rule functions of `nnunet/post_rules.py` with the frozen s8 parameters (`nnunet/label_rule_counts.py`). "
             f"MISSION B: < {MIN_CHANGED} changed volumes → no pre-training arm. Exploratory.", "",
             f"Spine centre found in {sum(d['spine_found'] for d in data)} / {n} volumes.", "",
             "| rule | volumes changed | median changed voxels (changed vols) | volumes losing tumour | "
             "tumour voxels removed (% of all) | tumour pieces removed | verdict |", "|---|---|---|---|---|---|---|"]
    tv = sum(d["tum_vox"] for d in data)
    for r in RULES:
        ch = [d["rules"][r]["changed_vox"] for d in data]
        nch = sum(x > 0 for x in ch)
        med = int(np.median([x for x in ch if x > 0])) if nch else 0
        lt = sum(d["rules"][r]["tum_vox_removed"] > 0 for d in data)
        trem = sum(d["rules"][r]["tum_vox_removed"] for d in data)
        tp = sum(d["rules"][r]["tum_pieces_removed"] for d in data)
        verdict = "reference" if r == "ccPost" else ("arm possible" if nch >= MIN_CHANGED else "generator obeys")
        lines.append(f"| {r} | {nch} | {med} | {lt} | {trem} ({100 * trem / max(tv, 1):.2f} %) | {tp} | {verdict} |")
    open(f"{pub}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
