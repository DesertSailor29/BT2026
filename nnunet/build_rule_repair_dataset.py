#!/usr/bin/env python
"""MISSION B (autopilot): pre-training twin of a post-hoc rule confirmed on liver test -> a repair dataset.

The rule function of post_rules.py, unchanged and with the frozen s8 parameters, is applied to each of the 999
synthetic training labels of Dataset101 (as nnunet/label_rule_counts.py, which counted the changes first; the build
asserts it changes exactly the volumes that count recorded). N stays 999, only labels change; images and the 20 real
val cases are linked from D101, real_split.json copied, plans copied from D101 (build_repair_dataset.copy_plans), so
the arm differs from `all` only in its labels.

  --rule pos  -> Dataset113_LiverSynthRepairPos: liver / tumour pieces < 50 % inside the spine-relative 2D position
                 prior (union of the 20 real LiTS val liver labels, 2 mm bins, 10 mm margin) -> background
(tumTouch, the other confirmed liver rule, is not built: its label twin changes the same 598 volumes and the same
tumour pieces as ccRepair, Dataset103, which is trained; notes/AUTOPILOT.md s10.)

  python nnunet/build_rule_repair_dataset.py --rule pos --workers 12 --plans
  nnUNetv2_preprocess -d 113 -plans_name nnUNetPlans -c 3d_fullres -np 12
  python nnunet/make_splits.py --target liver --dataset-id 113 --arm posRepair --all-synth
  python nnunet/build_rule_repair_dataset.py --rule pos --cases liver_0000 liver_0001 --out /tmp/x   # fake test
Exploratory.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

from multiprocessing import Pool

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import post_rules as pr  # noqa: E402
from build_repair_dataset import copy_plans, link  # noqa: E402

VERSION = "rulerepair1"
SRC = "Dataset101_LiverSynth"
DATASETS = {"pos": (113, "Dataset113_LiverSynthRepairPos")}
COUNTS = f"{pr.REPO}/reports/data/post_rules/liver_label_counts/per_volume.csv"


def repair(args):
    rule, src_img, src_lab, dst_lab, cache, support = args
    if os.path.exists(dst_lab) and os.path.exists(cache):
        return json.load(open(cache))
    nii = nib.load(src_lab)
    lab = np.asarray(nii.dataobj).astype(np.uint8)
    img = nib.load(src_img).get_fdata(dtype=np.float32)
    sp = pr.spacing_of(nii.affine)
    centre = pr.spine_centre(img, sp, nii.affine)
    del img
    fixed = pr.position(lab, centre, sp, nii.affine, support).astype(np.uint8)
    out = nib.Nifti1Image(fixed, nii.affine, nii.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, dst_lab + ".tmp.nii.gz")
    os.replace(dst_lab + ".tmp.nii.gz", dst_lab)
    rec = {"version": VERSION, "id": os.path.basename(dst_lab)[: -len(".nii.gz")], "spine_found": centre is not None,
           "changed_vox": int((fixed != lab).sum()), "tum_vox_before": int((lab == 2).sum()),
           "tum_vox_after": int((fixed == 2).sum()), "liver_vox_removed": int(((lab == 1) & (fixed != 1)).sum())}
    json.dump(rec, open(cache + ".tmp", "w"))
    os.replace(cache + ".tmp", cache)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", required=True, choices=list(DATASETS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plans", action="store_true", help="also copy Dataset101's plans (needs nnU-Net)")
    ap.add_argument("--cases", nargs="+", help="subset (fake test; needs --out)")
    ap.add_argument("--out", help="raw output dir instead of $nnUNet_raw/<name> (fake test)")
    a = ap.parse_args()
    assert not a.cases or a.out, "--cases needs --out (never a partial dataset under nnUNet_raw)"
    pr.set_target("liver")
    did, name = DATASETS[a.rule]
    raw_src = f"{os.environ['nnUNet_raw']}/{SRC}"
    raw_dst = a.out or f"{os.environ['nnUNet_raw']}/{name}"
    for d in ("imagesTr", "labelsTr", "repair_cache"):
        os.makedirs(f"{raw_dst}/{d}", exist_ok=True)
    bins = np.load(f"{pr.OUT}/pos_bins.npz")
    assert len(bins.files) == 20, bins.files
    support = pr.prior_support([bins[c] for c in bins.files])
    used = {"rule": a.rule, "version": VERSION, "post_rules": "s8 frozen (liverfrozen1)", "prior": "union of 20 val"}
    rf = f"{raw_dst}/repair_rules.json"
    if os.path.exists(rf) and json.load(open(rf)) != used:
        raise SystemExit(f"{raw_dst} was built with other rules: {json.load(open(rf))}")
    json.dump(used, open(rf, "w"), indent=1)

    labs = sorted(glob.glob(f"{raw_src}/labelsTr/*.nii.gz"))
    real = [f for f in labs if os.path.basename(f).startswith("lits_")]
    synth = [f for f in labs if f not in real]
    if a.cases:
        synth = [f for f in synth if os.path.basename(f)[: -len(".nii.gz")] in a.cases]
        real = []
    else:
        assert len(synth) == 999 and len(real) == 20, (len(synth), len(real))
    for f in real:
        c = os.path.basename(f)[: -len(".nii.gz")]
        link(f, f"{raw_dst}/labelsTr/{c}.nii.gz")
        link(f"{raw_src}/imagesTr/{c}_0000.nii.gz", f"{raw_dst}/imagesTr/{c}_0000.nii.gz")
    jobs = []
    for f in synth:
        c = os.path.basename(f)[: -len(".nii.gz")]
        img = f"{raw_src}/imagesTr/{c}_0000.nii.gz"
        link(img, f"{raw_dst}/imagesTr/{c}_0000.nii.gz")
        jobs.append((a.rule, img, f, f"{raw_dst}/labelsTr/{c}.nii.gz", f"{raw_dst}/repair_cache/{c}.json", support))
    print(f"{name}: {len(jobs)} synthetic labels to repair ({a.rule}), {len(real)} real linked", flush=True)
    recs = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(repair, jobs, chunksize=1), 1):
            recs.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}, changed so far {sum(x['changed_vox'] > 0 for x in recs)}", flush=True)
    recs.sort(key=lambda r: r["id"])
    # the written labels must change exactly what label_rule_counts.py counted (same function, same prior)
    cnt = {row["case"]: int(row[f"{a.rule}_changed_vox"]) for row in csv.DictReader(open(COUNTS))}
    bad = [r["id"] for r in recs if r["changed_vox"] != cnt[r["id"]]]
    assert not bad, f"build vs label_rule_counts: {bad[:10]}"
    nchg = sum(r["changed_vox"] > 0 for r in recs)
    print(f"counts match label_rule_counts ({nchg} volumes changed)", flush=True)
    if a.cases:
        return
    n = len(glob.glob(f"{raw_dst}/labelsTr/*.nii.gz"))
    assert n == len(glob.glob(f"{raw_dst}/imagesTr/*_0000.nii.gz")) == 1019, n
    shutil.copy(f"{raw_src}/real_split.json", f"{raw_dst}/real_split.json")
    src_ds = json.load(open(f"{raw_src}/dataset.json"))
    what = ("liver / tumour pieces < 50 % inside the spine-relative position prior of the 20 real val liver labels "
            "relabelled to background")
    json.dump(dict(src_ds, numTraining=n, description=f"{src_ds.get('description', SRC)}; {what} (posRepair arm)"),
              open(f"{raw_dst}/dataset.json", "w"), indent=1)
    with open(f"{raw_dst}/repair_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    tb, ta = sum(r["tum_vox_before"] for r in recs), sum(r["tum_vox_after"] for r in recs)
    txt = (f"# {name}: repair summary\n\n{len(recs)} synthetic labels from {SRC}; {what}.\n\n"
           f"Volumes changed: {nchg}; tumour voxels {tb} -> {ta} ({100 * (tb - ta) / max(tb, 1):.2f} % removed); "
           f"liver voxels removed {sum(r['liver_vox_removed'] for r in recs)}; tumour-free volumes "
           f"{sum(r['tum_vox_after'] == 0 for r in recs)} (was {sum(r['tum_vox_before'] == 0 for r in recs)}).\n")
    open(f"{raw_dst}/repair_summary.md", "w").write(txt)
    print(txt, flush=True)
    if a.plans:
        copy_plans(did, name, f"{os.environ['nnUNet_preprocessed']}/{name}")


if __name__ == "__main__":
    main()
