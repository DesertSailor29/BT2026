#!/usr/bin/env python
"""Build Dataset112_PancreasSynthThick: the 999 synthetic pancreas pairs of Dataset102 re-sliced to real slice thicknesses.

Why (autopilot plan E1, notes/AUTOPILOT.md): the pancreas baseline misses most of the gland on real MSD07 scans. One
measured gap is slice thickness: every synthetic scan has 1.0 mm slices, the real ones mostly 2.5 mm (38/238 test
cases >= 5 mm, gland Dice 0.16 vs 0.30). nnU-Net resamples a real 2.5 / 5 mm scan to the 1.0 mm target spacing with
its anisotropic rule (z separately, order 0 = slices repeated), so the model meets thick, partial-volume-blurred,
stair-stepped scans at test time that it never saw in training.

What it does, per synthetic volume (labels and images of Dataset102, real val cases untouched):
- slice thickness t drawn from the slice spacings of the 43 MSD07 *val* cases (the test cases are never read),
  seeded by the case id (random.Random(f"{seed}-{id}")); t <= the native 1.0 mm -> the pair is linked unchanged;
- image: slab average over t mm along z (each 1.0 mm slice weighted by its overlap with the slab, i.e. a thick
  reconstruction with spacing = thickness), slabs start at a random offset in [0, t);
- label: the label of the 1.0 mm slice at each slab centre (nearest), so image and label share the thick grid;
- written at spacing (0.8, 0.8, t) with the matching affine. nnU-Net preprocessing then resamples them to the plan's
  target spacing exactly as it does with real test scans (same code path).
Plans are *copied* from Dataset102 (same patch size, target spacing, CT normalisation), so the arm differs from `all`
only in the training images. Trainer, epochs and evaluation are those of `all`.

Resumable (existing outputs are skipped). Writes thick_report.csv (id, thickness, slices) + thick_summary.md.

  python build_thick_dataset.py --workers 16 [--limit 5]        # raw dataset
  python build_thick_dataset.py --workers 16 --plans            # + copy Dataset102's plans (needs nnU-Net)
  nnUNetv2_preprocess -d 112 -plans_name nnUNetPlans -c 3d_fullres -np 16
  python make_splits.py --target pancreas --dataset-id 112 --arm thickSlice --all-synth
"""
import argparse
import csv
import glob
import json
import os
import random
import shutil
import subprocess
from collections import Counter
from multiprocessing import Pool

import nibabel as nib
import numpy as np

SRC_ID, SRC = 102, "Dataset102_PancreasSynth"
DST_ID, DST = 112, "Dataset112_PancreasSynthThick"
VERSION = "thick1"


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def slab_weights(nz, r, offset):
    """(n_slabs, nz) overlap weights of slabs r voxels wide (voxel k covers [k-0.5, k+0.5]) and the slab centres."""
    starts = []
    s = -0.5 + offset
    while s + r <= nz - 0.5 + 1e-6:
        starts.append(s)
        s += r
    starts = np.asarray(starts)
    k = np.arange(nz)
    lo = np.maximum(starts[:, None], k[None, :] - 0.5)
    hi = np.minimum(starts[:, None] + r, k[None, :] + 0.5)
    w = np.clip(hi - lo, 0, None)
    return w / w.sum(1, keepdims=True), starts + r / 2


def thicken(args):
    src_img, src_lab, dst_img, dst_lab, t, offset_frac = args
    cid = os.path.basename(dst_lab)[: -len(".nii.gz")]
    if os.path.exists(dst_img) and os.path.exists(dst_lab):
        return {"id": cid, "thickness_mm": t, "slices": nib.load(dst_lab).shape[2], "status": "skip"}
    img, lab = nib.load(src_img), nib.load(src_lab)
    sz = float(img.header.get_zooms()[2])
    if t <= sz * 1.01:
        link(src_img, dst_img)
        link(src_lab, dst_lab)
        return {"id": cid, "thickness_mm": sz, "slices": img.shape[2], "status": "native"}
    a = np.asarray(img.dataobj, dtype=np.float32)
    l = np.asarray(lab.dataobj).astype(np.uint8)
    r = t / sz
    w, centres = slab_weights(a.shape[2], r, offset_frac * r)
    thick = np.rint(np.tensordot(a, w.T, axes=([2], [0]))).astype(np.int16)
    thick_lab = l[:, :, np.clip(np.floor(centres + 0.5).astype(int), 0, a.shape[2] - 1)]
    aff = img.affine.copy()
    aff[:3, 3] = img.affine[:3, :3] @ np.array([0.0, 0.0, centres[0]]) + img.affine[:3, 3]
    aff[:3, 2] = img.affine[:3, 2] * r
    for arr, d in ((thick, dst_img), (thick_lab, dst_lab)):
        o = nib.Nifti1Image(arr, aff)
        o.set_qform(aff, code=1)
        o.set_sform(aff, code=1)
        nib.save(o, d + ".tmp.nii.gz")
        os.replace(d + ".tmp.nii.gz", d)
    return {"id": cid, "thickness_mm": t, "slices": thick.shape[2], "status": "thick"}


def copy_plans(pre_dst):
    """Dataset102's plans -> Dataset112 (+ fingerprint, dataset.json), as build_repair_dataset.copy_plans."""
    os.makedirs(pre_dst, exist_ok=True)
    shutil.copy(f"{os.environ['nnUNet_raw']}/{DST}/dataset.json", f"{pre_dst}/dataset.json")
    if os.path.exists(f"{pre_dst}/nnUNetPlans.json"):
        print("plans already copied")
        return
    cmd = ["nnUNetv2_move_plans_between_datasets", "-s", str(SRC_ID), "-t", str(DST_ID), "-sp", "nnUNetPlans",
           "-tp", "nnUNetPlans"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    fp = f"{os.environ['nnUNet_preprocessed']}/{SRC}/dataset_fingerprint.json"
    if os.path.exists(fp):
        shutil.copy(fp, f"{pre_dst}/dataset_fingerprint.json")
    plans = json.load(open(f"{pre_dst}/nnUNetPlans.json"))
    assert plans["dataset_name"] == DST, plans["dataset_name"]
    print(f"plans copied to {pre_dst}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N synthetic volumes (testing)")
    ap.add_argument("--plans", action="store_true", help="also copy Dataset102's plans (needs nnU-Net)")
    a = ap.parse_args()

    raw_src, raw_dst = f"{os.environ['nnUNet_raw']}/{SRC}", f"{os.environ['nnUNet_raw']}/{DST}"
    used = {"version": VERSION, "seed": a.seed, "source": SRC}
    rules_f = f"{raw_dst}/thick_rules.json"
    if os.path.exists(rules_f) and json.load(open(rules_f)) != used:
        raise SystemExit(f"{DST} was built with {json.load(open(rules_f))}, now {used}: delete it and rebuild")
    for d in ("imagesTr", "labelsTr"):
        os.makedirs(f"{raw_dst}/{d}", exist_ok=True)
    json.dump(used, open(rules_f, "w"), indent=1)
    shutil.copy(f"{raw_src}/real_split.json", f"{raw_dst}/real_split.json")

    # real val cases: linked unchanged; their slice spacings are the thickness distribution
    real = sorted(glob.glob(f"{raw_src}/labelsTr/msd_*.nii.gz"))
    val_t = []
    for f in real:
        c = os.path.basename(f)[: -len(".nii.gz")]
        link(f, f"{raw_dst}/labelsTr/{c}.nii.gz")
        link(f"{raw_src}/imagesTr/{c}_0000.nii.gz", f"{raw_dst}/imagesTr/{c}_0000.nii.gz")
        val_t.append(round(float(nib.load(f).header.get_zooms()[2]), 3))
    n_val = len(json.load(open(f"{raw_src}/real_split.json"))["val"])
    assert len(real) == n_val, f"{len(real)} real labels in {SRC}, real_split.json has {n_val} val cases"
    print(f"{len(real)} real val cases linked; slice spacings {dict(sorted(Counter(val_t).items()))}", flush=True)

    synth = sorted(f for f in glob.glob(f"{raw_src}/labelsTr/pancreas_*.nii.gz"))
    if a.limit:
        synth = synth[: a.limit]
    jobs = []
    for f in synth:
        cid = os.path.basename(f)[: -len(".nii.gz")]
        rng = random.Random(f"{a.seed}-{cid}")
        t = rng.choice(val_t)
        jobs.append((f"{raw_src}/imagesTr/{cid}_0000.nii.gz", f, f"{raw_dst}/imagesTr/{cid}_0000.nii.gz",
                     f"{raw_dst}/labelsTr/{cid}.nii.gz", t, rng.random()))
    recs = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(thicken, jobs, chunksize=2), 1):
            recs.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {dict(Counter(x['status'] for x in recs))}", flush=True)
    recs.sort(key=lambda r: r["id"])

    n = len(glob.glob(f"{raw_dst}/labelsTr/*.nii.gz"))
    assert n == len(glob.glob(f"{raw_dst}/imagesTr/*_0000.nii.gz")) == len(real) + len(jobs), n
    src_ds = json.load(open(f"{raw_src}/dataset.json"))
    json.dump(dict(src_ds, numTraining=n, description=f"{src_ds.get('description', SRC)}; synthetic images re-sliced "
                   f"to MSD07 val slice thicknesses (slab average, {VERSION}); thickSlice arm"),
              open(f"{raw_dst}/dataset.json", "w"), indent=1)
    with open(f"{raw_dst}/thick_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["id", "thickness_mm", "slices", "status"])
        w.writeheader()
        w.writerows(recs)
    cnt = Counter(r["thickness_mm"] for r in recs)
    txt = "\n".join([f"# {DST}: thickness summary", "",
                     f"{len(recs)} synthetic pairs from {SRC}, thickness drawn from the {len(real)} MSD07 val slice "
                     f"spacings (seed {a.seed}); real val cases unchanged.", "",
                     "| thickness mm | volumes |", "|---|---|", *[f"| {k} | {v} |" for k, v in sorted(cnt.items())], ""])
    open(f"{raw_dst}/thick_summary.md", "w").write(txt)
    print(txt, flush=True)
    if a.plans:
        copy_plans(f"{os.environ['nnUNet_preprocessed']}/{DST}")


if __name__ == "__main__":
    main()
