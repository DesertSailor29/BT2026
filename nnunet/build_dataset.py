#!/usr/bin/env python
"""Build one nnU-Net raw dataset per target: all synthetic MAISI pairs + a fixed set of real validation cases.

Design (PLAN_training.md, decided 2026-11-07):
- Training is synthetic only. Every experiment arm is one fold in splits_final.json (make_splits.py):
  train = the arm's synthetic ids, val = the same real cases. So the dataset is preprocessed once for all arms.
- Real validation cases (used by nnU-Net for checkpoint_best) come from the thesis train split (LiTS imagesTr,
  91 cases), drawn with a fixed seed. Everything else in LiTS (71 remaining train + 40 thesis test = 111) is test.
  The split is written to <dataset>/real_split.json and never changes.
- Synthetic volumes: only MAISI's constant padding is cropped (in-plane 16..495, z 64..447 of the 512^3 volume,
  pure -1000 HU in the image and 0 in the label; checked per volume, a volume with content there is kept whole).
  No anatomy crop.

Resumable: existing output files are skipped. Real data must be RAS (raw_data/LiTS_RAS, GUIDE 6b).

  python build_dataset.py --target liver      # Dataset101_LiverSynth
"""
import argparse
import glob
import json
import os
import random
import re
from multiprocessing import Pool

import nibabel as nib
import numpy as np

PROJ = os.environ.get("PROJ", "/work/Research")
TARGETS = {
    # target: (dataset id, name, real dir, real case glob, label names)
    "liver": (101, "LiverSynth", f"{PROJ}/raw_data/LiTS_RAS", {"background": 0, "liver": 1, "tumor": 2}),
    "pancreas": (102, "PancreasSynth", f"{PROJ}/raw_data/Task07_Pancreas", {"background": 0, "pancreas": 1, "tumor": 2}),
}
CROP = (slice(16, 496), slice(16, 496), slice(64, 448))  # MAISI Path A content box: 480x480x384 inside 512^3
AIR = -1000


def crop_synth(args):
    src_img, dst_img, src_lab, dst_lab = args
    if os.path.exists(dst_img) and os.path.exists(dst_lab):
        return "skip"
    img, lab = nib.load(src_img), nib.load(src_lab)
    a, l = np.asarray(img.dataobj), np.asarray(lab.dataobj)
    inside = np.zeros(a.shape, bool)
    inside[CROP] = True
    clean = bool((a[~inside] == AIR).all() and (l[~inside] == 0).all())
    if clean:
        img, lab = img.slicer[CROP], lab.slicer[CROP]
    out_img = nib.Nifti1Image(np.asarray(img.dataobj).astype(np.int16), img.affine)
    out_lab = nib.Nifti1Image(np.asarray(lab.dataobj).astype(np.uint8), lab.affine)
    for o, d in ((out_img, dst_img), (out_lab, dst_lab)):
        o.set_qform(o.affine, code=1)
        o.set_sform(o.affine, code=1)
        nib.save(o, d + ".tmp.nii.gz")
        os.replace(d + ".tmp.nii.gz", d)
    return "cropped" if clean else "kept whole (content in padding)"


def copy_real(args):
    src_img, dst_img, src_lab, dst_lab = args
    for s, d, dt in ((src_img, dst_img, np.int16), (src_lab, dst_lab, np.uint8)):
        if os.path.exists(d):
            continue
        im = nib.load(s)
        o = nib.Nifti1Image(np.asarray(im.dataobj).astype(dt), im.affine)
        o.set_qform(im.affine, code=1)
        o.set_sform(im.affine, code=1)
        nib.save(o, d + ".tmp.nii.gz")
        os.replace(d + ".tmp.nii.gz", d)
    return "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS, default="liver")
    ap.add_argument("--pool", default=f"{os.environ.get('MAISI_RUNS', PROJ + '/maisi_runs')}/pool")
    ap.add_argument("--val-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N synthetic volumes (testing)")
    a = ap.parse_args()

    did, name, real_dir, labels = TARGETS[a.target]
    out = f"{os.environ['nnUNet_raw']}/Dataset{did:03d}_{name}"
    os.makedirs(f"{out}/imagesTr", exist_ok=True)
    os.makedirs(f"{out}/labelsTr", exist_ok=True)

    # --- real split (fixed once, then reused) ---
    split_file = f"{out}/real_split.json"
    if os.path.exists(split_file):
        split = json.load(open(split_file))
    else:
        if a.target != "liver":
            raise SystemExit("real split for pancreas not implemented yet")
        ids_dir = f"{PROJ}/raw_data/LiTS"  # original split (same ids as LiTS_RAS, complete before the conversion ends)
        tr = sorted(int(re.search(r"volume-(\d+)_0000", f).group(1)) for f in glob.glob(f"{ids_dir}/imagesTr/volume-*_0000.nii*"))
        ts = sorted(int(re.search(r"volume-(\d+)_0000", f).group(1)) for f in glob.glob(f"{ids_dir}/imagesTs/volume-*_0000.nii*"))
        assert len(tr) == 91 and len(ts) == 40, (len(tr), len(ts))
        val = sorted(random.Random(a.seed).sample(tr, a.val_n))
        split = {"source": real_dir, "seed": a.seed, "val": val,
                 "test": sorted([i for i in tr if i not in val] + ts), "thesis_test": ts}
        json.dump(split, open(split_file, "w"), indent=1)
    print(f"real: {len(split['val'])} val, {len(split['test'])} test")

    def real_path(sub, i, lab=False):
        f = glob.glob(f"{real_dir}/{sub}/volume-{i}{'' if lab else '_0000'}.nii*")
        return f[0] if len(f) == 1 else None

    real_jobs, real_missing = [], []
    for i in split["val"]:
        si, sl = real_path("imagesTr", i), real_path("labelsTr", i, True)
        if si and sl:
            real_jobs.append((si, f"{out}/imagesTr/lits_{i:03d}_0000.nii.gz", sl, f"{out}/labelsTr/lits_{i:03d}.nii.gz"))
        else:
            real_missing.append(i)
    if real_missing:
        print(f"WARNING: {len(real_missing)} real val cases not in {real_dir} yet (rerun later): {real_missing}")

    sdir = f"{a.pool}/pairs/{a.target}"
    synth = sorted(glob.glob(f"{sdir}/imagesTr/{a.target}_*_0000.nii.gz"))
    if a.limit:
        synth = synth[: a.limit]
    synth_jobs = []
    for f in synth:
        cid = os.path.basename(f)[: -len("_0000.nii.gz")]
        synth_jobs.append((f, f"{out}/imagesTr/{cid}_0000.nii.gz", f"{sdir}/labelsTr/{cid}.nii.gz", f"{out}/labelsTr/{cid}.nii.gz"))
    print(f"synthetic: {len(synth_jobs)}")

    with Pool(a.workers) as p:
        for r in p.imap_unordered(copy_real, real_jobs):
            pass
        counts = {}
        for k, r in enumerate(p.imap_unordered(crop_synth, synth_jobs, chunksize=2), 1):
            counts[r] = counts.get(r, 0) + 1
            if k % 50 == 0 or k == len(synth_jobs):
                print(f"{k}/{len(synth_jobs)} {counts}", flush=True)

    n = len(glob.glob(f"{out}/labelsTr/*.nii.gz"))
    if real_missing:
        print("not writing dataset.json: real val cases missing, rerun when the RAS conversion is done")
        return
    json.dump({"channel_names": {"0": "CT"}, "labels": labels, "numTraining": n, "file_ending": ".nii.gz",
               "description": f"MAISI synthetic {a.target} pool (train) + {len(split['val'])} real val cases"},
              open(f"{out}/dataset.json", "w"), indent=1)
    print(f"done: {out} ({n} cases)")


if __name__ == "__main__":
    main()
