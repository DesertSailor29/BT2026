#!/usr/bin/env python
"""Build Dataset103_LiverSynthRepair: Dataset101 with off-liver tumour components relabelled to background.

Repair arm (PLAN_filtering_options.md, decided 2026-11-10/12): every synthetic label of Dataset101 goes through
maisi/filter_metrics.repair_off_organ (a tumour component is off-organ if < 50% of its voxels lie within 15 mm of
the liver; off-organ components -> 0). N stays 999, only labels change. Images and the 20 real validation cases are
hard-linked from Dataset101 (no copies; symlink fallback across file systems), real_split.json is copied, so the
validation and test cases are identical to Dataset101.

Plans are *copied* from Dataset101 (nnUNetv2_move_plans_between_datasets), not re-planned, so patch size, spacing and
CT normalisation are exactly those of the `all` arm. The labels may differ, the images don't, so the copied intensity
statistics are the right ones.

  python build_repair_dataset.py            # raw dataset + repair_report.csv + repair_summary.md (resumable)
  python build_repair_dataset.py --plans    # also copy Dataset101's plans into nnUNet_preprocessed/Dataset103_*
  nnUNetv2_preprocess -d 103 -plans_name nnUNetPlans -c 3d_fullres -np 16
  python make_splits.py --target liver --dataset-id 103 --arm repair --all-synth

Memory ~1.5 GB per worker (cropped 480x480x384 label). nnunet/run_queue.py runs all of this.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maisi"))
from filter_metrics import THESIS_MM, repair_class, repair_off_organ  # noqa: E402
from orient import spacing_of  # noqa: E402

SRC_ID, DST_ID = 101, 103
SRC, DST = "Dataset101_LiverSynth", "Dataset103_LiverSynthRepair"
KEEP_FRAC = 0.5


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def repair_one(args):
    src_lab, dst_lab, cache = args
    if os.path.exists(dst_lab) and os.path.exists(cache):
        return json.load(open(cache))
    nii = nib.load(src_lab)
    lab = np.asarray(nii.dataobj).astype(np.uint8)
    spacing = spacing_of(nii.affine)
    fixed, rows = repair_off_organ(lab, spacing, THESIS_MM, KEEP_FRAC)
    out = nib.Nifti1Image(fixed, nii.affine)
    out.set_qform(nii.affine, code=1)
    out.set_sform(nii.affine, code=1)
    nib.save(out, dst_lab + ".tmp.nii.gz")
    os.replace(dst_lab + ".tmp.nii.gz", dst_lab)
    vml = float(np.prod(spacing)) / 1000.0
    rec = {"id": os.path.basename(dst_lab)[: -len(".nii.gz")], "class": repair_class(rows),
           "components": len(rows), "removed": sum(not r["on_organ"] for r in rows),
           "removed_ml": round(sum(r["ml"] for r in rows if not r["on_organ"]), 3),
           "tumour_ml_before": round(float((lab == 2).sum()) * vml, 3),
           "tumour_ml_after": round(float((fixed == 2).sum()) * vml, 3)}
    json.dump(rec, open(cache + ".tmp", "w"))
    os.replace(cache + ".tmp", cache)
    return rec


def copy_plans(raw_dst, pre_dst):
    """Dataset101's plans -> Dataset103 (same patch size / spacing / normalisation), + its fingerprint for reference."""
    if os.path.exists(f"{pre_dst}/nnUNetPlans.json"):
        print("plans already copied")
        return
    os.makedirs(pre_dst, exist_ok=True)
    cmd = ["nnUNetv2_move_plans_between_datasets", "-s", str(SRC_ID), "-t", str(DST_ID),
           "-sp", "nnUNetPlans", "-tp", "nnUNetPlans"]
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
    ap.add_argument("--plans", action="store_true", help="also copy Dataset101's plans (needs nnU-Net)")
    a = ap.parse_args()

    raw_src, raw_dst = f"{os.environ['nnUNet_raw']}/{SRC}", f"{os.environ['nnUNet_raw']}/{DST}"
    pre_dst = f"{os.environ['nnUNet_preprocessed']}/{DST}"
    for d in ("imagesTr", "labelsTr", "repair_cache"):
        os.makedirs(f"{raw_dst}/{d}", exist_ok=True)
    src_ds = json.load(open(f"{raw_src}/dataset.json"))

    imgs = sorted(glob.glob(f"{raw_src}/imagesTr/*_0000.nii.gz"))
    for f in imgs:
        link(f, f"{raw_dst}/imagesTr/{os.path.basename(f)}")
    labs = sorted(glob.glob(f"{raw_src}/labelsTr/*.nii.gz"))
    real = [f for f in labs if os.path.basename(f).startswith("lits_")]
    synth = [f for f in labs if f not in real]
    for f in real:
        link(f, f"{raw_dst}/labelsTr/{os.path.basename(f)}")
    shutil.copy(f"{raw_src}/real_split.json", f"{raw_dst}/real_split.json")
    print(f"{len(imgs)} images linked, {len(real)} real labels linked, {len(synth)} synthetic labels to repair", flush=True)

    jobs = [(f, f"{raw_dst}/labelsTr/{os.path.basename(f)}",
             f"{raw_dst}/repair_cache/{os.path.basename(f)[: -len('.nii.gz')]}.json") for f in synth]
    recs = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(repair_one, jobs, chunksize=2), 1):
            recs.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {dict(Counter(x['class'] for x in recs))}", flush=True)
    recs.sort(key=lambda r: r["id"])

    n = len(glob.glob(f"{raw_dst}/labelsTr/*.nii.gz"))
    assert n == len(labs) == len(imgs), (n, len(labs), len(imgs))
    ds = dict(src_ds, numTraining=n,
              description=f"{src_ds.get('description', SRC)}; tumour components with < {KEEP_FRAC:.0%} of voxels "
                          f"within {THESIS_MM:g} mm of the liver relabelled to background (repair arm)")
    json.dump(ds, open(f"{raw_dst}/dataset.json", "w"), indent=1)

    if recs:
        with open(f"{raw_dst}/repair_report.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, list(recs[0]))
            w.writeheader()
            w.writerows(recs)
    cls = Counter(r["class"] for r in recs)
    removed = [r for r in recs if r["removed"]]
    txt = "\n".join([
        f"# {DST}: repair summary", "",
        f"{len(recs)} synthetic labels from {SRC}; component off-organ if < {KEEP_FRAC:.0%} within {THESIS_MM:g} mm.",
        "", "| class | volumes |", "|---|---|", *[f"| {c} | {cls[c]} |" for c in ("no_tumour", "clean", "repaired", "tumour_free")],
        "", f"Volumes changed: {len(removed)}; components removed: {sum(r['removed'] for r in recs)}, "
            f"{sum(r['removed_ml'] for r in recs):.1f} ml tumour label -> background in total.",
        f"Tumour-free training volumes: {sum(r['tumour_ml_after'] == 0 for r in recs)} after repair "
        f"(was {sum(r['tumour_ml_before'] == 0 for r in recs)}).", ""])
    open(f"{raw_dst}/repair_summary.md", "w").write(txt)
    print(txt, flush=True)

    if a.plans:
        copy_plans(raw_dst, pre_dst)


if __name__ == "__main__":
    main()
