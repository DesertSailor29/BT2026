#!/usr/bin/env python
"""Build a repair-arm dataset: Dataset101 with the failing tumour components of one criterion relabelled.

PLAN_training.md "Filter rules" (fixed 2026-11-12). The decision per component is filter_metrics' (same functions,
same cached measurements, same cut from rules.json), so a repair arm and its reject arm judge identical components:
    --criterion touch -> Dataset103_LiverSynthRepairTouch: components not touching the liver -> background
    --criterion auc   -> Dataset104_LiverSynthRepairAUC:   invisible components (sep < real min) -> liver if >= 50%
                                                            of their outer shell is liver, else background
N stays 999, only labels change. Each Dataset101 label is relabelled with filter_metrics.apply_repair, which recomputes
the components and checks them against the metrics cache (Dataset101 labels are the pool labels cropped at MAISI's
empty padding, so the components and their numbering are the same). Images and the 20 real validation cases are
hard-linked from Dataset101 (symlink fallback), real_split.json is copied: validation and test are identical to D101.

Plans are *copied* from Dataset101 (nnUNetv2_move_plans_between_datasets), not re-planned, so patch size, spacing and
CT normalisation are exactly those of the `all` arm (the images are the same, so its intensity statistics apply).

The dataset records the rules it was built with (repair_rules.json). If filter_metrics' rules.json changes later (e.g.
an excluded real lesion moves the AUC cut), the build refuses: delete the dataset (raw + preprocessed) and rebuild.

  python build_repair_dataset.py --criterion touch           # raw dataset + repair_report.csv + repair_summary.md
  python build_repair_dataset.py --criterion touch --plans   # also copy Dataset101's plans
  nnUNetv2_preprocess -d 103 -plans_name nnUNetPlans -c 3d_fullres -np 16
  python make_splits.py --target liver --dataset-id 103 --arm repairTouch --all-synth

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
from filter_metrics import CRITERIA, ORGAN, TUMOR, VERSION, apply_repair, fails, repair_value, volume_class  # noqa: E402

SRC_ID, SRC = 101, "Dataset101_LiverSynth"
DATASETS = {"touch": (103, "Dataset103_LiverSynthRepairTouch"), "auc": (104, "Dataset104_LiverSynthRepairAUC")}
POOL = os.environ.get("MAISI_RUNS", f"{os.environ.get('PROJ', '/work/Research')}/maisi_runs") + "/pool"


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def repair_one(args):
    src_lab, dst_lab, cache, metrics_json, criterion, cut = args
    if os.path.exists(dst_lab) and os.path.exists(cache):
        return json.load(open(cache))
    rows = json.load(open(metrics_json))["components"]
    nii = nib.load(src_lab)
    lab = np.asarray(nii.dataobj).astype(np.uint8)
    fixed = apply_repair(lab, rows, criterion, cut)
    out = nib.Nifti1Image(fixed, nii.affine)
    out.set_qform(nii.affine, code=1)
    out.set_sform(nii.affine, code=1)
    nib.save(out, dst_lab + ".tmp.nii.gz")
    os.replace(dst_lab + ".tmp.nii.gz", dst_lab)
    vml = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0
    failed = [r for r in rows if fails(r, criterion, cut)]
    rec = {"id": os.path.basename(dst_lab)[: -len(".nii.gz")], "class": volume_class(rows, criterion, cut),
           "components": len(rows), "relabelled": len(failed),
           "to_background": sum(repair_value(r, criterion) == 0 for r in failed),
           "to_organ": sum(repair_value(r, criterion) == ORGAN for r in failed),
           "relabelled_ml": round(sum(r["ml"] for r in failed), 3),
           "tumour_ml_before": round(float((lab == TUMOR).sum()) * vml, 3),
           "tumour_ml_after": round(float((fixed == TUMOR).sum()) * vml, 3),
           "organ_ml_added": round(float(((fixed == ORGAN) & (lab != ORGAN)).sum()) * vml, 3)}
    json.dump(rec, open(cache + ".tmp", "w"))
    os.replace(cache + ".tmp", cache)
    return rec


def copy_plans(did, name, pre_dst):
    """Dataset101's plans -> this dataset (same patch size / spacing / normalisation), + its fingerprint."""
    if os.path.exists(f"{pre_dst}/nnUNetPlans.json"):
        print("plans already copied")
        return
    os.makedirs(pre_dst, exist_ok=True)
    cmd = ["nnUNetv2_move_plans_between_datasets", "-s", str(SRC_ID), "-t", str(did), "-sp", "nnUNetPlans", "-tp", "nnUNetPlans"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    fp = f"{os.environ['nnUNet_preprocessed']}/{SRC}/dataset_fingerprint.json"
    if os.path.exists(fp):
        shutil.copy(fp, f"{pre_dst}/dataset_fingerprint.json")
    plans = json.load(open(f"{pre_dst}/nnUNetPlans.json"))
    assert plans["dataset_name"] == name, plans["dataset_name"]
    print(f"plans copied to {pre_dst}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--criterion", required=True, choices=list(DATASETS))
    ap.add_argument("--metrics", default=f"{POOL}/filter_metrics/liver", help="filter_metrics output (caches + rules.json)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--plans", action="store_true", help="also copy Dataset101's plans (needs nnU-Net)")
    a = ap.parse_args()

    did, name = DATASETS[a.criterion]
    rules = json.load(open(f"{a.metrics}/rules.json"))
    assert rules["version"] == VERSION, f"rules.json is from another filter_metrics version: {rules['version']}"
    cut = rules["auc_cut"]
    if a.criterion == "auc" and cut is None:
        raise SystemExit("rules.json has no AUC cut (filter_metrics ran without images?)")
    used = {"criterion": a.criterion, "version": VERSION, "auc_cut": cut if a.criterion == "auc" else None,
            "excluded": rules["excluded"] if a.criterion == "auc" else None}

    raw_src, raw_dst = f"{os.environ['nnUNet_raw']}/{SRC}", f"{os.environ['nnUNet_raw']}/{name}"
    pre_dst = f"{os.environ['nnUNet_preprocessed']}/{name}"
    rules_f = f"{raw_dst}/repair_rules.json"
    if os.path.exists(rules_f) and json.load(open(rules_f)) != used:
        raise SystemExit(f"{name} was built with other rules ({rules_f}) than {a.metrics}/rules.json now gives "
                         f"({used}). Delete {raw_dst} and {pre_dst} and rebuild.")
    for d in ("imagesTr", "labelsTr", "repair_cache"):
        os.makedirs(f"{raw_dst}/{d}", exist_ok=True)
    json.dump(used, open(rules_f, "w"), indent=1)
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
    print(f"{name}: {len(imgs)} images linked, {len(real)} real labels linked, {len(synth)} synthetic labels "
          f"(criterion {a.criterion}{f', cut {cut}' if a.criterion == 'auc' else ''})", flush=True)

    jobs = []
    for f in synth:
        cid = os.path.basename(f)[: -len(".nii.gz")]
        mj = f"{a.metrics}/per_volume/{cid}.json"
        assert os.path.exists(mj), f"no filter_metrics cache for {cid}: run filter_metrics.py first"
        jobs.append((f, f"{raw_dst}/labelsTr/{cid}.nii.gz", f"{raw_dst}/repair_cache/{cid}.json", mj, a.criterion, cut))
    recs = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(repair_one, jobs, chunksize=2), 1):
            recs.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {dict(Counter(x['class'] for x in recs))}", flush=True)
    recs.sort(key=lambda r: r["id"])

    n = len(glob.glob(f"{raw_dst}/labelsTr/*.nii.gz"))
    assert n == len(labs) == len(imgs), (n, len(labs), len(imgs))
    what = ("tumour components not touching the liver relabelled to background" if a.criterion == "touch" else
            f"tumour components with separability < {cut} (real val minimum) relabelled to their surroundings' label")
    json.dump(dict(src_ds, numTraining=n, description=f"{src_ds.get('description', SRC)}; {what} "
                   f"(repair{CRITERIA[a.criterion]} arm)"), open(f"{raw_dst}/dataset.json", "w"), indent=1)

    if recs:
        with open(f"{raw_dst}/repair_report.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, list(recs[0]))
            w.writeheader()
            w.writerows(recs)
    cls = Counter(r["class"] for r in recs)
    changed = [r for r in recs if r["relabelled"]]
    txt = "\n".join([
        f"# {name}: repair summary", "", f"{len(recs)} synthetic labels from {SRC}; {what}.", "",
        "| class | volumes |", "|---|---|", *[f"| {c} | {cls[c]} |" for c in ("no_tumour", "clean", "repaired", "tumour_free")],
        "", f"Volumes changed: {len(changed)}; components relabelled: {sum(r['relabelled'] for r in recs)} "
            f"({sum(r['to_background'] for r in recs)} to background, {sum(r['to_organ'] for r in recs)} to liver), "
            f"{sum(r['relabelled_ml'] for r in recs):.1f} ml in total.",
        f"Tumour-free training volumes: {sum(r['tumour_ml_after'] == 0 for r in recs)} after repair "
        f"(was {sum(r['tumour_ml_before'] == 0 for r in recs)}).", ""])
    open(f"{raw_dst}/repair_summary.md", "w").write(txt)
    print(txt, flush=True)

    if a.plans:
        copy_plans(did, name, pre_dst)


if __name__ == "__main__":
    main()
