#!/usr/bin/env python
"""Tumour-size test (PLAN_followups.md, part 3): split the ccAucRepair training set (Dataset104 labels) by the
measured tumour burden per volume. CPU; writes id lists only (no fold is registered, nothing is trained).

Burden = tumour voxels (label 2) in the Dataset104 label x voxel volume, i.e. exactly the tumour the network sees
(cc + AUC already applied, so every remaining tumour lies in the main liver structure and is visible).
    sizeLarge   the half of the tumour-bearing volumes with the larger burden (ties broken by id)
    sizeSmall   the other half
    sizeRand    a random half of the tumour-bearing volumes (seed 0): control for "half the tumour volumes"
Every list also holds all tumour-free volumes (no tumour after the repair), so the arms differ only in which
tumour-bearing volumes they get. The split uses the synthetic labels only. The 20 real val cases (Dataset101
labelsTr/lits_*) are measured as a reference for the table; the real test cases are never read.

Writes <metrics>/size/{burden.csv, ids_sizeLarge.json, ids_sizeSmall.json, ids_sizeRand.json, size_summary.md}
(metrics = $MAISI_RUNS/pool/filter_metrics/liver). Resumable is not needed (~5 min with 8 workers).

  python nnunet/size_split.py --workers 8
Then, only once decided (registers new folds of Dataset104; append-only):
  python nnunet/make_splits.py --target liver --dataset-id 104 --arm ccAucRepair_sizeLarge --ids <metrics>/size/ids_sizeLarge.json
"""
import argparse
import csv
import glob
import json
import os
import random
from multiprocessing import Pool

import nibabel as nib
import numpy as np

RAW = os.environ["nnUNet_raw"]
D104 = f"{RAW}/Dataset104_LiverSynthRepairCCAUC/labelsTr"
D101 = f"{RAW}/Dataset101_LiverSynth/labelsTr"
METRICS = (os.environ.get("MAISI_RUNS", f"{os.environ.get('PROJ', '/work/Research')}/maisi_runs")
           + "/pool/filter_metrics/liver")
SEED = 0


def burden(path):
    n = nib.load(path)
    lab = np.asarray(n.dataobj)
    vml = float(np.prod(n.header.get_zooms()[:3])) / 1000.0
    return os.path.basename(path)[: -len(".nii.gz")], round(float((lab == 2).sum()) * vml, 4), \
        round(float((lab == 1).sum()) * vml, 2)


def q(v, ps=(10, 25, 50, 75, 90)):
    return " / ".join(f"{np.percentile(v, p):.1f}" for p in ps) if len(v) else "–"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=D104)
    ap.add_argument("--real-val", default=D101)
    ap.add_argument("--out", default=f"{METRICS}/size")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    synth = sorted(glob.glob(f"{a.labels}/liver_*.nii.gz"))
    real = sorted(glob.glob(f"{a.real_val}/lits_*.nii.gz"))
    assert synth, f"no liver_*.nii.gz in {a.labels}"
    with Pool(a.workers) as p:
        rows = p.map(burden, synth)
        rrows = p.map(burden, real)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/burden.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "tumour_ml", "liver_ml", "group"])
        tum = sorted([r for r in rows if r[1] > 0], key=lambda r: (r[1], r[0]))
        free = sorted(r[0] for r in rows if r[1] == 0)
        half = len(tum) // 2
        small, large = [r[0] for r in tum[:half]], [r[0] for r in tum[half:]]
        rand = sorted(random.Random(SEED).sample([r[0] for r in tum], len(large)))
        grp = {**{i: "small" for i in small}, **{i: "large" for i in large}, **{i: "tumour_free" for i in free}}
        for r in rows:
            w.writerow([r[0], r[1], r[2], grp[r[0]]])
        for r in rrows:
            w.writerow([r[0], r[1], r[2], "real_val"])
    split_ml = tum[half][1]
    lists = {"sizeLarge": large + free, "sizeSmall": small + free, "sizeRand": rand + free}
    which = {"sizeLarge": f"the {len(large)} with the larger tumour burden (>= {split_ml} ml)",
             "sizeSmall": f"the {len(small)} with the smaller tumour burden (< {split_ml} ml)",
             "sizeRand": f"a random {len(rand)} (seed {SEED})"}
    for k, ids in lists.items():
        json.dump({"rule": f"tumour-bearing volumes of Dataset104: {which[k]}; + all {len(free)} tumour-free volumes",
                   "n_pass": len(ids), "n_total": len(rows), "split_ml": split_ml, "ids": sorted(ids)},
                  open(f"{a.out}/ids_{k}.json", "w"), indent=1)
    b = {r[0]: r[1] for r in rows}
    rv = [r[1] for r in rrows if r[1] > 0]
    L = ["# Tumour-size split of the ccAucRepair training set (Dataset104)", "",
         f"`nnunet/size_split.py`. {len(rows)} volumes: {len(tum)} with tumour, {len(free)} tumour-free. "
         f"Median split at {split_ml} ml tumour per volume. Every list = its tumour volumes + all {len(free)} "
         "tumour-free volumes.", "",
         "| set | volumes | tumour volumes | tumour ml per tumour volume p10/25/50/75/90 | total tumour ml |",
         "|---|---|---|---|---|"]
    for k, ids in lists.items():
        t = [b[i] for i in ids if b[i] > 0]
        L.append(f"| {k} | {len(ids)} | {len(t)} | {q(t)} | {sum(t):.0f} |")
    t_all = [r[1] for r in tum]
    L += [f"| all (ccAucRepair) | {len(rows)} | {len(tum)} | {q(t_all)} | {sum(t_all):.0f} |",
          f"| real val (reference) | {len(rrows)} cases | {len(rv)} | {q(rv)} | {sum(rv):.0f} |", "",
          "Real test cases are not read. Train each arm as a new fold of Dataset104 (make_splits.py --dataset-id 104 "
          "--ids ...), 1000 epochs as every arm."]
    open(f"{a.out}/size_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
