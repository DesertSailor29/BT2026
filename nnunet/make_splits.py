#!/usr/bin/env python
"""Register an experiment arm as a new fold in splits_final.json (needs a preprocessed dataset).

Every fold: train = the arm's synthetic case ids, val = the dataset's fixed real validation cases (real_split.json).
Folds are append-only: an arm keeps its fold number forever (arms.json maps fold -> arm + how it was built).

  python make_splits.py --target liver --arm all                       # every synthetic case
  python make_splits.py --target liver --arm all_rep2                  # same cases again: run-to-run variation
  python make_splits.py --target liver --arm f15mm --ids f15mm.json    # filter output (JSON list or {"ids": [...]})
  python make_splits.py --target liver --arm f15mm_k300 --ids f15mm.json --k 300 --seed 0   # random K of a list
  python make_splits.py --target liver --list
Then train:  nnUNetv2_train <id> 3d_fullres <fold>
"""
import argparse
import glob
import json
import os
import random

from build_dataset import TARGETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS, default="liver")
    ap.add_argument("--arm")
    ap.add_argument("--ids", help="JSON list of synthetic case ids (or a dict with key 'ids')")
    ap.add_argument("--k", type=int, default=0, help="random K of the list (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    did, name, _, _ = TARGETS[a.target]
    ds = f"Dataset{did:03d}_{name}"
    raw, pre = f"{os.environ['nnUNet_raw']}/{ds}", f"{os.environ['nnUNet_preprocessed']}/{ds}"
    arms_f, splits_f = f"{pre}/arms.json", f"{pre}/splits_final.json"
    arms = json.load(open(arms_f)) if os.path.exists(arms_f) else []
    splits = json.load(open(splits_f)) if os.path.exists(splits_f) else []
    assert len(arms) == len(splits), "arms.json and splits_final.json out of sync"
    if a.list or not a.arm:
        for i, arm in enumerate(arms):
            print(f"fold {i}: {arm['arm']:20s} n_train={arm['n_train']}  {arm.get('ids_file') or ''}")
        return
    if any(x["arm"] == a.arm for x in arms):
        raise SystemExit(f"arm {a.arm} already registered; folds are append-only")

    real = json.load(open(f"{raw}/real_split.json"))
    val = [f"lits_{i:03d}" for i in real["val"]]
    synth = sorted(os.path.basename(f)[: -len(".nii.gz")] for f in glob.glob(f"{raw}/labelsTr/{a.target}_*.nii.gz"))
    if a.ids:
        d = json.load(open(a.ids))
        ids = sorted(d["ids"] if isinstance(d, dict) else d)
        missing = set(ids) - set(synth)
        assert not missing, f"{len(missing)} ids not in the dataset, e.g. {sorted(missing)[:3]}"
    else:
        assert a.arm == "all" or a.arm.startswith("all_rep"), "--ids is required unless --arm all / all_repN"
        ids = synth
    if a.k:
        assert a.k <= len(ids), (a.k, len(ids))
        ids = sorted(random.Random(a.seed).sample(ids, a.k))

    splits.append({"train": ids, "val": val})
    arms.append({"arm": a.arm, "n_train": len(ids), "ids_file": a.ids, "k": a.k, "seed": a.seed})
    json.dump(splits, open(splits_f, "w"), indent=1)
    json.dump(arms, open(arms_f, "w"), indent=1)
    print(f"fold {len(splits) - 1}: {a.arm}, {len(ids)} train, {len(val)} val")


if __name__ == "__main__":
    main()
