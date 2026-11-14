#!/usr/bin/env python
"""The fixed MSD Task07 Pancreas val/test split (pancreas target; the counterpart of Dataset101's real_split.json).

The 281 labelled MSD07 cases (imagesTr/labelsTr; imagesTs has no labels) are split once with a fixed seed:
    val   VAL_N cases: the real reference of the filter rules (AUC cut) and, later, nnU-Net's checkpoint selection
    test  the rest: never read for a decision (real_lesion_metrics.py describes them, nothing more)
The split is written to maisi/msd07_split.json (case numbers only: public dataset, no voxels) on first use and never
changes afterwards; a later call with another seed / val_n refuses instead of silently re-splitting.

Real case names are msd_<NNN> (MSD's pancreas_<NNN>) so they can't be mixed up with the pool's pancreas_<NNNN>.

    python maisi/msd07_split.py --root $PROJ/raw_data/Task07_Pancreas      # create (or show) the split
"""
import argparse
import glob
import json
import os
import random
import re
from pathlib import Path

SPLIT = Path(__file__).resolve().parent / "msd07_split.json"
VAL_N, SEED = 43, 42  # 43/281 = 15%, the same fraction as liver (20/131); user 2026-11-14
EXPECTED = 281
ROOT = f"{os.environ.get('PROJ', '/work/Research')}/raw_data/Task07_Pancreas"


def case_name(i):
    return f"msd_{i:03d}"


def labelled_ids(root):
    files = [f for f in glob.glob(f"{root}/labelsTr/pancreas_*.nii.gz") if not os.path.basename(f).startswith("._")]
    ids = sorted(int(re.fullmatch(r"pancreas_(\d+)\.nii\.gz", os.path.basename(f)).group(1)) for f in files)
    for i in ids:
        if not os.path.exists(f"{root}/imagesTr/pancreas_{i:03d}.nii.gz"):
            raise SystemExit(f"{root}: label pancreas_{i:03d} has no image")
    return ids


def load_or_make(root=ROOT, val_n=VAL_N, seed=SEED, path=SPLIT, expected=EXPECTED):
    """The split dict; created from the labelled ids under `root` if `path` doesn't exist yet."""
    path = Path(path)
    if path.exists():
        s = json.loads(path.read_text())
        if (s["seed"], s["val_n"]) != (seed, val_n):
            raise SystemExit(f"{path} has seed {s['seed']} / val_n {s['val_n']}, asked for {seed} / {val_n}; "
                             "the split is fixed (delete the file only if nothing has used it yet)")
        return s
    ids = labelled_ids(root)
    if expected and len(ids) != expected:
        raise SystemExit(f"{root}: {len(ids)} labelled cases, expected {expected}")
    val = sorted(random.Random(seed).sample(ids, val_n))
    s = {"source": "MSD Task07 Pancreas imagesTr/labelsTr", "seed": seed, "val_n": val_n,
         "rule": f"random.Random({seed}).sample(sorted labelled ids, {val_n}) = val; the rest = test",
         "val": val, "test": [i for i in ids if i not in val]}
    path.write_text(json.dumps(s, indent=1) + "\n")
    print(f"wrote {path}: {len(val)} val, {len(s['test'])} test", flush=True)
    return s


def cases(split, which, root=ROOT):
    """[(case name, label path, image path)] of 'val' or 'test'."""
    return [(case_name(i), f"{root}/labelsTr/pancreas_{i:03d}.nii.gz", f"{root}/imagesTr/pancreas_{i:03d}.nii.gz")
            for i in split[which]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--val-n", type=int, default=VAL_N)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    s = load_or_make(a.root, a.val_n, a.seed)
    print(f"val {len(s['val'])}: {s['val']}\ntest {len(s['test'])}")
