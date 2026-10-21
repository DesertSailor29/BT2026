"""Bring NIfTI volumes into one voxel orientation (RAS, nibabel's canonical) so real and synthetic data match.

Only flips/permutes voxel axes (no resampling, values unchanged); the affine is updated so every voxel
keeps its world position. Used by postprocess.py for MAISI pairs, and as a CLI for real datasets:

    python orient.py --in  /work/Research/raw_data/LiTS --out /work/Research/raw_data/LiTS_RAS
    python orient.py --in  <dir> --check        # only report the orientations found

Mirrors the folder structure of --in, keeps file names, on-disk data types and compression.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation

TARGET = "RAS"


def axcodes(affine):
    return "".join(nib.aff2axcodes(affine))


def reorient_array(array, affine):
    """Return (array, affine) in RAS voxel order. The array view is made contiguous."""
    ornt = io_orientation(affine)
    new_affine = affine @ inv_ornt_aff(ornt, array.shape)
    return np.ascontiguousarray(apply_orientation(array, ornt)), new_affine


def spacing_of(affine):
    return tuple(float(v) for v in np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))


def exact_dtype(data, disk_dtype):
    """Integer-valued data goes back to the file's integer type, so it is saved without rescaling (bit-exact)."""
    if np.issubdtype(disk_dtype, np.integer) and np.issubdtype(data.dtype, np.floating):
        info = np.iinfo(disk_dtype)
        if data.min() >= info.min and data.max() <= info.max and not np.any(np.mod(data, 1)):
            return data.astype(disk_dtype)
    return data.astype(np.float32) if data.dtype == np.float64 else data


def reorient_file(src, dst):
    img = nib.load(str(src))
    before = axcodes(img.affine)
    data = exact_dtype(np.asanyarray(img.dataobj), img.get_data_dtype())
    data, affine = reorient_array(data, img.affine)
    out = nib.Nifti1Image(data, affine, header=img.header)
    out.set_data_dtype(data.dtype)
    out.set_qform(out.affine, code=1)
    out.set_sform(out.affine, code=1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    return before


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", dest="dst", type=Path, default=None)
    p.add_argument("--check", action="store_true", help="only count orientations, write nothing")
    args = p.parse_args()

    files = sorted(f for f in args.src.rglob("*") if f.name.endswith((".nii", ".nii.gz")))
    if not files:
        sys.exit(f"no .nii/.nii.gz files under {args.src}")
    if not args.check and args.dst is None:
        sys.exit("give --out (or --check)")
    if args.dst is not None and args.dst.resolve() == args.src.resolve():
        sys.exit("--out must differ from --in (originals are never overwritten)")

    counts = Counter()
    for i, f in enumerate(files, 1):
        if args.check:
            counts[axcodes(nib.load(str(f)).affine)] += 1
            continue
        dst = args.dst / f.relative_to(args.src)
        if dst.exists():
            counts["skipped (exists)"] += 1
            continue
        counts[reorient_file(f, dst)] += 1
        print(f"[{i}/{len(files)}] {f.relative_to(args.src)}", flush=True)
    print(("orientations found: " if args.check else f"done, now all {TARGET}. original orientations: ") + str(dict(counts)))


if __name__ == "__main__":
    main()
