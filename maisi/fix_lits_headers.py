"""Repair the broken NIfTI headers in the LiTS training release (run once, before orient.py).

Two known defects in the released files (the split script only moved files, so every copy has them):

  * volume-28..47 (= 3D-IRCADb-01 patients 1..20): image AND label carry an identity affine (1 mm isotropic, no
    direction). The true affines are in lits_headers_28_47.json, derived from the IRCAD DICOMs after a voxel-exact
    match (see the note in that file). Read with the identity header, L/R is mirrored and the spacing is wrong.
  * volume-48..52: the label carries an identity affine (with a 1 mm offset) while the image is correct; the voxel
    grids are identical, so the label gets the image's affine.

Only the header changes (affine, sform/qform code 1, units mm); voxel values are verified bit-exact after writing.
Files are rewritten in place via a temporary file. Idempotent: files that already have the right affine are skipped.

    python fix_lits_headers.py --lits $PROJ/raw_data/LiTS --dry-run
    python fix_lits_headers.py --lits $PROJ/raw_data/LiTS
"""

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

TABLE = Path(__file__).with_name("lits_headers_28_47.json")
LABEL_ONLY = range(48, 53)


def is_identity(affine):
    """Identity rotation/scale (any translation): the placeholder header of the broken files."""
    return np.allclose(affine[:3, :3], np.eye(3))


def find(lits, name):
    hits = [p for p in lits.glob(f"*/{name}") if p.is_file()] + [p for p in lits.glob(f"*/{name}.gz") if p.is_file()]
    if len(hits) != 1:
        sys.exit(f"expected exactly one {name}[.gz] under {lits}/*/, found {len(hits)}")
    return hits[0]


def rewrite(path, affine, dry_run):
    """Set the affine, keep the stored voxels and dtype. Returns 'fixed', 'ok' or raises on anything unexpected."""
    img = nib.load(str(path))
    if np.allclose(img.affine, affine, atol=1e-4):
        return "ok"
    if not is_identity(img.affine):
        raise SystemExit(f"{path.name}: affine is neither identity nor the expected one, not touching it:\n{img.affine}")
    slope, inter = img.header.get_slope_inter()
    if slope not in (None, 1) or inter not in (None, 0):
        raise SystemExit(f"{path.name}: intensity scaling {slope}/{inter} set, a header-only rewrite would change values")
    if dry_run:
        return "would fix"
    data = np.asanyarray(img.dataobj)
    out = nib.Nifti1Image(data, affine, header=img.header)
    out.set_data_dtype(img.get_data_dtype())
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    out.header.set_xyzt_units("mm", "sec")
    tmp = path.with_name(path.name.replace(".nii", ".tmp.nii"))
    nib.save(out, str(tmp))
    back = nib.load(str(tmp))
    if not (np.array_equal(np.asanyarray(back.dataobj), data) and back.get_data_dtype() == img.get_data_dtype()
            and np.allclose(back.affine, affine, atol=1e-4)):
        tmp.unlink()
        raise SystemExit(f"{path.name}: verification failed after writing, original left unchanged")
    os.replace(tmp, path)
    return "fixed"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lits", type=Path, required=True, help="split LiTS folder with imagesTr/labelsTr/imagesTs/labelsTs")
    p.add_argument("--dry-run", action="store_true", help="only report what would change")
    args = p.parse_args()

    table = json.loads(TABLE.read_text())["volumes"]
    jobs = []
    for n, entry in sorted(table.items(), key=lambda kv: int(kv[0])):
        affine = np.array(entry["affine"], float)
        jobs += [(find(args.lits, f"volume-{n}_0000.nii"), affine), (find(args.lits, f"volume-{n}.nii"), affine)]
    for n in LABEL_ONLY:
        jobs.append((find(args.lits, f"volume-{n}.nii"), nib.load(str(find(args.lits, f"volume-{n}_0000.nii"))).affine))

    counts = {}
    for path, affine in jobs:
        result = rewrite(path, affine, args.dry_run)
        counts[result] = counts.get(result, 0) + 1
        print(f"{result:9s} {path.relative_to(args.lits)}  -> {''.join(nib.aff2axcodes(affine))} "
              f"{tuple(round(float(v), 4) for v in np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))} mm", flush=True)
    print(f"done: {counts} ({len(jobs)} files)")


if __name__ == "__main__":
    main()
