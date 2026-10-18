"""Turn one raw MAISI output pair into a training pair, plus quality metrics for the filtering study.

Inputs (raw, kept untouched): sample_*_image.nii.gz (float HU) and sample_*_label.nii.gz (132-class mask).
Outputs, in nnU-Net naming and RAS voxel orientation (see orient.py):
    pairs/<target>/imagesTr/<job_id>_0000.nii.gz   int16 HU
    pairs/<target>/labelsTr/<job_id>.nii.gz        uint8: organ 1, tumour 2, rest 0

Can also be run on its own, e.g. to redo one job:
    python postprocess.py --run-dir $MAISI_RUNS/trial10 --job liver_0003
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from orient import axcodes, reorient_array, spacing_of

# Same organ groups MAISI uses for its own check (scripts/sample.py, LDMSampler.label_int_dict).
MAISI_QC_LABELS = {
    "liver": [1],
    "spleen": [3],
    "pancreas": [4],
    "kidney": [5, 14],
    "lung": [28, 29, 30, 31],
    "brain": [22],
    "hepatic tumor": [26],
    "bone lesion": [128],
    "lung tumor": [23],
    "colon cancer primaries": [27],
    "pancreatic tumor": [24],
    "bone": list(range(33, 57)) + list(range(63, 98)) + [120, 122, 127],
}

TUMOR_DISTANCE_MM = 15.0  # thesis filter: keep if >= 90% of tumour voxels lie within 15 mm of the organ


def save_nifti(array, affine, path):
    out = nib.Nifti1Image(array, affine)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))


def tumor_fraction_near_organ(organ, tumor, spacing, max_mm):
    """Fraction of tumour voxels within max_mm of the nearest organ voxel (0 if no organ nearby)."""
    idx = np.argwhere(tumor)
    margin = np.ceil((max_mm + 1) / np.asarray(spacing)).astype(int)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, tumor.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    organ_box, tumor_box = organ[box], tumor[box]
    if not organ_box.any():
        return 0.0
    dist = ndimage.distance_transform_edt(~organ_box, sampling=spacing)
    return float((dist[tumor_box] <= max_mm).mean())


def components(mask):
    """Number of 26-connected components and the share of voxels in the largest one."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return 0, None
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), float(sizes.max() / sizes.sum())


def maisi_quality_check(image, label, maisi_dir):
    """Re-run MAISI's own image check (organ median HU inside the real-CT range) and return per-organ results."""
    if maisi_dir is None:
        return None
    sys.path.insert(0, str(maisi_dir))
    try:
        from scripts.quality_check import is_outlier
    finally:
        sys.path.pop(0)
    stats = json.loads((Path(maisi_dir) / "configs" / "image_median_statistics_ct.json").read_text())
    res = is_outlier(stats, image, label, MAISI_QC_LABELS)
    return {k: {kk: (vv.item() if isinstance(vv, np.generic) else vv) for kk, vv in v.items()} for k, v in res.items()}


def process(job, image_path, label_path, run_dir, maisi_dir=None):
    img_nii = nib.load(str(image_path))
    lab_nii = nib.load(str(label_path))
    image = np.asarray(img_nii.dataobj, dtype=np.float32)
    label = np.asarray(lab_nii.dataobj).astype(np.int16)
    if image.shape != label.shape:
        raise ValueError(f"image {image.shape} and label {label.shape} differ")
    raw_axcodes = axcodes(lab_nii.affine)
    image, affine = reorient_array(image, lab_nii.affine)
    label, _ = reorient_array(label, lab_nii.affine)
    spacing = spacing_of(affine)
    voxel_ml = float(np.prod(spacing)) / 1000.0

    organ = label == job["organ_label"]
    tumor = label == job["tumor_label"]
    remapped = np.zeros(label.shape, np.uint8)
    remapped[organ] = 1
    remapped[tumor] = 2

    out_root = Path(run_dir) / "pairs" / job["target"]
    image_out = out_root / "imagesTr" / f"{job['job_id']}_0000.nii.gz"
    label_out = out_root / "labelsTr" / f"{job['job_id']}.nii.gz"
    save_nifti(np.clip(np.rint(image), -1024, 3071).astype(np.int16), affine, image_out)
    save_nifti(remapped, affine, label_out)

    organ_n, organ_largest = components(organ)
    tumor_n, tumor_largest = components(tumor)
    qc = maisi_quality_check(image, label, maisi_dir)

    return {
        "pair_image": str(image_out),
        "pair_label": str(label_out),
        "shape": list(image.shape),
        "spacing": list(spacing),
        "raw_axcodes": raw_axcodes,
        "axcodes": axcodes(affine),
        "pair_bytes": image_out.stat().st_size + label_out.stat().st_size,
        "hu_min": float(image.min()),
        "hu_max": float(image.max()),
        "labels_present": int(len(np.unique(label))),
        "organ_ml": float(organ.sum() * voxel_ml),
        "tumor_ml": float(tumor.sum() * voxel_ml),
        "tumor_present": bool(tumor.any()),
        "organ_components": organ_n,
        "organ_largest_share": organ_largest,
        "tumor_components": tumor_n,
        "tumor_largest_share": tumor_largest,
        "organ_median_hu": float(np.median(image[organ])) if organ.any() else None,
        "tumor_median_hu": float(np.median(image[tumor])) if tumor.any() else None,
        f"tumor_frac_within_{int(TUMOR_DISTANCE_MM)}mm": tumor_fraction_near_organ(organ, tumor, spacing, TUMOR_DISTANCE_MM) if tumor.any() else None,
        "maisi_qc_recomputed_pass": None if qc is None else not any(v.get("is_outlier", False) for v in qc.values()),
        "maisi_qc_recomputed": qc,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--job", required=True, help="job id, e.g. liver_0003")
    p.add_argument("--maisi-dir", type=Path, default=None)
    args = p.parse_args()

    job = json.loads((args.run_dir / "jobs" / f"{args.job}.json").read_text())
    raw = args.run_dir / "raw" / args.job
    image = sorted(raw.glob("sample_*_image.nii.gz"))
    label = sorted(raw.glob("sample_*_label.nii.gz"))
    if len(image) != 1 or len(label) != 1:
        sys.exit(f"expected one image and one label in {raw}, found {len(image)} and {len(label)}")
    print(json.dumps(process(job, image[0], label[0], args.run_dir, args.maisi_dir), indent=2))


if __name__ == "__main__":
    main()
