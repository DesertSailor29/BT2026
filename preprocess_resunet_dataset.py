#!/usr/bin/env python3
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import nibabel as nib  # type: ignore
import numpy as np  # type: ignore
import SimpleITK as sitk  # type: ignore


# ========= CONFIGURATION =========
BASE_DIR = Path.cwd()

LITS_IMAGES = Path("./LiTS/imagesTr").resolve()
LITS_LABELS = Path("./LiTS/labelsTr").resolve()
MAISI_IMAGES = Path("./Maisi/images").resolve()
MAISI_LABELS = Path("./Maisi/labels").resolve()

SHARED_VAL_SIZE = 20
SHARED_VAL_SEED = 42

OUTPUT_ROOT = (BASE_DIR / "preprocessed_resunet").resolve()



# Which datasets to build/preprocess
DATASET_SPECS = [
    {
        "id": 3,
        "name": "LiTS_ResNet",
        "maisi_fraction": 0.0,
        "plans_json": "./nnUNet_preprocessed/Dataset003_LiTS_Train/nnUNetPlans.json",
        "dataset_json": "./nnUNet_preprocessed/Dataset003_LiTS_Train/dataset.json",
    },
    {
        "id": 4,
        "name": "LiTSMaisi100_30_ResNet",
        "maisi_fraction": 0.3,
        "plans_json": "./nnUNet_preprocessed/Dataset004_LiTSMaisi100_30_Train/nnUNetPlans.json",
        "dataset_json": "./nnUNet_preprocessed/Dataset004_LiTSMaisi100_30_Train/dataset.json",
    },
]
"""
    {"id": 5, "name": "LiTSMaisi100_50_Train", "maisi_fraction": 0.5},
    {"id": 6, "name": "LiTSMaisi100_80_Train", "maisi_fraction": 0.8},
    {"id": 7, "name": "LiTSMaisi100_100_Train", "maisi_fraction": 1.0},
"""
    
# nnU-Net config to mirror
CONFIG_NAME = "3d_fullres"

# Save as .nii.gz after preprocessing
OVERWRITE = False


Pair = Tuple[Path, Path]


class PreparedCase(TypedDict):
    case_id: str
    image: Path
    label: Path
    source: Literal["LiTS", "MAISI"]
    split_role: Literal["train", "val"]


@dataclass
class PreprocessConfig:
    config_name: str
    target_spacing_zyx: Tuple[float, float, float]
    clip_lower: float
    clip_upper: float
    mean: float
    std: float
    labels: Dict[str, int]
    patch_size_zyx: Optional[Tuple[int, int, int]] = None


# ========= DATASET BUILDING: SAME LOGIC AS YOUR nnU-Net SCRIPT =========

def create_shared_validation(lits_pairs: List[Pair]) -> List[Pair]:
    if len(lits_pairs) < SHARED_VAL_SIZE:
        raise ValueError(
            f"Need at least {SHARED_VAL_SIZE} LiTS cases, found only {len(lits_pairs)}."
        )
    rng = np.random.default_rng(SHARED_VAL_SEED)
    val_indices = sorted(rng.choice(len(lits_pairs), SHARED_VAL_SIZE, replace=False).tolist())
    return [lits_pairs[i] for i in val_indices]


def remove_validation_cases(all_pairs: List[Pair], val_pairs: List[Pair]) -> List[Pair]:
    val_image_names = {img.name for img, _ in val_pairs}
    return [p for p in all_pairs if p[0].name not in val_image_names]


def validate_nifti_files(pairs: List[Pair], dataset_name: str) -> List[int]:
    print(f"\nValidating {dataset_name} ({len(pairs)} cases)")
    bad_indices: List[int] = []

    for i, (img_path, label_path) in enumerate(pairs):
        try:
            if img_path.stat().st_size == 0 or label_path.stat().st_size == 0:
                print(f"  BAD {i:03d}: empty file")
                bad_indices.append(i)
                continue

            img_nii = nib.load(str(img_path))
            lab_nii = nib.load(str(label_path))

            if tuple(img_nii.shape) != tuple(lab_nii.shape):
                print(f"  WARN {i:03d}: shape mismatch {img_nii.shape} vs {lab_nii.shape}")

        except Exception as e:
            print(f"  BAD {i:03d}: {img_path.name} failed to load: {e}")
            bad_indices.append(i)

    print(f"Finished validation: {len(pairs) - len(bad_indices)} OK, {len(bad_indices)} bad")
    return bad_indices


def find_lits_pairs(images_dir: Path, labels_dir: Path) -> List[Pair]:
    bad_volumes = {
        "volume-61.nii", "volume-61.nii.gz",
        "volume-63.nii", "volume-63.nii.gz",
        "volume-64.nii", "volume-64.nii.gz",
        "volume-65.nii", "volume-65.nii.gz",
        "volume-66.nii", "volume-66.nii.gz",
        "volume-67.nii", "volume-67.nii.gz",
        "volume-73.nii", "volume-73.nii.gz",
    }

    images = sorted([p for p in images_dir.iterdir() if p.name.lower().endswith((".nii", ".nii.gz"))])
    labels = sorted([p for p in labels_dir.iterdir() if p.name.lower().endswith((".nii", ".nii.gz"))])

    def normalize_base_name(p: Path) -> str:
        name = p.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return p.stem

    label_map = {}
    for lab in labels:
        if lab.name in bad_volumes:
            continue
        label_map[normalize_base_name(lab)] = lab

    pairs: List[Pair] = []
    for img in images:
        img_base = normalize_base_name(img)
        if not img_base.endswith("_0000"):
            continue
        base_name = img_base[:-5]
        label_file = label_map.get(base_name)
        if label_file is None:
            continue
        if label_file.stat().st_size == 0:
            continue
        pairs.append((img, label_file))

    print(f"LiTS good pairs: {len(pairs)}")
    return pairs


def find_maisi_pairs(images_dir: Path, labels_dir: Path) -> List[Pair]:
    images = sorted([p for p in images_dir.iterdir() if p.name.lower().endswith((".nii", ".nii.gz"))])
    labels = sorted([p for p in labels_dir.iterdir() if p.name.lower().endswith((".nii", ".nii.gz"))])

    def normalize_base_name(p: Path) -> str:
        name = p.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return p.stem

    label_map = {normalize_base_name(p): p for p in labels}
    pairs: List[Pair] = []

    for img in images:
        img_base = normalize_base_name(img)
        if not img_base.endswith("_0000"):
            continue
        base_name = img_base[:-5]
        label_file = label_map.get(base_name)
        if label_file is not None:
            pairs.append((img, label_file))

    print(f"MAISI valid pairs: {len(pairs)}")
    return pairs


def balance_maisi(maisi_pairs: List[Pair], fraction: float, seed: int = 42) -> List[Pair]:
    if fraction <= 0.0:
        return []
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    n_total = len(maisi_pairs)
    n_take = max(1, int(round(n_total * fraction)))
    n_take = min(n_take, n_total)

    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(n_total, size=n_take, replace=False).tolist())
    subset = [maisi_pairs[i] for i in idx]

    print(f"MAISI subset: {n_take}/{n_total} ({fraction*100:.0f}%)")
    return subset


def make_case_records(
    lits_train_pairs: List[Pair],
    lits_val_pairs: List[Pair],
    maisi_pairs: Optional[List[Pair]] = None,
) -> List[PreparedCase]:
    cases: List[PreparedCase] = []

    for i, (img, lab) in enumerate(lits_train_pairs):
        cases.append({
            "case_id": f"case_{i:03d}",
            "image": img,
            "label": lab,
            "source": "LiTS",
            "split_role": "train",
        })

    offset = len(cases)
    for j, (img, lab) in enumerate(lits_val_pairs):
        cases.append({
            "case_id": f"case_{offset + j:03d}",
            "image": img,
            "label": lab,
            "source": "LiTS",
            "split_role": "val",
        })

    if maisi_pairs:
        offset = len(cases)
        for k, (img, lab) in enumerate(maisi_pairs):
            cases.append({
                "case_id": f"case_{offset + k:03d}",
                "image": img,
                "label": lab,
                "source": "MAISI",
                "split_role": "train",
            })

    return cases


# ========= PREPROCESSING =========

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_config(plans: dict, dataset_json: dict, config_name: str) -> PreprocessConfig:
    cfg = plans["configurations"][config_name]

    spacing_list = [float(x) for x in cfg["spacing"]]
    if len(spacing_list) != 3:
        raise ValueError(f"{config_name}: expected 3 spacing values, got {spacing_list}")
    spacing: Tuple[float, float, float] = (
        spacing_list[0],
        spacing_list[1],
        spacing_list[2],
    )

    patch_size: Optional[Tuple[int, int, int]] = None
    if "patch_size" in cfg:
        patch_list = [int(x) for x in cfg["patch_size"]]
        if len(patch_list) != 3:
            raise ValueError(f"{config_name}: expected 3 patch values, got {patch_list}")
        patch_size = (
            patch_list[0],
            patch_list[1],
            patch_list[2],
        )

    fg = plans["foreground_intensity_properties_per_channel"]["0"]
    clip_lower = float(fg["percentile_00_5"])
    clip_upper = float(fg["percentile_99_5"])
    mean = float(fg["mean"])
    std = float(fg["std"])

    labels = {str(k): int(v) for k, v in dataset_json["labels"].items()}

    return PreprocessConfig(
        config_name=config_name,
        target_spacing_zyx=spacing,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        mean=mean,
        std=std,
        labels=labels,
        patch_size_zyx=patch_size,
    )


def read_sitk(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def assert_same_geometry(img: sitk.Image, seg: sitk.Image, case_id: str) -> None:
    if img.GetSize() != seg.GetSize():
        raise ValueError(f"{case_id}: size mismatch: {img.GetSize()} vs {seg.GetSize()}")
    if tuple(round(x, 6) for x in img.GetSpacing()) != tuple(round(x, 6) for x in seg.GetSpacing()):
        raise ValueError(f"{case_id}: spacing mismatch: {img.GetSpacing()} vs {seg.GetSpacing()}")
    if tuple(round(x, 6) for x in img.GetOrigin()) != tuple(round(x, 6) for x in seg.GetOrigin()):
        raise ValueError(f"{case_id}: origin mismatch")
    if tuple(round(x, 6) for x in img.GetDirection()) != tuple(round(x, 6) for x in seg.GetDirection()):
        raise ValueError(f"{case_id}: direction mismatch")


def compute_new_size(old_size, old_spacing, new_spacing):
    return [
        int(np.round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(old_size, old_spacing, new_spacing)
    ]


def resample_image(
    image: sitk.Image,
    new_spacing_xyz: Tuple[float, float, float],
    is_label: bool,
) -> sitk.Image:
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = compute_new_size(old_size, old_spacing, new_spacing_xyz)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing_xyz)
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkUInt8)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputPixelType(sitk.sitkFloat32)

    return resampler.Execute(image)


def normalize_ct_array(arr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    arr = np.clip(arr, cfg.clip_lower, cfg.clip_upper)
    arr = (arr - cfg.mean) / max(cfg.std, 1e-8)
    return arr.astype(np.float32, copy=False)


def write_nifti_from_array_like(
    arr_zyx: np.ndarray,
    reference_img: sitk.Image,
    out_path: Path,
    is_label: bool,
) -> None:
    out = sitk.GetImageFromArray(arr_zyx)
    out.SetSpacing(reference_img.GetSpacing())
    out.SetOrigin(reference_img.GetOrigin())
    out.SetDirection(reference_img.GetDirection())

    if is_label:
        out = sitk.Cast(out, sitk.sitkUInt8)
    else:
        out = sitk.Cast(out, sitk.sitkFloat32)

    sitk.WriteImage(out, str(out_path), useCompression=True)


def preprocess_case(
    case_id: str,
    image_path: Path,
    label_path: Path,
    cfg: PreprocessConfig,
    out_images_dir: Path,
    out_labels_dir: Path,
    overwrite: bool = False,
) -> dict:
    img_out = out_images_dir / f"{case_id}_0000.nii.gz"
    lbl_out = out_labels_dir / f"{case_id}.nii.gz"

    if img_out.exists() and lbl_out.exists() and not overwrite:
        return {
            "case_id": case_id,
            "image": str(img_out),
            "label": str(lbl_out),
            "status": "skipped_existing",
        }

    img = read_sitk(image_path)
    seg = read_sitk(label_path)
    assert_same_geometry(img, seg, case_id)

    # nnU-Net spacing is listed z,y,x in plans. SimpleITK uses x,y,z.
    target_spacing_xyz = tuple(reversed(cfg.target_spacing_zyx))

    img_r = resample_image(img, target_spacing_xyz, is_label=False)
    seg_r = resample_image(seg, target_spacing_xyz, is_label=True)

    img_arr = sitk.GetArrayFromImage(img_r).astype(np.float32, copy=False)
    seg_arr = sitk.GetArrayFromImage(seg_r).astype(np.uint8, copy=False)

    img_arr = normalize_ct_array(img_arr, cfg)

    write_nifti_from_array_like(img_arr, img_r, img_out, is_label=False)
    write_nifti_from_array_like(seg_arr, seg_r, lbl_out, is_label=True)

    return {
        "case_id": case_id,
        "image_in": str(image_path),
        "label_in": str(label_path),
        "image_out": str(img_out),
        "label_out": str(lbl_out),
        "orig_spacing_xyz": list(img.GetSpacing()),
        "new_spacing_xyz": list(img_r.GetSpacing()),
        "orig_size_xyz": list(img.GetSize()),
        "new_size_xyz": list(img_r.GetSize()),
        "status": "done",
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_dataset_json_like_nnunet(case_records: List[PreparedCase], out_dir: Path) -> None:
    training_entries = []
    for rec in case_records:
        cid = rec["case_id"]
        training_entries.append({
            "image": f"./images/{cid}_0000.nii.gz",
            "label": f"./labels/{cid}.nii.gz",
        })

    obj = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "liver": 1,
            "tumor": 2,
        },
        "numTraining": len(case_records),
        "file_ending": ".nii.gz",
        "training": training_entries,
    }

    with (out_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_split_like_nnunet(case_records: List[PreparedCase], out_dir: Path) -> None:
    train_ids = [r["case_id"] for r in case_records if r["split_role"] == "train"]
    val_ids = [r["case_id"] for r in case_records if r["split_role"] == "val"]

    split_obj = [{"train": train_ids, "val": val_ids}]
    with (out_dir / "splits_final.json").open("w", encoding="utf-8") as f:
        json.dump(split_obj, f, indent=2)

    with (out_dir / "train_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            [{"case_id": cid} for cid in train_ids],
            f,
            indent=2,
        )

    with (out_dir / "val_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            [{"case_id": cid} for cid in val_ids],
            f,
            indent=2,
        )


def save_selection_report(
    out_path: Path,
    dataset_name: str,
    case_records: List[PreparedCase],
) -> None:
    train_cases = [r for r in case_records if r["split_role"] == "train"]
    val_cases = [r for r in case_records if r["split_role"] == "val"]
    maisi_cases = [r for r in case_records if r["source"] == "MAISI"]

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Train cases: {len(train_cases)}\n")
        f.write(f"Val cases: {len(val_cases)}\n")
        f.write(f"MAISI train cases: {len(maisi_cases)}\n\n")

        f.write("Validation case IDs:\n")
        for r in val_cases:
            f.write(f"{r['case_id']} -> {r['image'].name}\n")

        f.write("\nMAISI selected:\n")
        for r in maisi_cases:
            f.write(f"{r['case_id']} -> {r['image'].name}\n")


def preprocess_dataset_spec(
    spec: dict,
    lits_train_pairs: List[Pair],
    lits_val_pairs: List[Pair],
    maisi_pairs_full: List[Pair],
    cfg: PreprocessConfig,
) -> None:
    ds_id = spec["id"]
    ds_name = spec["name"]
    maisi_fraction = float(spec["maisi_fraction"])

    print(f"\n{'=' * 80}")
    print(f"Dataset{ds_id:03d}_{ds_name}")
    print(f"{'=' * 80}")

    maisi_subset = balance_maisi(maisi_pairs_full, maisi_fraction, seed=SHARED_VAL_SEED)

    case_records = make_case_records(
        lits_train_pairs=lits_train_pairs,
        lits_val_pairs=lits_val_pairs,
        maisi_pairs=maisi_subset,
    )

    out_dir = OUTPUT_ROOT / f"Dataset{ds_id:03d}_{ds_name}"
    out_images_dir = out_dir / "images"
    out_labels_dir = out_dir / "labels"

    if OVERWRITE and out_dir.exists():
        shutil.rmtree(out_dir)

    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)

    results: List[dict] = []
    for idx, rec in enumerate(case_records, start=1):
        print(f"[{idx}/{len(case_records)}] {rec['case_id']} | {rec['source']} | {rec['split_role']}")
        result = preprocess_case(
            case_id=rec["case_id"],
            image_path=rec["image"],
            label_path=rec["label"],
            cfg=cfg,
            out_images_dir=out_images_dir,
            out_labels_dir=out_labels_dir,
            overwrite=OVERWRITE,
        )
        result["source"] = rec["source"]
        result["split_role"] = rec["split_role"]
        results.append(result)

    save_dataset_json_like_nnunet(case_records, out_dir)
    save_split_like_nnunet(case_records, out_dir)
    save_selection_report(out_dir / "selection_report.txt", ds_name, case_records)

    meta = {
        "dataset_id": ds_id,
        "dataset_name": ds_name,
        "config_name": cfg.config_name,
        "target_spacing_zyx": list(cfg.target_spacing_zyx),
        "clip_lower": cfg.clip_lower,
        "clip_upper": cfg.clip_upper,
        "mean": cfg.mean,
        "std": cfg.std,
        "patch_size_zyx": None if cfg.patch_size_zyx is None else list(cfg.patch_size_zyx),
        "num_cases": len(case_records),
        "num_train": sum(1 for r in case_records if r["split_role"] == "train"),
        "num_val": sum(1 for r in case_records if r["split_role"] == "val"),
        "results": results,
    }
    with (out_dir / "preprocessing_run.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Finished: {out_dir}")


def main() -> None:
    lits_pairs_full = find_lits_pairs(LITS_IMAGES, LITS_LABELS)
    bad_lits = validate_nifti_files(lits_pairs_full, "LiTS full")
    if bad_lits:
        lits_pairs_full = [p for i, p in enumerate(lits_pairs_full) if i not in bad_lits]
        print(f"LiTS after dropping bad cases: {len(lits_pairs_full)}")

    lits_val_pairs = create_shared_validation(lits_pairs_full)
    lits_train_pairs = remove_validation_cases(lits_pairs_full, lits_val_pairs)

    print(f"Shared LiTS validation: {len(lits_val_pairs)}")
    print(f"LiTS train pool: {len(lits_train_pairs)}")

    maisi_pairs_full = find_maisi_pairs(MAISI_IMAGES, MAISI_LABELS)
    bad_maisi = validate_nifti_files(maisi_pairs_full, "MAISI full")
    if bad_maisi:
        maisi_pairs_full = [p for i, p in enumerate(maisi_pairs_full) if i not in bad_maisi]
        print(f"MAISI after dropping bad cases: {len(maisi_pairs_full)}")

    ensure_dir(OUTPUT_ROOT)

    for spec in DATASET_SPECS:
        plans = load_json(Path(spec["plans_json"]).resolve())
        dataset_json = load_json(Path(spec["dataset_json"]).resolve())
        cfg = build_config(plans, dataset_json, CONFIG_NAME)

        print("\nUsing preprocessing config:")
        print(f"  dataset: {spec['name']}")
        print(f"  config: {CONFIG_NAME}")
        print(f"  target spacing zyx: {cfg.target_spacing_zyx}")
        print(f"  patch size zyx: {cfg.patch_size_zyx}")
        print(f"  clip: [{cfg.clip_lower}, {cfg.clip_upper}]")
        print(f"  mean/std: {cfg.mean} / {cfg.std}")

        preprocess_dataset_spec(
            spec=spec,
            lits_train_pairs=lits_train_pairs,
            lits_val_pairs=lits_val_pairs,
            maisi_pairs_full=maisi_pairs_full,
            cfg=cfg,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()