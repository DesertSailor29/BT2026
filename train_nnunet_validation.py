#!/usr/bin/env python3
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal, TypedDict

import nibabel as nib  # type: ignore
import numpy as np     # type: ignore


# ========= CONFIGURATION =========
BASE_DIR = Path.cwd()

LITS_IMAGES = Path("./LiTS/imagesTr").resolve()
LITS_LABELS = Path("./LiTS/labelsTr").resolve()
MAISI_IMAGES = Path("./Maisi/images").resolve()
MAISI_LABELS = Path("./Maisi/labels").resolve()

# Shared validation: fixed 20 LiTS cases across every dataset
SHARED_VAL_SIZE = 20
SHARED_VAL_SEED = 42

# Dataset IDs / names
LITS_TRAIN_ID = 3
LITS_TRAIN_NAME = "LiTS_Train"

MIXED_100_50_ID = 4
MIXED_100_50_NAME = "LiTSMaisi100_50_Train"

MIXED_100_80_ID = 5
MIXED_100_80_NAME = "LiTSMaisi100_80_Train"

MIXED_100_30_ID = 6
MIXED_100_30_NAME = "LiTSMaisi100_30_Train"

CONFIG = "3d_fullres"
TRAINER = "nnUNetTrainer2000epochs"
FOLD = "0"

# Safer defaults for 32 GB RAM host
NUM_PROCESSES_PREPROCESS = 8
OMP_NUM_THREADS = 4
NUM_GPUS = 2


Pair = Tuple[Path, Path]
class PreparedCase(TypedDict):
    case_id: str
    image: Path
    label: Path
    source: Literal["LiTS", "MAISI"]
    split_role: Literal["train", "val"]


def setup_nnunet_dirs(base_dir: Path) -> Dict[str, Path]:
    """Create standard nnUNet directory structure and export env vars."""
    os.environ["nnUNet_raw"] = str(base_dir / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(base_dir / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(base_dir / "nnUNet_results")

    dirs = {
        "raw": Path(os.environ["nnUNet_raw"]),
        "preprocessed": Path(os.environ["nnUNet_preprocessed"]),
        "results": Path(os.environ["nnUNet_results"]),
    }

    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    print("✅ nnUNet directories configured")
    for k, v in dirs.items():
        print(f"   {k}: {v}")
    return dirs


def configure_runtime() -> None:
    """Set environment variables for preprocessing and multi-GPU training."""
    os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(NUM_GPUS))
    os.environ["nnUNet_compile"] = "F"

    # Optional but sometimes useful:
    os.environ.setdefault("MKL_NUM_THREADS", str(OMP_NUM_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(OMP_NUM_THREADS))

    print("✅ Runtime environment configured")
    print(f"   CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"   OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    print(f"   nnUNet_compile={os.environ['nnUNet_compile']}")


def install_custom_trainer() -> None:
    """
    Install a custom trainer into the actual nnUNet v2 trainer package path
    so nnUNetv2_train can import it by name.
    """
    try:
        import nnunetv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "nnunetv2 is not importable in this Python environment. "
            "Activate the correct environment before running this script."
        ) from e

    nnunet_file = nnunetv2.__file__
    if nnunet_file is None:
        raise RuntimeError("nnunetv2.__file__ is None; cannot locate nnUNet installation.")
    nnunet_root = Path(nnunet_file).resolve().parent
    trainer_dir = nnunet_root / "training" / "nnUNetTrainer"
    trainer_dir.mkdir(parents=True, exist_ok=True)

    trainer_file = trainer_dir / f"{TRAINER}.py"
    trainer_code = '''import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer2000epochs(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.max_num_epochs = 2000
        print("Loaded nnUNetTrainer2000epochs with max_num_epochs=2000")
'''

    trainer_file.write_text(trainer_code, encoding="utf-8")
    print(f"✅ Custom trainer installed: {trainer_file}")

    # Try a direct import check
    module_name = f"nnunetv2.training.nnUNetTrainer.{TRAINER}"
    try:
        __import__(module_name, fromlist=[TRAINER])
        print(f"✅ Trainer import check passed: {module_name}")
    except Exception as e:
        raise RuntimeError(
            f"Custom trainer file was written, but import failed for {module_name}: {e}"
        ) from e


def create_shared_validation(lits_pairs: List[Pair]) -> List[Pair]:
    """Select fixed LiTS validation cases."""
    if len(lits_pairs) < SHARED_VAL_SIZE:
        raise ValueError(
            f"Need at least {SHARED_VAL_SIZE} LiTS cases, found only {len(lits_pairs)}."
        )

    rng = np.random.default_rng(SHARED_VAL_SEED)
    val_indices = sorted(rng.choice(len(lits_pairs), SHARED_VAL_SIZE, replace=False).tolist())
    val_pairs = [lits_pairs[i] for i in val_indices]

    print(f"✅ Shared validation selected: {len(val_pairs)} LiTS cases (seed={SHARED_VAL_SEED})")
    return val_pairs


def remove_validation_cases(all_pairs: List[Pair], val_pairs: List[Pair]) -> List[Pair]:
    """Remove chosen validation cases from the training pool."""
    val_image_names = {img.name for img, _ in val_pairs}
    train_pairs = [p for p in all_pairs if p[0].name not in val_image_names]

    print(f"✅ Training pool after removing validation cases: {len(train_pairs)} cases")
    return train_pairs


def validate_nifti_files(pairs: List[Pair], dataset_name: str) -> List[int]:
    """Basic NIfTI validation. Returns indices of unreadable/empty cases."""
    print(f"\n🔍 Validating {dataset_name} ({len(pairs)} cases)")
    bad_indices: List[int] = []

    for i, (img_path, label_path) in enumerate(pairs):
        try:
            if img_path.stat().st_size == 0 or label_path.stat().st_size == 0:
                print(f"  ❌ {i:03d}: empty file")
                bad_indices.append(i)
                continue

            img_nii = nib.load(str(img_path))
            lab_nii = nib.load(str(label_path))

            img_shape = img_nii.shape
            lab_shape = lab_nii.shape

            if len(img_shape) != len(lab_shape) or tuple(img_shape) != tuple(lab_shape):
                print(f"  ⚠️  {i:03d}: shape mismatch {img_shape} vs {lab_shape}")

            print(f"  ok {i:03d}: {img_path.name} | shape={img_shape}")

        except Exception as e:
            print(f"  ❌ {i:03d}: {img_path.name} failed to load: {e}")
            bad_indices.append(i)

    if bad_indices:
        print(f"⚠️  {len(bad_indices)} problematic cases in {dataset_name}")
    else:
        print("✅ Validation passed")

    return bad_indices


def find_lits_pairs(images_dir: Path, labels_dir: Path) -> List[Pair]:
    """LiTS pairing: volume-xxx_0000.nii.gz -> segmentation/volume-xxx.nii.gz style pairing."""
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

    print(f"LITS DEBUG: {len(images)} images, {len(labels)} labels")

    def normalize_base_name(p: Path) -> str:
        name = p.name
        if name.endswith(".nii.gz"):
            stem = name[:-7]
        elif name.endswith(".nii"):
            stem = name[:-4]
        else:
            stem = p.stem
        return stem

    label_map = {}
    for lab in labels:
        if lab.name in bad_volumes:
            continue
        label_map[normalize_base_name(lab)] = lab

    pairs: List[Pair] = []
    skipped = 0

    for img in images:
        img_base = normalize_base_name(img)
        if not img_base.endswith("_0000"):
            continue

        base_name = img_base[:-5]
        label_file = label_map.get(base_name)

        if label_file is None:
            skipped += 1
            print(f"⏭️  Missing label for {img.name}")
            continue

        if label_file.stat().st_size == 0:
            skipped += 1
            print(f"⏭️  Empty label for {img.name}")
            continue

        pairs.append((img, label_file))

    print(f"✅ LiTS good pairs: {len(pairs)} (skipped {skipped})")
    return pairs


def find_maisi_pairs(images_dir: Path, labels_dir: Path) -> List[Pair]:
    """MAISI pairing: maisi_001_0000.nii.gz -> maisi_001.nii.gz"""
    images = sorted([p for p in images_dir.iterdir() if p.name.lower().endswith((".nii", ".nii.gz"))])
    labels = sorted([p for p in labels_dir.iterdir() if p.name.lower().endswith((".nii", ".nii.gz"))])

    print(f"MAISI DEBUG: {len(images)} images, {len(labels)} labels")

    def normalize_base_name(p: Path) -> str:
        name = p.name
        if name.endswith(".nii.gz"):
            stem = name[:-7]
        elif name.endswith(".nii"):
            stem = name[:-4]
        else:
            stem = p.stem
        return stem

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

    print(f"✅ MAISI valid pairs: {len(pairs)}")
    return pairs


def filter_by_size(pairs: List[Pair], max_slices: int = 600) -> List[Pair]:
    """Optionally filter giant volumes."""
    filtered: List[Pair] = []
    skipped = 0

    for img_path, label_path in pairs:
        try:
            img_nii = nib.load(str(img_path))
            if len(img_nii.shape) >= 3 and img_nii.shape[2] <= max_slices:
                filtered.append((img_path, label_path))
            else:
                skipped += 1
                print(f"⏭️  Skipped giant case {img_path.name} ({img_nii.shape})")
        except Exception as e:
            skipped += 1
            print(f"❌ Skipping unreadable {img_path.name}: {e}")

    print(f"✅ Size filter: kept {len(filtered)}, skipped {skipped}")
    return filtered


def make_case_records(
    lits_train_pairs: List[Pair],
    lits_val_pairs: List[Pair],
    maisi_pairs: Optional[List[Pair]] = None,
) -> List[PreparedCase]:
    """
    Build explicit case records with stable nnUNet case IDs.
    Convention:
      - LiTS train IDs start at 000
      - LiTS val IDs continue after LiTS train
      - MAISI IDs continue after all LiTS
    """
    cases: List[PreparedCase] = []

    # LiTS train
    for i, (img, lab) in enumerate(lits_train_pairs):
        cases.append({
            "case_id": f"case_{i:03d}",
            "image": img,
            "label": lab,
            "source": "LiTS",
            "split_role": "train",
        })

    # LiTS val
    offset = len(cases)
    for j, (img, lab) in enumerate(lits_val_pairs):
        cases.append({
            "case_id": f"case_{offset + j:03d}",
            "image": img,
            "label": lab,
            "source": "LiTS",
            "split_role": "val",
        })

    # MAISI train-only
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


def prepare_dataset(
    dataset_id: int,
    dataset_name: str,
    case_records: List[PreparedCase],
    nnunet_raw: Path,
    copy_files: bool = True,
) -> Path:
    """Create DatasetXXX folder and dataset.json."""
    ds_folder = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = ds_folder / "imagesTr"
    labels_tr = ds_folder / "labelsTr"

    shutil.rmtree(ds_folder, ignore_errors=True)
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    training_entries = []

    for rec in case_records:
        case_id = rec["case_id"]
        img_src = rec["image"]
        lab_src = rec["label"]

        img_dst = images_tr / f"{case_id}_0000.nii.gz"
        lab_dst = labels_tr / f"{case_id}.nii.gz"

        if copy_files:
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lab_src, lab_dst)
        else:
            img_dst.symlink_to(img_src)
            lab_dst.symlink_to(lab_src)

        training_entries.append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz",
        })

    dataset_json = {
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

    with open(ds_folder / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"✅ Prepared Dataset{dataset_id:03d}_{dataset_name} with {len(case_records)} cases")
    return ds_folder


def write_manual_split(
    dataset_id: int,
    dataset_name: str,
    case_records: List[PreparedCase],
    nnunet_preprocessed: Path,
) -> Path:
    """
    Write splits_final.json for nnUNet v2.
    Fold 0 will use this train/val assignment.
    """
    ds_preprocessed = nnunet_preprocessed / f"Dataset{dataset_id:03d}_{dataset_name}"
    ds_preprocessed.mkdir(parents=True, exist_ok=True)

    train_ids = [str(r["case_id"]) for r in case_records if r["split_role"] == "train"]
    val_ids = [str(r["case_id"]) for r in case_records if r["split_role"] == "val"]

    split_obj = [{"train": train_ids, "val": val_ids}]
    split_path = ds_preprocessed / "splits_final.json"

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, indent=2)

    print(f"✅ Wrote manual split: {split_path}")
    print(f"   train={len(train_ids)} | val={len(val_ids)}")
    return split_path


def run_command(cmd: List[str], description: str) -> None:
    """Run a command and show stdout/stderr on failure."""
    print(f"\n🚀 {description}")
    print("$ " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print("❌ Command failed")
        print("STDOUT:")
        print(result.stdout[-4000:] if result.stdout else "<empty>")
        print("STDERR:")
        print(result.stderr[-4000:] if result.stderr else "<empty>")
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")

    print("✅ Done")
    if result.stdout:
        print(result.stdout[-1200:])


def balance_maisi(
    maisi_pairs: List[Pair],
    fraction: float,
    seed: int = 42,
) -> List[Pair]:
    """Sample a stable fraction of MAISI."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    n_total = len(maisi_pairs)
    n_take = max(1, int(round(n_total * fraction)))
    n_take = min(n_take, n_total)

    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(n_total, size=n_take, replace=False).tolist())
    subset = [maisi_pairs[i] for i in idx]

    print(f"✅ MAISI subset: {n_take}/{n_total} ({fraction*100:.0f}%)")
    return subset


def save_selection_report(
    out_path: Path,
    dataset_name: str,
    lits_train_count: int,
    lits_val_count: int,
    maisi_pairs: List[Pair],
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"LiTS train: {lits_train_count}\n")
        f.write(f"LiTS val: {lits_val_count}\n")
        f.write(f"MAISI train: {len(maisi_pairs)}\n\n")
        f.write("Selected MAISI files:\n")
        for img, _ in maisi_pairs:
            f.write(f"{img.name}\n")


def preprocess_dataset(dataset_id: int) -> None:
    run_command(
        ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-np", str(NUM_PROCESSES_PREPROCESS)],
        f"Preprocess Dataset{dataset_id:03d}",
    )


def train_dataset(dataset_id: int) -> None:
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        CONFIG,
        FOLD,
        "-tr",
        TRAINER,
        "-num_gpus",
        str(NUM_GPUS),
    ]
    run_command(cmd, f"Train Dataset{dataset_id:03d}")


def main() -> None:
    configure_runtime()
    dirs = setup_nnunet_dirs(BASE_DIR)
    install_custom_trainer()

    # -------- 1) Load and validate LiTS --------
    lits_pairs_full = find_lits_pairs(LITS_IMAGES, LITS_LABELS)
    bad_lits = validate_nifti_files(lits_pairs_full, "LiTS full")
    if bad_lits:
        lits_pairs_full = [p for i, p in enumerate(lits_pairs_full) if i not in bad_lits]
        print(f"✅ LiTS after dropping bad cases: {len(lits_pairs_full)}")

    # Optional: uncomment if giant volumes are a practical problem on your machine
    # lits_pairs_full = filter_by_size(lits_pairs_full, max_slices=600)

    # Shared validation split, actually used everywhere
    lits_val_pairs = create_shared_validation(lits_pairs_full)
    lits_train_pairs = remove_validation_cases(lits_pairs_full, lits_val_pairs)

    # -------- 2) Load and validate MAISI --------
    maisi_pairs_full = find_maisi_pairs(MAISI_IMAGES, MAISI_LABELS)
    bad_maisi = validate_nifti_files(maisi_pairs_full, "MAISI full")
    if bad_maisi:
        maisi_pairs_full = [p for i, p in enumerate(maisi_pairs_full) if i not in bad_maisi]
        print(f"✅ MAISI after dropping bad cases: {len(maisi_pairs_full)}")

    # Optional: uncomment if giant volumes are a practical problem on your machine
    # maisi_pairs_full = filter_by_size(maisi_pairs_full, max_slices=600)

    # -------- 3) Define all datasets --------
    dataset_specs = [
        {
            "id": LITS_TRAIN_ID,
            "name": LITS_TRAIN_NAME,
            "maisi_fraction": 0.0,
        },
        {
            "id": MIXED_100_50_ID,
            "name": MIXED_100_50_NAME,
            "maisi_fraction": 0.5,
        },
        {
            "id": MIXED_100_80_ID,
            "name": MIXED_100_80_NAME,
            "maisi_fraction": 0.8,
        },
        {
            "id": MIXED_100_30_ID,
            "name": MIXED_100_30_NAME,
            "maisi_fraction": 0.3,
        },
    ]

    # -------- 4) Prepare, preprocess, and write manual split for each dataset --------
    for spec in dataset_specs:
        ds_id = spec["id"]
        ds_name = spec["name"]
        maisi_fraction = spec["maisi_fraction"]

        print(f"\n{'='*80}")
        print(f"📁 Dataset{ds_id:03d}_{ds_name}")
        print(f"{'='*80}")

        if maisi_fraction > 0.0:
            maisi_subset = balance_maisi(maisi_pairs_full, float(maisi_fraction), seed=SHARED_VAL_SEED)
        else:
            maisi_subset = []

        case_records = make_case_records(
            lits_train_pairs=lits_train_pairs,
            lits_val_pairs=lits_val_pairs,
            maisi_pairs=maisi_subset,
        )

        prepare_dataset(ds_id, ds_name, case_records, dirs["raw"], copy_files=True)
        preprocess_dataset(ds_id)
        write_manual_split(ds_id, ds_name, case_records, dirs["preprocessed"])

        report_path = BASE_DIR / f"dataset_{ds_id:03d}_selection_report.txt"
        save_selection_report(
            report_path,
            ds_name,
            lits_train_count=len(lits_train_pairs),
            lits_val_count=len(lits_val_pairs),
            maisi_pairs=maisi_subset,
        )
        print(f"✅ Selection report saved: {report_path}")

    # -------- 5) Train all datasets on fold 0 using the same shared LiTS validation --------
    print(f"\n{'='*80}")
    print("🎓 TRAINING")
    print(f"{'='*80}")
    print("All runs use the same manually defined validation split:")
    print(f"  - {len(lits_val_pairs)} LiTS validation cases")
    print("  - MAISI is train-only in the mixed datasets")
    print(f"  - Trainer: {TRAINER}")
    print(f"  - GPUs: {NUM_GPUS}")

    for spec in dataset_specs:
        train_dataset(spec["id"])

    print("\n🎉 Finished successfully")
    print("Summary:")
    print("  Dataset003: LiTS only")
    print("  Dataset004: LiTS + 50% MAISI")
    print("  Dataset005: LiTS + 80% MAISI")
    print("  Dataset006: LiTS + 30% MAISI")
    print("  Shared validation: {len(lits_val_pairs)} LiTS cases across all datasets")


if __name__ == "__main__":
    main()