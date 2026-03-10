#!/usr/bin/env python3
import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

# ========= CONFIGURATION =========
BASE_DIR = Path.cwd()  # Uses current UCloud session dir - adjust if needed
LITS_IMAGES = Path("./LiTS/imagesTr").resolve()
LITS_LABELS = Path("./LiTS/labelsTr").resolve()
MAISI_IMAGES = Path("./Maisi/images").resolve() 
MAISI_LABELS = Path("./Maisi/labels").resolve()

# Dataset IDs/names (nnUNet format: Dataset001_Name)
LITS_DATASET_ID = 1
LITS_DATASET_NAME = "LiTS"
COMBINED_DATASET_ID = 2  
COMBINED_DATASET_NAME = "LiTSMaisiCombined"

# Training settings (V100 16GB perfect for 3d_fullres)
CONFIG = "3d_fullres"
TRAINER = "nnUNetTrainer"
FOLDS = ["0", "1", "2", "3", "4"]  # Full 5-fold CV (nnUNet standard)
NUM_PROCESSES_PREPROCESS = 8  # Safe for 46GB RAM
# =================================

def setup_nnunet_dirs(base_dir: Path) -> Dict[str, Path]:
    """Create standard nnUNet directory structure."""
    dirs = {
        "raw": base_dir / "nnUNet_raw",
        "preprocessed": base_dir / "nnUNet_preprocessed", 
        "results": base_dir / "nnUNet_results"
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "nnUNet_raw": str(dirs["raw"]),
        "nnUNet_preprocessed": str(dirs["preprocessed"]),
        "nnUNet_results": str(dirs["results"])
    })
    return dirs

def find_pairs(images_dir: Path, labels_dir: Path, mode: str) -> List[Tuple[Path, Path]]:
    """Find image-label pairs based on naming convention."""
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in ('.nii', '.nii.gz')]
    labels = [p for p in labels_dir.iterdir() if p.suffix.lower() in ('.nii', '.nii.gz')]
    
    pairs = []
    label_map = {p.stem: p for p in labels}
    
    for img in sorted(images):
        if mode == "lits":
            # volume-0_0000.nii -> volume-0.nii
            if img.stem.endswith("_0000"):
                base_name = img.stem[:-5]  # Remove _0000
                if base_name in label_map:
                    pairs.append((img, label_map[base_name]))
        elif mode == "maisi":
            # maisi_001_0000.nii.gz -> maisi_001.nii.gz  
            if img.stem.endswith("_0000"):
                base_name = img.stem[:-5]
                if base_name in label_map:
                    pairs.append((img, label_map[base_name]))
    
    print(f"Found {len(pairs)} valid {mode.upper()} pairs")
    return pairs

def prepare_dataset(dataset_id: int, dataset_name: str, pairs: List[Tuple[Path, Path]], 
                   nnunet_raw: Path, start_case: int = 0):
    """Create nnUNet DatasetXXX folder with symlinks and dataset.json."""
    ds_id_str = f"{dataset_id:03d}"
    ds_folder = nnunet_raw / f"Dataset{ds_id_str}_{dataset_name}"
    imagesTr = ds_folder / "imagesTr"
    labelsTr = ds_folder / "labelsTr"
    
    # Clean and recreate
    shutil.rmtree(ds_folder, ignore_errors=True)
    ds_folder.mkdir(parents=True)
    imagesTr.mkdir()
    labelsTr.mkdir()
    
    # Create symlinks: case_000_0000.nii.gz <- volume-0_0000.nii
    for i, (img_path, label_path) in enumerate(pairs):
        case_id = f"{start_case + i:03d}"
        img_link = imagesTr / f"case_{case_id}_0000.nii.gz"
        label_link = labelsTr / f"case_{case_id}.nii.gz"
        
        img_link.symlink_to(img_path)
        label_link.symlink_to(label_path)
    
    # Write dataset.json
    dataset_json = {
        "name": dataset_name,
        "description": f"{dataset_name} liver/tumor segmentation (CT)",
        "tensorImageSize": "4D",
        "reference": "LiTS + Maisi datasets", 
        "licence": "",
        "release": "1.0",
        "modality": {"0": "CT"},
        "labels": {
            "0": "background",
            "1": "liver", 
            "2": "tumor"
        },
        "numTraining": len(pairs),
        "numTest": 0,
        "training": [],
        "test": []
    }
    
    for i in range(len(pairs)):
        case_id = f"{start_case + i:03d}"
        dataset_json["training"].append({
            "image": f"./imagesTr/case_{case_id}_0000.nii.gz",
            "label": f"./labelsTr/case_{case_id}.nii.gz"
        })
    
    with open(ds_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"✅ Prepared Dataset{ds_id_str}_{dataset_name} with {len(pairs)} cases")
    return len(pairs)

def run_command(cmd: List[str], description: str):
    """Run nnUNet command with nice output."""
    print(f"\n🚀 {description}")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print("✅ Done!")
    if result.stdout:
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)

def main():
    print("🎯 nnUNet Liver/Tumor Segmentation Pipeline")
    print(f"LiTS:     {LITS_IMAGES} ({LITS_IMAGES.exists()})")
    print(f"Maisi:    {MAISI_IMAGES} ({MAISI_IMAGES.exists()})")
    print(f"Output:   {BASE_DIR}/nnUNet_*")
    
    # Setup directories & env vars
    dirs = setup_nnunet_dirs(BASE_DIR)
    
    # Find data pairs
    lits_pairs = find_pairs(LITS_IMAGES, LITS_LABELS, "lits")  # ~91 cases
    maisi_pairs = find_pairs(MAISI_IMAGES, MAISI_LABELS, "maisi")  # ~90 cases
    
    if not lits_pairs:
        raise RuntimeError("No LiTS pairs found!")
    if not maisi_pairs:
        raise RuntimeError("No Maisi pairs found!")
    
    # Prepare datasets
    num_lits = prepare_dataset(LITS_DATASET_ID, LITS_DATASET_NAME, lits_pairs, dirs["raw"])
    prepare_dataset(COMBINED_DATASET_ID, COMBINED_DATASET_NAME, 
                   lits_pairs + maisi_pairs, dirs["raw"])
    
    # Plan & preprocess
    run_command([
        "nnUNetv2_plan_and_preprocess", "-d", str(LITS_DATASET_ID),
        "--verify_dataset_integrity", "-np", str(NUM_PROCESSES_PREPROCESS)
    ], f"Preprocessing Dataset{LITS_DATASET_ID:03d}_{LITS_DATASET_NAME}")
    
    run_command([
        "nnUNetv2_plan_and_preprocess", "-d", str(COMBINED_DATASET_ID), 
        "--verify_dataset_integrity", "-np", str(NUM_PROCESSES_PREPROCESS)
    ], f"Preprocessing Dataset{COMBINED_DATASET_ID:03d}_{COMBINED_DATASET_NAME}")
    
    # Train (all 5 folds)
    for dataset_id, dataset_name in [(LITS_DATASET_ID, LITS_DATASET_NAME), 
                                   (COMBINED_DATASET_ID, COMBINED_DATASET_NAME)]:
        for fold in FOLDS:
            run_command([
                "nnUNetv2_train", str(dataset_id), CONFIG, fold,
                "--trainer_class_name", TRAINER
            ], f"Training Dataset{dataset_id:03d}_{dataset_name} fold {fold}")
    
    print("\n🎉 COMPLETE!")
    print(f"Results: {dirs['results']}")
    print("Use nnUNetv2_predict for inference later")

if __name__ == "__main__":
    main()
