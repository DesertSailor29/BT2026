#!/usr/bin/env python3
import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import nibabel as nib # type: ignore
import numpy as np # type: ignore

# ========= CONFIGURATION =========
BASE_DIR = Path.cwd()
LITS_IMAGES = Path("./LiTS/imagesTr").resolve()
LITS_LABELS = Path("./LiTS/labelsTr").resolve()
MAISI_IMAGES = Path("./Maisi/images").resolve() 
MAISI_LABELS = Path("./Maisi/labels").resolve()

# *** SHARED VALIDATION (20 LiTS cases) ***
SHARED_VAL_SIZE = 20
SHARED_VAL_SEED = 42

# *** NEW DATASET STRUCTURE (skipping 1-2) ***
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
FOLDS = ["0"]

# *** 32GB + 2x RTX 5070 OPTIMIZED ***
NUM_PROCESSES_PREPROCESS = 12  # Good for 32GB RAM
NUM_THREADS = 4  # Per GPU thread optimization

def setup_nnunet_dirs(base_dir: Path) -> Dict[str, Path]:
    """Create standard nnUNet directory structure with env vars first."""
    os.environ['nnUNet_raw'] = str(base_dir / "nnUNet_raw")
    os.environ['nnUNet_preprocessed'] = str(base_dir / "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = str(base_dir / "nnUNet_results")
    
    dirs = {
        "raw": Path(os.environ['nnUNet_raw']),
        "preprocessed": Path(os.environ['nnUNet_preprocessed']),
        "results": Path(os.environ['nnUNet_results'])
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ nnUNet env vars set: raw={dirs['raw']}")
    return dirs

def create_shared_validation(lits_pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    """Extract fixed LiTS validation set (20 cases)."""
    np.random.seed(SHARED_VAL_SEED)
    val_indices = np.random.choice(len(lits_pairs), SHARED_VAL_SIZE, replace=False)
    val_pairs = [lits_pairs[i] for i in sorted(val_indices)]
    print(f"✅ SHARED VALIDATION: {len(val_pairs)} LiTS cases (seed={SHARED_VAL_SEED})")
    return val_pairs

def remove_validation_cases(pairs: List[Tuple[Path, Path]], val_pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    """Remove shared validation cases from training pairs."""
    val_filenames = {p[0].name for p in val_pairs}
    train_pairs = [p for p in pairs if p[0].name not in val_filenames]
    print(f"   → Removed {len(pairs) - len(train_pairs)} validation cases")
    print(f"   → {len(train_pairs)} training cases remaining")
    return train_pairs

# ==== VALIDATION & PREPARATION FUNCTIONS ====
def validate_nifti_files(pairs: List[Tuple[Path, Path]], dataset_name: str):
    """FIXED: Safe validation that handles LiTS spacing issues."""
    print(f"\n🔍 Validating {dataset_name} ({len(pairs)} cases)...")
    bad_cases = []
    
    for i, (img_path, label_path) in enumerate(pairs):
        try:
            img_nii = nib.load(str(img_path))  # str() fixes Path issue
            label_nii = nib.load(str(label_path))
            
            # SAFE spacing extraction - handles LiTS tuple issues
            img_spacing = img_nii.header.get_zooms()
            if isinstance(img_spacing, tuple):
                img_spacing = [float(s) if np.isfinite(s) else 1.0 for s in img_spacing]
            else:
                img_spacing = [float(img_spacing)]
            
            label_spacing = label_nii.header.get_zooms()
            if isinstance(label_spacing, tuple):
                label_spacing = [float(s) if np.isfinite(s) else 1.0 for s in label_spacing]
            
            img_shape = img_nii.shape
            label_shape = label_nii.shape
            
            # Skip detailed print to avoid format errors, just check basics
            print(f"  case {i:03d}: img_shape={img_shape}, label_shape={label_shape}")
            
            # Check file sizes (catches empty files)
            if img_path.stat().st_size == 0 or label_path.stat().st_size == 0:
                print(f"  ❌ case {i}: Empty file!")
                bad_cases.append(i)
                continue
                
            # Basic shape match check
            if len(img_shape) != len(label_shape) or any(a != b for a, b in zip(img_shape, label_shape)):
                print(f"  ⚠️  case {i}: Shape mismatch {img_shape} vs {label_shape}")
            
        except Exception as e:
            print(f"❌ case {i}: {str(e)[:100]}")
            bad_cases.append(i)
    
    if bad_cases:
        print(f"\n⚠️  {len(bad_cases)} problematic cases found. Continuing anyway...")
        print(f"First 5: {bad_cases[:5]}")
    else:
        print("✅ All cases validated!")
    
    # Don't fail on LiTS known issues - let nnUNet handle it
    return bad_cases

# Remove blacklisted items, see if it can be handled on a better machine.
def find_lits_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """LiTS: Skip ALL known corrupt cases BY FILENAME."""
    # Skip these EXACT filenames (not case numbers)
    BAD_VOLUMES = {
        'volume-61.nii', 'volume-61.nii.gz',
        'volume-63.nii', 'volume-63.nii.gz', 
        'volume-64.nii', 'volume-64.nii.gz',  # ← NEW
        'volume-65.nii', 'volume-65.nii.gz',
        'volume-66.nii', 'volume-66.nii.gz',
        'volume-67.nii', 'volume-67.nii.gz',
        'volume-73.nii', 'volume-73.nii.gz'
    }
    
    images = [p for p in images_dir.iterdir() if p.name.lower().endswith(('.nii.gz', '.nii'))]
    labels = [p for p in labels_dir.iterdir() if p.name.lower().endswith(('.nii.gz', '.nii'))]
    
    print(f"LITS DEBUG: {len(images)} images, {len(labels)} labels")
    pairs = []
    label_map = {p.stem.split('.')[0]: p for p in labels if p.name not in BAD_VOLUMES}
    
    skipped = 0
    for img in sorted(images):
        img_base = img.stem.split('.')[0]
        if img_base.endswith("_0000"):
            base_name = img_base[:-5]
            label_file = label_map.get(base_name)
            
            # Skip empty OR known bad files
            if (label_file and label_file.stat().st_size > 0 and 
                label_file.name not in BAD_VOLUMES):
                pairs.append((img, label_file))
            else:
                skipped += 1
                print(f"⏭️  Skipping {base_name} (empty/bad)")
    
    print(f"✅ {len(pairs)} GOOD pairs (skipped {skipped} bad)")
    return pairs



def find_maisi_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Maisi: maisi_001_0000.nii.gz → maisi_001.nii.gz"""  
    images = [p for p in images_dir.iterdir() if p.name.lower().endswith(('.nii.gz', '.nii'))]
    labels = [p for p in labels_dir.iterdir() if p.name.lower().endswith(('.nii.gz', '.nii'))]
    
    print(f"MAISI DEBUG: {len(images)} images, {len(labels)} labels")
    pairs = []
    label_map = {p.stem.split('.')[0]: p for p in labels}
    
    for img in sorted(images):
        img_base = img.stem.split('.')[0]
        if img_base.endswith("_0000"):
            base_name = img_base[:-5]
            if base_name in label_map:
                pairs.append((img, label_map[base_name]))
    
    print(f"Found {len(pairs)} valid MAISI pairs")
    return pairs


def prepare_dataset(dataset_id: int, dataset_name: str, pairs: List[Tuple[Path, Path]], 
                   nnunet_raw: Path, start_case: int = 0, copy_files: bool = True):
    """Create nnUNet DatasetXXX folder. Copy files to avoid symlink issues."""
    ds_id_str = f"{dataset_id:03d}"
    ds_folder = nnunet_raw / f"Dataset{ds_id_str}_{dataset_name}"
    imagesTr = ds_folder / "imagesTr"
    labelsTr = ds_folder / "labelsTr"
    
    shutil.rmtree(ds_folder, ignore_errors=True)
    ds_folder.mkdir(parents=True)
    imagesTr.mkdir()
    labelsTr.mkdir()
    
    for i, (img_path, label_path) in enumerate(pairs):
        case_id = f"{start_case + i:03d}"
        img_dest = imagesTr / f"case_{case_id}_0000.nii.gz"
        label_dest = labelsTr / f"case_{case_id}.nii.gz"
        
        if copy_files:
            shutil.copy2(img_path, img_dest)
            shutil.copy2(label_path, label_dest)
            print(f"📋 Copied case_{case_id}")
        else:
            img_dest.symlink_to(img_path)
            label_dest.symlink_to(label_path)
            print(f"🔗 Symlinked case_{case_id}")
    
    # Complete dataset.json with all required fields
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1, "tumor": 2},
        "numTraining": len(pairs),
        "file_ending": ".nii.gz"
    }
    
    for i in range(len(pairs)):
        case_id = f"{start_case + i:03d}"
        dataset_json.setdefault("training", []).append({
            "image": f"./imagesTr/case_{case_id}_0000.nii.gz",
            "label": f"./labelsTr/case_{case_id}.nii.gz"
        })
    
    with open(ds_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"✅ Prepared Dataset{ds_id_str}_{dataset_name} with {len(pairs)} cases")
    return len(pairs)

# Remove to see, if a better machine can handle it.
def filter_by_size(pairs: List[Tuple[Path, Path]], max_slices: int = 600) -> List[Tuple[Path, Path]]:
    """Skip giant volumes >600 slices"""
    filtered = []
    skipped = 0
    for img_path, label_path in pairs:
        try:
            img_nii = nib.load(str(img_path))
            if img_nii.shape[2] <= max_slices:  # Z-dimension
                filtered.append((img_path, label_path))
            else:
                skipped += 1
                print(f"⏭️  Skipped giant case {img_path.name} ({img_nii.shape[2]} slices)")
        except Exception as e:
            print(f"❌ Skipping unreadable {img_path}: {e}")
            continue  # Keep if unreadable
            
    print(f"✅ Filtered {len(filtered)} cases (skipped {skipped} giants)")
    return filtered


def run_command(cmd: List[str], description: str):
    """Run nnUNet command with full error output."""
    print(f"\n🚀 {description}")
    print(f"$ {' '.join(cmd)}")
    print(f"ENV: nnUNet_raw={os.environ.get('nnUNet_raw')}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True, env=os.environ)
        print("✅ Done!")
        if result.stdout:
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout[:1000] if e.stdout else 'None'}")
        print(f"STDERR: {e.stderr[:1000] if e.stderr else 'None'}")
        print("\n💡 Try running manually: cd /work/Bachelor\\ Thesis &&", ' '.join(cmd))
        raise

def balance_datasets(lits_pairs: List[Tuple[Path, Path]], maisi_pairs: List[Tuple[Path, Path]], 
                    target_n: int = 0, seed: int = 42) -> List[Tuple[Path, Path]]:
    """Select target_n MAISI + ALL LiTS with fixed seed."""
    np.random.seed(seed)
    
    # Use ALL LiTS (100%)
    n_lits = len(lits_pairs)
    
    # target_n=0 → 50% MAISI, else use target_n
    if target_n == 0:
        target_n = int(len(maisi_pairs) * 0.5)
    n_maisi = min(target_n, len(maisi_pairs))
    
    print(f"\n  Balancing: {n_lits} LiTS (100%) + {n_maisi}/{len(maisi_pairs)} MAISI ({n_maisi/len(maisi_pairs)*100:.0f}%)")
    
    # ALL LiTS + random MAISI subset
    maisi_sample = np.random.choice(len(maisi_pairs), n_maisi, replace=False)
    balanced_lits = lits_pairs  # 100% LiTS
    balanced_maisi = [maisi_pairs[i] for i in maisi_sample]
    
    # Log selected MAISI cases
    print("\n📋 SELECTED MAISI CASES (indices in original list):")
    for idx, orig_idx in enumerate(maisi_sample):
        case_name = maisi_pairs[orig_idx][0].stem
        print(f"  MAISI[{orig_idx:3d}] → case_{n_lits+idx:03d}: {case_name}")
    
    print(f"\n✅ {len(balanced_lits)} LiTS + {len(balanced_maisi)} MAISI = {len(balanced_lits)+len(balanced_maisi)} total (seed={seed})")
    
    # Save selection
    with open("selected_maisi_50pct.txt", "w") as f:
        f.write(f"MAISI subset for Dataset004 (seed={seed}):\n")
        f.write(f"LiTS: {len(balanced_lits)} (100%)\n")
        f.write(f"MAISI: {len(balanced_maisi)}/{len(maisi_pairs)} ({len(balanced_maisi)/len(maisi_pairs)*100:.1f}%)\n\n")
        f.write("MAISI indices: " + str(sorted(maisi_sample.tolist())) + "\n")
    
    print("💾 Selection saved: selected_maisi_50pct.txt")
    return balanced_lits + balanced_maisi



#!/usr/bin/env python3
# [Keep ALL your imports and functions EXACTLY as-is until main()]

def main():
    # *** ENV SETUP + TRAINER FIRST ***
    os.environ['NNUNET_NUM_PROCESSES'] = str(NUM_PROCESSES_PREPROCESS)
    os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
    os.environ['NNUNET_RANDOMIZE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['nnUNet_compile'] = 'F'

    dirs = setup_nnunet_dirs(BASE_DIR)
    
    # Create trainer
    print("\n🔧 Creating nnUNetTrainer2000epochs.py")
    trainer_code = '''import torch
from nnunetv2.training.nnunet.trainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainer2000epochs(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.max_num_epochs = 2000
        print(f"🔥 2000 epochs trainer loaded!")
'''
    with open(dirs["results"] / "nnUNetTrainer2000epochs.py", 'w') as f: 
        f.write(trainer_code)

    # *** MASTER Dataset003 (creates validation split) ***
    print("\n📁 1. Dataset003_LiTS_Train (MASTER VALIDATION SOURCE)")
    lits_pairs_full = find_lits_pairs(LITS_IMAGES, LITS_LABELS)
    bad_cases = validate_nifti_files(lits_pairs_full, LITS_TRAIN_NAME)
    if bad_cases: 
        lits_pairs_full = [p for i,p in enumerate(lits_pairs_full) if i not in bad_cases]
    
    prepare_dataset(LITS_TRAIN_ID, LITS_TRAIN_NAME, lits_pairs_full, dirs["raw"])
    print("\n🚀 Preprocess Dataset003 (creates master splits_final.pkl)")
    run_command(["nnUNetv2_plan_and_preprocess", "-d", "3", "-np", str(NUM_PROCESSES_PREPROCESS)], "Preprocess 003")
    
    # Wait for splits_final.pkl to be created
    import time
    splits_003_path = dirs["preprocessed"] / "Dataset003_LiTS_Train" / "splits_final.pkl"
    while not splits_003_path.exists():
        print("⏳ Waiting for splits_final.pkl...")
        time.sleep(2)
    print(f"✅ Master splits ready: {splits_003_path}")

    # *** GET MAISI PAIRS ONCE ***
    maisi_pairs_full = find_maisi_pairs(MAISI_IMAGES, MAISI_LABELS)

    # *** Datasets 4-6 with COPIED splits ***
    dataset_configs = [
        (4, MIXED_100_50_NAME, lambda: balance_datasets(lits_pairs_full, maisi_pairs_full, 0, 42)),
        (5, MIXED_100_80_NAME, lambda: balance_datasets(lits_pairs_full, maisi_pairs_full, int(len(maisi_pairs_full)*0.8), 42)),
        (6, MIXED_100_30_NAME, lambda: balance_datasets(lits_pairs_full, maisi_pairs_full, int(len(maisi_pairs_full)*0.3), 42))
    ]
    
    for ds_id, ds_name, get_pairs in dataset_configs:
        print(f"\n📁 Dataset{ds_id:03d}_{ds_name}")
        mixed_pairs = get_pairs()
        bad_cases = validate_nifti_files(mixed_pairs, ds_name)
        if bad_cases: 
            mixed_pairs = [p for i,p in enumerate(mixed_pairs) if i not in bad_cases]
        prepare_dataset(ds_id, ds_name, mixed_pairs, dirs["raw"])
        
        print(f"\n🚀 Preprocess Dataset{ds_id:03d}")
        run_command(["nnUNetv2_plan_and_preprocess", "-d", str(ds_id), "-np", str(NUM_PROCESSES_PREPROCESS)], f"Preprocess {ds_id}")
        
        # CRITICAL: Copy splits_final.pkl from Dataset003
        ds_preprocessed = dirs["preprocessed"] / f"Dataset{ds_id:03d}_{ds_name}"
        shutil.copy2(splits_003_path, ds_preprocessed / "splits_final.pkl")
        print(f"✅ SAME fold_0 validation → Dataset{ds_id:03d}")

    # *** TRAIN ALL (IDENTICAL Dataset003 fold_0 validation!) ***
    print("\n🎓 TRAINING (2000 epochs, IDENTICAL Dataset003 fold_0 validation)")
    for ds_id, ds_name in [(3, LITS_TRAIN_NAME), (4, MIXED_100_50_NAME), (5, MIXED_100_80_NAME), (6, MIXED_100_30_NAME)]:
        print(f"\n🎓 Dataset{ds_id:03d}_{ds_name}")
        run_command(["nnUNetv2_train", str(ds_id), CONFIG, FOLDS[0], "--trainer_class_name", TRAINER], f"Train {ds_id}")

    print("\n🎉 PERFECT! All 4 models = SAME LiTS fold_0 validation + 2000 epochs")