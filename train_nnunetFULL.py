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

LITS_DATASET_ID = 1
LITS_DATASET_NAME = "LiTS"
COMBINED_DATASET_ID = 2  
COMBINED_DATASET_NAME = "LiTSMaisiCombined"
LITS_FULL_DATASET_ID = 3
LITS_FULL_DATASET_NAME = "LiTS_Full"
FULL_MIXED_DATASET_ID = 4
FULL_MIXED_DATASET_NAME = "LiTSMaisiFullMixed"
MIXED_DATASET_100_50_ID = 5
MIXED_DATASET_100_50_NAME = "LiTSMaisi100_50"


CONFIG = "3d_fullres"
TRAINER = "nnUNetTrainer"
FOLDS = ["0"] # Only fold 0, since comperative evaluation can be done with only one fold.
NUM_PROCESSES_PREPROCESS = 8 # System limit has been reached before, so I set it to 8 for safety

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



def main():
    print("🎯 Full LiTS Pipeline: Dataset003_LiTS_Full (all valid cases)")
    
    # Optimal env for 12 cores / 250GB / A100
    os.environ['NNUNET_NUM_PROCESSES'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NNUNET_RANDOMIZE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['nnUNet_compile'] = 'F'

    dirs = setup_nnunet_dirs(BASE_DIR)
    """
    # 1. LiTS FULL dataset (Dataset003)
    print("\n📁 1. Preparing Dataset003_LiTS_Full (all valid cases)...")
    lits_pairs = find_lits_pairs(LITS_IMAGES, LITS_LABELS)
    
    # Remove slice filtering to use ALL cases (or keep with max_slices=9999)
    # lits_pairs = filter_by_size(lits_pairs, max_slices=9999)  
    prepare_dataset(LITS_FULL_DATASET_ID, LITS_FULL_DATASET_NAME, lits_pairs, dirs["raw"], copy_files=True)
    validate_nifti_files(lits_pairs, "LiTS_Full")

    # 2. Preprocess Dataset003
    print("\n🚀 2. Preprocessing Dataset003_LiTS_Full ...")
    run_command([
        "nnUNetv2_plan_and_preprocess", "-d", "3", 
        "-np", "1",
    ], "Preprocess Dataset003_LiTS_Full")

    # 3. Train Dataset003 (fold 0)
    print("\n🎓 3. Training Dataset003_LiTS_Full model...")
    run_command([
        "nnUNetv2_train", "3", CONFIG, FOLDS[0],
    ], "Train Dataset003_LiTS_Full fold 0")

    print("\n🎉 COMPLETE!")
    print(f"LiTS_Full model: {dirs['results']}/Dataset003_LiTS_Full/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0")
    """

    """
    # 4. FULL MIXED dataset (Dataset004)  
    print("\n📁 4. Preparing Dataset004_LiTSMaisiFullMixed (ALL cases)...")
    lits_pairs_full = find_lits_pairs(LITS_IMAGES, LITS_LABELS)
    maisi_pairs_full = find_maisi_pairs(MAISI_IMAGES, MAISI_LABELS)
    mixed_pairs = lits_pairs_full + maisi_pairs_full  # ← Just concatenate!

    bad_cases = validate_nifti_files(mixed_pairs, "LiTSMaisiFullMixed")
    if bad_cases:
        print(f"🗑️  Removing {len(bad_cases)} bad cases before dataset prep...")
        mixed_pairs = [p for i, p in enumerate(mixed_pairs) if i not in bad_cases]
        print(f"✅ {len(mixed_pairs)} good cases remaining")
    prepare_dataset(FULL_MIXED_DATASET_ID, FULL_MIXED_DATASET_NAME, mixed_pairs, dirs["raw"], copy_files=True)

    # 5. Preprocess Dataset004
    print("\n🚀 5. Preprocessing Dataset004_LiTSMaisiFullMixed ...")
    run_command([
        "nnUNetv2_plan_and_preprocess", "-d", "4", 
        "-np", "1",
    ], "Preprocess Dataset004_LiTSMaisiFullMixed")

    # 6. Train Dataset004 (fold 0)
    print("\n🎓 6. Training Dataset004_LiTSMaisiFullMixed model...")
    run_command([
        "nnUNetv2_train", "4", CONFIG, FOLDS[0],
    ], "Train Dataset004_LiTSMaisiFullMixed fold 0")

    print("\n🎉 COMPLETE!")
    print(f"FullMixed model: {dirs['results']}/Dataset004_LiTSMaisiFullMixed/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0")
    """

    # 5. MIXED dataset (Dataset005) with 100% LiTS + 50% MAISI  
    print("\n📁 5. Preparing Dataset005_LiTSMaisi100_50 (ALL cases)...")
    lits_pairs_full = find_lits_pairs(LITS_IMAGES, LITS_LABELS)
    maisi_pairs_full = find_maisi_pairs(MAISI_IMAGES, MAISI_LABELS)
    mixed_pairs = lits_pairs_full + maisi_pairs_full  # ← Just concatenate!

    mixed_pairs = balance_datasets(lits_pairs_full, maisi_pairs_full, target_n=0, seed=42)
    bad_cases = validate_nifti_files(mixed_pairs, "LiTSMaisi100_50")
    if bad_cases:
        print(f"🗑️  Removing {len(bad_cases)} bad cases before dataset prep...")
        mixed_pairs = [p for i, p in enumerate(mixed_pairs) if i not in bad_cases]
        print(f"✅ {len(mixed_pairs)} good cases remaining")
    prepare_dataset(MIXED_DATASET_100_50_ID, MIXED_DATASET_100_50_NAME, mixed_pairs, dirs["raw"], copy_files=True)

    # 5. Preprocess Dataset005
    print("\n🚀 5. Preprocessing Dataset005_LiTSMaisi100_50 ...")
    run_command([
        "nnUNetv2_plan_and_preprocess", "-d", "5", 
        "-np", "1",
    ], "Preprocess Dataset005_LiTSMaisi100_50")

    # 6. Train Dataset005 (fold 0)
    print("\n🎓 6. Training Dataset005_LiTSMaisi100_50 model...")
    run_command([
        "nnUNetv2_train", "5", CONFIG, FOLDS[0],
    ], "Train Dataset005_LiTSMaisi100_50 fold 0")

    print("\n🎉 COMPLETE!")
    print(f"100_50 model: {dirs['results']}/Dataset005_LiTSMaisi100_50/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0")

if __name__ == "__main__":
    main()
