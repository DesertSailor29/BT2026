#!/usr/bin/env python3
"""
Compare multiple model predictions vs ground truth LiTS segmentations + VISUALS.
Liver=1, Tumor=2. Outputs CSV + publication-ready figures.

UPDATED: Layout B (GT vs each model per case), max tumor slice, fewer images.
"""

from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff


# ========== CONFIGURE YOUR PATHS HERE ==========
GT_DIR = Path("./LiTS/labelsTs")                    # Ground truth labels
CT_DIR = Path("./LiTS/imagesTs")                    # CT images (for overlays)
PRED_DIRS = [                                   # List ALL your prediction folders
    Path("./Predictions/Dataset001_LiTS"),
    Path("./Predictions/Dataset002_LiTSMaisiCombined"),
    Path("./Predictions/Dataset003_LiTS_Full"),
    Path("./Predictions/Dataset004_LiTSMaisiFullMixed"),
    Path("./Predictions/Dataset005_LiTSMaisi100_50"),
]
OUTPUT_CSV = Path("./Predictions/liver_tumor_metrics.csv")
FIG_DIR = Path("./Predictions/figures")
SAVE_FIGS = True                               # Set False to disable figures
# ==============================================


# Setup plotting
plt.style.use('default')
sns.set_palette("husl")
if SAVE_FIGS:
    FIG_DIR.mkdir(exist_ok=True)


def find_max_tumor_slice(gt: np.ndarray) -> int:
    """Return index of axial slice with largest tumor area (label 2)."""
    tumor_per_slice = np.sum(gt == 2, axis=(1, 2))
    if tumor_per_slice.sum() == 0:
        return gt.shape[0] // 2  # Fallback: middle slice
    return int(np.argmax(tumor_per_slice))


def compute_dice(pred, gt, label):
    """Standard Dice coefficient."""
    pred_voxels = np.sum(pred == label)
    gt_voxels = np.sum(gt == label)
    intersection = np.sum((pred == label) & (gt == label))
    if pred_voxels + gt_voxels == 0:
        return 1.0
    return 2.0 * intersection / (pred_voxels + gt_voxels)


def compute_hausdorff(pred, gt, label, percentile=100):
    """Hausdorff distance (full or 95th percentile)."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    if not pred_mask.any() or not gt_mask.any():
        return np.nan
    
    pred_coords = np.argwhere(pred_mask)
    gt_coords = np.argwhere(gt_mask)
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.nan
    
    # Voxel distances first
    hd12 = directed_hausdorff(gt_coords, pred_coords)[0]
    hd21 = directed_hausdorff(pred_coords, gt_coords)[0]
    hd_voxel = max(hd12, hd21)
    
    if percentile < 100:
        dist12 = np.min(np.linalg.norm(gt_coords[:, None] - pred_coords[None, :], axis=-1), axis=1)
        dist21 = np.min(np.linalg.norm(pred_coords[:, None] - gt_coords[None, :], axis=-1), axis=1)
        hd12_p95 = np.percentile(dist12, percentile)
        hd21_p95 = np.percentile(dist21, percentile)
        hd_voxel = max(hd12_p95, hd21_p95)
    
    return hd_voxel


def save_case_comparison(case_id, info):
    """Layout B: GT + one row per model, all on same max-tumor slice."""
    gt_path = info["gt_path"]
    ct_path = info["ct_path"]
    slice_idx = info["slice_idx"]
    models = info["models"]  # {model_name: {"pred_path": Path, **metrics}}

    gt_img = sitk.ReadImage(str(gt_path))
    gt = sitk.GetArrayFromImage(gt_img)

    ct = None
    if ct_path and ct_path.exists():
        ct_img = sitk.ReadImage(str(ct_path))
        ct = sitk.GetArrayFromImage(ct_img)
        min_shape = np.minimum(gt.shape, ct.shape)
        gt = gt[:min_shape[0], :min_shape[1], :min_shape[2]]
        ct = ct[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Colormap: bg black, liver green, tumor red
    colors = ['black', '#00FF00', '#FF0000']
    cmap = ListedColormap(colors)

    n_models = len(models)
    n_rows = 1 + n_models
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(f"{case_id} - axial slice {slice_idx}", fontsize=14)

    # GT row
    ax = axes[0]
    if ct is not None:
        ax.imshow(ct[slice_idx], cmap='gray', vmin=-200, vmax=300)
        ax.imshow(gt[slice_idx], cmap=cmap, alpha=0.4, vmin=0, vmax=2)
        ax.set_title("Ground Truth")
    else:
        ax.imshow(gt[slice_idx], cmap=cmap, vmin=0, vmax=2)
        ax.set_title("Ground Truth (labels only)")
    ax.axis('off')

    # Model rows
    for row_idx, (model_name, m) in enumerate(models.items(), start=1):
        pred_img = sitk.ReadImage(str(m["pred_path"]))
        pred = sitk.GetArrayFromImage(pred_img)
        min_shape = np.minimum(gt.shape, pred.shape)
        pred = pred[:min_shape[0], :min_shape[1], :min_shape[2]]

        ax = axes[row_idx]
        if ct is not None:
            ax.imshow(ct[slice_idx], cmap='gray', vmin=-200, vmax=300)
            ax.imshow(pred[slice_idx], cmap=cmap, alpha=0.4, vmin=0, vmax=2)
        else:
            ax.imshow(pred[slice_idx], cmap=cmap, vmin=0, vmax=2)

        dice = m['dice_avg']
        hd95 = m['hd95_avg']
        ax.set_title(f"{model_name}\nDice={dice:.3f}, HD95={hd95:.1f}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{case_id}_all_models_slice{slice_idx}.png", dpi=200, bbox_inches='tight')
    plt.close()


def save_summary_plots(df):
    """Boxplots and heatmaps from CSV results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison Summary (LiTS Liver/Tumor)', fontsize=16, fontweight='bold')
    
    # Dice boxplot
    df.boxplot(column='dice_avg', by='model', ax=axes[0,0], patch_artist=True)
    axes[0,0].set_title('Dice Score (Average)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # HD95 boxplot
    df.boxplot(column='hd95_avg', by='model', ax=axes[0,1], patch_artist=True)
    axes[0,1].set_title('HD95 Distance (mm)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Dice heatmap
    pivot_dice = df.pivot(index='model', columns='case_id', values='dice_avg')
    sns.heatmap(pivot_dice, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Dice'}, ax=axes[1,0])
    axes[1,0].set_title('Dice Heatmap (Model × Case)')
    
    # HD95 heatmap
    pivot_hd = df.pivot(index='model', columns='case_id', values='hd95_avg')
    sns.heatmap(pivot_hd, annot=True, fmt='.1f', cmap='Reds', 
                cbar_kws={'label': 'HD95 (mm)'}, ax=axes[1,1])
    axes[1,1].set_title('HD95 Heatmap (Model × Case)')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "summary_all_models.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Summary plot saved: {FIG_DIR / 'summary_all_models.png'}")


def process_pair(gt_path, pred_path, ct_path, case_id, model):
    """Compute metrics + slice index - NO plotting here."""
    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))
    
    gt = sitk.GetArrayFromImage(gt_img)
    pred = sitk.GetArrayFromImage(pred_img)
    spacing = np.array(gt_img.GetSpacing())[::-1]
    
    if gt.shape != pred.shape:
        print(f"⚠️ {case_id}: shape {gt.shape} vs {pred.shape} - cropping")
        min_shape = np.minimum(gt.shape, pred.shape)
        gt = gt[:min_shape[0], :min_shape[1], :min_shape[2]]
        pred = pred[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # Max tumor slice
    slice_idx = find_max_tumor_slice(gt)
    
    metrics = {}
    
    # Dice
    metrics['dice_liver'] = compute_dice(pred, gt, 1)
    metrics['dice_tumor'] = compute_dice(pred, gt, 2)
    metrics['dice_avg'] = np.nanmean([metrics['dice_liver'], metrics['dice_tumor']])
    
    # Hausdorff (mm)
    for label, name in [(1, 'liver'), (2, 'tumor')]:
        hd_full = compute_hausdorff(pred, gt, label, 100)
        hd95 = compute_hausdorff(pred, gt, label, 95)
        metrics[f'hd_full_{name}'] = hd_full * np.prod(spacing) if not np.isnan(hd_full) else np.nan
        metrics[f'hd95_{name}'] = hd95 * np.prod(spacing) if not np.isnan(hd95) else np.nan
    
    metrics['hd95_avg'] = np.nanmean([metrics['hd95_liver'], metrics['hd95_tumor']])
    metrics['slice_idx'] = slice_idx
    
    return metrics


def main():
    
    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT directory not found: {GT_DIR}")
    
    # Fixed glob pattern - handles .nii AND .nii.gz
    gt_files = sorted(GT_DIR.glob("*.nii")) + sorted(GT_DIR.glob("*.nii.gz"))
    gt_files = sorted(set(gt_files))  # Remove duplicates
    
    print(f"Found {len(gt_files)} ground truth files in {GT_DIR}")
    if len(gt_files) == 0:
        print("Available files:")
        for f in GT_DIR.iterdir():
            print(f"  {f.name}")
        return
    
    case_results = {}  # {case_id: {"gt_path": Path, "ct_path": Path|None,
                    #           "slice_idx": int, "models": {model_name: {...metrics, "pred_path": Path}}}}
    all_results = []
    
    for i, pred_dir in enumerate(PRED_DIRS):
        if not pred_dir.exists():
            print(f"⚠️  Prediction directory {i+1} not found: {pred_dir}")
            continue
            
        model_name = pred_dir.name
        print(f"\n🔄 Processing model: {model_name}")
        
        model_results = []
        missing_files = 0
        
        for gt_file in gt_files:
            pred_file = pred_dir / gt_file.name
            ct_file = CT_DIR / gt_file.name.replace('labels', 'images') if CT_DIR.exists() else None
            
            if not pred_file.exists():
                missing_files += 1
                continue
            
            case_id = gt_file.stem
            metrics = process_pair(gt_file, pred_file, ct_file, case_id, model_name)
            metrics['case_id'] = case_id
            metrics['model'] = model_name
            model_results.append(metrics)
            
            # Build case_results structure
            if case_id not in case_results:
                case_results[case_id] = {
                    "gt_path": gt_file,
                    "ct_path": ct_file,
                    "slice_idx": metrics['slice_idx'],
                    "models": {}
                }
            case_results[case_id]["models"][model_name] = {
                "pred_path": pred_file,
                **{k: v for k, v in metrics.items() if k != 'case_id' and k != 'model'}
            }
        
        print(f"  Processed {len(model_results)}/{len(gt_files)} cases "
              f"({missing_files} missing)")
        all_results.extend(model_results)
    
    if not all_results:
        print("❌ No valid prediction pairs found!")
        return
    
    # Save CSV
    df = pd.DataFrame(all_results)
    cols = ['model', 'case_id', 'slice_idx', 'dice_liver', 'dice_tumor', 'dice_avg',
            'hd_full_liver', 'hd_full_tumor', 
            'hd95_liver', 'hd95_tumor', 'hd95_avg']
    df = df[cols].round(4)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Results saved: {OUTPUT_CSV}")
    
    # Save case comparison figures
    if SAVE_FIGS:
        print(f"\n🎨 Saving {len(case_results)} case comparison figures...")
        for case_id, info in case_results.items():
            save_case_comparison(case_id, info)
        print(f"✅ Case figures saved to: {FIG_DIR}")
        print("   → 1 PNG per case")
    
    # Summary plots
    save_summary_plots(df)
    
    print("\n📊 Model Averages:")
    summary = df.groupby('model')[['dice_avg', 'hd95_avg']].mean()
    print(summary.round(4))


if __name__ == "__main__":
    main()
