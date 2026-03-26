#!/usr/bin/env python3
"""
Compare multiple model predictions vs ground truth LiTS segmentations + visuals.
Liver=1, Tumor=2. Outputs CSV + publication-ready figures.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

GT_DIR = Path("./LiTS/labelsTs")
CT_DIR = Path("./LiTS/imagesTs")
PRED_DIRS = [
    Path("./Predictions/Dataset001_LiTS"),
    Path("./Predictions/Dataset002_LiTSMaisiCombined"),
    Path("./Predictions/Dataset003_LiTS_Full"),
    Path("./Predictions/Dataset004_LiTSMaisiFullMixed"),
    Path("./Predictions/Dataset005_LiTSMaisi100_50"),
    Path("./Predictions/Dataset006_LiTSMaisi100_80"),
    Path("./Predictions/Dataset007_LiTSMaisi100_30"),
]
OUTPUT_CSV = Path("./Predictions/liver_tumor_metrics.csv")
FIG_DIR = Path("./Predictions/figures")
SAVE_FIGS = True

plt.style.use("default")
sns.set_palette("husl")
if SAVE_FIGS:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def case_id_from_path(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def find_max_tumor_slice(gt: np.ndarray) -> int:
    tumor_per_slice = np.sum(gt == 2, axis=(1, 2))
    if tumor_per_slice.sum() == 0:
        return gt.shape[0] // 2
    return int(np.argmax(tumor_per_slice))


def compute_dice(pred, gt, label):
    pred_voxels = np.sum(pred == label)
    gt_voxels = np.sum(gt == label)
    intersection = np.sum((pred == label) & (gt == label))
    if pred_voxels + gt_voxels == 0:
        return 1.0
    return 2.0 * intersection / (pred_voxels + gt_voxels)


def load_binary_image(img, label):
    return sitk.Cast(img == label, sitk.sitkUInt8)


def surface_distances_mm(pred_img, gt_img, label):
    pred_bin = load_binary_image(pred_img, label)
    gt_bin = load_binary_image(gt_img, label)

    dim = gt_bin.GetDimension()
    pred_bin = sitk.ConstantPad(pred_bin, [1] * dim, [1] * dim)
    gt_bin = sitk.ConstantPad(gt_bin, [1] * dim, [1] * dim)

    pred_surf = sitk.LabelContour(pred_bin)
    gt_surf = sitk.LabelContour(gt_bin)

    pred_distmap = sitk.Abs(
        sitk.SignedMaurerDistanceMap(pred_surf, squaredDistance=False, useImageSpacing=True)
    )
    gt_distmap = sitk.Abs(
        sitk.SignedMaurerDistanceMap(gt_surf, squaredDistance=False, useImageSpacing=True)
    )

    gt_to_pred = sitk.GetArrayViewFromImage(pred_distmap * sitk.Cast(gt_surf, sitk.sitkFloat32))
    pred_to_gt = sitk.GetArrayViewFromImage(gt_distmap * sitk.Cast(pred_surf, sitk.sitkFloat32))

    d1 = gt_to_pred[gt_to_pred > 0]
    d2 = pred_to_gt[pred_to_gt > 0]
    return np.concatenate([d1, d2]) if (len(d1) or len(d2)) else np.array([])


def compute_hd_and_hd95(pred_img, gt_img, label):
    dists = surface_distances_mm(pred_img, gt_img, label)
    if dists.size == 0:
        return np.nan, np.nan
    return float(np.max(dists)), float(np.percentile(dists, 95))


def save_case_comparison(case_id, info):
    gt_path = info["gt_path"]
    ct_path = info["ct_path"]
    slice_idx = info["slice_idx"]
    models = info["models"]

    gt_img = sitk.ReadImage(str(gt_path))
    gt = sitk.GetArrayFromImage(gt_img)

    ct = None
    if ct_path and ct_path.exists():
        ct_img = sitk.ReadImage(str(ct_path))
        ct = sitk.GetArrayFromImage(ct_img)
        min_shape = np.minimum(gt.shape, ct.shape)
        gt = gt[:min_shape[0], :min_shape[1], :min_shape[2]]
        ct = ct[:min_shape[0], :min_shape[1], :min_shape[2]]

    colors = ["black", "#00FF00", "#FF0000"]
    cmap = ListedColormap(colors)

    n_models = len(models)
    n_rows = 1 + n_models
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    max_valid_slice = gt.shape[0] - 1
    if ct is not None:
        max_valid_slice = min(max_valid_slice, ct.shape[0] - 1)
    for m in models.values():
        pred_img = sitk.ReadImage(str(m["pred_path"]))
        pred = sitk.GetArrayFromImage(pred_img)
        min_shape = np.minimum(gt.shape, pred.shape)
        max_valid_slice = min(max_valid_slice, min_shape[0] - 1)
    slice_idx = int(max(0, min(slice_idx, max_valid_slice)))

    fig.suptitle(f"{case_id} - axial slice {slice_idx}", fontsize=14)

    ax = axes[0]
    if ct is not None:
        ax.imshow(ct[slice_idx], cmap="gray", vmin=-200, vmax=300)
        ax.imshow(gt[slice_idx], cmap=cmap, alpha=0.4, vmin=0, vmax=2)
        ax.set_title("Ground Truth")
    else:
        ax.imshow(gt[slice_idx], cmap=cmap, vmin=0, vmax=2)
        ax.set_title("Ground Truth (labels only)")
    ax.axis("off")

    for row_idx, (model_name, m) in enumerate(models.items(), start=1):
        pred_img = sitk.ReadImage(str(m["pred_path"]))
        pred = sitk.GetArrayFromImage(pred_img)
        min_shape = np.minimum(gt.shape, pred.shape)
        pred = pred[:min_shape[0], :min_shape[1], :min_shape[2]]

        ax = axes[row_idx]
        if ct is not None:
            ax.imshow(ct[slice_idx], cmap="gray", vmin=-200, vmax=300)
            ax.imshow(pred[slice_idx], cmap=cmap, alpha=0.4, vmin=0, vmax=2)
        else:
            ax.imshow(pred[slice_idx], cmap=cmap, vmin=0, vmax=2)

        ax.set_title(f"{model_name}\nDice={m['dice_avg']:.3f}, HD95={m['hd95_avg']:.1f}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{case_id}_all_models_slice{slice_idx}.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_summary_plots(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Model Comparison Summary (LiTS Liver/Tumor)", fontsize=16, fontweight="bold")

    df.boxplot(column="dice_avg", by="model", ax=axes[0, 0], patch_artist=True)
    axes[0, 0].set_title("Dice Score (Average)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    df.boxplot(column="hd95_avg", by="model", ax=axes[0, 1], patch_artist=True)
    axes[0, 1].set_title("HD95 Distance (mm)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    pivot_dice = df.pivot(index="model", columns="case_id", values="dice_avg")
    sns.heatmap(pivot_dice, annot=True, fmt=".3f", cmap="RdYlGn", cbar_kws={"label": "Dice"}, ax=axes[1, 0])
    axes[1, 0].set_title("Dice Heatmap (Model × Case)")

    pivot_hd = df.pivot(index="model", columns="case_id", values="hd95_avg")
    sns.heatmap(pivot_hd, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "HD95 (mm)"}, ax=axes[1, 1])
    axes[1, 1].set_title("HD95 Heatmap (Model × Case)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "summary_all_models.png", dpi=200, bbox_inches="tight")
    plt.close()


def process_pair(gt_path, pred_path, ct_path, case_id, model):
    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))

    gt = sitk.GetArrayFromImage(gt_img)
    pred = sitk.GetArrayFromImage(pred_img)

    if gt.shape != pred.shape:
        print(f"⚠️ {case_id}: shape {gt.shape} vs {pred.shape} - cropping")
        min_shape = np.minimum(gt.shape, pred.shape)
        gt = gt[:min_shape[0], :min_shape[1], :min_shape[2]]
        pred = pred[:min_shape[0], :min_shape[1], :min_shape[2]]
        gt_img = sitk.GetImageFromArray(gt)
        pred_img = sitk.GetImageFromArray(pred)
        gt_img.CopyInformation(sitk.ReadImage(str(gt_path)))
        pred_img.CopyInformation(sitk.ReadImage(str(pred_path)))

    slice_idx = find_max_tumor_slice(gt)

    metrics = {}
    metrics["dice_liver"] = compute_dice(pred, gt, 1)
    metrics["dice_tumor"] = compute_dice(pred, gt, 2)
    metrics["dice_avg"] = np.nanmean([metrics["dice_liver"], metrics["dice_tumor"]])

    for label, name in [(1, "liver"), (2, "tumor")]:
        hd_full, hd95 = compute_hd_and_hd95(pred_img, gt_img, label)
        metrics[f"hd_full_{name}"] = hd_full
        metrics[f"hd95_{name}"] = hd95

    metrics["hd95_avg"] = np.nanmean([metrics["hd95_liver"], metrics["hd95_tumor"]])
    metrics["slice_idx"] = slice_idx
    return metrics


def main():
    if not GT_DIR.exists():
        raise FileNotFoundError(f"GT directory not found: {GT_DIR}")

    gt_files = sorted(GT_DIR.glob("*.nii")) + sorted(GT_DIR.glob("*.nii.gz"))
    gt_files = sorted(set(gt_files))

    print(f"Found {len(gt_files)} ground truth files in {GT_DIR}")
    if len(gt_files) == 0:
        print("Available files:")
        for f in GT_DIR.iterdir():
            print(f"  {f.name}")
        return

    case_results = {}
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
            ct_file = CT_DIR / gt_file.name.replace("labels", "images") if CT_DIR.exists() else None

            if not pred_file.exists():
                missing_files += 1
                continue

            case_id = case_id_from_path(gt_file)
            metrics = process_pair(gt_file, pred_file, ct_file, case_id, model_name)
            metrics["case_id"] = case_id
            metrics["model"] = model_name
            model_results.append(metrics)

            if case_id not in case_results:
                case_results[case_id] = {
                    "gt_path": gt_file,
                    "ct_path": ct_file,
                    "slice_idx": metrics["slice_idx"],
                    "models": {},
                }
            case_results[case_id]["models"][model_name] = {
                "pred_path": pred_file,
                **{k: v for k, v in metrics.items() if k not in {"case_id", "model"}},
            }

        print(f"  Processed {len(model_results)}/{len(gt_files)} cases ({missing_files} missing)")
        all_results.extend(model_results)

    if not all_results:
        print("❌ No valid prediction pairs found!")
        return

    df = pd.DataFrame(all_results)
    cols = [
        "model", "case_id", "slice_idx", "dice_liver", "dice_tumor", "dice_avg",
        "hd_full_liver", "hd_full_tumor", "hd95_liver", "hd95_tumor", "hd95_avg",
    ]
    df = df[cols].round(4)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Results saved: {OUTPUT_CSV}")

    if SAVE_FIGS:
        print(f"\n🎨 Saving {len(case_results)} case comparison figures...")
        for case_id, info in case_results.items():
            save_case_comparison(case_id, info)
        print(f"✅ Case figures saved to: {FIG_DIR}")

    save_summary_plots(df)
    print("\n📊 Model Averages:")
    summary = df.groupby("model")[["dice_avg", "hd95_avg"]].mean()
    print(summary.round(4))


if __name__ == "__main__":
    main()
