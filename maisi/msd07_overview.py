"""Check MSD Task07 Pancreas, write per-case stats and axial overview PNGs with the labels overlaid.

    python msd07_overview.py --root $PROJ/raw_data/Task07_Pancreas           # all checks, stats.csv, overview/
    python msd07_overview.py --root ... --single pancreas_123              # pick the large single-case figure

Checks: file counts (imagesTr/labelsTr/imagesTs, macOS `._*` files ignored), dataset.json labels, label values
within {0,1,2}, image/label shape + affine match. Writes <root>/stats.csv (one row per labelled case) and
<root>/overview/: contact sheets of --per-page cases (one axial slice each, at the slice with the largest tumour
area, or largest pancreas area if there is no tumour) and single_<case>.png. Volumes are reoriented to RAS in
memory and shown in radiological convention (patient right on image left, anterior up). Nothing is modified.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch
from scipy import ndimage

from orient import axcodes, reorient_array, spacing_of

WINDOW = (40 - 200, 40 + 200)  # W400 / L40
COLORS = {1: (0.0, 0.85, 0.0), 2: (1.0, 0.1, 0.1)}  # pancreas green, tumour red
ALPHA = 0.4
MARGIN_MM = 25
EXPECTED = {"imagesTr": 281, "labelsTr": 281, "imagesTs": 139}
EXPECTED_LABELS = {"0": "background", "1": "pancreas", "2": "cancer"}


def nii_files(folder):
    return sorted(f for f in folder.glob("*.nii.gz") if not f.name.startswith("._"))


def case_id(path):
    return path.name[: -len(".nii.gz")]


def display(slice2d):
    """RAS axial slice (x, y) -> image rows anterior..posterior, columns patient right..left (radiological)."""
    return slice2d.T[::-1, ::-1]


def analyse(args):
    """Check one labelled case; return (stats row, list of problems, data for its overview tile)."""
    image_path, label_path = args
    cid = case_id(image_path)
    problems = []
    img, lab = nib.load(str(image_path)), nib.load(str(label_path))
    if img.shape != lab.shape:
        problems.append(f"shape image {img.shape} != label {lab.shape}")
    if not np.allclose(img.affine, lab.affine, atol=1e-3):
        problems.append("affine image != label")
    image = np.asanyarray(img.dataobj).astype(np.float32)
    label = np.rint(np.asanyarray(lab.dataobj)).astype(np.int16)
    values = sorted(int(v) for v in np.unique(label))
    if not set(values) <= {0, 1, 2}:
        problems.append(f"label values {values}")

    row = {"case": cid, "shape": "x".join(map(str, img.shape)), "axcodes": axcodes(img.affine)}
    sx, sy, sz = spacing_of(img.affine)
    row.update(spacing_x=round(sx, 4), spacing_y=round(sy, 4), spacing_z=round(sz, 4))
    row.update(hu_min=float(image.min()), hu_max=float(image.max()), label_values=" ".join(map(str, values)))
    tile = None
    if img.shape == lab.shape:
        image, affine = reorient_array(image, img.affine)
        label, _ = reorient_array(label, lab.affine)
        sx, sy, sz = spacing_of(affine)
        ml = sx * sy * sz / 1000
        tumour = label == 2
        n_comp = int(ndimage.label(tumour, structure=np.ones((3, 3, 3), bool))[1]) if tumour.any() else 0
        row.update(pancreas_ml=round(float((label == 1).sum() * ml), 2), tumour_ml=round(float(tumour.sum() * ml), 2),
                   tumour_components=n_comp)
        if (label > 0).any():
            tile = make_tile(cid, image, label, (sx, sy, sz), row)
        else:
            problems.append("empty label")
    return row, problems, tile


def make_tile(cid, image, label, spacing, row):
    """Axial slice with the largest tumour area (largest pancreas area without tumour), cropped to the label."""
    sx, sy, _ = spacing
    target = 2 if row["tumour_ml"] > 0 else 1
    k = int(np.argmax((label == target).sum(axis=(0, 1))))
    xs, ys = np.nonzero((label > 0).any(axis=2))
    mx, my = round(MARGIN_MM / sx), round(MARGIN_MM / sy)
    x0, x1 = max(xs.min() - mx, 0), min(xs.max() + mx + 1, label.shape[0])
    y0, y1 = max(ys.min() - my, 0), min(ys.max() + my + 1, label.shape[1])
    return {"case": cid, "z": k, "nz": label.shape[2], "aspect": sy / sx,
            "image": display(image[x0:x1, y0:y1, k]).astype(np.int16),
            "label": display(label[x0:x1, y0:y1, k]).astype(np.uint8),
            "full_image": display(image[:, :, k]).astype(np.int16),
            "full_label": display(label[:, :, k]).astype(np.uint8),
            "crop": (x0, x1, y0, y1), "pancreas_ml": row["pancreas_ml"], "tumour_ml": row["tumour_ml"]}


def draw(ax, image, label, aspect):
    ax.imshow(image, cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1], aspect=aspect, interpolation="nearest")
    rgba = np.zeros(label.shape + (4,), np.float32)
    for v, c in COLORS.items():
        rgba[label == v] = (*c, ALPHA)
    ax.imshow(rgba, aspect=aspect, interpolation="nearest")
    for v, c in COLORS.items():
        if (label == v).any():
            ax.contour(label == v, levels=[0.5], colors=[c], linewidths=0.6)
    ax.set_xticks([]), ax.set_yticks([])


def title(t):
    return f"{t['case']}  z {t['z']}/{t['nz']}\npanc {t['pancreas_ml']:.0f} ml · tum {t['tumour_ml']:.1f} ml"


def contact_sheets(tiles, out_dir, per_page):
    cols = 8
    pages = math.ceil(len(tiles) / per_page)
    for p in range(pages):
        chunk = tiles[p * per_page:(p + 1) * per_page]
        rows = math.ceil(len(chunk) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.75), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, t in zip(axes.ravel(), chunk):
            draw(ax, t["image"], t["label"], t["aspect"])
            ax.axis("on")
            ax.set_title(title(t), fontsize=7)
        fig.suptitle(f"MSD Task07 Pancreas, axial slice at max tumour area (max pancreas if no tumour), cropped to "
                     f"label + {MARGIN_MM} mm, W400/L40, radiological (patient right = image left). "
                     f"Green pancreas, red tumour. Page {p + 1}/{pages}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = out_dir / f"contact_sheet_{p + 1:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"wrote {path}", flush=True)


def single_figure(t, out_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 8.4))
    draw(a, t["full_image"], t["full_label"], t["aspect"])
    x0, x1, y0, y1 = t["crop"]
    h, w = t["full_image"].shape  # rows = y (flipped), cols = x (flipped)
    a.add_patch(plt.Rectangle((w - x1 - 0.5, h - y1 - 0.5), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1))
    a.set_title("full axial slice", fontsize=11)
    for side, x, ha in (("R", 0.01, "left"), ("L", 0.99, "right")):
        a.text(x, 0.5, side, transform=a.transAxes, color="yellow", fontsize=14, ha=ha, va="center")
    a.text(0.5, 0.99, "A", transform=a.transAxes, color="yellow", fontsize=14, ha="center", va="top")
    draw(b, t["image"], t["label"], t["aspect"])
    b.set_title(f"zoom: label bbox + {MARGIN_MM} mm", fontsize=11)
    b.legend(handles=[Patch(color=COLORS[1], label="pancreas"), Patch(color=COLORS[2], label="tumour")],
             loc="lower right", fontsize=10)
    fig.suptitle(f"{title(t).replace(chr(10), '  ·  ')}  ·  W400/L40, radiological convention", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"single_{t['case']}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def check_layout(root):
    problems = []
    counts = {d: len(nii_files(root / d)) for d in EXPECTED}
    for d, n in EXPECTED.items():
        if counts[d] != n:
            problems.append(f"{d}: {counts[d]} files, expected {n}")
    mac = len(list(root.rglob("._*")))
    ds = json.loads((root / "dataset.json").read_text())
    if ds.get("labels") != EXPECTED_LABELS:
        problems.append(f"dataset.json labels {ds.get('labels')}")
    images = {case_id(f) for f in nii_files(root / "imagesTr")}
    labels = {case_id(f) for f in nii_files(root / "labelsTr")}
    if images != labels:
        problems.append(f"imagesTr/labelsTr ids differ: {sorted(images ^ labels)[:10]}")
    print(f"counts {counts}, macOS ._ files {mac}, dataset.json labels {ds.get('labels')}, "
          f"numTraining {ds.get('numTraining')}, numTest {ds.get('numTest')}", flush=True)
    return problems, sorted(images & labels)


def describe(rows):
    sp = {a: [r[f"spacing_{a}"] for r in rows] for a in "xyz"}
    lines = [f"orientations: {dict(Counter(r['axcodes'] for r in rows))}"]
    for a, v in sp.items():
        lines.append(f"spacing {a}: median {statistics.median(v):.3f}, range {min(v):.3f}-{max(v):.3f} mm")
    nz = [int(r["shape"].split("x")[2]) for r in rows]
    lines.append(f"slices: median {statistics.median(nz):.0f}, range {min(nz)}-{max(nz)}; "
                 f"in-plane shapes {dict(Counter('x'.join(r['shape'].split('x')[:2]) for r in rows))}")
    hu = [r["hu_min"] for r in rows], [r["hu_max"] for r in rows]
    lines.append(f"HU min range {min(hu[0]):.0f}..{max(hu[0]):.0f}, HU max range {min(hu[1]):.0f}..{max(hu[1]):.0f}")
    for key in ("pancreas_ml", "tumour_ml"):
        v = [r[key] for r in rows if key in r]
        lines.append(f"{key}: median {statistics.median(v):.1f}, range {min(v):.1f}-{max(v):.1f}")
    no_tumour = [r["case"] for r in rows if r.get("tumour_ml", 0) == 0]
    comps = Counter(r.get("tumour_components", 0) for r in rows)
    lines.append(f"cases without tumour: {len(no_tumour)} {no_tumour}")
    lines.append(f"tumour components per case: {dict(sorted(comps.items()))}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="the extracted Task07_Pancreas folder")
    p.add_argument("--per-page", type=int, default=48)
    p.add_argument("--single", default=None, help="case for the large figure (default: median tumour volume)")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    problems, cases = check_layout(args.root)
    pairs = [(args.root / "imagesTr" / f"{c}.nii.gz", args.root / "labelsTr" / f"{c}.nii.gz") for c in cases]
    rows, tiles = [], []
    with Pool(args.workers) as pool:
        for i, (row, probs, tile) in enumerate(pool.imap(analyse, pairs), 1):
            rows.append(row)
            problems += [f"{row['case']}: {x}" for x in probs]
            if tile is not None:
                tiles.append(tile)
            if i % 25 == 0 or i == len(pairs):
                print(f"[{i}/{len(pairs)}] analysed", flush=True)

    stats_path = args.root / "stats.csv"
    fields = ["case", "shape", "spacing_x", "spacing_y", "spacing_z", "axcodes", "hu_min", "hu_max", "label_values",
              "pancreas_ml", "tumour_ml", "tumour_components"]
    with open(stats_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {stats_path}")
    for line in describe(rows):
        print(line)

    out = args.root / "overview"
    out.mkdir(exist_ok=True)
    contact_sheets(tiles, out, args.per_page)
    if args.single:
        pick = next(t for t in tiles if t["case"] == args.single)
    else:
        with_tumour = sorted((t for t in tiles if t["tumour_ml"] > 0), key=lambda t: t["tumour_ml"])
        pick = with_tumour[len(with_tumour) // 2]
    single_figure(pick, out)

    print(f"problems: {len(problems)}")
    for x in problems:
        print("  " + x)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
