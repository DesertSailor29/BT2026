#!/usr/bin/env python
"""Contact sheets for the filter sanity checks (PLAN_training.md "Filter rules"), in the shared figure style (viz.py).

Reads filter_metrics.py's output (caches + rules.json) and writes <metrics>/figures/:
    real_least_visible.png   the 6 least visible real val lesions (check 2: is the minimum a label error?)
    real_not_touching.png    real val lesions that don't touch the organ (check 1), if any
    synth_fail_touch.png     failing-touch components: full axial + coronal slice through the component (where is it?)
    synth_near_miss.png      fail touch but within 3 mm of the organ, zoomed (is the no-tolerance rule right?)
    synth_fail_auc.png       failing-auc components, zoomed (is the label really invisible?)
    synth_auc_borderline.png components just above the cut, zoomed (what "barely visible" looks like)
Synthetic components are a seeded random sample (--n per sheet). Zoomed panels come in pairs (image alone, then with
the label) with the component centred; in full slices it is circled (white). All inputs are RAS (pairs and the real val cases are saved RAS).

  python filter_figures.py --metrics $MAISI_RUNS/pool/filter_metrics/liver --pool $MAISI_RUNS/pool --target liver
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_metrics import NEAR_MM, fails, image_of  # noqa: E402
from orient import axcodes, spacing_of  # noqa: E402
from viz import CAPTION, draw, legend_handles, plane  # noqa: E402

ZOOM_MM = 40.0


def load(label_path):
    nii = nib.load(label_path)
    assert axcodes(nii.affine) == "RAS", f"{label_path} is {axcodes(nii.affine)}, expected RAS"
    label = np.asarray(nii.dataobj).astype(np.uint8)
    image = np.asarray(nib.load(image_of(label_path)).dataobj, dtype=np.float32)
    return image, label, spacing_of(nii.affine)


def zoom_panels(axes, image, label, spacing, c, title):
    """Image alone (to judge visibility without the fill) + the same field with the label."""
    x, y, z = c["centroid"]
    rx, ry = (int(round(ZOOM_MM / s)) for s in spacing[:2])
    sl = (slice(max(x - rx, 0), x + rx + 1), slice(max(y - ry, 0), y + ry + 1), slice(z, z + 1))
    img, lab, aspect = plane(image[sl], label[sl], spacing, "axial", 0)
    draw(axes[0], img, None, aspect)
    draw(axes[1], img, lab, aspect)
    axes[0].set_title(title, fontsize=6)
    axes[1].set_title("label", fontsize=6)


def full_panel(ax, image, label, spacing, c, view, title):
    x, y, z = c["centroid"]
    if view == "axial":
        img, lab, aspect = plane(image, label, spacing, "axial", z)
        row, col = image.shape[1] - 1 - y, image.shape[0] - 1 - x
    else:
        img, lab, aspect = plane(image, label, spacing, "coronal", y)
        row, col = image.shape[2] - 1 - z, image.shape[0] - 1 - x
    draw(ax, img, lab, aspect)
    r = max(6.0, 2.5 * (c["voxels"] ** (1 / 3)))
    ax.add_patch(plt.Circle((col, row), r, fill=False, color="white", linewidth=0.6))
    ax.set_title(title, fontsize=6)


def sheet(items, path, header, full=False):
    """items = [(label_path, component row, title)], drawn in the given order: zoomed image + label panels, or axial +
    coronal full slices when full."""
    if not items:
        return False
    per = 2
    cols = 4 if full else 6
    n = len(items) * per
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows + 0.6), dpi=130, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    panels = iter(axes.ravel())
    cache = {}
    for lp, c, title in items:
        if lp not in cache:
            cache.clear()
            cache[lp] = load(lp)
        image, label, spacing = cache[lp]
        if full:
            for view in ("axial", "coronal"):
                ax = next(panels)
                ax.axis("on")
                full_panel(ax, image, label, spacing, c, view, f"{title} ({view})")
        else:
            pair = [next(panels), next(panels)]
            for ax in pair:
                ax.axis("on")
            zoom_panels(pair, image, label, spacing, c, title)
    fig.suptitle(f"{header}\n{CAPTION}" + ("" if full else f" Axial, {2 * ZOOM_MM:g} mm field centred on the component."),
                 fontsize=8)
    fig.legend(handles=legend_handles("liver", "tumour"), loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path, flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, help="filter_metrics output dir")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--target", default="liver")
    ap.add_argument("--n", type=int, default=12, help="components per synthetic sheet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    m = Path(a.metrics)
    out = m / "figures"
    out.mkdir(exist_ok=True)
    rules = json.loads((m / "rules.json").read_text())
    cut = rules["auc_cut"]
    excl = {(e["case"], e["component"]) for e in rules["excluded"]}
    real = [json.loads(p.read_text()) for p in sorted((m / "real_reference").glob("*.json"))]
    real_l = [(c["label_path"], l, c["case"]) for c in real for l in c["lesions"]]
    lab_dir = f"{a.pool}/pairs/{a.target}/labelsTr"
    syn = [(f"{lab_dir}/{r['id']}.nii.gz", c, r["id"])
           for r in (json.loads(p.read_text()) for p in sorted((m / "per_volume").glob("*.json"))) for c in r["components"]]
    rng = random.Random(a.seed)
    sample = lambda xs: sorted(rng.sample(xs, min(a.n, len(xs))), key=lambda t: t[0])  # noqa: E731

    def t(name, c):
        s = f", sep {c['sep']}" if c.get("sep") is not None else ""
        return f"{name} c{c['component']} z{c['centroid'][2]}\n{c['ml']} ml, d {c['min_dist_mm']} mm{s}"

    scored = sorted([x for x in real_l if x[1].get("sep") is not None], key=lambda x: x[1]["sep"])
    sheet([(lp, c, t(n, c) + (" EXCL" if (n, c["component"]) in excl else "")) for lp, c, n in scored[:6]],
          out / "real_least_visible.png", f"Real val lesions, least visible first (cut = {cut})")
    sheet([(lp, c, t(n, c)) for lp, c, n in real_l if not c["touches"]], out / "real_not_touching.png",
          "Real val lesions not touching the liver")
    sheet([(lp, c, t(n, c)) for lp, c, n in sample([x for x in syn if fails(x[1], "touch")])],
          out / "synth_fail_touch.png", f"Synthetic components failing touch (random {a.n}, seed {a.seed})", full=True)
    sheet([(lp, c, t(n, c)) for lp, c, n in sample([x for x in syn if not x[1]["touches"] and x[1]["min_dist_mm"] <= NEAR_MM])],
          out / "synth_near_miss.png", f"Near misses: fail touch, within {NEAR_MM:g} mm of the liver")
    if cut is not None:
        sheet([(lp, c, t(n, c)) for lp, c, n in sample([x for x in syn if fails(x[1], "auc", cut)])],
              out / "synth_fail_auc.png", f"Synthetic components failing auc (sep < {cut}; random {a.n})")
        above = sorted([x for x in syn if x[1].get("sep") is not None and x[1]["sep"] >= cut], key=lambda x: x[1]["sep"])
        sheet([(lp, c, t(n, c)) for lp, c, n in above[: a.n]], out / "synth_auc_borderline.png",
              f"Synthetic components just above the cut (sep >= {cut}, lowest {a.n})")


if __name__ == "__main__":
    main()
