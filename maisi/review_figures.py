#!/usr/bin/env python
"""Review figures for the filter sanity checks: each example shown without and with the label, one row per example,
in the shared figure style (viz.py). Unlike filter_figures.py (random contact sheets in the metrics dir), these are
fixed, large-panel examples meant for the repo / supervisor (default out: reports/figures/filter_review/).

    real_invisible_lesions.png  real val lesions with the lowest separability (sep < --real-max-sep): is anything
                                visible under the tumour label? (decides real_reference_exclusions.json)
    maisi_off_liver.png         MAISI tumour components far from the liver (fail touch, >= --far-mm), largest first
    maisi_invisible_label.png   MAISI tumour components with sep below --sep (default: the real minimum without the
                                excluded case), largest first: a tumour label with no lesion in the image
    maisi_stray_island.png      MAISI tumour components that "touch the liver" only through a small detached piece
                                of liver label (passes touch, but not attached to the main liver), largest first
    README.md                   what each figure shows + the numbers in the titles

Rows: [image | image + label] in an axial field of 2 x --zoom-mm centred on the component, at the slice where the
component is largest; synthetic rows add a full axial and coronal slice with the component circled (white) for
context. sep = 1 - AUC of core vs shell HU (0.5 = invisible, 1 = perfectly separable), see filter_metrics.py.
Examples are chosen deterministically (largest components first, one per volume), not cherry-picked.

  python maisi/review_figures.py --metrics $MAISI_RUNS/pool/filter_metrics/liver --pool $MAISI_RUNS/pool
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_figures import load  # noqa: E402
from viz import CAPTION, draw, legend_handles, plane  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FULL26 = np.ones((3, 3, 3), bool)


def component_mask(label, c, margin_mm, spacing):
    """Boolean box mask of this component (26-connected tumour, matched by voxel count, nearest centroid) and the box
    offset. Local re-labelling keeps it cheap on 512^3 volumes."""
    x, y, z = c["centroid"]
    r = [int(np.ceil(margin_mm / s)) for s in spacing]
    lo = [max(v - rv, 0) for v, rv in zip((x, y, z), r)]
    hi = [min(v + rv + 1, n) for v, rv, n in zip((x, y, z), r, label.shape)]
    box = label[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == 2
    lab, n = ndimage.label(box, FULL26)
    if n == 0:
        return None, lo
    sizes = ndimage.sum(box, lab, range(1, n + 1))
    cents = ndimage.center_of_mass(box, lab, range(1, n + 1))
    want = np.array([x, y, z]) - lo
    best = min(range(n), key=lambda i: (abs(sizes[i] - c["voxels"]), np.linalg.norm(np.array(cents[i]) - want)))
    return lab == best + 1, lo


def best_slice(label, c, spacing):
    """Axial index where the component has its largest cross-section (centroid z if it can't be matched)."""
    extent = max(30.0, 3 * (c["ml"] * 1000) ** (1 / 3))  # mm, generous for elongated components
    m, lo = component_mask(label, c, extent, spacing)
    if m is None or not m.any():
        return c["centroid"][2]
    return lo[2] + int(np.argmax(m.sum(axis=(0, 1))))


def zoom(image, label, spacing, c, z, zoom_mm):
    x, y = c["centroid"][:2]
    rx, ry = (int(round(zoom_mm / s)) for s in spacing[:2])
    sl = (slice(max(x - rx, 0), x + rx + 1), slice(max(y - ry, 0), y + ry + 1), slice(z, z + 1))
    return plane(image[sl], label[sl], spacing, "axial", 0)


def circle(ax, image_shape, c, view, z):
    x, y, zz = c["centroid"]
    if view == "axial":
        row, col = image_shape[1] - 1 - y, image_shape[0] - 1 - x
    else:
        row, col = image_shape[2] - 1 - z, image_shape[0] - 1 - x
    r = min(max(8.0, 2.5 * (c["voxels"] ** (1 / 3))), 40.0)
    ax.add_patch(plt.Circle((col, row), r, fill=False, color="white", linewidth=0.8))


def figure(rows, path, header, zoom_mm, context):
    """rows = [(label_path, component, title)]. Columns: zoom image, zoom image + label (+ full axial, full coronal)."""
    if not rows:
        print("nothing for", path.name, flush=True)
        return 0
    ncol = 4 if context else 2
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.6 * ncol, 3.6 * len(rows) + 1.0), dpi=110, squeeze=False)
    for (lp, c, title), ax in zip(rows, axes):
        image, label, spacing = load(lp)
        z = best_slice(label, c, spacing)
        img, lab, aspect = zoom(image, label, spacing, c, z, zoom_mm)
        draw(ax[0], img, None, aspect)
        draw(ax[1], img, lab, aspect)
        ax[0].set_title(f"{title}\nimage only (z {z})", fontsize=7)
        ax[1].set_title("image + label", fontsize=7)
        if context:
            c_view = dict(c, centroid=[c["centroid"][0], c["centroid"][1], z])
            for a, view, idx in ((ax[2], "axial", z), (ax[3], "coronal", c["centroid"][1])):
                im2, lb2, asp2 = plane(image, label, spacing, view, idx)
                draw(a, im2, lb2, asp2)
                circle(a, image.shape, c_view, view, z)
                a.set_title(f"{view} (whole slice, component circled)", fontsize=7)
    fig.suptitle(f"{header}\n{CAPTION}\nZoom: axial {2 * zoom_mm:g} mm field centred on the component, "
                 "slice of its largest cross-section.", fontsize=8)
    fig.legend(handles=legend_handles("liver", "tumour"), loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.7 / fig.get_figheight()))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path, flush=True)
    return len(rows)


def one_per_volume(items, n):
    """Largest components first, at most one per volume, n in total."""
    seen, out = set(), []
    for it in sorted(items, key=lambda t: -t[1]["ml"]):
        if it[2] not in seen:
            seen.add(it[2])
            out.append(it)
        if len(out) == n:
            break
    return out


def touches_only_island(label_path, c):
    """True if the component touches liver label but not the main (largest) liver component."""
    _, label, spacing = load(label_path)
    m, lo = component_mask(label, c, max(30.0, 3 * (c["ml"] * 1000) ** (1 / 3)), spacing)
    if m is None:
        return False
    liver, n = ndimage.label(label == 1, FULL26)
    if n == 0:
        return False
    main = 1 + int(np.argmax(ndimage.sum(label == 1, liver, range(1, n + 1))))
    ring = ndimage.binary_dilation(m, FULL26) & ~m
    hi = [a + b for a, b in zip(lo, m.shape)]
    touched = liver[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][ring]
    touched = set(np.unique(touched[touched > 0]).tolist())
    return bool(touched) and main not in touched


def island_check(args):
    return touches_only_island(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, help="filter_metrics output dir (liver)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--target", default="liver")
    ap.add_argument("--out", default=str(REPO / "reports/figures/filter_review"))
    ap.add_argument("--n", type=int, default=6, help="examples per MAISI figure")
    ap.add_argument("--real-max-sep", type=float, default=0.65, help="real lesions shown: sep below this")
    ap.add_argument("--real-max-n", type=int, default=10)
    ap.add_argument("--sep", type=float, default=None,
                    help="invisible-label threshold for MAISI (default: lowest real sep outside --exclude-case)")
    ap.add_argument("--exclude-case", default="lits_049", help="real case left out when deriving the default --sep")
    ap.add_argument("--far-mm", type=float, default=30.0)
    ap.add_argument("--zoom-mm", type=float, default=40.0, help="half width of the zoom field")
    ap.add_argument("--workers", type=int, default=12, help="processes (island search, then one per figure); ~2 GB each")
    a = ap.parse_args()

    m, out = Path(a.metrics), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    real = [json.loads(p.read_text()) for p in sorted((m / "real_reference").glob("*.json"))]
    real_l = [(r["label_path"], l, r["case"]) for r in real for l in r["lesions"] if l.get("sep") is not None]
    lab_dir = f"{a.pool}/pairs/{a.target}/labelsTr"
    syn = [(f"{lab_dir}/{r['id']}.nii.gz", c, r["id"])
           for r in (json.loads(p.read_text()) for p in sorted((m / "per_volume").glob("*.json"))) for c in r["components"]]
    sep_cut = a.sep if a.sep is not None else min(l["sep"] for _, l, n in real_l if n != a.exclude_case)

    def t(name, c):
        s = f"\nsep {c['sep']:.3f}, HU core {c['core_hu']:g} / shell {c['shell_hu']:g}" if c.get("sep") is not None else ""
        return f"{name} c{c['component']}: {c['ml']:.2f} ml, {c['min_dist_mm']:g} mm to liver{s}"

    jobs = []

    def add_job(rows, *args, **kw):
        jobs.append((rows, args, kw))
        return len(rows)

    readme = ["# Filter review figures", "",
              f"Made by `maisi/review_figures.py` from `{m}` (pool `{a.pool}`). Shared figure style (`maisi/viz.py`): "
              "W400/L40, radiological, green liver, red tumour. Each example: image alone, then image + label. "
              "sep = separability of tumour core vs surrounding HU (0.5 = invisible, 1 = fully separable); "
              "real val lesions: median 0.90, synthetic components: median 0.71.", ""]

    # 1. real lesions with (almost) no visible contrast
    rows = sorted([x for x in real_l if x[1]["sep"] < a.real_max_sep], key=lambda x: x[1]["sep"])[: a.real_max_n]
    k = add_job([(lp, c, t(n, c)) for lp, c, n in rows], out / "real_invisible_lesions.png",
               f"Real LiTS val lesions with the lowest separability (sep < {a.real_max_sep}), least visible first",
               a.zoom_mm, context=False)
    cases = sorted({n for _, _, n in rows})
    readme += ["## real_invisible_lesions.png", "",
               f"The {k} least visible real validation lesions (sep < {a.real_max_sep}; cases: {', '.join(cases)}). "
               "If nothing is visible under the label, the lesion is a label error and can go into "
               "`maisi/real_reference_exclusions.json` (it sets the AUC cut).", ""]

    # 2. MAISI: tumour far from the liver
    rows = one_per_volume([x for x in syn if not x[1]["touches"] and x[1]["min_dist_mm"] >= a.far_mm], a.n)
    n_far = len({n for _, c, n in syn if not c["touches"] and c["min_dist_mm"] >= a.far_mm})
    k = add_job([(lp, c, t(n, c)) for lp, c, n in rows], out / "maisi_off_liver.png",
               f"MAISI issue: tumour label away from the liver (>= {a.far_mm:g} mm), largest per volume",
               a.zoom_mm, context=True)
    readme += ["## maisi_off_liver.png", "",
               f"{k} examples (largest component per volume) of tumour components >= {a.far_mm:g} mm from the liver; "
               f"{n_far}/999 volumes have at least one. LiTS labels only liver tumours, so these are wrong labels "
               "(rule touch: rejectTouch drops the volume, repairTouch sets them to background).", ""]

    # 3. MAISI: tumour label without a visible lesion, at the liver (off-liver ones are in figure 2; many of the big
    #    pelvic blobs there are invisible too)
    inv = lambda c: c.get("sep") is not None and c["sep"] < sep_cut  # noqa: E731
    at_liver = lambda c: c["touches"] and c["contact"] >= 0.5  # noqa: E731
    rows = one_per_volume([x for x in syn if inv(x[1]) and at_liver(x[1])], a.n)
    n_inv = len({n for _, c, n in syn if inv(c)})
    n_inv_liver = len({n for _, c, n in syn if inv(c) and at_liver(c)})
    k = add_job([(lp, c, t(n, c)) for lp, c, n in rows], out / "maisi_invisible_label.png",
               f"MAISI issue: tumour label at the liver with no visible lesion (sep < {sep_cut:.3f}), largest per volume",
               a.zoom_mm, context=True)
    readme += ["## maisi_invisible_label.png", "",
               f"{k} examples of tumour components with sep < {sep_cut:.3f} (the least visible real lesion outside "
               f"{a.exclude_case}) that sit at the liver (touch it, contact >= 0.5); {n_inv_liver}/999 volumes have "
               f"one at the liver, {n_inv}/999 anywhere (many big off-liver blobs in maisi_off_liver.png are invisible "
               "too). The label says tumour, the image shows tissue like its surroundings (rule AUC).", ""]

    # 4. MAISI: touches the liver only via a stray liver island
    cand = sorted([x for x in syn if x[1]["touches"] and x[1]["contact"] < 0.1], key=lambda x: -x[1]["ml"])
    with ProcessPoolExecutor(a.workers) as ex:
        hits = list(ex.map(island_check, [(x[0], x[1]) for x in cand], chunksize=1))
    print(f"island search: {sum(hits)}/{len(cand)} candidates touch only a liver island", flush=True)
    found, seen = [], set()
    for x, hit in zip(cand, hits):
        if hit and x[2] not in seen and len(found) < a.n:
            found.append(x)
            seen.add(x[2])
    k = add_job([(lp, c, t(n, c) + f", contact {c['contact']:.2f}") for lp, c, n in found],
               out / "maisi_stray_island.png",
               "MAISI issue: tumour 'touches the liver' only through a detached piece of liver label",
               a.zoom_mm, context=True)
    readme += ["## maisi_stray_island.png", "",
               f"{k} examples (largest first, one per volume) of the {sum(hits)}/{len(cand)} touching components "
               "with contact < 0.1 whose "
               "only liver contact is a small liver-label island separate from the main liver. They pass the touch "
               "rule although they sit outside the liver: a loophole of the rule, kept as found (rules were fixed "
               "in advance).", ""]

    with ProcessPoolExecutor(min(a.workers, len(jobs))) as ex:
        futs = [ex.submit(figure, rows, *args, **kw) for rows, args, kw in jobs]
        for f in futs:
            f.result()
    (out / "README.md").write_text("\n".join(readme))
    print("wrote", out / "README.md", flush=True)


if __name__ == "__main__":
    main()
