#!/usr/bin/env python
"""What each main-arm rule (cc, ccAuc; maisi/cc_rules.py) does to a pool, a few examples per kind of violation, in
the shared figure style (viz.py). Written for the pancreas pool (same rules as liver, applied unchanged), works for
liver too. Reads only caches (filter_metrics.py, cc_rules.py, real_lesion_metrics.py); nothing is decided here.

    cc_tumour_far.png        tumour components outside the main structure, >= --far-mm from the organ (cc -> background)
    cc_tumour_near.png       tumour components outside the main structure, closer than --far-mm (incl. touching only a
                             detached organ piece: passes the old touch rule, fails cc)
    cc_organ_pieces.png      detached organ pieces (organ label not connected to the main structure; cc -> background),
                             the largest piece of each volume
    ccauc_invisible.png      tumour components inside the main structure with sep < cut (ccAuc -> surroundings' label)
    ccauc_borderline.png     inside the main structure, lowest sep >= cut: what just passes
    real_least_visible.png   the real val lesions with the lowest sep (they set the cut; EXCL = excluded)
    real_cc_changes.png      real cases (val + test) whose label cc would change (cost of the rule; descriptive)
    README.md                counts behind every figure + how the examples were chosen

Rows: [image | image + label] in an axial field of 2 x --zoom-mm centred on the object at the slice of its largest
cross-section, then the whole axial and coronal slice with the object circled (white). Examples: one per volume,
ranked by size, then --n picked at evenly spaced ranks (largest, ..., smallest), so typical cases show, not only the
extremes. Deterministic.

  python maisi/cc_review_figures.py --target pancreas --metrics $MAISI_RUNS/pool/filter_metrics/pancreas \\
      --pool $MAISI_RUNS/pool --real $PROJ/raw_data/msd07_lesion_metrics
"""
import argparse
import json
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_metrics import ORGAN, STRUCT26, image_of  # noqa: E402
from orient import axcodes, spacing_of  # noqa: E402
from review_figures import best_slice  # noqa: E402
from viz import CAPTION, draw, legend_handles, plane  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def load_pair(label_path, image_path):
    nii = nib.load(label_path)
    assert axcodes(nii.affine) == "RAS", f"{label_path} is {axcodes(nii.affine)}, expected RAS"
    label = np.asarray(nii.dataobj).astype(np.uint8)
    image = np.asarray(nib.load(image_path).dataobj, dtype=np.float32)
    assert image.shape == label.shape, (image_path, image.shape, label.shape)
    return image, label, spacing_of(nii.affine)


def main_structure(label):
    from cc_rules import main_structure as ms
    return ms(label)


def organ_piece(label, spacing):
    """Largest detached organ piece: pseudo component row (centroid, voxels, ml) + z of its largest cross-section."""
    det = (label == ORGAN) & ~main_structure(label)
    if not det.any():
        return None, None
    pieces, n = ndimage.label(det, STRUCT26)
    sizes = np.bincount(pieces.ravel(), minlength=n + 1)[1:]
    big = int(np.argmax(sizes)) + 1
    m = pieces == big
    c = {"centroid": [int(round(v)) for v in ndimage.center_of_mass(m)], "voxels": int(sizes[big - 1]),
         "ml": round(float(sizes[big - 1]) * float(np.prod(spacing)) / 1000, 3)}
    return c, int(np.argmax(m.sum(axis=(0, 1))))


def circle(ax, shape, x, y, z, view, voxels):
    row, col = (shape[1] - 1 - y, shape[0] - 1 - x) if view == "axial" else (shape[2] - 1 - z, shape[0] - 1 - x)
    r = min(max(8.0, 2.5 * (voxels ** (1 / 3))), 40.0)
    ax.add_patch(plt.Circle((col, row), r, fill=False, color="white", linewidth=0.8))


def figure(rows, path, header, organ, zoom_mm):
    """rows = [(label_path, image_path, component row or None (= organ piece), title)]."""
    if not rows:
        print("nothing for", Path(path).name, flush=True)
        return 0
    fig, axes = plt.subplots(len(rows), 4, figsize=(14.4, 4.4 * len(rows) + 1.0), dpi=110, squeeze=False)
    for (lp, ip, c, title), ax in zip(rows, axes):
        image, label, spacing = load_pair(lp, ip)
        if c is None:
            c, z = organ_piece(label, spacing)
            if c is None:
                for a in ax:
                    a.axis("off")
                ax[0].set_title(f"{title}\n(no detached piece found)", fontsize=7)
                continue
            title += f"\nlargest detached piece {c['ml']} ml"
        else:
            z = best_slice(label, c, spacing)
        x, y = c["centroid"][:2]
        rx, ry = (int(round(zoom_mm / s)) for s in spacing[:2])
        sl = (slice(max(x - rx, 0), x + rx + 1), slice(max(y - ry, 0), y + ry + 1), slice(z, z + 1))
        img, lab, aspect = plane(image[sl], label[sl], spacing, "axial", 0)
        draw(ax[0], img, None, aspect)
        draw(ax[1], img, lab, aspect)
        wrapped = "\n".join(textwrap.fill(line, 62) for line in title.split("\n"))
        ax[0].set_title(f"{wrapped}\nimage only (z {z})", fontsize=7, loc="left")
        ax[1].set_title("image + label", fontsize=7)
        for a, view, idx in ((ax[2], "axial", z), (ax[3], "coronal", y)):
            im2, lb2, asp2 = plane(image, label, spacing, view, idx)
            draw(a, im2, lb2, asp2)
            circle(a, image.shape, x, y, z, view, c["voxels"])
            a.set_title(f"{view} (whole slice, circled)", fontsize=7)
        del image, label
    fig.suptitle(f"{header}\n{CAPTION}\nZoom: axial {2 * zoom_mm:g} mm field centred on the object, slice of its "
                 "largest cross-section.", fontsize=8)
    fig.legend(handles=legend_handles(organ, "tumour"), loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path, flush=True)
    return len(rows)


def spread(items, n, size=lambda it: it["ml"], key=lambda it: it["id"]):
    """One per volume (its largest), ranked by size, n picked at evenly spaced ranks."""
    best = {}
    for it in items:
        if key(it) not in best or size(it) > size(best[key(it)]):
            best[key(it)] = it
    ranked = sorted(best.values(), key=lambda it: -size(it))
    if len(ranked) <= n:
        return ranked
    return [ranked[int(round(i))] for i in np.linspace(0, len(ranked) - 1, n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="pancreas", choices=["liver", "pancreas"])
    ap.add_argument("--metrics", required=True, help="filter_metrics dir (with cc/ from cc_rules.py)")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--real", help="real_lesion_metrics output dir (val + test, with the cc fields)")
    ap.add_argument("--out", help="default reports/figures/<target>_rules")
    ap.add_argument("--n", type=int, default=6, help="examples per figure")
    ap.add_argument("--far-mm", type=float, default=30.0)
    ap.add_argument("--zoom-mm", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=6, help="one process per figure, ~2-3 GB each")
    a = ap.parse_args()

    m = Path(a.metrics)
    out = Path(a.out or REPO / f"reports/figures/{a.target}_rules")
    out.mkdir(parents=True, exist_ok=True)
    organ = a.target
    rules = json.loads((m / "rules.json").read_text())
    cut = rules["auc_cut"]
    excl = {(e["case"], e.get("component")) for e in rules["excluded"]}
    lab_dir = f"{a.pool}/pairs/{a.target}/labelsTr"
    vols = [json.loads(p.read_text()) for p in sorted((m / "per_volume").glob("*.json"))]
    ccs = {r["id"]: r for r in (json.loads(p.read_text()) for p in sorted((m / "cc").glob("*.json")))}
    missing = [v["id"] for v in vols if v["id"] not in ccs]
    if missing:
        raise SystemExit(f"{len(missing)} volumes without a cc cache (run cc_rules.py --target {a.target}): {missing[:5]}")
    N = len(vols)
    comps = []
    for v in vols:
        lp = f"{lab_dir}/{v['id']}.nii.gz"
        for c, inm in zip(v["components"], ccs[v["id"]]["in_main"]):
            comps.append({**c, "id": v["id"], "lp": lp, "ip": image_of(lp), "in_main": inm})
    vml = {i: r["voxel_ml"] for i, r in ccs.items()}

    def t(c, extra=""):
        s = (f"\nsep {c['sep']:.3f}, HU core {c['core_hu']:g} / shell {c['shell_hu']:g}" if c.get("sep") is not None
             else "\nsep: unscored (< 10 voxels)")
        return f"{c['id']} c{c['component']}: {c['ml']:.3f} ml, {c['min_dist_mm']:g} mm to {organ}{extra}{s}"

    jobs, readme = [], ["# What the cc / ccAuc rules do: example figures", "",
                        f"Made by `maisi/cc_review_figures.py` from `{m}` (pool `{a.pool}`, {N} {a.target} volumes). "
                        f"Rules unchanged from liver (`maisi/cc_rules.py`); AUC cut **{cut}** = the least visible real "
                        f"val lesion ({rules['auc_cut_from']['case'] if rules.get('auc_cut_from') else '-'}). Shared "
                        f"figure style (`maisi/viz.py`): W400/L40, radiological, green {organ}, red tumour. Each row: "
                        "image alone, image + label, whole axial + coronal slice with the object circled. Examples: one "
                        f"per volume, ranked by size, {a.n} at evenly spaced ranks (largest ... smallest), so typical "
                        "cases show too.", ""]

    def add(name, rows, header, text):
        jobs.append((rows, str(out / name), header))
        readme.extend([f"## {name}", "", text, ""])

    # cc: tumour outside the main structure
    outside = [c for c in comps if not c["in_main"]]
    far = [c for c in outside if c["min_dist_mm"] >= a.far_mm]
    near = [c for c in outside if c["min_dist_mm"] < a.far_mm]
    nvol = lambda xs: len({c["id"] for c in xs})  # noqa: E731
    add("cc_tumour_far.png", [(c["lp"], c["ip"], c, t(c)) for c in spread(far, a.n)],
        f"cc: tumour outside the main {organ} structure, >= {a.far_mm:g} mm away (-> background)",
        f"{len(far)} components in {nvol(far)} volumes are outside the main structure and >= {a.far_mm:g} mm from "
        f"the {organ} ({sum(c['ml'] for c in far):.1f} ml in total). cc sets them to background.")
    tn = [c for c in near if c["touches"]]
    add("cc_tumour_near.png", [(c["lp"], c["ip"], c, t(c, ", touches a detached piece" if c["touches"] else ""))
                               for c in spread(near, a.n)],
        f"cc: tumour outside the main {organ} structure, < {a.far_mm:g} mm away (-> background)",
        f"{len(near)} components in {nvol(near)} volumes are outside the main structure but closer than "
        f"{a.far_mm:g} mm; {len(tn)} of them touch {organ} label, i.e. only a detached piece (they pass the old touch "
        f"rule). Near misses (not touching, <= 3 mm): {sum(c['min_dist_mm'] <= 3 and not c['touches'] for c in near)}.")

    # cc: detached organ pieces
    pv = [{"id": i, "ml": r["liver_removed_vox"] * vml[i]} for i, r in ccs.items() if r["liver_removed_vox"] > 0]
    ml = np.array([p["ml"] for p in pv]) if pv else np.zeros(0)
    add("cc_organ_pieces.png", [(f"{lab_dir}/{p['id']}.nii.gz", image_of(f"{lab_dir}/{p['id']}.nii.gz"), None,
                                 f"{p['id']}: {p['ml']:.3f} ml {organ} removed in total") for p in spread(pv, a.n)],
        f"cc: detached {organ} pieces (not connected to the main structure; -> background)",
        f"{len(pv)}/{N} volumes have {organ} label outside the main structure"
        + (f" (total per volume: median {np.median(ml):.3f} ml, p95 {np.percentile(ml, 95):.3f}, max {ml.max():.2f}; "
           f">= 0.1 ml in {int((ml >= 0.1).sum())}, >= 1 ml in {int((ml >= 1).sum())})" if len(ml) else "")
        + ". The row shows each volume's largest piece.")

    # ccAuc: invisible tumour inside the main structure
    if cut is not None:
        inv = [c for c in comps if c["in_main"] and c.get("sep") is not None and c["sep"] < cut]
        to_org = [c for c in inv if c["contact"] >= 0.5]
        add("ccauc_invisible.png", [(c["lp"], c["ip"], c, t(c, f", -> {organ if c['contact'] >= 0.5 else 'background'}"))
                                    for c in spread(inv, a.n)],
            f"ccAuc: tumour inside the main {organ} structure with no visible lesion (sep < {cut})",
            f"{len(inv)} components in {nvol(inv)} volumes ({sum(c['ml'] for c in inv):.1f} ml); ccAuc relabels "
            f"{len(to_org)} to {organ} (>= 50% of the outer shell is {organ}) and {len(inv) - len(to_org)} to background.")
        above = sorted([c for c in comps if c["in_main"] and c.get("sep") is not None and c["sep"] >= cut],
                       key=lambda c: c["sep"])
        low = []
        for c in above:  # lowest sep, one per volume
            if c["id"] not in {x["id"] for x in low}:
                low.append(c)
            if len(low) == a.n:
                break
        add("ccauc_borderline.png", [(c["lp"], c["ip"], c, t(c)) for c in low],
            f"ccAuc: just passes (inside the main structure, lowest sep >= {cut})",
            f"The {len(low)} least visible components that pass (one per volume): what 'barely visible' looks like. "
            f"Scored in-structure components: {sum(c['in_main'] and c.get('sep') is not None for c in comps)}, "
            f"unscored (pass): {sum(c['in_main'] and c.get('sep') is None for c in comps)}.")

    # real val lesions: least visible
    real = [json.loads(p.read_text()) for p in sorted((m / "real_reference").glob("*.json"))]
    rl = sorted([{**l, "id": r["case"], "lp": r["label_path"], "ip": r.get("image_path") or image_of(r["label_path"])}
                 for r in real for l in r["lesions"] if l.get("sep") is not None], key=lambda l: l["sep"])
    add("real_least_visible.png",
        [(l["lp"], l["ip"], l, t(l, " EXCL" if (l["id"], l["component"]) in excl or (l["id"], None) in excl else ""))
         for l in rl[: a.n]],
        f"Real val lesions, least visible first (cut = {cut})",
        f"The {min(a.n, len(rl))} least visible of {len(rl)} scored real val lesions ({len(real)} cases). The first "
        "non-excluded one sets the cut; if nothing is visible under its label it may be a label error "
        "(real_reference_exclusions.json, with a reason).")

    # real cases (val + test): what cc would change
    if a.real:
        rc = [json.loads(p.read_text()) for p in sorted((Path(a.real) / "cases").glob("*.json"))]
        rc = [r for r in rc if "cc" in r]
        rows, val_cases = [], {r["case"] for r in real}
        ch = sorted([r for r in rc if r["cc"]["organ_removed_ml"] + r["cc"]["tumour_outside_ml"] > 0],
                    key=lambda r: -(r["cc"]["organ_removed_ml"] + r["cc"]["tumour_outside_ml"]))
        for r in ch[: a.n]:
            ip = r.get("image_path") or image_of(r["label_path"])
            split = "val" if r["case"] in val_cases else "test"
            what = (f"{r['case']} ({split}): {r['cc']['organ_removed_ml']} ml {organ} in {r['cc']['pieces']} detached "
                    f"pieces, {r['cc']['tumour_outside']} tumour comps ({r['cc']['tumour_outside_ml']} ml) outside")
            outl = [l for l in r["lesions"] if l["component"] in r["cc"].get("outside_components", [])]
            if outl and (not r["cc"]["pieces"] or r["cc"]["tumour_outside_ml"] >= r["cc"]["organ_removed_ml"]):
                c = {**max(outl, key=lambda l: l["ml"]), "id": r["case"]}
                rows.append((r["label_path"], ip, c, f"{what}\nshown: tumour c{c['component']}, {c['ml']} ml"))
            else:
                rows.append((r["label_path"], ip, None, what))
        add("real_cc_changes.png", rows, "Real cases (val + test) whose label cc would change (descriptive)",
            f"{len(ch)}/{len(rc)} real cases: cc would remove {organ} label (detached pieces) in "
            f"{sum(r['cc']['pieces'] > 0 for r in rc)} and tumour outside the main structure in "
            f"{sum(r['cc']['tumour_outside'] > 0 for r in rc)}. Real labels are the reference, so this is a cost of the "
            "rule (it would delete real anatomy), not a repair. Largest changes first.")

    with ProcessPoolExecutor(max(1, min(a.workers, len(jobs)))) as ex:
        futs = [ex.submit(figure, rows, path, header, organ, a.zoom_mm) for rows, path, header in jobs]
        for f in futs:
            f.result()
    (out / "README.md").write_text("\n".join(readme))
    print("wrote", out / "README.md", flush=True)


if __name__ == "__main__":
    main()
