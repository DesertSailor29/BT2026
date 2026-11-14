#!/usr/bin/env python
"""Liver checks for the MAISI pool (PLAN_training.md "Amendment candidates", 2026-11-14). CPU. DESCRIPTIVE ONLY:
nothing here changes a filter decision. It measures, counts and draws figures so the user can decide.

Per volume (the liver pool, and all 131 real LiTS: the 20 val cases are the reference, the 111 test cases are
described only), measured by the same code:
  pieces   26-connected pieces of the liver label. Main liver = the largest piece. Main structure = the 26-connected
           component of liver + tumour that holds the main liver. A liver piece outside it is "detached" (a piece
           joined to the main liver only through tumour is "attached via tumour"). The liver is one organ, so a
           detached piece is a label error unless real livers have them too.
  tumour   per tumour component (numbered as filter_metrics.py numbers them; cross-checked against its cache):
           touches (touch v1: 26-adjacent to any liver voxel), in_main (touch v2: inside the main structure),
           touches_main (26-adjacent to the main liver itself).
  liver    is the liver label really liver? Inside the label, core = voxels >= 2 mm from its edge (label edges are
           1.5 mm steps): HU median, IQR, p5 / p95, share of fat (< -30 HU), air (< -200 HU), bone (> 300 HU).
           Edge: median HU gradient (HU/mm, Gaussian 1 mm) on the main liver's surface (not next to tumour), and its
           ratio to the core's median gradient (a label that follows no organ edge has a weak edge). Shape: liver
           ml, main share, centroid relative to the body centroid (mm to the patient's right), craniocaudal extent.

Counts (summary.md): for each tumour rule (touch v1, touch v2, AUC at --cuts, touch v2 or AUC) the volumes that stay
unchanged / get repaired with tumour left / are tumour-free after repair (deleted under "repair, delete tumour-free")
/ had no tumour; the "keep the main structure" repair (detached liver pieces and tumour outside -> background);
detached pieces in the pool vs real; each liver measure: real val range vs the pool, volumes outside the range.

Blind review (the user judges whether the liver measures flag visibly bad livers): --blind-n flagged (any liver
measure outside the real val range) + --blind-n unflagged pool volumes, shuffled, shown under codes only
(blind/B01..; the key stays in <out>/blind_key.json, not in the repo). The user fills blind/marks.csv
(ok / bad / unsure), then `--score marks.csv` compares the marks with the flags. Look at the blind sheets BEFORE
the open figures (open/), which show the flagged extremes by name.

Outputs: <out> (default <metrics>/liver_check): pool/<id>.json, real/<case>.json caches (resumable), volumes.csv,
pieces.csv, tumours.csv, summary.md, blind_key.json. Figures (default reports/figures/liver_review): README.md,
open/*.jpg, blind/B*.jpg + marks.csv.

Memory ~3 GB per worker on a 512^3 pair. Run detached (CLAUDE.md):
    python maisi/liver_metrics.py --workers 14
    python maisi/liver_metrics.py --score reports/figures/liver_review/blind/marks.csv     # after the review
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_metrics import ORGAN, STRUCT26, TUMOR, distance_to_organ_mm, write_json  # noqa: E402
from orient import spacing_of  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROJ = os.environ.get("PROJ", "/work/Research")
CORE_MM, GRAD_SIGMA_MM = 2.0, 1.0
FAT_HU, AIR_HU, BONE_HU, BODY_HU = -30, -200, 300, -500
LVERSION = f"liver1 cc26 core={CORE_MM:g}mm grad_sigma={GRAD_SIGMA_MM:g}mm fat<{FAT_HU} air<{AIR_HU} bone>{BONE_HU}"
# (key, description, decimals). FLAG = measures whose real val range defines "flagged" for the blind review.
MEASURES = [
    ("liver_ml", "liver volume (ml, all pieces)", 0),
    ("main_share", "main piece / all liver label", 4),
    ("hu_median", "core HU median", 1),
    ("hu_iqr", "core HU IQR", 1),
    ("frac_fat", f"core share < {FAT_HU} HU (fat)", 4),
    ("frac_air", f"core share < {AIR_HU} HU (air)", 4),
    ("frac_bone", f"core share > {BONE_HU} HU (bone)", 4),
    ("edge_grad", "edge gradient median (HU/mm)", 1),
    ("edge_ratio", "edge / core gradient", 3),
    ("right_mm", "liver centroid right of body centroid (mm)", 1),
    ("cc_extent_mm", "main liver craniocaudal extent (mm)", 1),
]
FLAG = [m for m, _, _ in MEASURES if m != "edge_grad"]  # raw gradient depends on noise/resolution; the ratio doesn't


# --- measurement ---------------------------------------------------------------------------------------------------
def load_pair(label_path, image_path, with_image=True):
    nii = nib.load(label_path)
    codes = "".join(aff2axcodes(nii.affine))
    assert codes == "RAS", f"{label_path} is {codes}, expected RAS"
    label = np.asarray(nii.dataobj).astype(np.uint8)
    img = np.asarray(nib.load(image_path).dataobj, dtype=np.float32) if with_image else None
    assert img is None or img.shape == label.shape, (image_path, img.shape, label.shape)
    return img, label, spacing_of(nii.affine)


def z_best(mask, z0):
    """Axial index (global) of the largest cross-section of `mask` (a crop starting at axial index z0)."""
    return int(z0 + np.argmax(mask.sum(axis=(0, 1))))


def gradient_mm(im, spacing):
    """|grad| in HU/mm, Gaussian derivative with sigma GRAD_SIGMA_MM (>= 0.5 voxel) per axis."""
    sig = [max(GRAD_SIGMA_MM / s, 0.5) for s in spacing]
    g = np.zeros(im.shape, np.float32)
    for ax, s in enumerate(spacing):
        order = [0, 0, 0]
        order[ax] = 1
        d = ndimage.gaussian_filter(im, sig, order=order, output=np.float32)
        g += (d / s) ** 2
    return np.sqrt(g)


def liver_record(label, img, spacing):
    """All measurements of one volume (no decisions). Coordinates in the rows are global voxel indices (RAS)."""
    sp = np.asarray(spacing, float)
    voxel_ml = float(np.prod(sp)) / 1000.0
    rec = {"spacing": [round(float(s), 4) for s in spacing]}
    fg = label > 0
    if not fg.any():
        rec.update(liver_ml=0.0, n_pieces=0, pieces=[], tumours=[])
        return rec
    margin = np.ceil(4 * GRAD_SIGMA_MM / sp).astype(int) + 3
    box = ndimage.find_objects(fg.astype(np.uint8))[0]
    sl = tuple(slice(max(b.start - m, 0), min(b.stop + m, n)) for b, m, n in zip(box, margin, label.shape))
    off = np.array([s.start for s in sl])
    lab = label[sl]
    liver, tumor = lab == ORGAN, lab == TUMOR

    # pieces + main structure (no liver at all: every tumour component is outside, no liver measures)
    pieces, npieces = ndimage.label(liver, STRUCT26)
    sizes = np.bincount(pieces.ravel(), minlength=npieces + 1)
    sizes[0] = 0
    union, _ = ndimage.label(liver | tumor, STRUCT26)
    main = int(np.argmax(sizes)) if npieces else 0
    main_u = int(union.ravel()[np.flatnonzero(pieces.ravel() == main)[0]]) if npieces else -1
    main_mask = pieces == main if npieces else np.zeros_like(liver)
    piece_u = ndimage.maximum(union, pieces, range(1, npieces + 1)) if npieces else []  # union label per piece
    others = np.argwhere((pieces > 0) & ~main_mask)
    mind = np.full(npieces + 1, np.inf)
    if len(others):
        np.minimum.at(mind, pieces[tuple(others.T)], distance_to_organ_mm(main_mask, others, sp))
    prow = []
    for p, pbox in enumerate(ndimage.find_objects(pieces), 1):
        if p == main or pbox is None:
            continue
        pm = pieces[pbox] == p
        cen = [int(round(v)) + int(s.start) + int(o) for v, s, o in zip(ndimage.center_of_mass(pm), pbox, off)]
        prow.append({"piece": p, "voxels": int(sizes[p]), "ml": round(sizes[p] * voxel_ml, 4),
                     "attached_via_tumour": bool(piece_u[p - 1] == main_u), "min_dist_mm": round(float(mind[p]), 2),
                     "centroid": cen, "z_best": z_best(pm, pbox[2].start + off[2])})
    prow.sort(key=lambda r: -r["voxels"])
    liver_vox = int(sizes.sum())
    det = [r for r in prow if not r["attached_via_tumour"]]

    # tumour components (same numbering as filter_metrics: raster order is kept by the crop)
    comp, n = ndimage.label(tumor, STRUCT26)
    trow = []
    for c, cbox in enumerate(ndimage.find_objects(comp), 1):
        bb = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, d)) for s, d in zip(cbox, comp.shape))
        cc = comp[bb] == c
        ring = ndimage.binary_dilation(cc, STRUCT26) & ~cc
        first = np.argwhere(cc)[0]
        trow.append({"component": c, "voxels": int(cc.sum()), "ml": round(float(cc.sum()) * voxel_ml, 4),
                     "touches": bool(liver[bb][ring].any()), "touches_main": bool(main_mask[bb][ring].any()),
                     "in_main": bool(union[bb][tuple(first)] == main_u),
                     "centroid": [int(round(v)) + int(s.start) + int(o)
                                  for v, s, o in zip(ndimage.center_of_mass(cc), bb, off)],
                     "z_best": z_best(cc, bb[2].start + off[2])})
    if not npieces:  # figures show the tumour's slices instead
        rec.update(liver_ml=0.0, n_pieces=0, pieces=[], tumours=trow, z_liver=z_best(tumor, off[2]),
                   y_liver=int(round(ndimage.center_of_mass(tumor)[1])) + int(off[1]))
        return rec

    # liver measures
    r = [int(np.ceil(CORE_MM / s)) for s in sp]
    core = ndimage.minimum_filter(liver.view(np.uint8), size=[2 * k + 1 for k in r], mode="constant", cval=0) > 0
    if core.sum() < 10:
        core = liver
    v = img[sl][core]
    p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
    # gradient only in the liver's box (+ margin): far tumour blobs would otherwise make the box ~ the whole volume
    lb = ndimage.find_objects(liver.view(np.uint8))[0]
    ls = tuple(slice(max(b.start - m, 0), min(b.stop + m, n)) for b, m, n in zip(lb, margin, liver.shape))
    g = gradient_mm(img[tuple(slice(s.start + o, s.stop + o) for s, o in zip(ls, off))], sp)
    mm = main_mask[ls]
    surface = mm & ~ndimage.binary_erosion(mm, border_value=0)
    surface &= ~ndimage.binary_dilation(tumor[ls], STRUCT26, iterations=2)
    edge = float(np.median(g[surface])) if surface.any() else None
    core_g = float(np.median(g[core[ls]]))
    del g
    sub = img[::4, ::4, ::4] > BODY_HU
    wx = sub.sum(axis=(1, 2))
    body_cx = float((np.arange(len(wx)) * wx).sum() / max(wx.sum(), 1)) * 4
    liver_cx = float(ndimage.center_of_mass(liver)[0]) + off[0]
    zs = np.flatnonzero(main_mask.any(axis=(0, 1)))
    cy = int(round(ndimage.center_of_mass(main_mask)[1])) + int(off[1])
    rec.update({
        "liver_ml": round(liver_vox * voxel_ml, 2), "main_ml": round(sizes[main] * voxel_ml, 2),
        "main_share": round(sizes[main] / liver_vox, 4), "n_pieces": int(npieces),
        "n_detached": len(det), "detached_ml": round(sum(x["ml"] for x in det), 3),
        "n_attached_via_tumour": len(prow) - len(det),
        "core_vox": int(v.size), "hu_median": round(float(p50), 1), "hu_iqr": round(float(p75 - p25), 1),
        "hu_p5": round(float(p5), 1), "hu_p95": round(float(p95), 1),
        "frac_fat": round(float((v < FAT_HU).mean()), 4), "frac_air": round(float((v < AIR_HU).mean()), 4),
        "frac_bone": round(float((v > BONE_HU).mean()), 4),
        "edge_grad": None if edge is None else round(edge, 1), "core_grad": round(core_g, 1),
        "edge_ratio": None if edge is None or core_g <= 0 else round(edge / core_g, 3),
        "right_mm": round((body_cx - liver_cx) * sp[0], 1),
        "cc_extent_mm": round(float(zs[-1] - zs[0] + 1) * sp[2], 1),
        "z_liver": z_best(main_mask, off[2]), "y_liver": cy,
        "tumor_ml": round(float(tumor.sum()) * voxel_ml, 3),
        "pieces": prow, "tumours": trow,
    })
    return rec


def measure(args):
    kind, name, label_path, image_path, out, extra = args
    cache = Path(out) / kind / f"{name}.json"
    if cache.exists() and json.loads(cache.read_text()).get("version") == LVERSION:
        return kind, name, "skip"
    img, label, spacing = load_pair(label_path, image_path)
    rec = {"name": name, "kind": kind, "version": LVERSION, "label_path": label_path, "image_path": image_path,
           **extra, **liver_record(label, img, spacing)}
    write_json(cache, rec)
    return kind, name, (f"{rec['n_pieces']} liver pieces ({rec.get('n_detached', 0)} detached), "
                        f"{len(rec['tumours'])} tumour comps, {sum(not t['in_main'] for t in rec['tumours'])} outside main")


# --- decisions to count (not to apply) ----------------------------------------------------------------------------
def classify(fails):
    """no_tumour | unchanged | repaired (tumour left) | tumour_free (every component fails)."""
    if not fails:
        return "no_tumour"
    nf = sum(fails)
    return "unchanged" if nf == 0 else "tumour_free" if nf == len(fails) else "repaired"


def rules(cuts):
    """name -> per-component fail function on a merged tumour row (liver_metrics row + filter_metrics sep)."""
    def auc(c, cut):
        return c.get("sep") is not None and c["sep"] < cut
    out = {"touch v1 (any liver voxel; as fixed)": lambda c: not c["touches"],
           "touch v2 (main structure)": lambda c: not c["in_main"]}
    for cut in cuts:
        out[f"AUC sep < {cut:g}"] = lambda c, cut=cut: auc(c, cut)
    for cut in cuts:
        out[f"touch v2 or AUC < {cut:g}"] = lambda c, cut=cut: not c["in_main"] or auc(c, cut)
    return out


def merge_sep(rec, metrics_dir):
    """Add filter_metrics' sep to the tumour rows; returns a list of mismatch messages (numbering, voxels, touch)."""
    p = Path(metrics_dir) / "per_volume" / f"{rec['name']}.json"
    if not p.exists():
        return [f"{rec['name']}: no filter_metrics cache"]
    fm = json.loads(p.read_text())["components"]
    if len(fm) != len(rec["tumours"]):
        return [f"{rec['name']}: {len(rec['tumours'])} vs {len(fm)} components"]
    bad = []
    for t, f in zip(rec["tumours"], fm):
        if t["voxels"] != f["voxels"] or t["touches"] != f["touches"]:
            bad.append(f"{rec['name']} c{t['component']}: voxels {t['voxels']}/{f['voxels']} touches "
                       f"{t['touches']}/{f['touches']}")
        t["sep"] = f.get("sep")
    return bad


# --- tables + summary ---------------------------------------------------------------------------------------------
def pct(v, ps=(5, 50, 95), d=3):
    v = [x for x in v if x is not None]
    return " / ".join(f"{np.percentile(v, p):.{d}f}" for p in ps) if v else "-"


def val_ranges(val):
    rng = {}
    for m, _, _ in MEASURES:
        v = [r[m] for r in val if r.get(m) is not None]
        if v:
            rng[m] = (min(v), max(v))
    return rng


def outside(rec, rng):
    """Liver measures of `rec` outside the real val range: [(measure, 'low'|'high', value)]."""
    out = []
    for m in FLAG:
        if m in rng and rec.get(m) is not None:
            lo, hi = rng[m]
            if rec[m] < lo:
                out.append((m, "low", rec[m]))
            elif rec[m] > hi:
                out.append((m, "high", rec[m]))
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def tables(out, metrics_dir, cuts):
    out = Path(out)
    pool = [json.loads(p.read_text()) for p in sorted((out / "pool").glob("*.json"))]
    real = [json.loads(p.read_text()) for p in sorted((out / "real").glob("*.json"))]
    val = [r for r in real if r.get("split") == "val"]
    test = [r for r in real if r.get("split") == "test"]
    mism = [m for r in pool for m in merge_sep(r, metrics_dir)]
    rng = val_ranges(val)
    R = rules(cuts)
    for r in pool:
        r["outside"] = outside(r, rng)
        for name, f in R.items():
            r["cls::" + name] = classify([f(t) for t in r["tumours"]])
        lost_t = sum(not t["in_main"] for t in r["tumours"])
        r["cc_change"] = ("unchanged" if not lost_t and not r.get("n_detached") else "liver+tumour" if lost_t and
                          r.get("n_detached") else "tumour only" if lost_t else "liver only")
        r["cc_tumour_free"] = bool(r["tumours"]) and lost_t == len(r["tumours"])
    for r in real:
        r["outside"] = outside(r, rng)

    # CSVs
    scal = lambda r: {k: v for k, v in r.items() if not isinstance(v, (list, dict))}  # noqa: E731
    write_csv(out / "volumes.csv", [{**scal(r), "outside": ";".join(f"{m}:{s}" for m, s, _ in r["outside"])}
                                    for r in pool + real])
    write_csv(out / "pieces.csv", [{"name": r["name"], "kind": r["kind"], "split": r.get("split", ""), **p}
                                   for r in pool + real for p in r["pieces"]])
    write_csv(out / "tumours.csv", [{"name": r["name"], **t} for r in pool for t in r["tumours"]])

    N = len(pool)
    L = ["# Liver checks: counts and measures (descriptive only)", "",
         f"`maisi/liver_metrics.py`, version `{LVERSION}`. Pool: {N} volumes; real LiTS: {len(val)} val "
         f"(the reference) + {len(test)} test (descriptive only). Nothing here changes a filter decision.", ""]
    if mism:
        L += [f"**Cross-check with filter_metrics failed for {len(mism)} components/volumes** (first 10):", ""]
        L += [f"- {m}" for m in mism[:10]] + [""]
    else:
        L += ["Cross-check with the filter_metrics cache: component numbering, voxels and touch v1 identical.", ""]

    # 1. tumour rules
    L += ["## 1. Tumour rules: what repair and reject would do", "",
          "Per volume: *unchanged* = no component fails; *repaired* = some fail, tumour left; *tumour-free* = every "
          "component fails (repair leaves no tumour: kept as a tumour-free volume by repair-keep, deleted by "
          "repair-delete); *no tumour* = none generated. Reject keeps unchanged + no tumour.", "",
          "| rule | no tumour | unchanged | repaired | tumour-free after repair | reject keeps | "
          "repair-keep keeps | repair-delete keeps |", "|---|---|---|---|---|---|---|---|"]
    for name in R:
        c = Counter(r["cls::" + name] for r in pool)
        L.append(f"| {name} | {c['no_tumour']} | {c['unchanged']} | {c['repaired']} | {c['tumour_free']} | "
                 f"{c['no_tumour'] + c['unchanged']} | {N} | {N - c['tumour_free']} |")
    comps = [t for r in pool for t in r["tumours"]]
    v1p_v2f = [(r["name"], t) for r in pool for t in r["tumours"] if t["touches"] and not t["in_main"]]
    L += ["", f"Tumour components: {len(comps)}. Touch v1 passes but v2 fails (touches only a detached liver piece): "
          f"**{len(v1p_v2f)} components in {len({n for n, _ in v1p_v2f})} volumes** "
          f"({sum(t['ml'] for _, t in v1p_v2f):.1f} ml). In the main structure but not touching the main liver "
          f"itself (joined via another liver piece / tumour): "
          f"{sum(t['in_main'] and not t['touches_main'] for t in comps)}.", ""]
    for name, rr in (("val", val), ("test", test)):
        rc = [t for r in rr for t in r["tumours"]]
        if rc:
            L.append(f"Real {name} lesions in the main structure (pass touch v2): "
                     f"{sum(t['in_main'] for t in rc)}/{len(rc)} (touch v1: {sum(t['touches'] for t in rc)}/{len(rc)}).")
    L.append("")

    # 2. main-structure repair
    cc = Counter(r["cc_change"] for r in pool)
    L += ["## 2. \"Keep the main structure\" (liver + tumour connected to the main liver; the rest -> background)", "",
          "| volumes | count |", "|---|---|",
          *[f"| {k} | {cc[k]} |" for k in ("unchanged", "liver only", "tumour only", "liver+tumour")],
          f"| tumour-free after this repair (had tumour) | {sum(r['cc_tumour_free'] for r in pool)} |",
          f"| reject (keep only unchanged) keeps | {cc['unchanged']} |", ""]

    # 3. pieces
    L += ["## 3. Liver label pieces (the liver is one organ)", "",
          "| set | volumes | > 1 piece | with detached piece | pieces / volume (max) | detached ml p50 / p95 / max "
          "| largest detached piece (ml) | min main share | attached only via tumour |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, rr in (("pool", pool), ("real val", val), ("real test", test)):
        if not rr:
            continue
        det = [p["ml"] for r in rr for p in r["pieces"] if not p["attached_via_tumour"]]
        big = max(rr, key=lambda r: max([p["ml"] for p in r["pieces"] if not p["attached_via_tumour"]] or [0]))
        bigml = max([p["ml"] for p in big["pieces"] if not p["attached_via_tumour"]] or [0])
        L.append(f"| {name} | {len(rr)} | {sum(r['n_pieces'] > 1 for r in rr)} | {sum(r.get('n_detached', 0) > 0 for r in rr)} "
                 f"| {max(r['n_pieces'] for r in rr)} | {pct(det, (50, 95, 100))} | {bigml:.2f} ({big['name']}) "
                 f"| {min(r.get('main_share', 1) for r in rr):.4f} | {sum(r.get('n_attached_via_tumour', 0) for r in rr)} |")
    for lim in (0.1, 1.0, 10.0):
        L.append(f"- pool volumes with a detached piece >= {lim:g} ml: "
                 f"{sum(any(p['ml'] >= lim and not p['attached_via_tumour'] for p in r['pieces']) for r in pool)}; "
                 f"real val {sum(any(p['ml'] >= lim and not p['attached_via_tumour'] for p in r['pieces']) for r in val)}, "
                 f"real test {sum(any(p['ml'] >= lim and not p['attached_via_tumour'] for p in r['pieces']) for r in test)}")
    L.append("")

    # 4. liver measures
    L += ["## 4. Is the liver label liver? Measures vs the 20 real val livers", "",
          "A pool volume is *outside* if a measure is below the min or above the max of the real val livers (same "
          "principle as the AUC cut: unlike every real liver). Flag measures: " + ", ".join(FLAG) + ".", "",
          "| measure | real val min / max | real test p5 / median / p95 | pool p5 / median / p95 | pool below | "
          "pool above | real test outside |", "|---|---|---|---|---|---|---|"]
    for m, desc, d in MEASURES:
        if m not in rng:
            continue
        lo, hi = rng[m]
        below = sum(r.get(m) is not None and r[m] < lo for r in pool)
        above = sum(r.get(m) is not None and r[m] > hi for r in pool)
        tout = sum(r.get(m) is not None and not lo <= r[m] <= hi for r in test)
        L.append(f"| {desc} | {lo:.{d}f} / {hi:.{d}f} | {pct([r.get(m) for r in test], d=d)} | "
                 f"{pct([r.get(m) for r in pool], d=d)} | {below} | {above} | {tout}/{len(test)} |")
    flagged = [r for r in pool if r["outside"]]
    nout = Counter(len(r["outside"]) for r in pool)
    L += ["", f"**Pool volumes outside on >= 1 flag measure: {len(flagged)}/{N}** (by number of measures: "
          + ", ".join(f"{k}: {nout[k]}" for k in sorted(nout) if k) + "). Real test livers outside on >= 1: "
          f"{sum(bool(r['outside']) for r in test)}/{len(test)} (how often a real liver would be flagged).", ""]
    v2 = "touch v2 (main structure)"
    L += ["Overlap with the tumour rules (volumes):", "", "| | touch v2 ok | touch v2 fails |", "|---|---|---|"]
    for name, sel in (("liver ok", lambda r: not r["outside"]), ("liver flagged", lambda r: bool(r["outside"]))):
        sub = [r for r in pool if sel(r)]
        ok = sum(r["cls::" + v2] in ("unchanged", "no_tumour") for r in sub)
        L.append(f"| {name} | {ok} | {len(sub) - ok} |")
    L += ["", "Most extreme pool volumes per flag measure (beyond the real val range):", ""]
    for m in FLAG:
        for side in ("low", "high"):
            xs = sorted([(v, r["name"]) for r in pool for mm, s, v in r["outside"] if mm == m and s == side],
                        reverse=side == "high")
            if xs:
                L.append(f"- {m} {side} ({len(xs)}): " + ", ".join(f"{n} {v:g}" for v, n in xs[:6]))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "summary.md", flush=True)
    return pool, real, rng


# --- figures ------------------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save(fig, path):
    fig.savefig(path, pil_kwargs={"quality": 85}) if str(path).endswith(".jpg") else fig.savefig(path)


def view_panels(axes, rec, img, label, spacing, with_label_only=False):
    """Axial at the main liver's largest cross-section, coronal through its centroid: [image, image+label] each."""
    from viz import draw, plane
    out = []
    for view, idx in (("axial", rec["z_liver"]), ("coronal", rec["y_liver"])):
        im2, lb2, asp = plane(img, label, spacing, view, idx)
        out.append((view, idx, im2, lb2, asp))
    i = 0
    for view, idx, im2, lb2, asp in out:
        if not with_label_only:
            draw(axes[i], im2, None, asp)
            axes[i].set_title(f"{view} {idx}, image", fontsize=7)
            i += 1
        draw(axes[i], im2, lb2, asp)
        axes[i].set_title(f"{view} {idx}, image + label", fontsize=7)
        i += 1


def blind_page(args):
    rows, path = args  # rows = [(code, rec)]
    plt = _plt()
    from viz import CAPTION, legend_handles
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.6 * len(rows) + 1), dpi=100, squeeze=False)
    for (code, rec), ax in zip(rows, axes):
        img, label, sp = load_pair(rec["label_path"], rec["image_path"])
        view_panels(ax, rec, img, label, sp)
        ax[0].set_title(f"{code}:  " + ax[0].get_title(), fontsize=10, fontweight="bold")
    fig.suptitle(f"Blind liver review. Mark each code ok / bad / unsure in marks.csv: is the green label the "
                 f"liver?\n{CAPTION}", fontsize=8)
    fig.legend(handles=legend_handles("liver", "tumour"), loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.6 / fig.get_figheight()))
    save(fig, path)
    plt.close(fig)
    return str(path)


def component_page(args):
    """rows = [(rec, comp_row, title)]: zoom image, zoom image+label, whole axial + coronal (circled)."""
    rows, path, header, zoom_mm = args
    plt = _plt()
    from review_figures import circle, zoom
    from viz import CAPTION, draw, legend_handles, plane
    fig, axes = plt.subplots(len(rows), 4, figsize=(14.4, 3.6 * len(rows) + 1), dpi=100, squeeze=False)
    for (rec, c, title), ax in zip(rows, axes):
        img, label, sp = load_pair(rec["label_path"], rec["image_path"])
        z = c["z_best"]
        im2, lb2, asp = zoom(img, label, sp, c, z, zoom_mm)
        draw(ax[0], im2, None, asp)
        draw(ax[1], im2, lb2, asp)
        ax[0].set_title(f"{title}\nimage only (z {z})", fontsize=7)
        ax[1].set_title("image + label", fontsize=7)
        cz = dict(c, centroid=[c["centroid"][0], c["centroid"][1], z])
        for a, view, idx in ((ax[2], "axial", z), (ax[3], "coronal", c["centroid"][1])):
            im3, lb3, asp3 = plane(img, label, sp, view, idx)
            draw(a, im3, lb3, asp3)
            circle(a, img.shape, cz, view, z)
            a.set_title(f"{view} (whole slice, circled)", fontsize=7)
    fig.suptitle(f"{header}\n{CAPTION}\nZoom: axial {2 * zoom_mm:g} mm field, slice of the largest cross-section.",
                 fontsize=8)
    fig.legend(handles=legend_handles("liver", "tumour"), loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.7 / fig.get_figheight()))
    save(fig, path)
    plt.close(fig)
    return str(path)


def extremes_page(args):
    """rows = [(syn_rec, real_rec, title)]: pool axial + coronal with label | the most extreme real val liver."""
    rows, path = args
    plt = _plt()
    from viz import CAPTION, legend_handles
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.6 * len(rows) + 1), dpi=100, squeeze=False)
    for (syn, real, title), ax in zip(rows, axes):
        for rec, a in ((syn, ax[:2]), (real, ax[2:])):
            img, label, sp = load_pair(rec["label_path"], rec["image_path"])
            view_panels(a, rec, img, label, sp, with_label_only=True)
            a[0].set_title(f"{rec['name']}: " + a[0].get_title(), fontsize=7)
        ax[0].set_title(f"{title}\n" + ax[0].get_title(), fontsize=7)
    fig.suptitle(f"Liver measures beyond the real val range: pool (left) vs the most extreme real val liver on the "
                 f"same side (right)\n{CAPTION}", fontsize=8)
    fig.legend(handles=legend_handles("liver", "tumour"), loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.6 / fig.get_figheight()))
    save(fig, path)
    plt.close(fig)
    return str(path)


def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def blind_set(out, pool, n, seed, resample):
    key_path = Path(out) / "blind_key.json"
    if key_path.exists() and not resample:
        return json.loads(key_path.read_text())
    rnd = random.Random(seed)
    flagged = sorted(r["name"] for r in pool if r["outside"])
    clean = sorted(r["name"] for r in pool if not r["outside"])
    pick = [(x, True) for x in rnd.sample(flagged, min(n, len(flagged)))] + \
           [(x, False) for x in rnd.sample(clean, min(n, len(clean)))]
    rnd.shuffle(pick)
    by = {r["name"]: r for r in pool}
    key = {"seed": seed, "n_flagged_total": len(flagged), "n_clean_total": len(clean),
           "cases": {f"B{i:02d}": {"name": x, "flagged": f, "outside": by[x]["outside"]}
                     for i, (x, f) in enumerate(pick, 1)}}
    write_json(key_path, key)
    return key


def figures(out, fig_out, pool, real, rng, a):
    fig_out = Path(fig_out)
    (fig_out / "open").mkdir(parents=True, exist_ok=True)
    (fig_out / "blind").mkdir(parents=True, exist_ok=True)
    by = {r["name"]: r for r in pool + real}
    val = [r for r in real if r.get("split") == "val"]
    jobs = []

    # blind sheets (codes only)
    key = blind_set(out, pool, a.blind_n, a.seed, a.resample)
    codes = sorted(key["cases"])
    for i, ch in enumerate(chunks(codes, 4), 1):
        jobs.append((blind_page, ([(c, by[key["cases"][c]["name"]]) for c in ch], fig_out / "blind" / f"B_page{i:02d}.jpg")))
    marks = fig_out / "blind" / "marks.csv"
    if not marks.exists() or a.resample:
        with open(marks, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code", "verdict", "note"])
            w.writerows([[c, "", ""] for c in codes])

    # open: detached liver pieces (largest per volume), pool + real
    def ptitle(r, p):
        return (f"{r['name']} piece {p['piece']}: {p['ml']:.2f} ml, {p['min_dist_mm']:g} mm from the main liver "
                f"({r['n_detached']} detached, main {r['main_ml']:.0f} ml)")

    def detached_first(recs, n):
        xs = [(r, max((p for p in r["pieces"] if not p["attached_via_tumour"]), key=lambda p: p["ml"]))
              for r in recs if r.get("n_detached")]
        return sorted(xs, key=lambda x: -x[1]["ml"])[:n]

    for name, recs, n in (("detached_liver_pieces_pool", pool, a.n), ("detached_liver_pieces_real", real, a.n)):
        rows = [(r, p, ptitle(r, p)) for r, p in detached_first(recs, n)]
        if rows:
            jobs.append((component_page, (rows, fig_out / "open" / f"{name}.jpg",
                                          f"Detached pieces of liver label ({'MAISI pool' if recs is pool else 'real LiTS'}), "
                                          "largest per volume", a.zoom_mm)))
    # open: tumours that pass v1 but fail v2
    v2only = sorted([(r, t) for r in pool for t in r["tumours"] if t["touches"] and not t["in_main"]],
                    key=lambda x: -x[1]["ml"])
    seen, rows = set(), []
    for r, t in v2only:
        if r["name"] not in seen and len(rows) < a.n:
            seen.add(r["name"])
            rows.append((r, t, f"{r['name']} c{t['component']}: {t['ml']:.2f} ml"
                         + (f", sep {t['sep']:.3f}" if t.get("sep") is not None else "")))
    if rows:
        jobs.append((component_page, (rows, fig_out / "open" / "tumour_on_detached_piece.jpg",
                                      "Tumour touching only a detached liver piece (passes touch v1, fails v2)",
                                      a.zoom_mm)))
    # open: extremes per flag measure and side, next to the most extreme real val liver on that side
    ext = []
    for m in FLAG:
        if m not in rng:
            continue
        for side in ("low", "high"):
            xs = [r for r in pool if any(mm == m and s == side for mm, s, _ in r["outside"])]
            if not xs:
                continue
            syn = (min if side == "low" else max)(xs, key=lambda r: r[m])
            ref = (min if side == "low" else max)([r for r in val if r.get(m) is not None], key=lambda r: r[m])
            ext.append((syn, ref, f"{m} {side}: pool {syn[m]:g} ({len(xs)} beyond) vs real val {ref[m]:g}"))
    for i, ch in enumerate(chunks(ext, 5), 1):
        jobs.append((extremes_page, (ch, fig_out / "open" / f"liver_extremes_{i:02d}.jpg")))

    with ProcessPoolExecutor(max(1, min(a.workers, len(jobs)))) as ex:
        for f in [ex.submit(fn, arg) for fn, arg in jobs]:
            print("wrote", f.result(), flush=True)

    fl = sum(v["flagged"] for v in key["cases"].values())
    readme = [
        "# Liver review figures", "",
        f"Made by `maisi/liver_metrics.py` (version `{LVERSION}`). Shared figure style (`maisi/viz.py`): W400/L40, "
        "radiological, green liver, red tumour. Numbers: `summary.md` next to the caches on UCloud "
        f"(`{out}`), copied into NOTES.md.", "",
        "## 1. Blind review first (`blind/`)", "",
        f"{len(codes)} pool volumes under codes B01..: {fl} flagged by the liver measures (outside the real val range "
        f"on >= 1 measure; {key['n_flagged_total']} in the pool) and {len(codes) - fl} not flagged (of "
        f"{key['n_clean_total']}), shuffled. Each row: axial slice of the liver's largest cross-section and coronal "
        "slice through its centroid, image alone and with the label.", "",
        "Fill `blind/marks.csv`, one verdict per code: **ok** (the green label is a liver, roughly where the image "
        "shows it), **bad** (it covers bowel, fat, other organs or no visible liver; or the anatomy is broken), "
        "**unsure**. Then on UCloud: `python maisi/liver_metrics.py --score reports/figures/liver_review/blind/marks.csv`. "
        "The key is in `blind_key.json` on UCloud, not in the repo. **Don't open `open/` before marking.**", "",
        "## 2. Open figures (`open/`)", "",
        "- `detached_liver_pieces_pool.jpg` / `_real.jpg`: the largest detached piece of liver label per volume "
        "(not connected to the main liver, not even through tumour), largest volumes first. Real livers show what "
        "annotation specks look like.",
        f"- `tumour_on_detached_piece.jpg`: tumour components that touch only a detached piece of liver label "
        f"({len(v2only)} components; pass touch v1, fail v2), largest per volume.",
        "- `liver_extremes_*.jpg`: per liver measure and side, the most extreme pool volume beyond the real val "
        "range, next to the most extreme real val liver on the same side.", ""]
    (fig_out / "README.md").write_text("\n".join(readme))
    print("wrote", fig_out / "README.md", flush=True)


def score(out, marks_path):
    key = json.loads((Path(out) / "blind_key.json").read_text())["cases"]
    with open(marks_path) as f:
        marks = {r["code"].strip(): (r["verdict"] or "").strip().lower() for r in csv.DictReader(f)}
    unknown = sorted(set(marks) - set(key))
    if unknown:
        raise SystemExit(f"codes not in the key: {unknown}")
    t = Counter((key[c]["flagged"], marks.get(c) or "missing") for c in key)
    verdicts = ["ok", "bad", "unsure", "missing"]
    L = ["# Blind liver review: marks vs liver flags", "", f"Marks: `{marks_path}`. Key: {len(key)} codes.", "",
         "| | " + " | ".join(verdicts) + " |", "|---|" + "---|" * len(verdicts)]
    for f, name in ((True, "flagged"), (False, "not flagged")):
        L.append(f"| {name} | " + " | ".join(str(t[(f, v)]) for v in verdicts) + " |")
    tp, fp = t[(True, "bad")], t[(True, "ok")]
    fn, tn = t[(False, "bad")], t[(False, "ok")]
    if tp + fn:
        L.append(f"\nOf the livers marked bad, flagged: {tp}/{tp + fn}. ")
    if tp + fp:
        L.append(f"Of the flagged livers marked ok/bad, bad: {tp}/{tp + fp}.")
    L += ["", "| code | volume | flagged | verdict | measures outside the real val range |", "|---|---|---|---|---|"]
    for c in sorted(key):
        k = key[c]
        L.append(f"| {c} | {k['name']} | {'yes' if k['flagged'] else 'no'} | {marks.get(c) or '-'} | "
                 + ", ".join(f"{m} {s} ({v:g})" for m, s, v in k["outside"]) + " |")
    p = Path(out) / "blind_score.md"
    p.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=f"{os.environ.get('MAISI_RUNS', PROJ + '/maisi_runs')}/pool")
    ap.add_argument("--target", default="liver")
    ap.add_argument("--metrics", help="filter_metrics dir (default <pool>/filter_metrics/<target>)")
    ap.add_argument("--raw", default=f"{os.environ.get('nnUNet_raw', PROJ + '/nnunet_data/raw')}/Dataset101_LiverSynth",
                    help="nnU-Net raw dataset with the real LiTS cases (labelsTr = val, labelsTs_real = test)")
    ap.add_argument("--out", help="default <metrics>/liver_check")
    ap.add_argument("--fig-out", default=str(REPO / "reports/figures/liver_review"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pool volumes (testing)")
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.504, 0.547], help="AUC cuts to count")
    ap.add_argument("--blind-n", type=int, default=30, help="flagged and unflagged volumes each in the blind review")
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--resample", action="store_true", help="draw a new blind set (overwrites key + marks.csv)")
    ap.add_argument("--n", type=int, default=8, help="rows per open component figure")
    ap.add_argument("--zoom-mm", type=float, default=40.0)
    ap.add_argument("--tables-only", action="store_true", help="no measuring: tables (+ figures) from the caches")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--score", metavar="MARKS_CSV", help="compare the blind marks with the flags, then exit")
    a = ap.parse_args()

    metrics = a.metrics or f"{a.pool}/filter_metrics/{a.target}"
    out = a.out or f"{metrics}/liver_check"
    if a.score:
        score(out, a.score)
        return
    if not a.tables_only:
        lab = Path(a.pool) / "pairs" / a.target / "labelsTr"
        files = sorted(lab.glob(f"{a.target}_*.nii.gz"))[: a.limit or None]
        if not files:
            raise SystemExit(f"no labels in {lab}")
        jobs = [("pool", f.name[: -len(".nii.gz")], str(f),
                 str(lab.parent / "imagesTr" / (f.name[: -len(".nii.gz")] + "_0000.nii.gz")), out, {}) for f in files]
        raw = Path(a.raw)
        split = json.loads((raw / "real_split.json").read_text())
        for sub, key in (("Tr", "val"), ("Ts_real", "test")):
            for i in split[key]:
                case = f"lits_{i:03d}"
                jobs.append(("real", case, str(raw / f"labels{sub}" / f"{case}.nii.gz"),
                             str(raw / f"images{sub}" / f"{case}_0000.nii.gz"), out, {"split": key}))
        jobs.sort(key=lambda j: -os.path.getsize(j[3]))  # big first, so the tail is short
        print(f"{len(files)} pool + {len(jobs) - len(files)} real -> {out} ({a.workers} workers)", flush=True)
        with Pool(a.workers) as pool:
            for i, (kind, name, msg) in enumerate(pool.imap_unordered(measure, jobs, chunksize=1), 1):
                if msg != "skip" or i == len(jobs):
                    print(f"[{i}/{len(jobs)}] {kind} {name}: {msg}", flush=True)
    pool_recs, real, rng = tables(out, metrics, a.cuts)
    if not a.no_figures:
        figures(out, a.fig_out, pool_recs, real, rng, a)


if __name__ == "__main__":
    main()
