#!/usr/bin/env python
"""Filter metrics for the MAISI pool (Option B): one row per synthetic volume, CPU only.

Family 1a, tumour components vs the organ (labels only, PLAN_filtering_options.md "Repair instead of reject"):
for every 26-connected tumour component: size, min / median distance to the organ and the fraction of its voxels
within --max-mm (15) of the organ. A component is **on-organ** if >= --keep-frac (0.5) of its voxels lie within
--max-mm, else **off-organ**. The repair "delete off-organ components" (relabel them to background) puts every
volume in one class:

    no_tumour    no tumour label to begin with
    clean        nothing to delete
    repaired     some components deleted, tumour left in the organ
    tumour_free  every component deleted: the volume becomes a tumour-free training case

and the summary cross-tabs this against the thesis rule (keep a volume if >= 90% of all its tumour voxels lie within
15 mm of the organ), so it shows whether a repair arm differs from the reject arm. Distances are exact Euclidean in
mm (nearest organ boundary voxel, KD-tree), the same quantity as postprocess.tumor_fraction_near_organ; the volume
value is cross-checked against raw/<id>/result.json when that exists.

Inputs: <pool>/pairs/<target>/labelsTr/<target>_*.nii.gz (RAS, organ 1, tumour 2; postprocess.py).
Outputs in --out (default <pool>/filter_metrics/<target>):
    per_volume/<id>.json   per-volume cache incl. the component list (resumable: existing files are skipped)
    metrics.csv            one row per volume          components.csv   one row per tumour component
    summary.md             counts + cross-tab           ids_thesis15mm.json   passing ids + scores (make_splits --ids)

Memory ~1.5 GB per worker on a 512^3 label (a 12 GB CPU session: --workers 4). Run detached (CLAUDE.md):
    python filter_metrics.py --target liver --workers 8
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orient import spacing_of  # noqa: E402

ORGAN, TUMOR = 1, 2
THESIS_MM, THESIS_FRAC = 15.0, 0.9
STRUCT26 = np.ones((3, 3, 3), bool)


def distance_to_organ_mm(organ, points, spacing):
    """Euclidean distance (mm) from each voxel index in `points` (n x 3) to the nearest organ voxel; inf if no organ.

    The nearest organ voxel of an outside point is always a boundary voxel, so a KD-tree on the organ boundary gives
    the exact value (organ voxels themselves get 0)."""
    if not organ.any():
        return np.full(len(points), np.inf)
    box = ndimage.find_objects(organ.astype(np.uint8))[0]
    sub = organ[box]
    boundary = sub & ~ndimage.binary_erosion(sub, border_value=0)
    offset = np.array([s.start for s in box])
    tree = cKDTree((np.argwhere(boundary) + offset) * np.asarray(spacing))
    d, _ = tree.query(points * np.asarray(spacing))
    d[organ[tuple(points.T)]] = 0.0
    return d


def tumour_components(label, spacing, max_mm, keep_frac):
    """Per-component table + the labelled component array (0 = no tumour). Shared by metrics and repair."""
    organ, tumor = label == ORGAN, label == TUMOR
    comp, n = ndimage.label(tumor, structure=STRUCT26)
    if n == 0:
        return [], comp, np.zeros(0)
    pts = np.argwhere(comp > 0)
    ids = comp[tuple(pts.T)]
    dist = distance_to_organ_mm(organ, pts, spacing)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    rows = []
    for c in range(1, n + 1):
        d = dist[ids == c]
        frac = float((d <= max_mm).mean())
        rows.append({
            "component": c,
            "voxels": int(d.size),
            "ml": round(d.size * voxel_ml, 4),
            "min_dist_mm": round(float(d.min()), 2),
            "median_dist_mm": round(float(np.median(d)), 2),
            f"frac_within_{max_mm:g}mm": round(frac, 4),
            "on_organ": frac >= keep_frac,
        })
    return rows, comp, dist


def repair_off_organ(label, spacing, max_mm=THESIS_MM, keep_frac=0.5):
    """Repaired copy of a label: off-organ tumour components -> background (0). For the repair-arm dataset."""
    rows, comp, _ = tumour_components(label, spacing, max_mm, keep_frac)
    out = label.copy()
    drop = [r["component"] for r in rows if not r["on_organ"]]
    if drop:
        out[np.isin(comp, drop)] = 0
    return out, rows


def repair_class(rows):
    if not rows:
        return "no_tumour"
    kept = sum(r["on_organ"] for r in rows)
    return "clean" if kept == len(rows) else "tumour_free" if kept == 0 else "repaired"


def measure(args):
    path, out_dir, pool, max_mm, keep_frac = args
    vid = os.path.basename(path)[: -len(".nii.gz")]
    cache = Path(out_dir) / "per_volume" / f"{vid}.json"
    if cache.exists():
        return vid, "skip"
    nii = nib.load(path)
    label = np.asarray(nii.dataobj).astype(np.uint8)
    spacing = spacing_of(nii.affine)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    rows, _, dist = tumour_components(label, spacing, max_mm, keep_frac)
    on = [r for r in rows if r["on_organ"]]
    tumor_vox = int(dist.size)
    thesis = float((dist <= THESIS_MM).mean()) if tumor_vox else None
    rec = {
        "id": vid,
        "organ_ml": round(float((label == ORGAN).sum()) * voxel_ml, 2),
        "tumor_ml": round(tumor_vox * voxel_ml, 3),
        "tumor_components": len(rows),
        "thesis_frac15": None if thesis is None else round(thesis, 4),
        "thesis_pass": None if thesis is None else thesis >= THESIS_FRAC,
        "on_components": len(on),
        "off_components": len(rows) - len(on),
        "tumor_ml_on": round(sum(r["ml"] for r in on), 3),
        "tumor_ml_off": round(sum(r["ml"] for r in rows if not r["on_organ"]), 3),
        "off_share": round(sum(r["voxels"] for r in rows if not r["on_organ"]) / tumor_vox, 4) if tumor_vox else None,
        "repair_class": repair_class(rows),
        "max_mm": max_mm,
        "keep_frac": keep_frac,
        "components": rows,
    }
    res = Path(pool) / "raw" / vid / "result.json"
    if res.exists():
        m = json.loads(res.read_text()).get("metrics") or {}
        rec["result_json_frac15"] = m.get("tumor_frac_within_15mm")
        rec["maisi_qc_pass"] = m.get("maisi_qc_recomputed_pass")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, cache)
    return vid, rec["repair_class"]


def write_tables(out_dir, target):
    recs = [json.loads(p.read_text()) for p in sorted((Path(out_dir) / "per_volume").glob("*.json"))]
    if not recs:
        return
    vol_keys = [k for k in recs[0] if k != "components"]
    vol_keys += [k for k in ("result_json_frac15", "maisi_qc_pass") if k not in vol_keys]
    with open(Path(out_dir) / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, vol_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    comp_rows = [{"id": r["id"], **c} for r in recs for c in r["components"]]
    if comp_rows:
        with open(Path(out_dir) / "components.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(comp_rows[0]))
            w.writeheader()
            w.writerows(comp_rows)

    passing = {r["id"]: r["thesis_frac15"] for r in recs if r["thesis_pass"]}
    (Path(out_dir) / "ids_thesis15mm.json").write_text(json.dumps({
        "rule": f">= {THESIS_FRAC:.0%} of tumour voxels within {THESIS_MM:g} mm of the organ (thesis)",
        "n_pass": len(passing), "n_total": len(recs), "ids": sorted(passing), "scores": passing}, indent=1))

    classes = ["no_tumour", "clean", "repaired", "tumour_free"]
    tab = Counter((r["repair_class"], r["thesis_pass"]) for r in recs)
    mm, kf = recs[0]["max_mm"], recs[0]["keep_frac"]
    lines = [
        f"# Filter metrics: {target} ({len(recs)} volumes)", "",
        f"Repair = delete tumour components with < {kf:.0%} of their voxels within {mm:g} mm of the organ.",
        f"Thesis rule = keep a volume if >= {THESIS_FRAC:.0%} of all tumour voxels lie within {THESIS_MM:g} mm.", "",
        "| repair class | thesis pass | thesis fail | no tumour | total |", "|---|---|---|---|---|",
    ]
    for c in classes:
        p, f_, n = tab[(c, True)], tab[(c, False)], tab[(c, None)]
        lines.append(f"| {c} | {p} | {f_} | {n} | {p + f_ + n} |")
    fail = [r for r in recs if r["thesis_pass"] is False]
    lines += ["", f"Thesis rule fails: {len(fail)}. Of these, after repair:"]
    for c in classes[1:]:
        lines.append(f"- {c}: {sum(r['repair_class'] == c for r in fail)}")
    straddle = sum(r["repair_class"] == "clean" for r in fail)
    lines += [
        "", f"'clean' among the thesis fails ({straddle}) = components that straddle the {mm:g} mm line "
        "(each mostly on-organ, but < 90% overall): component-level repair leaves them unchanged.",
        f"Thesis passes changed by repair (small off-organ pieces deleted): "
        f"{sum(r['thesis_pass'] is True and r['repair_class'] == 'repaired' for r in recs)}.",
    ]
    off = [c for r in recs for c in r["components"] if not c["on_organ"]]
    if off:
        ml = np.array([c["ml"] for c in off])
        d = np.array([c["median_dist_mm"] for c in off])
        lines += ["", f"Off-organ components: {len(off)}, volume median {np.median(ml):.2f} ml (max {ml.max():.1f}), "
                  f"median distance to the organ {np.median(d):.0f} mm (IQR {np.percentile(d, 25):.0f}-{np.percentile(d, 75):.0f})."]
    chk = [(r["thesis_frac15"], r["result_json_frac15"]) for r in recs
           if r.get("result_json_frac15") is not None and r["thesis_frac15"] is not None]
    if chk:
        diff = max(abs(a - b) for a, b in chk)
        lines += ["", f"Cross-check vs result.json tumor_frac_within_15mm: {len(chk)} volumes, max |diff| {diff:.4f}."]
    (Path(out_dir) / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="liver", choices=["liver", "pancreas"])
    ap.add_argument("--pool", default=f"{os.environ.get('MAISI_RUNS', os.environ.get('PROJ', '/work/Research') + '/maisi_runs')}/pool")
    ap.add_argument("--out", help="default <pool>/filter_metrics/<target>")
    ap.add_argument("--max-mm", type=float, default=THESIS_MM)
    ap.add_argument("--keep-frac", type=float, default=0.5, help="on-organ if this share of a component is within --max-mm")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N volumes (testing)")
    a = ap.parse_args()

    out = a.out or f"{a.pool}/filter_metrics/{a.target}"
    files = sorted(glob.glob(f"{a.pool}/pairs/{a.target}/labelsTr/{a.target}_*.nii.gz"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        raise SystemExit(f"no labels in {a.pool}/pairs/{a.target}/labelsTr")
    stale = [p for p in Path(out, "per_volume").glob("*.json")
             if (lambda r: (r["max_mm"], r["keep_frac"]) != (a.max_mm, a.keep_frac))(json.loads(p.read_text()))]
    if stale:
        raise SystemExit(f"{len(stale)} cached volumes in {out} used other --max-mm/--keep-frac; use another --out")
    print(f"{len(files)} labels -> {out} ({a.workers} workers)", flush=True)
    jobs = [(f, out, a.pool, a.max_mm, a.keep_frac) for f in files]
    with Pool(a.workers) as pool:
        for i, (vid, cls) in enumerate(pool.imap_unordered(measure, jobs), 1):
            if cls != "skip" or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {vid}: {cls}", flush=True)
    write_tables(out, a.target)


if __name__ == "__main__":
    main()
