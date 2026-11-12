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

Family 2, label-image consistency (needs the images; skip with --no-image): for every tumour component the
**separability AUC** of the image HU between the tumour and its surroundings: core = the component eroded by one voxel
(the label edge is a 1.5 mm step in MAISI masks, so edge voxels are unreliable), shell = voxels 2..10 mm outside the
component, not tumour, restricted to the organ when the organ makes up >= 25% of the shell (else any tissue).
AUC = P(core HU > shell HU) (+ half the ties), exact on integer HU; sep = max(AUC, 1 - AUC) in [0.5, 1], so hypo- and
hyperdense lesions count alike and 0.5 = the label is invisible in the image. Components with < 10 measurable voxels
get no score. Volume score = voxel-weighted mean of the component seps (volumes without a scored component: no score).
The same score per lesion on the **real validation cases** (Dataset101 labelsTr/lits_*, the 20 real val cases; the
test cases are never read) gives the reference distribution; cut = its --auc-pct (5th) percentile. A volume passes the
AUC filter if its score >= cut or it has no score (nothing to judge) -> ids_auc.json.

Inputs: <pool>/pairs/<target>/labelsTr/<target>_*.nii.gz (RAS, organ 1, tumour 2; postprocess.py) and imagesTr.
Outputs in --out (default <pool>/filter_metrics/<target>):
    per_volume/<id>.json   per-volume cache incl. the component list (resumable: existing files are skipped,
                           volumes cached without family 2 get it added)
    real_reference/<case>.json   per-lesion scores of the real val cases (cached)
    metrics.csv            one row per volume          components.csv   one row per tumour component
    summary.md             counts + cross-tabs          ids_thesis15mm.json, ids_auc.json   passing ids + scores
                                                        (make_splits --ids)

Memory ~2.5 GB per worker on a 512^3 pair (a 12 GB CPU session: --workers 3). Run detached (CLAUDE.md):
    python filter_metrics.py --target liver --workers 16
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
AUC_GAP_MM, AUC_SHELL_MM, AUC_MIN_VOX, AUC_ORGAN_SHARE = 2.0, 10.0, 10, 0.25
AUC_PARAMS = f"core=erode1 shell={AUC_GAP_MM:g}-{AUC_SHELL_MM:g}mm organ>={AUC_ORGAN_SHARE:g} min={AUC_MIN_VOX}"
HU_LO, HU_HI = -1024, 3071


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


def auc_int(a, b):
    """P(a > b) + 0.5 P(a == b) for HU samples rounded to integers: the exact Mann-Whitney AUC with ties, O(n)."""
    n = HU_HI - HU_LO + 1
    ha = np.bincount(np.clip(np.rint(a), HU_LO, HU_HI).astype(np.int64) - HU_LO, minlength=n)
    hb = np.bincount(np.clip(np.rint(b), HU_LO, HU_HI).astype(np.int64) - HU_LO, minlength=n)
    below = np.cumsum(hb) - hb
    return float((ha * (below + 0.5 * hb)).sum() / (ha.sum() * hb.sum()))


def component_separability(img, label, comp, spacing):
    """{component id: scores or None} for every tumour component (family 2, see the module docstring)."""
    organ, tumor = label == ORGAN, label == TUMOR
    sp = np.asarray(spacing, float)
    pad = np.ceil((AUC_SHELL_MM + 1) / sp).astype(int)
    out = {}
    for c, box in enumerate(ndimage.find_objects(comp), 1):
        if box is None:
            continue
        sl = tuple(slice(max(s.start - p, 0), min(s.stop + p, dim)) for s, p, dim in zip(box, pad, comp.shape))
        cc = comp[sl] == c
        core = ndimage.binary_erosion(cc, border_value=0)
        eroded = core.sum() >= AUC_MIN_VOX
        if not eroded:
            core = cc
        d = ndimage.distance_transform_edt(~cc, sampling=sp)
        shell = (d > AUC_GAP_MM) & (d <= AUC_SHELL_MM) & ~tumor[sl]
        on_organ = shell & organ[sl]
        kind = "organ" if on_organ.sum() >= max(AUC_MIN_VOX, AUC_ORGAN_SHARE * shell.sum()) else "any"
        if kind == "organ":
            shell = on_organ
        if core.sum() < AUC_MIN_VOX or shell.sum() < AUC_MIN_VOX:
            out[c] = None
            continue
        im = img[sl]
        t, s = im[core], im[shell]
        auc = auc_int(t, s)
        out[c] = {"auc": round(auc, 4), "sep": round(max(auc, 1 - auc), 4), "shell": kind, "core_eroded": bool(eroded),
                  "core_vox": int(t.size), "shell_vox": int(s.size),
                  "core_hu": round(float(np.median(t)), 1), "shell_hu": round(float(np.median(s)), 1)}
    return out


def weighted_sep(rows):
    """Voxel-weighted mean sep over the scored components, and the share of tumour voxels that were scored."""
    scored = [r for r in rows if r.get("sep") is not None]
    tot = sum(r["voxels"] for r in rows)
    if not scored:
        return None, 0.0 if tot else None
    w = np.array([r["voxels"] for r in scored], float)
    return round(float((w * [r["sep"] for r in scored]).sum() / w.sum()), 4), round(float(w.sum() / tot), 4)


def load_image(path, shape):
    nii = nib.load(path)
    img = nii.get_fdata(dtype=np.float32, caching="unchanged")
    assert img.shape == shape, (path, img.shape, shape)
    return img


def measure(args):
    path, out_dir, pool, max_mm, keep_frac, with_image = args
    vid = os.path.basename(path)[: -len(".nii.gz")]
    cache = Path(out_dir) / "per_volume" / f"{vid}.json"
    old = json.loads(cache.read_text()) if cache.exists() else None
    if old is not None and (not with_image or old.get("auc_params") == AUC_PARAMS):
        return vid, "skip"
    nii = nib.load(path)
    label = np.asarray(nii.dataobj).astype(np.uint8)
    spacing = spacing_of(nii.affine)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    rows, comp, dist = tumour_components(label, spacing, max_mm, keep_frac)
    if with_image:
        img_path = path.replace("/labelsTr/", "/imagesTr/")[: -len(".nii.gz")] + "_0000.nii.gz"
        sep = component_separability(load_image(img_path, label.shape), label, comp, spacing)
        for r in rows:
            r.update(sep.get(r["component"]) or {"auc": None, "sep": None})
    del comp
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
    }
    if with_image:
        score, share = weighted_sep(rows)
        rec.update({"auc_score": score, "auc_scored_share": share, "auc_params": AUC_PARAMS})
    rec["components"] = rows
    res = Path(pool) / "raw" / vid / "result.json"
    if res.exists():
        m = json.loads(res.read_text()).get("metrics") or {}
        rec["result_json_frac15"] = m.get("tumor_frac_within_15mm")
        rec["maisi_qc_pass"] = m.get("maisi_qc_recomputed_pass")
    write_json(cache, rec)
    return vid, rec["repair_class"] + (f" sep={rec['auc_score']}" if with_image else "")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def measure_real(args):
    """Per-lesion family-2 scores of one real validation case (same code path as the synthetic volumes)."""
    label_path, out_dir = args
    case = os.path.basename(label_path)[: -len(".nii.gz")]
    cache = Path(out_dir) / "real_reference" / f"{case}.json"
    if cache.exists() and json.loads(cache.read_text()).get("auc_params") == AUC_PARAMS:
        return case, "skip"
    nii = nib.load(label_path)
    label = np.asarray(nii.dataobj).astype(np.uint8)
    spacing = spacing_of(nii.affine)
    rows, comp, _ = tumour_components(label, spacing, THESIS_MM, 0.5)
    img_path = label_path.replace("/labelsTr/", "/imagesTr/")[: -len(".nii.gz")] + "_0000.nii.gz"
    sep = component_separability(load_image(img_path, label.shape), label, comp, spacing)
    for r in rows:
        r.update(sep.get(r["component"]) or {"auc": None, "sep": None})
    write_json(cache, {"case": case, "spacing": spacing, "auc_params": AUC_PARAMS, "lesions": rows})
    return case, f"{len(rows)} lesions"


def union_keys(dicts, skip=()):
    keys = []
    for d in dicts:
        keys += [k for k in d if k not in keys and k not in skip]
    return keys


def write_tables(out_dir, target, auc_pct):
    recs = [json.loads(p.read_text()) for p in sorted((Path(out_dir) / "per_volume").glob("*.json"))]
    if not recs:
        return
    vol_keys = union_keys(recs, skip=("components",))
    with open(Path(out_dir) / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, vol_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    comp_rows = [{"id": r["id"], **c} for r in recs for c in r["components"]]
    if comp_rows:
        with open(Path(out_dir) / "components.csv", "w", newline="") as f:
            w = csv.DictWriter(f, union_keys(comp_rows))
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
    lines += auc_section(out_dir, recs, auc_pct)
    (Path(out_dir) / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def auc_section(out_dir, recs, auc_pct):
    """Family 2: real reference, cut, ids_auc.json, and how the AUC filter overlaps the thesis rule and repair."""
    if not all(r.get("auc_params") == AUC_PARAMS for r in recs):
        return ["", "Family 2 (separability AUC): not computed for every volume (--no-image?), no ids_auc.json."]
    real = [json.loads(p.read_text()) for p in sorted((Path(out_dir) / "real_reference").glob("*.json"))]
    lesions = [l for c in real for l in c["lesions"] if l.get("sep") is not None]
    if len(lesions) < 5:
        return ["", f"Family 2: only {len(lesions)} scored real lesions, no cut, no ids_auc.json."]
    real_sep = np.array([l["sep"] for l in lesions])
    cut = float(np.percentile(real_sep, auc_pct))
    passing = {r["id"]: r["auc_score"] for r in recs if r["auc_score"] is None or r["auc_score"] >= cut}
    (Path(out_dir) / "ids_auc.json").write_text(json.dumps({
        "rule": f"volume-weighted separability AUC >= {cut:.4f} = {auc_pct:g}th percentile of {len(lesions)} real "
                f"lesions ({len(real)} real val cases); volumes without a scored tumour pass",
        "cut": cut, "pct": auc_pct, "auc_params": AUC_PARAMS, "n_pass": len(passing), "n_total": len(recs),
        "ids": sorted(passing), "scores": passing}, indent=1))
    syn = np.array([r["auc_score"] for r in recs if r["auc_score"] is not None])
    q = lambda v: " / ".join(f"{np.percentile(v, p):.3f}" for p in (5, 25, 50, 75, 95))  # noqa: E731
    unscored_real = sum(l.get("sep") is None for c in real for l in c["lesions"])
    kinds = Counter(c.get("shell") for r in recs for c in r["components"] if c.get("sep") is not None)
    lines = ["", "## Family 2: label-image consistency (separability AUC)", "",
             f"Params: {AUC_PARAMS}. sep = max(AUC, 1 - AUC); 0.5 = tumour label invisible in the image.", "",
             "| | n | p5 / p25 / median / p75 / p95 |", "|---|---|---|",
             f"| real val lesions ({len(real)} cases) | {len(real_sep)} | {q(real_sep)} |",
             f"| synthetic volumes (weighted) | {len(syn)} | {q(syn) if len(syn) else '-'} |", "",
             f"Real lesions without a score (< {AUC_MIN_VOX} measurable voxels): {unscored_real}. "
             f"Synthetic shells: organ {kinds['organ']}, any tissue {kinds['any']}.",
             f"**Cut = real p{auc_pct:g} = {cut:.4f}. Pass: {len(passing)}/{len(recs)}** "
             f"(incl. {sum(v is None for v in passing.values())} without a scored tumour).", "",
             "| | AUC pass | AUC fail |", "|---|---|---|"]
    for name, f_ in (("thesis pass", lambda r: r["thesis_pass"] is True), ("thesis fail", lambda r: r["thesis_pass"] is False),
                     ("no tumour", lambda r: r["thesis_pass"] is None)):
        sub = [r for r in recs if f_(r)]
        lines.append(f"| {name} | {sum(r['id'] in passing for r in sub)} | {sum(r['id'] not in passing for r in sub)} |")
    lines += ["", "| repair class | AUC pass | AUC fail |", "|---|---|---|"]
    for c in ["no_tumour", "clean", "repaired", "tumour_free"]:
        sub = [r for r in recs if r["repair_class"] == c]
        lines.append(f"| {c} | {sum(r['id'] in passing for r in sub)} | {sum(r['id'] not in passing for r in sub)} |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="liver", choices=["liver", "pancreas"])
    ap.add_argument("--pool", default=f"{os.environ.get('MAISI_RUNS', os.environ.get('PROJ', '/work/Research') + '/maisi_runs')}/pool")
    ap.add_argument("--out", help="default <pool>/filter_metrics/<target>")
    ap.add_argument("--max-mm", type=float, default=THESIS_MM)
    ap.add_argument("--keep-frac", type=float, default=0.5, help="on-organ if this share of a component is within --max-mm")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N volumes (testing)")
    ap.add_argument("--no-image", action="store_true", help="family 1a only (labels), no separability AUC")
    ap.add_argument("--real-dir", help="nnU-Net raw dataset with the real val cases in imagesTr/labelsTr "
                    "(default $nnUNet_raw/Dataset101_LiverSynth for liver)")
    ap.add_argument("--auc-pct", type=float, default=5.0, help="cut = this percentile of the real lesion seps")
    a = ap.parse_args()

    out = a.out or f"{a.pool}/filter_metrics/{a.target}"
    with_image = not a.no_image
    files = sorted(glob.glob(f"{a.pool}/pairs/{a.target}/labelsTr/{a.target}_*.nii.gz"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        raise SystemExit(f"no labels in {a.pool}/pairs/{a.target}/labelsTr")
    stale = [p for p in Path(out, "per_volume").glob("*.json")
             if (lambda r: (r["max_mm"], r["keep_frac"]) != (a.max_mm, a.keep_frac))(json.loads(p.read_text()))]
    if stale:
        raise SystemExit(f"{len(stale)} cached volumes in {out} used other --max-mm/--keep-frac; use another --out")
    real_jobs = []
    if with_image:
        if a.target != "liver" and not a.real_dir:
            raise SystemExit("--real-dir is required for pancreas")
        raw = os.environ.get("nnUNet_raw", f"{os.environ.get('PROJ', '/work/Research')}/nnunet_data/raw")
        real_dir = a.real_dir or f"{raw}/Dataset101_LiverSynth"
        # labelsTr holds only the real *validation* cases; the test cases live in labelsTs_real and are never read
        real_jobs = [(f, out) for f in sorted(glob.glob(f"{real_dir}/labelsTr/lits_*.nii.gz"))]
        if not real_jobs:
            raise SystemExit(f"no real val labels in {real_dir}/labelsTr")
    print(f"{len(files)} labels -> {out} ({a.workers} workers, family 2 {'on' if with_image else 'off'}, "
          f"{len(real_jobs)} real val cases)", flush=True)
    jobs = [(f, out, a.pool, a.max_mm, a.keep_frac, with_image) for f in files]
    with Pool(a.workers) as pool:
        for case, msg in pool.imap_unordered(measure_real, real_jobs):
            print(f"[real] {case}: {msg}", flush=True)
        for i, (vid, cls) in enumerate(pool.imap_unordered(measure, jobs), 1):
            if cls != "skip" or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {vid}: {cls}", flush=True)
    write_tables(out, a.target, a.auc_pct)


if __name__ == "__main__":
    main()
