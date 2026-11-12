#!/usr/bin/env python
"""Filter metrics for the MAISI pool (Option B): per tumour component pass / fail for the reject and repair arms. CPU.

Rules: PLAN_training.md "Filter rules" (fixed 2026-11-12, before looking at the pool metrics). Unit = tumour component
(26-connected tumour voxels). A criterion decides pass / fail per component, and the same decision drives both arms:
    reject C   keep the volumes in which every component passes (tumour-free volumes pass) -> ids_reject<C>.json
    repair C   keep every volume, relabel its failing components (nnunet/build_repair_dataset.py --criterion C)
Criteria (a component fails if it is unlike every real lesion):
    touch   fails if none of its voxels is 26-adjacent to an organ voxel.                   repair -> background
    auc     fails if its separability sep < cut = the minimum sep over the real val lesions.  repair -> organ if >= 50%
            of its outer 26-neighbour shell is organ, else background. Unscored components pass.
Separability: core = the component eroded by one voxel (MAISI label edges are 1.5 mm steps, so edge voxels are
unreliable), shell = voxels 2..10 mm outside it, not tumour, restricted to the organ when the component sits in the
organ (contact >= 50%, the same "surroundings" as the auc repair) or the organ is >= 25% of the shell; else any tissue. AUC = P(core HU > shell HU) (+ half the ties), exact on integer HU; sep = max(AUC, 1 - AUC)
in [0.5, 1] (hypo- and hyperdense alike; 0.5 = the label is invisible in the image). Components with < 10 measurable
voxels get no score.
Descriptive only (no decisions): contact = organ share of the outer 26-neighbour shell, min / median distance to the
organ (exact Euclidean mm), near misses (fails touch but within 3 mm), the thesis volume rule (>= 90% of all tumour
voxels within 15 mm; ids_thesis15mm.json, cross-checked against raw/<id>/result.json), MAISI QC.

Real reference: every lesion of the real validation cases (Dataset101 labelsTr/lits_*, the 20 val cases; the test
cases live in labelsTs_real and are never read), measured by the same code. Lesions listed in
real_reference_exclusions.json (next to this script, each with a reason; sanity check 2 of the plan) are left out of
the cut.

Outputs in --out (default <pool>/filter_metrics/<target>):
    per_volume/<id>.json, real_reference/<case>.json   caches (no decisions in them, so the cut can change freely;
                                                       resumable: existing files of the same VERSION are skipped)
    metrics.csv  components.csv  real_lesions.csv      tables incl. the decisions
    ids_rejectTouch.json  ids_rejectAUC.json  ids_thesis15mm.json   passing ids (make_splits --ids)
    rules.json                                         cut + version, read by build_repair_dataset.py
    summary.md                                         real reference, classes per criterion, cross-tabs, sanity checks

Memory ~2.5 GB per worker on a 512^3 pair. Run detached (CLAUDE.md):
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
NEAR_MM = 3.0
STRUCT26 = np.ones((3, 3, 3), bool)
AUC_GAP_MM, AUC_SHELL_MM, AUC_MIN_VOX, AUC_ORGAN_SHARE = 2.0, 10.0, 10, 0.25
SURROUND_SHARE = 0.5  # contact >= this: the component's surroundings are the organ (auc shell + auc repair label)
VERSION = (f"cc3 touch=adj26 auc: core=erode1 shell={AUC_GAP_MM:g}-{AUC_SHELL_MM:g}mm organ if contact>={SURROUND_SHARE:g} "
           f"or organ>={AUC_ORGAN_SHARE:g} min={AUC_MIN_VOX}")
NOIMG = " noimage"
CRITERIA = {"touch": "Touch", "auc": "AUC"}  # criterion -> arm suffix (rejectTouch, repairAUC, ...)
EXCLUSIONS = Path(__file__).resolve().parent / "real_reference_exclusions.json"
HU_LO, HU_HI = -1024, 3071
MIN_REJECT_KEEP, MIN_REPAIR_CHANGE = 100, 20  # sanity check 4


# --- measurements ------------------------------------------------------------------------------------------------
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


def auc_int(a, b):
    """P(a > b) + 0.5 P(a == b) for HU samples rounded to integers: the exact Mann-Whitney AUC with ties, O(n)."""
    n = HU_HI - HU_LO + 1
    ha = np.bincount(np.clip(np.rint(a), HU_LO, HU_HI).astype(np.int64) - HU_LO, minlength=n)
    hb = np.bincount(np.clip(np.rint(b), HU_LO, HU_HI).astype(np.int64) - HU_LO, minlength=n)
    below = np.cumsum(hb) - hb
    return float((ha * (below + 0.5 * hb)).sum() / (ha.sum() * hb.sum()))


def separability(cc, im, organ, tumor, sp, contact):
    """Family-2 scores of one component (cc = its mask in a crop; im/organ/tumor = the same crop), or None."""
    core = ndimage.binary_erosion(cc, border_value=0)
    eroded = core.sum() >= AUC_MIN_VOX
    if not eroded:
        core = cc
    d = ndimage.distance_transform_edt(~cc, sampling=sp)
    shell = (d > AUC_GAP_MM) & (d <= AUC_SHELL_MM) & ~tumor
    on_organ = shell & organ
    in_organ = contact >= SURROUND_SHARE or on_organ.sum() >= AUC_ORGAN_SHARE * shell.sum()
    kind = "organ" if in_organ and on_organ.sum() >= AUC_MIN_VOX else "any"
    if kind == "organ":
        shell = on_organ
    if core.sum() < AUC_MIN_VOX or shell.sum() < AUC_MIN_VOX:
        return None
    t, s = im[core], im[shell]
    auc = auc_int(t, s)
    return {"auc": round(auc, 4), "sep": round(max(auc, 1 - auc), 4), "shell": kind, "core_eroded": bool(eroded),
            "core_vox": int(t.size), "shell_vox": int(s.size),
            "core_hu": round(float(np.median(t)), 1), "shell_hu": round(float(np.median(s)), 1)}


def component_rows(label, spacing, img=None):
    """One row per tumour component (numbered as ndimage.label numbers them) + the tumour voxels' organ distances.

    Rows hold measurements only, no decisions. With `img`, the separability fields are added (None if unscored)."""
    organ, tumor = label == ORGAN, label == TUMOR
    comp, n = ndimage.label(tumor, structure=STRUCT26)
    if n == 0:
        return [], np.zeros(0)
    pts = np.argwhere(comp > 0)
    ids = comp[tuple(pts.T)]
    dist = distance_to_organ_mm(organ, pts, spacing)
    order = np.argsort(ids, kind="stable")
    bounds = np.searchsorted(ids[order], np.arange(1, n + 2))
    sp = np.asarray(spacing, float)
    pad = np.ceil((AUC_SHELL_MM + 1) / sp).astype(int) if img is not None else np.ones(3, int)
    voxel_ml = float(np.prod(sp)) / 1000.0
    rows = []
    for c, box in enumerate(ndimage.find_objects(comp), 1):
        sl = tuple(slice(max(s.start - p, 0), min(s.stop + p, dim)) for s, p, dim in zip(box, pad, comp.shape))
        cc = comp[sl] == c
        ring = ndimage.binary_dilation(cc, STRUCT26) & ~cc
        org = organ[sl]
        d = dist[order[bounds[c - 1]: bounds[c]]]
        row = {
            "component": c,
            "voxels": int(d.size),
            "ml": round(d.size * voxel_ml, 4),
            "centroid": [int(round(v)) + int(s.start) for v, s in zip(ndimage.center_of_mass(cc), sl)],
            "touches": bool(org[ring].any()),
            "contact": round(float(org[ring].mean()), 4) if ring.any() else 0.0,
            "min_dist_mm": round(float(d.min()), 2),
            "median_dist_mm": round(float(np.median(d)), 2),
            f"frac_within_{THESIS_MM:g}mm": round(float((d <= THESIS_MM).mean()), 4),
        }
        if img is not None:
            row.update(separability(cc, img[sl], org, tumor[sl], sp, row["contact"]) or {"auc": None, "sep": None})
        rows.append(row)
    return rows, dist


# --- decisions (shared with nnunet/build_repair_dataset.py) -------------------------------------------------------
def fails(c, criterion, cut=None):
    """Does tumour component row `c` fail `criterion`? (PLAN_training.md "Filter rules")"""
    if criterion == "touch":
        return not c["touches"]
    if criterion == "auc":
        if cut is None:
            raise ValueError("criterion auc needs the cut (rules.json)")
        return c.get("sep") is not None and c["sep"] < cut
    raise ValueError(criterion)


def repair_value(c, criterion):
    """New label of a failing component: background for touch; the surroundings' label for auc."""
    if criterion == "touch":
        return 0
    return ORGAN if c["contact"] >= SURROUND_SHARE else 0


def volume_class(rows, criterion, cut=None):
    """no_tumour | clean (nothing fails: passes reject) | repaired (some fail) | tumour_free (all fail)."""
    if not rows:
        return "no_tumour"
    nf = sum(fails(c, criterion, cut) for c in rows)
    return "clean" if nf == 0 else "tumour_free" if nf == len(rows) else "repaired"


def apply_repair(label, rows, criterion, cut=None):
    """Repaired copy of `label`: failing components relabelled. `rows` = the cached rows of this volume; the
    components are recomputed and checked against them (count and voxels per component), so the repair acts on exactly
    the components filter_metrics judged. Raises ValueError on a mismatch."""
    comp, n = ndimage.label(label == TUMOR, structure=STRUCT26)
    counts = np.bincount(comp.ravel(), minlength=n + 1)[1:]
    if n != len(rows) or any(int(k) != r["voxels"] or r["component"] != i
                             for i, (k, r) in enumerate(zip(counts, rows), 1)):
        raise ValueError(f"components differ from the metrics cache ({n} vs {len(rows)})")
    lut = np.full(n + 1, 255, np.uint8)
    for r in rows:
        if fails(r, criterion, cut):
            lut[r["component"]] = repair_value(r, criterion)
    if (lut == 255).all():
        return label.copy()
    new = lut[comp]
    return np.where(new == 255, label, new).astype(label.dtype)


# --- per-volume jobs ----------------------------------------------------------------------------------------------
def load_image(path, shape):
    img = nib.load(path).get_fdata(dtype=np.float32, caching="unchanged")
    assert img.shape == shape, (path, img.shape, shape)
    return img


def image_of(label_path):
    return label_path.replace("/labelsTr/", "/imagesTr/")[: -len(".nii.gz")] + "_0000.nii.gz"


def cache_ok(cache, with_image):
    if not cache.exists():
        return False
    v = json.loads(cache.read_text()).get("version")
    return v == VERSION or (not with_image and v == VERSION + NOIMG)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def measure(args):
    path, out_dir, pool, with_image = args
    vid = os.path.basename(path)[: -len(".nii.gz")]
    cache = Path(out_dir) / "per_volume" / f"{vid}.json"
    if cache_ok(cache, with_image):
        return vid, "skip"
    nii = nib.load(path)
    label = np.asarray(nii.dataobj).astype(np.uint8)
    spacing = spacing_of(nii.affine)
    img = load_image(image_of(path), label.shape) if with_image else None
    rows, dist = component_rows(label, spacing, img)
    del img
    voxel_ml = float(np.prod(spacing)) / 1000.0
    thesis = float((dist <= THESIS_MM).mean()) if dist.size else None
    rec = {
        "id": vid,
        "version": VERSION + ("" if with_image else NOIMG),
        "organ_ml": round(float((label == ORGAN).sum()) * voxel_ml, 2),
        "tumor_ml": round(dist.size * voxel_ml, 3),
        "tumor_components": len(rows),
        "thesis_frac15": None if thesis is None else round(thesis, 4),
        "thesis_pass": None if thesis is None else thesis >= THESIS_FRAC,
    }
    res = Path(pool) / "raw" / vid / "result.json"
    if res.exists():
        m = json.loads(res.read_text()).get("metrics") or {}
        rec["result_json_frac15"] = m.get("tumor_frac_within_15mm")
        rec["maisi_qc_pass"] = m.get("maisi_qc_recomputed_pass")
    rec["components"] = rows
    write_json(cache, rec)
    return vid, f"{len(rows)} components, {sum(not r['touches'] for r in rows)} not touching"


def measure_real(args):
    """The same component rows for one real validation case (the reference)."""
    label_path, out_dir, with_image = args
    case = os.path.basename(label_path)[: -len(".nii.gz")]
    cache = Path(out_dir) / "real_reference" / f"{case}.json"
    if cache_ok(cache, with_image):
        return case, "skip"
    nii = nib.load(label_path)
    label = np.asarray(nii.dataobj).astype(np.uint8)
    spacing = spacing_of(nii.affine)
    img = load_image(image_of(label_path), label.shape) if with_image else None
    rows, _ = component_rows(label, spacing, img)
    write_json(cache, {"case": case, "label_path": label_path, "spacing": spacing,
                       "version": VERSION + ("" if with_image else NOIMG), "lesions": rows})
    return case, f"{len(rows)} lesions"


# --- tables + summary ---------------------------------------------------------------------------------------------
def union_keys(dicts, skip=()):
    keys = []
    for d in dicts:
        keys += [k for k in d if k not in keys and k not in skip]
    return keys


def write_csv(path, rows, skip=()):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, union_keys(rows, skip), extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def load_exclusions(path, target):
    if not path or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text()).get(target, [])


def q5(v):
    return " / ".join(f"{np.percentile(v, p):.3f}" for p in (5, 25, 50, 75, 95)) if len(v) else "-"


def write_tables(out_dir, target, exclusions_path):
    out = Path(out_dir)
    recs = [json.loads(p.read_text()) for p in sorted((out / "per_volume").glob("*.json"))]
    real = [json.loads(p.read_text()) for p in sorted((out / "real_reference").glob("*.json"))]
    if not recs:
        return
    with_image = all(r["version"] == VERSION for r in recs) and real and all(c["version"] == VERSION for c in real)
    excl = load_exclusions(exclusions_path, target)
    excl_keys = {(e["case"], e["component"]) for e in excl}

    # real reference -> cut
    real_rows = [{"case": c["case"], **l, "excluded": (c["case"], l["component"]) in excl_keys}
                 for c in real for l in c["lesions"]]
    scored = [l for l in real_rows if l.get("sep") is not None and not l["excluded"]]
    cut = cut_from = None
    if with_image and len(scored) >= 5:
        low = min(scored, key=lambda l: l["sep"])
        cut, cut_from = low["sep"], {k: low[k] for k in ("case", "component", "ml", "sep", "core_hu", "shell_hu")}
    crits = ["touch"] + (["auc"] if cut is not None else [])

    # decisions
    for r in recs:
        for k in crits:
            nf = [c for c in r["components"] if fails(c, k, cut)]
            r[f"fail_{k}"] = len(nf)
            r[f"class_{k}"] = volume_class(r["components"], k, cut)
            r[f"pass_reject{CRITERIA[k]}"] = not nf
        r["near_miss"] = sum(not c["touches"] and c["min_dist_mm"] <= NEAR_MM for c in r["components"])
    comp_rows = []
    for r in recs:
        for c in r["components"]:
            row = {"id": r["id"], **c}
            for k in crits:
                row[f"fail_{k}"] = fails(c, k, cut)
                row[f"repair_to_{k}"] = repair_value(c, k) if row[f"fail_{k}"] else None
            comp_rows.append(row)
    write_csv(out / "metrics.csv", recs, skip=("components",))
    write_csv(out / "components.csv", comp_rows)
    write_csv(out / "real_lesions.csv", real_rows)

    def ids_file(name, rule, passing, extra=None):
        (out / name).write_text(json.dumps({"rule": rule, **(extra or {}), "n_pass": len(passing),
                                            "n_total": len(recs), "ids": sorted(passing)}, indent=1))

    ids_file("ids_thesis15mm.json", f">= {THESIS_FRAC:.0%} of tumour voxels within {THESIS_MM:g} mm of the organ "
             "(thesis; descriptive, no arm)", [r["id"] for r in recs if r["thesis_pass"]])
    ids_file("ids_rejectTouch.json", "every tumour component touches the organ (26-adjacent); tumour-free volumes pass",
             [r["id"] for r in recs if r["pass_rejectTouch"]], {"version": VERSION})
    if cut is not None:
        ids_file("ids_rejectAUC.json", f"every scored tumour component has sep >= {cut} (minimum of the real val "
                 "lesions); unscored components and tumour-free volumes pass",
                 [r["id"] for r in recs if r["pass_rejectAUC"]], {"version": VERSION, "cut": cut})
    else:
        (out / "ids_rejectAUC.json").unlink(missing_ok=True)
    write_json(out / "rules.json", {
        "version": VERSION, "target": target, "auc_cut": cut, "auc_cut_from": cut_from,
        "real_cases": len(real), "real_lesions": len(real_rows), "real_lesions_scored": len(scored),
        "excluded": excl, "surround_share": SURROUND_SHARE, "criteria": crits})

    lines = summary_lines(target, recs, real, real_rows, scored, cut, cut_from, crits, excl)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def summary_lines(target, recs, real, real_rows, scored, cut, cut_from, crits, excl):
    n = len(recs)
    L = [f"# Filter metrics: {target} ({n} synthetic volumes, {len(real)} real val cases)", "",
         "Rules: PLAN_training.md \"Filter rules\". A component fails touch if no voxel is 26-adjacent to the organ; "
         "it fails auc if its sep < the minimum sep of the real val lesions. reject = keep volumes without a failing "
         "component; repair = relabel the failing components (touch -> background, auc -> surroundings' label).", "",
         "## Real reference (sanity checks 1 + 2)", ""]
    touch_real = [l for l in real_rows if not l["excluded"]]
    nt = sum(l["touches"] for l in touch_real)
    L.append(f"Real lesions: {len(real_rows)} ({len(excl)} excluded by real_reference_exclusions.json), "
             f"**touching the organ: {nt}/{len(touch_real)}**.")
    for l in [l for l in touch_real if not l["touches"]]:
        L.append(f"- NOT touching: {l['case']} component {l['component']}, {l['ml']} ml, min distance "
                 f"{l['min_dist_mm']} mm -> view it (label error?)")
    if cut is not None:
        L += ["", f"Scored real lesions: {len(scored)} (unscored, < {AUC_MIN_VOX} measurable voxels: "
              f"{sum(l.get('sep') is None for l in real_rows)}). sep p5/p25/median/p75/p95: "
              f"{q5([l['sep'] for l in scored])}.", "", "Least visible real lesions (view them; a label error goes "
              "into real_reference_exclusions.json with a reason):", "",
              "| case | component | ml | sep | core HU | shell HU | shell |", "|---|---|---|---|---|---|---|"]
        for l in sorted(scored, key=lambda l: l["sep"])[:5]:
            L.append(f"| {l['case']} | {l['component']} | {l['ml']} | {l['sep']} | {l['core_hu']} | {l['shell_hu']} "
                     f"| {l['shell']} |")
        L += ["", f"**AUC cut = {cut}** (from {cut_from['case']} component {cut_from['component']})."]
        for e in excl:
            L.append(f"- excluded: {e['case']} component {e['component']}: {e.get('reason', '(no reason given)')}")
    else:
        L += ["", "No AUC cut (images not measured, or < 5 scored real lesions): no auc decisions, no ids_rejectAUC.json."]

    L += ["", "## Decisions per criterion (sanity checks 3 + 4)", "",
          "| criterion | no tumour | clean (reject keeps) | repaired | tumour-free after repair | reject keeps | "
          "repair changes |", "|---|---|---|---|---|---|---|"]
    checks = []
    for k in crits:
        cls = Counter(r[f"class_{k}"] for r in recs)
        keep = sum(r[f"pass_reject{CRITERIA[k]}"] for r in recs)
        change = cls["repaired"] + cls["tumour_free"]
        L.append(f"| {k} | {cls['no_tumour']} | {cls['clean']} | {cls['repaired']} | {cls['tumour_free']} | "
                 f"{keep}/{n} | {change}/{n} |")
        checks += [f"reject{CRITERIA[k]} keeps {keep} ({'ok' if keep >= MIN_REJECT_KEEP else 'CHECK: < %d' % MIN_REJECT_KEEP})",
                   f"repair{CRITERIA[k]} changes {change} ({'ok' if change >= MIN_REPAIR_CHANGE else 'CHECK: < %d' % MIN_REPAIR_CHANGE})"]
    comps = [c for r in recs for c in r["components"]]
    for k in crits:
        f = [c for c in comps if fails(c, k, cut)]
        ml = np.array([c["ml"] for c in f]) if f else np.zeros(0)
        to = Counter(repair_value(c, k) for c in f)
        L.append(f"\n{k}: {len(f)}/{len(comps)} components fail, {ml.sum():.1f} ml"
                 + (f" (median {np.median(ml):.2f} ml, max {ml.max():.1f})" if len(ml) else "")
                 + f"; repair relabels {to[0]} to background, {to[ORGAN]} to organ.")
    near = [c for c in comps if not c["touches"] and c["min_dist_mm"] <= NEAR_MM]
    L += ["", f"Near misses (fail touch but within {NEAR_MM:g} mm of the organ): {len(near)} components in "
          f"{sum(r['near_miss'] > 0 for r in recs)} volumes, {sum(c['ml'] for c in near):.1f} ml. Many of them would "
          "mean the no-tolerance assumption is wrong (view them)."]
    off = [c for c in comps if not c["touches"]]
    if off:
        d = np.array([c["median_dist_mm"] for c in off])
        L.append(f"Non-touching components: median distance to the organ {np.median(d):.0f} mm "
                 f"(IQR {np.percentile(d, 25):.0f}-{np.percentile(d, 75):.0f}).")

    if "auc" in crits:
        L += ["", "Volumes failing touch x failing auc:", "", "| | auc ok | auc fails |", "|---|---|---|"]
        for name, t in (("touch ok", True), ("touch fails", False)):
            sub = [r for r in recs if r["pass_rejectTouch"] is t]
            L.append(f"| {name} | {sum(r['pass_rejectAUC'] for r in sub)} | {sum(not r['pass_rejectAUC'] for r in sub)} |")
        syn = [c["sep"] for c in comps if c.get("sep") is not None]
        L += ["", "| sep | n | p5 / p25 / median / p75 / p95 |", "|---|---|---|",
              f"| real val lesions | {len(scored)} | {q5([l['sep'] for l in scored])} |",
              f"| synthetic components | {len(syn)} | {q5(syn)} |",
              f"| synthetic touching | {sum(c['touches'] for c in comps if c.get('sep') is not None)} | "
              f"{q5([c['sep'] for c in comps if c.get('sep') is not None and c['touches']])} |",
              "", f"Unscored synthetic components (pass auc): {sum(c.get('sep') is None for c in comps)}."]

    L += ["", "## Descriptive", "",
          f"Contact share (organ part of the outer shell), touching components: real {q5([l['contact'] for l in touch_real if l['touches']])}, "
          f"synthetic {q5([c['contact'] for c in comps if c['touches']])} (p5/p25/median/p75/p95).", "",
          "Thesis rule (volume-level, no arm) vs rejectTouch:", "", "| | rejectTouch keeps | rejectTouch drops |",
          "|---|---|---|"]
    for name, t in (("thesis pass", True), ("thesis fail", False), ("no tumour", None)):
        sub = [r for r in recs if r["thesis_pass"] is t]
        L.append(f"| {name} | {sum(r['pass_rejectTouch'] for r in sub)} | {sum(not r['pass_rejectTouch'] for r in sub)} |")
    chk = [(r["thesis_frac15"], r["result_json_frac15"]) for r in recs
           if r.get("result_json_frac15") is not None and r["thesis_frac15"] is not None]
    if chk:
        L.append(f"\nCross-check thesis fraction vs result.json: {len(chk)} volumes, max |diff| "
                 f"{max(abs(a - b) for a, b in chk):.4f}.")
    qc = Counter(r.get("maisi_qc_pass") for r in recs)
    L += [f"MAISI QC pass {qc[True]}, fail {qc[False]}, unknown {qc[None]}.", "", "## Sanity check 4", ""]
    L += [f"- {c}" for c in checks]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="liver", choices=["liver", "pancreas"])
    ap.add_argument("--pool", default=f"{os.environ.get('MAISI_RUNS', os.environ.get('PROJ', '/work/Research') + '/maisi_runs')}/pool")
    ap.add_argument("--out", help="default <pool>/filter_metrics/<target>")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N volumes (testing)")
    ap.add_argument("--no-image", action="store_true", help="labels only (touch), no separability")
    ap.add_argument("--real-dir", help="nnU-Net raw dataset with the real val cases in imagesTr/labelsTr "
                    "(default $nnUNet_raw/Dataset101_LiverSynth for liver)")
    ap.add_argument("--exclusions", default=str(EXCLUSIONS), help="real lesions left out of the reference")
    ap.add_argument("--tables-only", action="store_true", help="only rebuild tables/ids/rules from the caches")
    a = ap.parse_args()

    out = a.out or f"{a.pool}/filter_metrics/{a.target}"
    if not a.tables_only:
        with_image = not a.no_image
        files = sorted(glob.glob(f"{a.pool}/pairs/{a.target}/labelsTr/{a.target}_*.nii.gz"))
        if a.limit:
            files = files[: a.limit]
        if not files:
            raise SystemExit(f"no labels in {a.pool}/pairs/{a.target}/labelsTr")
        if a.target != "liver" and not a.real_dir:
            raise SystemExit("--real-dir is required for pancreas")
        raw = os.environ.get("nnUNet_raw", f"{os.environ.get('PROJ', '/work/Research')}/nnunet_data/raw")
        real_dir = a.real_dir or f"{raw}/Dataset101_LiverSynth"
        # labelsTr holds only the real *validation* cases; the test cases live in labelsTs_real and are never read
        real_jobs = [(f, out, with_image) for f in sorted(glob.glob(f"{real_dir}/labelsTr/lits_*.nii.gz"))]
        if not real_jobs:
            raise SystemExit(f"no real val labels in {real_dir}/labelsTr")
        print(f"{len(files)} labels -> {out} ({a.workers} workers, separability {'on' if with_image else 'off'}, "
              f"{len(real_jobs)} real val cases)", flush=True)
        jobs = [(f, out, a.pool, with_image) for f in files]
        with Pool(a.workers) as pool:
            for case, msg in pool.imap_unordered(measure_real, real_jobs):
                print(f"[real] {case}: {msg}", flush=True)
            for i, (vid, msg) in enumerate(pool.imap_unordered(measure, jobs), 1):
                if msg != "skip" or i == len(jobs):
                    print(f"[{i}/{len(jobs)}] {vid}: {msg}", flush=True)
    write_tables(out, a.target, a.exclusions)


if __name__ == "__main__":
    main()
