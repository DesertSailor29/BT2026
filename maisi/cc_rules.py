#!/usr/bin/env python
"""cc and ccAuc rules (Decisions 2026-11-14): "the liver is one organ, a tumour lies in it". CPU.

    cc      main liver = the largest 26-connected liver piece; keep the 26-connected component of liver ∪ tumour that
            contains it (the "main structure"); every other liver and tumour voxel -> background.
    ccAuc   cc, then every tumour component inside the main structure with sep < cut (filter_metrics' separability,
            cut = minimum of the real val lesions, rules.json) -> its surroundings' label (liver if >= 50% of its
            outer 26-neighbour shell is liver, else background; the same as the auc repair). Unscored components pass.

A tumour component is connected, so it lies wholly inside or wholly outside the main structure. The same functions
drive the reject ids (volumes the rule leaves unchanged) and the repair datasets (nnunet/build_repair_dataset.py
--criterion cc / ccAuc), so "reject = the volumes the repair leaves unchanged" holds by construction.

Arms (Decisions 2026-11-14):
    ccReject      Dataset101 fold, ids_ccReject.json      (cc leaves the volume unchanged)
    ccRepair      Dataset103_LiverSynthRepairCC, all 999
    ccRepairDel   Dataset103 fold, ids_ccRepairDel.json   (not left tumour-free by cc)
    ccAucRepair   Dataset104_LiverSynthRepairCCAUC, all 999
    (backlog: ccAucReject = ids_ccAucReject.json)

Input: the Dataset101 synthetic labels (pool labels cropped at MAISI's empty padding: same components, same numbering
as filter_metrics' per_volume caches, which are checked). Cache per volume in <metrics>/cc/<id>.json holds
measurements only (main-structure membership per component, removed liver voxels), so the cut can change freely.

    python cc_rules.py --workers 40          # caches + ids_cc*.json + cc_rules.json + cc_summary.md in <metrics>
    python cc_rules.py --tables-only
"""
import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_metrics import ORGAN, STRUCT26, TUMOR, VERSION, repair_value  # noqa: E402
from filter_metrics import write_json as _wj  # noqa: E402


def write_json(path, obj):
    _wj(Path(path), obj)

CC_VERSION = "cc1 main=largest liver piece (26); keep its 26-component of liver|tumour; rest -> 0"
CRITERIA = ("cc", "ccAuc")
POOL = os.environ.get("MAISI_RUNS", f"{os.environ.get('PROJ', '/work/Research')}/maisi_runs") + "/pool"
FM = f"{POOL}/filter_metrics/liver"


def main_structure(label):
    """Bool mask of the main structure (all False if there is no liver). Works on the foreground box for speed."""
    keep = np.zeros(label.shape, bool)
    liver = label == ORGAN
    if not liver.any():
        return keep
    box = ndimage.find_objects((label > 0).astype(np.uint8))[0]
    lab = label[box]
    lv = lab == ORGAN
    pieces, n = ndimage.label(lv, STRUCT26)
    sizes = np.bincount(pieces.ravel(), minlength=n + 1)
    sizes[0] = 0
    main = int(np.argmax(sizes))  # first of equal sizes, as liver_metrics.py
    union, _ = ndimage.label(lab > 0, STRUCT26)
    mu = union.ravel()[np.flatnonzero(pieces.ravel() == main)[0]]
    keep[box] = union == mu
    return keep


def measure_cc(label, rows=None):
    """Measurements (no decisions): main-structure membership per tumour component + liver voxels outside it.
    `rows` (filter_metrics' cached component rows) are checked against the recomputed components."""
    comp, n = ndimage.label(label == TUMOR, STRUCT26)
    counts = np.bincount(comp.ravel(), minlength=n + 1)[1:]
    if rows is not None and (n != len(rows) or any(int(k) != r["voxels"] or r["component"] != i
                                                   for i, (k, r) in enumerate(zip(counts, rows), 1))):
        raise ValueError(f"components differ from the metrics cache ({n} vs {len(rows)})")
    keep = main_structure(label)
    in_main = [bool(v) for v in ndimage.maximum(keep, comp, range(1, n + 1))] if n else []
    return comp, keep, {"components": n, "voxels": [int(k) for k in counts], "in_main": in_main,
                        "liver_removed_vox": int(((label == ORGAN) & ~keep).sum())}


def decide(m, rows, criterion, cut=None):
    """New label per component: None = unchanged, else 0 / ORGAN. m = measure_cc's dict, rows = filter_metrics rows."""
    out = []
    for i, inm in enumerate(m["in_main"]):
        if not inm:
            out.append(0)
        elif criterion == "ccAuc" and rows[i].get("sep") is not None and rows[i]["sep"] < cut:
            out.append(repair_value(rows[i], "auc"))
        elif criterion in CRITERIA:
            out.append(None)
        else:
            raise ValueError(criterion)
    return out


def classify(m, new):
    """unchanged / changed, tumour_free (had tumour, none left), and the cc change kind."""
    lost = sum(v is not None for v in new)
    kept_t = sum(v is None for v in new)
    liver = m["liver_removed_vox"] > 0
    return {"changed": bool(lost or liver), "tumour_free": bool(m["components"]) and kept_t == 0,
            "kind": "unchanged" if not lost and not liver else "liver+tumour" if lost and liver
            else "tumour only" if lost else "liver only", "relabelled": lost,
            "to_organ": sum(v == ORGAN for v in new)}


def apply(label, rows, criterion, cut=None, cached=None):
    """Repaired copy of `label` + its classification. `cached` (the cc cache of this volume) is checked if given."""
    comp, keep, m = measure_cc(label, rows)
    if cached is not None:
        for k in ("components", "in_main", "liver_removed_vox", "voxels"):
            if cached[k] != m[k]:
                raise ValueError(f"cc cache mismatch in {k}")
    new = decide(m, rows, criterion, cut)
    fixed = np.where((label > 0) & ~keep, 0, label).astype(label.dtype)
    lut = np.array([255] + [255 if v is None else v for v in new], np.uint8)
    if (lut != 255).any():
        nl = lut[comp]
        fixed = np.where(nl == 255, fixed, nl).astype(label.dtype)
    cls = classify(m, new)
    if criterion == "ccAuc" and any(v == 0 for v, inm in zip(new, m["in_main"]) if inm):
        # an in-main tumour relabelled to background may have been the only link to a liver piece: report it
        cls["liver_detached_after_vox"] = int(((fixed == ORGAN) & ~main_structure(fixed)).sum())
    return fixed, m, cls


def measure_one(args):
    lab_path, metrics_json, cache = args
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get("version") == CC_VERSION:
            return c
    rows = json.load(open(metrics_json))["components"]
    lab = np.asarray(nib.load(lab_path).dataobj).astype(np.uint8)
    _, _, m = measure_cc(lab, rows)
    vml = float(np.prod(nib.load(lab_path).header.get_zooms()[:3])) / 1000.0
    rec = {"id": os.path.basename(lab_path)[: -len(".nii.gz")], "version": CC_VERSION, "voxel_ml": vml, **m}
    write_json(cache, rec)
    return rec


def tables(metrics):
    rules = json.load(open(f"{metrics}/rules.json"))
    assert rules["version"] == VERSION, rules["version"]
    cut = rules["auc_cut"]
    assert cut is not None, "rules.json has no AUC cut"
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{metrics}/cc/*.json"))]
    N = len(recs)
    res = {}
    for k in CRITERIA:
        cls = {}
        for r in recs:
            rows = json.load(open(f"{metrics}/per_volume/{r['id']}.json"))["components"]
            if len(rows) != r["components"] or any(x["voxels"] != v for x, v in zip(rows, r["voxels"])):
                raise SystemExit(f"{r['id']}: filter_metrics cache and cc cache disagree; rerun cc_rules.py")
            cls[r["id"]] = classify(r, decide(r, rows, k, cut))
        res[k] = cls

    def ids_file(name, rule, ids):
        write_json(f"{metrics}/{name}", {"rule": rule, "n_pass": len(ids), "n_total": N, "cc_version": CC_VERSION,
                                         "version": VERSION, "cut": cut, "ids": sorted(ids)})

    cc, ca = res["cc"], res["ccAuc"]
    ids_file("ids_ccReject.json", "cc leaves the volume unchanged (no detached liver, no tumour outside the main "
             "structure)", [i for i, c in cc.items() if not c["changed"]])
    ids_file("ids_ccRepairDel.json", "every volume except those left tumour-free by cc", [i for i, c in cc.items()
                                                                                          if not c["tumour_free"]])
    ids_file("ids_ccAucReject.json", f"ccAuc (cut {cut}) leaves the volume unchanged",
             [i for i, c in ca.items() if not c["changed"]])
    ids_file("ids_ccAucRepairDel.json", f"every volume except those left tumour-free by ccAuc (cut {cut})",
             [i for i, c in ca.items() if not c["tumour_free"]])

    counts = {}
    for k, cls in res.items():
        counts[k] = {"unchanged": sum(not c["changed"] for c in cls.values()),
                     "kind": dict(Counter(c["kind"] for c in cls.values())),
                     "tumour_free": sum(c["tumour_free"] for c in cls.values()),
                     "no_tumour": sum(r["components"] == 0 for r in recs),
                     "components_relabelled": sum(c["relabelled"] for c in cls.values()),
                     "to_organ": sum(c["to_organ"] for c in cls.values())}
    auc_extra = [(i, ca[i]["relabelled"] - cc[i]["relabelled"]) for i in ca if ca[i]["relabelled"] > cc[i]["relabelled"]]
    counts["ccAuc_in_liver"] = {"components": sum(n for _, n in auc_extra), "volumes": len(auc_extra)}
    write_json(f"{metrics}/cc_rules.json", {"cc_version": CC_VERSION, "version": VERSION, "auc_cut": cut,
                                            "excluded": rules["excluded"], "volumes": N, "counts": counts})
    L = ["# cc / ccAuc rules: counts", "", f"`maisi/cc_rules.py` ({CC_VERSION}); AUC cut {cut} "
         f"(from {rules['auc_cut_from']['case']} c{rules['auc_cut_from']['component']}); {N} volumes.", "",
         "| rule | unchanged (reject keeps) | liver only | tumour only | liver+tumour | tumour-free after repair | "
         "repair-delete keeps | components relabelled (to liver) |", "|---|---|---|---|---|---|---|---|"]
    for k in CRITERIA:
        c = counts[k]
        L.append(f"| {k} | {c['unchanged']} | {c['kind'].get('liver only', 0)} | {c['kind'].get('tumour only', 0)} | "
                 f"{c['kind'].get('liver+tumour', 0)} | {c['tumour_free']} | {N - c['tumour_free']} | "
                 f"{c['components_relabelled']} ({c['to_organ']}) |")
    L += ["", f"ccAuc on top of cc: {counts['ccAuc_in_liver']['components']} in-liver components relabelled in "
          f"{counts['ccAuc_in_liver']['volumes']} volumes. Volumes with no tumour at all: {counts['cc']['no_tumour']}.",
          "", "For ccAuc, 'tumour only' includes the AUC relabelling. Arms: ccReject = ids_ccReject.json (D101 fold), "
          "ccRepair = D103 all, ccRepairDel = ids_ccRepairDel.json (D103 fold), ccAucRepair = D104 all.", ""]
    open(f"{metrics}/cc_summary.md", "w").write("\n".join(L))
    print("\n".join(L), flush=True)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default=FM)
    ap.add_argument("--labels", default=f"{os.environ.get('nnUNet_raw', '')}/Dataset101_LiverSynth/labelsTr")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    if not a.tables_only:
        labs = sorted(glob.glob(f"{a.labels}/liver_*.nii.gz"))
        assert labs, f"no liver_*.nii.gz in {a.labels}"
        os.makedirs(f"{a.metrics}/cc", exist_ok=True)
        jobs = [(f, f"{a.metrics}/per_volume/{os.path.basename(f)[:-7]}.json",
                 f"{a.metrics}/cc/{os.path.basename(f)[:-7]}.json") for f in labs]
        with Pool(a.workers) as p:
            for i, _ in enumerate(p.imap_unordered(measure_one, jobs, chunksize=2), 1):
                if i % 100 == 0 or i == len(jobs):
                    print(f"{i}/{len(jobs)}", flush=True)
    tables(a.metrics)


if __name__ == "__main__":
    main()
