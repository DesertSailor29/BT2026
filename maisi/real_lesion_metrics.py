#!/usr/bin/env python
"""The filter metrics of filter_metrics.py (per tumour component: touch, contact, distance, separability) on ALL real
LiTS cases (20 val + 111 test), measured by the same code (component_rows). DESCRIPTIVE ONLY.
--target pancreas: the same on all 281 labelled MSD Task07 cases (val + test of maisi/msd07_split.json), output
$PROJ/raw_data/msd07_lesion_metrics; candidate cuts include the val cut from --metrics' rules.json.

Each case also gets what the cc rule (cc_rules.py) would do to its real label ("cc" in the cache): tumour components
outside the main structure, organ voxels in detached pieces (count, largest piece + its centroid). Real labels are
the reference, so anything cc removes there is a cost of the rule, not a repair.

The filter rules and the AUC cut come from the 20 val cases only (filter_metrics.py, plans/PLAN_training.md). This script
never writes into the filter_metrics dir and nothing reads its output for a decision: the 111 test cases describe
how real lesions look (for the paper and as a robustness check of the val-derived cut), they do not set it.

Output (--out, default $PROJ/raw_data/lits_lesion_metrics):
    cases/<case>.json   cache per case (same row format as filter_metrics' real_reference), resumable
    lesions.csv         one row per lesion, with split (val / test) and thesis_test
    summary.md          touch rate, sep percentiles per split, per-case minima, counts below candidate cuts, and the
                        synthetic pool's components against the same cuts (if --metrics is given)

  python maisi/real_lesion_metrics.py --workers 12 --metrics $MAISI_RUNS/pool/filter_metrics/liver
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_metrics import ORGAN, STRUCT26, VERSION, component_rows, load_image, write_json  # noqa: E402
from orient import spacing_of  # noqa: E402

PROJ = os.environ.get("PROJ", "/work/Research")
CUTS = (0.504, 0.547, 0.56, 0.60, 0.637, 0.65, 0.70)
CUTS_PANCREAS = (0.55, 0.60, 0.65, 0.70)  # + the val cut of rules.json


def cc_real(label, spacing):
    """What cc_rules' cc would change in a real label (descriptive)."""
    from scipy import ndimage

    from cc_rules import measure_cc
    comp, keep, m = measure_cc(label)
    vml = float(np.prod(spacing)) / 1000
    det = (label == ORGAN) & ~keep
    out = {"tumour_outside": sum(not v for v in m["in_main"]),
           "outside_components": [i for i, v in enumerate(m["in_main"], 1) if not v],  # = component_rows numbering
           "tumour_outside_ml": round(sum(k for k, v in zip(m["voxels"], m["in_main"]) if not v) * vml, 4),
           "organ_removed_ml": round(m["liver_removed_vox"] * vml, 4), "pieces": 0}
    if det.any():
        pieces, n = ndimage.label(det, STRUCT26)
        sizes = np.bincount(pieces.ravel(), minlength=n + 1)[1:]
        big = int(np.argmax(sizes)) + 1
        out.update(pieces=int(n), largest_piece_ml=round(float(sizes[big - 1]) * vml, 4),
                   largest_piece_centroid=[int(round(v)) for v in ndimage.center_of_mass(pieces == big)])
    return out


def measure(args):
    case, label_path, image_path, out = args
    cache = Path(out) / "cases" / f"{case}.json"
    if cache.exists() and json.loads(cache.read_text()).get("version") == VERSION:
        return case, "skip"
    nii = nib.load(label_path)
    label = np.asarray(nii.dataobj).astype(np.uint8)
    spacing = spacing_of(nii.affine)
    img = load_image(image_path, label.shape)
    rows, _ = component_rows(label, spacing, img)
    del img
    write_json(cache, {"case": case, "label_path": label_path, "image_path": image_path, "spacing": spacing,
                       "version": VERSION,
                       "liver_ml": round(float((label == 1).sum()) * float(np.prod(spacing)) / 1000, 1),
                       "lesions": rows, "cc": cc_real(label, spacing)})
    return case, f"{len(rows)} lesions"


def pct(v, ps=(5, 25, 50, 75, 95)):
    return " / ".join(f"{np.percentile(v, p):.3f}" for p in ps) if len(v) else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="liver", choices=["liver", "pancreas"])
    ap.add_argument("--raw", help="liver: Dataset101 raw dir (default $nnUNet_raw/Dataset101_LiverSynth); "
                    "pancreas: the MSD Task07 root (default $PROJ/raw_data/Task07_Pancreas)")
    ap.add_argument("--out", help="default $PROJ/raw_data/lits_lesion_metrics (pancreas: msd07_lesion_metrics)")
    ap.add_argument("--metrics", help="filter_metrics dir of the pool, to compare the synthetic components")
    ap.add_argument("--workers", type=int, default=8, help="~3-4 GB each on long LiTS volumes")
    a = ap.parse_args()

    jobs = []
    if a.target == "pancreas":
        import msd07_split
        root = a.raw or msd07_split.ROOT
        out = Path(a.out or f"{PROJ}/raw_data/msd07_lesion_metrics")
        split = msd07_split.load_or_make(root)
        for which in ("val", "test"):
            jobs += [(case, lab, img, str(out)) for case, lab, img in msd07_split.cases(split, which, root)]
        val = {msd07_split.case_name(i) for i in split["val"]}
        thesis_test, organ, n_all = set(), "pancreas", len(jobs)
        cuts = CUTS_PANCREAS
        if a.metrics and (Path(a.metrics) / "rules.json").exists():
            vc = json.loads((Path(a.metrics) / "rules.json").read_text())["auc_cut"]
            cuts = tuple(sorted({*cuts, *([vc] if vc is not None else [])}))
        ref_cut = min(cuts)
    else:
        raw, out = Path(a.raw or f"{os.environ.get('nnUNet_raw', PROJ + '/nnunet_data/raw')}/Dataset101_LiverSynth"), \
            Path(a.out or f"{PROJ}/raw_data/lits_lesion_metrics")
        split = json.loads((raw / "real_split.json").read_text())
        thesis_test = {f"lits_{i:03d}" for i in split["thesis_test"]}
        for sub, ids in (("Tr", split["val"]), ("Ts_real", split["test"])):
            for i in ids:
                case = f"lits_{i:03d}"
                jobs.append((case, str(raw / f"labels{sub}" / f"{case}.nii.gz"),
                             str(raw / f"images{sub}" / f"{case}_0000.nii.gz"), str(out)))
        val = {f"lits_{i:03d}" for i in split["val"]}
        organ, n_all, cuts, ref_cut = "liver", 131, CUTS, 0.547
    print(f"{len(jobs)} cases -> {out} ({a.workers} workers)", flush=True)
    with Pool(a.workers) as pool:
        for i, (case, msg) in enumerate(pool.imap_unordered(measure, sorted(jobs, key=lambda j: -os.path.getsize(j[2])),
                                                              chunksize=1), 1):
            print(f"[{i}/{len(jobs)}] {case}: {msg}", flush=True)

    recs = [json.loads(p.read_text()) for p in sorted((out / "cases").glob("*.json"))]
    rows = []
    for r in recs:
        for l in r["lesions"]:
            rows.append({"case": r["case"], "split": "val" if r["case"] in val else "test",
                         "thesis_test": int(r["case"] in thesis_test), **{k: v for k, v in l.items()}})
    with open(out / "lesions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def sel(split_name):
        return [x for x in rows if split_name == "all" or x["split"] == split_name]

    name = "MSD07" if a.target == "pancreas" else "LiTS"
    L = [f"# Real {name} lesions: filter metrics (descriptive only)", "",
         f"Same code as `filter_metrics.py` (version `{VERSION}`), all {len(recs)} {name} cases: {len(val)} val (the "
         f"filter reference) + {len(recs) - len(val)} test. Nothing here sets a filter rule: the cut comes from the val "
         "cases only.", "",
         "## Touch and separability per split", "",
         f"| split | cases | cases with tumour | lesions | touch {organ} | scored | sep min | sep p5 / p25 / median / p75 / p95 |",
         "|---|---|---|---|---|---|---|---|"]
    for s in ("val", "test", "all"):
        x = sel(s)
        cs = {r["case"] for r in recs if s == "all" or ("val" if r["case"] in val else "test") == s}
        sc = [l["sep"] for l in x if l.get("sep") is not None]
        L.append(f"| {s} | {len(cs)} | {len({l['case'] for l in x})} | {len(x)} | {sum(l['touches'] for l in x)}/{len(x)} "
                 f"| {len(sc)} | {min(sc) if sc else float('nan'):.3f} | {pct(sc)} |")
    nt = [l for l in rows if not l["touches"]]
    L += ["", f"Lesions not touching the {organ}: {len(nt)}"
          + (": " + ", ".join(f"{l['case']} c{l['component']} ({l['ml']} ml, {l['min_dist_mm']} mm)" for l in nt[:20])
             if nt else "."), ""]

    # weighting: sep by lesion size
    L += [f"## Separability by lesion size (all {n_all})", "", f"| size | lesions scored | sep p5 / median | below {ref_cut} |",
          "|---|---|---|---|"]
    for lo, hi, sname in ((0, 0.1, "< 0.1 ml"), (0.1, 1, "0.1-1 ml"), (1, 10, "1-10 ml"), (10, 1e9, ">= 10 ml")):
        sc = [l["sep"] for l in rows if l.get("sep") is not None and lo <= l["ml"] < hi]
        L.append(f"| {sname} | {len(sc)} | {pct(sc, (5, 50))} | {sum(v < ref_cut for v in sc)} |")

    # per-case minima: is the low end one case or many?
    L += ["", "## Least visible lesions: one case or many?", "",
          "| case | split | lesions scored | min sep | median sep | lesions < 0.60 |", "|---|---|---|---|---|---|"]
    per = {}
    for l in rows:
        if l.get("sep") is not None:
            per.setdefault(l["case"], []).append(l["sep"])
    for case, v in sorted(per.items(), key=lambda kv: min(kv[1]))[:15]:
        L.append(f"| {case} | {'val' if case in val else 'test'} | {len(v)} | {min(v):.3f} | {np.median(v):.3f} "
                 f"| {sum(x < 0.6 for x in v)} |")

    # candidate cuts
    syn = None
    if a.metrics:
        syn = [c for p in sorted((Path(a.metrics) / "per_volume").glob("*.json"))
               for c in [dict(cc, id=json.loads(p.read_text())["id"]) for cc in json.loads(p.read_text())["components"]]]
    L += ["", "## Candidate AUC cuts: real lesions below vs synthetic pool", "",
          "Real: share of scored lesions below the cut (a filter at this cut would call them 'invisible'). "
          "Synthetic: components below, and volumes aucReject would keep.", "",
          "| cut | real val below | real test below | real cases with one below | synthetic comps below | aucReject keeps |",
          "|---|---|---|---|---|---|"]
    for cut in cuts:
        rv = [l for l in sel("val") if l.get("sep") is not None]
        rt = [l for l in sel("test") if l.get("sep") is not None]
        cases = {l["case"] for l in rows if l.get("sep") is not None and l["sep"] < cut}
        sy = "-"
        if syn is not None:
            bad = [c for c in syn if c.get("sep") is not None and c["sep"] < cut]
            n_vol = len({json.loads(p.read_text())["id"] for p in (Path(a.metrics) / "per_volume").glob("*.json")})
            sy = f"{len(bad)} | {n_vol - len({c['id'] for c in bad})}"
        L.append(f"| {cut} | {sum(l['sep'] < cut for l in rv)}/{len(rv)} | {sum(l['sep'] < cut for l in rt)}/{len(rt)} "
                 f"| {len(cases)}/{n_all} | {sy} |")

    # what cc would change in real labels (a cost of the rule: real labels are the reference)
    cc = [r for r in recs if "cc" in r]
    if cc:
        L += ["", f"## What cc would change in real {organ} labels (real labels are the reference: a cost, not a repair)", "",
              f"| split | cases | tumour comps outside the main structure (cases) | cases with detached {organ} pieces | "
              f"detached {organ} ml: max / sum |", "|---|---|---|---|---|"]
        for s in ("val", "test", "all"):
            x = [r for r in cc if s == "all" or ("val" if r["case"] in val else "test") == s]
            ml = [r["cc"]["organ_removed_ml"] for r in x]
            L.append(f"| {s} | {len(x)} | {sum(r['cc']['tumour_outside'] for r in x)} "
                     f"({sum(r['cc']['tumour_outside'] > 0 for r in x)}) | {sum(r['cc']['pieces'] > 0 for r in x)} | "
                     f"{max(ml) if ml else 0:.3f} / {sum(ml):.3f} |")
        worst = sorted(cc, key=lambda r: -(r["cc"]["organ_removed_ml"] + r["cc"]["tumour_outside_ml"]))[:10]
        L += ["", "Largest changes:", "", f"| case | split | {organ} removed ml (pieces) | tumour outside: comps / ml |",
              "|---|---|---|---|"]
        for r in worst:
            if r["cc"]["organ_removed_ml"] + r["cc"]["tumour_outside_ml"] > 0:
                L.append(f"| {r['case']} | {'val' if r['case'] in val else 'test'} | {r['cc']['organ_removed_ml']} "
                         f"({r['cc']['pieces']}) | {r['cc']['tumour_outside']} / {r['cc']['tumour_outside_ml']} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "summary.md", flush=True)


if __name__ == "__main__":
    main()
