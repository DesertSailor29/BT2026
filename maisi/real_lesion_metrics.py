#!/usr/bin/env python
"""The filter metrics of filter_metrics.py (per tumour component: touch, contact, distance, separability) on ALL real
LiTS cases (20 val + 111 test), measured by the same code (component_rows). DESCRIPTIVE ONLY.

The filter rules and the AUC cut come from the 20 val cases only (filter_metrics.py, PLAN_training.md). This script
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
from filter_metrics import VERSION, component_rows, load_image, write_json  # noqa: E402
from orient import spacing_of  # noqa: E402

PROJ = os.environ.get("PROJ", "/work/Research")
CUTS = (0.504, 0.547, 0.56, 0.60, 0.637, 0.65, 0.70)


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
    write_json(cache, {"case": case, "label_path": label_path, "spacing": spacing, "version": VERSION,
                       "liver_ml": round(float((label == 1).sum()) * float(np.prod(spacing)) / 1000, 1),
                       "lesions": rows})
    return case, f"{len(rows)} lesions"


def pct(v, ps=(5, 25, 50, 75, 95)):
    return " / ".join(f"{np.percentile(v, p):.3f}" for p in ps) if len(v) else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=f"{os.environ.get('nnUNet_raw', PROJ + '/nnunet_data/raw')}/Dataset101_LiverSynth")
    ap.add_argument("--out", default=f"{PROJ}/raw_data/lits_lesion_metrics")
    ap.add_argument("--metrics", help="filter_metrics dir of the pool, to compare the synthetic components")
    ap.add_argument("--workers", type=int, default=8, help="~3-4 GB each on long LiTS volumes")
    a = ap.parse_args()

    raw, out = Path(a.raw), Path(a.out)
    split = json.loads((raw / "real_split.json").read_text())
    thesis_test = {f"lits_{i:03d}" for i in split["thesis_test"]}
    jobs = []
    for sub, ids in (("Tr", split["val"]), ("Ts_real", split["test"])):
        for i in ids:
            case = f"lits_{i:03d}"
            jobs.append((case, str(raw / f"labels{sub}" / f"{case}.nii.gz"),
                         str(raw / f"images{sub}" / f"{case}_0000.nii.gz"), str(out)))
    val = {f"lits_{i:03d}" for i in split["val"]}
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

    L = ["# Real LiTS lesions: filter metrics (descriptive only)", "",
         f"Same code as `filter_metrics.py` (version `{VERSION}`), all {len(recs)} LiTS cases: 20 val (the filter "
         "reference) + 111 test. Nothing here sets a filter rule: the cut comes from the val cases only.", "",
         "## Touch and separability per split", "",
         "| split | cases | cases with tumour | lesions | touch liver | scored | sep min | sep p5 / p25 / median / p75 / p95 |",
         "|---|---|---|---|---|---|---|---|"]
    for s in ("val", "test", "all"):
        x = sel(s)
        cs = {r["case"] for r in recs if s == "all" or ("val" if r["case"] in val else "test") == s}
        sc = [l["sep"] for l in x if l.get("sep") is not None]
        L.append(f"| {s} | {len(cs)} | {len({l['case'] for l in x})} | {len(x)} | {sum(l['touches'] for l in x)}/{len(x)} "
                 f"| {len(sc)} | {min(sc):.3f} | {pct(sc)} |")
    nt = [l for l in rows if not l["touches"]]
    L += ["", f"Lesions not touching the liver: {len(nt)}"
          + (": " + ", ".join(f"{l['case']} c{l['component']} ({l['ml']} ml, {l['min_dist_mm']} mm)" for l in nt[:20])
             if nt else "."), ""]

    # weighting: sep by lesion size
    L += ["## Separability by lesion size (all 131)", "", "| size | lesions scored | sep p5 / median | below 0.547 |",
          "|---|---|---|---|"]
    for lo, hi, name in ((0, 0.1, "< 0.1 ml"), (0.1, 1, "0.1-1 ml"), (1, 10, "1-10 ml"), (10, 1e9, ">= 10 ml")):
        sc = [l["sep"] for l in rows if l.get("sep") is not None and lo <= l["ml"] < hi]
        L.append(f"| {name} | {len(sc)} | {pct(sc, (5, 50))} | {sum(v < 0.547 for v in sc)} |")

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
          "Synthetic: components below, and volumes rejectAUC would keep.", "",
          "| cut | real val below | real test below | real cases with one below | synthetic comps below | rejectAUC keeps |",
          "|---|---|---|---|---|---|"]
    for cut in CUTS:
        rv = [l for l in sel("val") if l.get("sep") is not None]
        rt = [l for l in sel("test") if l.get("sep") is not None]
        cases = {l["case"] for l in rows if l.get("sep") is not None and l["sep"] < cut}
        sy = "-"
        if syn is not None:
            bad = [c for c in syn if c.get("sep") is not None and c["sep"] < cut]
            n_vol = len({json.loads(p.read_text())["id"] for p in (Path(a.metrics) / "per_volume").glob("*.json")})
            sy = f"{len(bad)} | {n_vol - len({c['id'] for c in bad})}"
        L.append(f"| {cut} | {sum(l['sep'] < cut for l in rv)}/{len(rv)} | {sum(l['sep'] < cut for l in rt)}/{len(rt)} "
                 f"| {len(cases)}/131 | {sy} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "summary.md", flush=True)


if __name__ == "__main__":
    main()
