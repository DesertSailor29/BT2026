#!/usr/bin/env python
"""Review figures for the cc rule applied to the test predictions (ccPost, nnunet/fp_location.py): what it catches,
what it wrongly removes, what it can't reach, and the lesions every run misses. Exploratory, for the report.

Step 1 (scan, parallel, cached): per (run, case) every reference lesion (26-connected) with detected / detected after
ccPost, and the largest predicted liver piece ccPost removes (with its share inside the reference liver).
Step 2 (select): edge cases chosen by fixed rules from the scan + fp_location's components.csv / per_case.csv
(largest of a kind, one per case; deterministic, not hand-picked).
Step 3 (draw, parallel): one PNG per example, shared CT style (maisi/viz.py):
    top row    full axial slice: CT | reference | prediction as saved | prediction after ccPost
    bottom row the same four, zoomed on the piece of interest (dashed white ring in the top row)
  slice = where the piece of interest is largest. Plus runs_<case>.png: one case, every run, saved vs ccPost.

Output: reports/figures/ccpost_review/ (PNGs + README.md with each example's numbers).

  python nnunet/ccpost_figures.py --workers 30
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "maisi"))
import predict_test as pt  # noqa: E402
from cc_rules import main_structure  # noqa: E402
from orient import reorient_array, spacing_of  # noqa: E402
from viz import CAPTION, draw, legend_handles, plane  # noqa: E402

VERSION = "ccfig1"
RUNS = ["101:0", "101:1", "101:2", "103:0", "103:1", "104:0"]
FPDIR = f"{os.environ['nnUNet_results']}/test_eval_liver/fp_location"
SCAN = f"{FPDIR}/review_scan"
OUT = f"{REPO}/reports/figures/ccpost_review"
S26 = pt.STRUCT26
ZOOM_MM = 35.0


def names():
    return {r: pt.arm_name(*map(int, r.split(":"))) for r in RUNS}


def pred_path(run):
    did, fold = map(int, run.split(":"))
    return f"{pt.res_dir(did)}/fold_{fold}/test_final"


def load(run, case):
    ref = nib.load(f"{pt.LAB}/{case}.nii.gz")
    r = np.asarray(ref.dataobj).astype(np.uint8)
    p = np.asarray(nib.load(f"{pred_path(run)}/{case}.nii.gz").dataobj).astype(np.uint8)
    return ref.affine, r, p


# ---------------------------------------------------------------- step 1: scan
def scan(args):
    run, case, out_f = args
    aff, r, p = load(run, case)
    voxel_ml = float(np.prod(spacing_of(aff))) / 1000.0
    keep = main_structure(p)
    post = np.where(keep, p, 0)
    lab, n = ndimage.label(r == 2, S26)
    les = []
    if n:
        idx = range(1, n + 1)
        vox = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        dn = ndimage.maximum(p == 2, lab, idx)
        dp = ndimage.maximum(post == 2, lab, idx)
        les = [{"id": i + 1, "ml": round(float(vox[i]) * voxel_ml, 4), "det": bool(dn[i]), "det_post": bool(dp[i])}
               for i in range(n)]
    removed = (p == 1) & ~keep
    liv = None
    if removed.any():
        ll, m = ndimage.label(removed, S26)
        sz = np.bincount(ll.ravel(), minlength=m + 1)
        sz[0] = 0
        k = int(np.argmax(sz))
        liv = {"id": k, "ml": round(float(sz[k]) * voxel_ml, 3),
               "ref_liver_share": round(float((r[ll == k] > 0).mean()), 3),
               "total_ml": round(float(removed.sum()) * voxel_ml, 3)}
    out = {"version": VERSION, "run": run, "case": case, "lesions": les, "removed_liver": liv}
    json.dump(out, open(out_f + ".tmp", "w"))
    os.replace(out_f + ".tmp", out_f)
    return run, case


# ---------------------------------------------------------------- step 2: select
def select(cases, nm):
    comps = list(csv.DictReader(open(f"{FPDIR}/components.csv")))
    for c in comps:
        c["ml"] = float(c["ml"])
        c["hit"] = c["hit"] == "True"
        c["in_pred_main"] = c["in_pred_main"] == "True"
        c["ref_organ_share"] = float(c["ref_organ_share"])
        c["component"] = int(c["component"])
    pc = {}
    for row in csv.DictReader(open(f"{FPDIR}/per_case.csv")):
        pc[(row["run"], row["case"], row["variant"])] = row
    sc = {(run, c): json.load(open(f"{SCAN}/{run.replace(':', '_')}/{c}.json")) for run in RUNS for c in cases}
    free = {c for c in cases if not sc[(RUNS[0], c)]["lesions"]}
    used = set()  # (case) already shown, so each example shows a different case where possible
    ex = []

    def add(cat, key, run, case, kind, comp, why, ml):
        used.add(case)
        ex.append({"cat": cat, "key": key, "run": run, "case": case, "kind": kind, "comp": comp, "why": why, "ml": ml,
                   "arm": nm[run]})

    def best(pool, sort_key):
        pool = [x for x in pool if x["case"] not in used] or pool
        return max(pool, key=sort_key) if pool else None

    def pick_comp(cat, key, run_filter, cond, why, sort_key=lambda x: x["ml"]):
        x = best([c for c in comps if run_filter(c["run"]) and cond(c)], sort_key)
        if x:
            add(cat, key, x["run"], x["case"], "pred", x["component"], why, x["ml"])

    is_ = lambda run: (lambda r: r == run)  # noqa: E731
    anyrun = lambda r: True  # noqa: E731
    fp_out = lambda c: not c["hit"] and not c["in_pred_main"]  # noqa: E731

    # A. caught: false-positive tumour outside the predicted main structure, removed by ccPost
    pick_comp("caught", "A1", is_("101:0"), lambda c: fp_out(c) and c["ref_organ_share"] < 0.5,
              "largest FP piece of `all` outside the predicted liver (and outside the real liver)")
    cnt = defaultdict(int)
    for c in comps:
        if c["run"] == "101:0" and fp_out(c):
            cnt[c["case"]] += 1
    shower = max((k for k in cnt if k not in used), key=lambda k: cnt[k])
    pick_comp("caught", "A2", is_("101:0"), lambda c: c["case"] == shower and fp_out(c),
              f"`all`'s case with the most FP pieces outside the predicted liver ({cnt[shower]} pieces; largest shown)")
    pick_comp("caught", "A3", anyrun, lambda c: fp_out(c) and c["ref_organ_share"] >= 0.5,
              "largest FP piece outside the predicted main structure but inside the real liver "
              "(predicted liver is broken into pieces)")
    pick_comp("caught", "A4", is_("104:0"), fp_out, "largest FP piece of ccAucRepair outside the predicted liver")
    pick_comp("caught", "A5", is_("101:0"), lambda c: fp_out(c) and c["case"] in free,
              "largest outside-FP piece of `all` in a tumour-free test case")

    # B. harm: ccPost removes something real
    pick_comp("harm", "B1", anyrun, lambda c: c["hit"] and not c["in_pred_main"],
              "largest correctly found tumour piece that ccPost removes (lies outside the predicted main structure)")
    pick_comp("harm", "B2", anyrun, lambda c: c["hit"] and not c["in_pred_main"],
              "second-largest correctly found tumour piece removed by ccPost (another case)")
    liv = [{"run": k[0], "case": k[1], **v["removed_liver"]} for k, v in sc.items() if v["removed_liver"]]
    x = best([v for v in liv if v["ref_liver_share"] >= 0.5], lambda v: v["ml"])
    if x:
        add("harm", "B3", x["run"], x["case"], "liver", x["id"],
            f"largest predicted liver piece removed by ccPost that lies in the real liver "
            f"({100 * x['ref_liver_share']:.0f} % inside)", x["ml"])
    drop = []
    for run in RUNS:
        for c in cases:
            a, b = pc[(run, c, "none")]["tumour_dice"], pc[(run, c, "ccPost")]["tumour_dice"]
            if a not in ("", "nan") and b not in ("", "nan"):
                drop.append((float(b) - float(a), run, c))
    d, run, c = min(x for x in drop if x[2] not in used)
    hits = [k for k in comps if k["run"] == run and k["case"] == c and not k["in_pred_main"]]
    if hits:
        h = max(hits, key=lambda k: (k["hit"], k["ml"]))
        add("harm", "B4", run, c, "pred", h["component"],
            f"largest tumour Dice loss under ccPost ({d:+.3f}); the removed piece is shown", h["ml"])

    # C. not reached: false-positive tumour inside the predicted main structure, kept by ccPost
    fp_in = lambda c: not c["hit"] and c["in_pred_main"]  # noqa: E731
    pick_comp("kept", "C1", is_("101:0"), fp_in, "largest FP piece of `all` inside the predicted liver (kept)")
    pick_comp("kept", "C2", is_("104:0"), fp_in, "largest FP piece of ccAucRepair inside the predicted liver (kept)")
    pick_comp("kept", "C3", is_("104:0"), lambda c: fp_in(c) and c["case"] in free,
              "largest kept FP piece of ccAucRepair in a tumour-free case")
    pick_comp("kept", "C4", anyrun, lambda c: fp_in(c) and c["ref_organ_share"] < 0.5,
              "largest kept FP piece that lies outside the real liver (the predicted liver reaches beyond it)")

    # D. missed: reference lesions without any predicted tumour (ccPost can't change these)
    def missed(run):
        return [{"run": run, "case": c, **l} for c in cases for l in sc[(run, c)]["lesions"] if not l["det"]]

    for key, run in (("D1", "101:0"), ("D2", "104:0")):
        x = best(missed(run), lambda v: v["ml"])
        add("missed", key, run, x["case"], "ref", x["id"], f"largest real lesion `{nm[run]}` misses", x["ml"])
    det = defaultdict(dict)
    for run in RUNS:
        for c in cases:
            for l in sc[(run, c)]["lesions"]:
                det[(c, l["id"])][run] = (l["det"], l["ml"])
    allm = [{"case": k[0], "id": k[1], "ml": v["101:0"][1]} for k, v in det.items() if not any(d[0] for d in v.values())]
    x = best(allm, lambda v: v["ml"])
    add("missed", "D3", "104:0", x["case"], "ref", x["id"], "largest real lesion that every run misses", x["ml"])
    x = best([{"case": k[0], "id": k[1], "ml": v["101:0"][1]} for k, v in det.items()
              if v["101:0"][0] and v["101:1"][0] and not v["104:0"][0]], lambda v: v["ml"])
    if x:
        add("missed", "D4", "104:0", x["case"], "ref", x["id"],
            "largest real lesion both `all` runs find but ccAucRepair misses", x["ml"])
    x = best([{"case": k[0], "id": k[1], "ml": v["101:0"][1]} for k, v in det.items()
              if not v["101:0"][0] and not v["101:1"][0] and v["104:0"][0]], lambda v: v["ml"])
    if x:
        add("missed", "D5", "101:0", x["case"], "ref", x["id"],
            "largest real lesion ccAucRepair finds but both `all` runs miss (shown: `all`)", x["ml"])

    # E. one case across all runs: biggest ccPost gain for `all`, and the case where the runs differ most
    gain = [(float(pc[("101:0", c, "ccPost")]["tumour_dice"]) - float(pc[("101:0", c, "none")]["tumour_dice"]), c)
            for c in cases if pc[("101:0", c, "none")]["tumour_dice"] not in ("", "nan")
            and pc[("101:0", c, "ccPost")]["tumour_dice"] not in ("", "nan")]
    g, gc = max(gain)
    spread = []
    for c in cases:
        v = [pc[(r, c, "none")]["tumour_dice"] for r in RUNS]
        if all(s not in ("", "nan") for s in v):
            v = list(map(float, v))
            spread.append((max(v) - min(v), c))
    s, scase = max(x for x in spread if x[1] != gc)
    runs_ex = [{"cat": "runs", "key": "E1", "case": gc, "why": f"largest tumour Dice gain of `all` under ccPost ({g:+.3f})"},
               {"cat": "runs", "key": "E2", "case": scase, "why": f"largest tumour Dice spread between the runs ({s:.3f})"}]
    return ex, runs_ex, pc


# ---------------------------------------------------------------- step 3: draw
def ras(arr, aff):
    return reorient_array(arr, aff)[0]


def target_mask(kind, comp, r, p, keep):
    if kind == "pred":
        lab, _ = ndimage.label(p == 2, S26)
    elif kind == "ref":
        lab, _ = ndimage.label(r == 2, S26)
    else:
        lab, _ = ndimage.label((p == 1) & ~keep, S26)
    return lab == comp


def dice_txt(row):
    d = row["tumour_dice"]
    d = "–" if d in ("", "nan") else f"{float(d):.2f}"
    return f"tumour Dice {d} · FP pieces {row['les_fp']} · lesions {row['les_det']}/{row['les_ref']}"


def ring(ax, cy, cx, r_mm, sp):
    ax.add_patch(Ellipse((cx, cy), 2 * r_mm / sp[0], 2 * r_mm / sp[1], fill=False, ec="white", ls="--", lw=1.0))


def crop(a, cy, cx, hy, hx):
    y0, x0 = max(int(cy - hy), 0), max(int(cx - hx), 0)
    return a[y0:int(cy + hy) + 1, x0:int(cx + hx) + 1]


def render(job):
    e, pcs, fname = job
    img_n = nib.load(f"{pt.IMG}/{e['case']}_0000.nii.gz")
    aff = img_n.affine
    img = np.asarray(img_n.dataobj).astype(np.int16)
    _, r, p = load(e["run"], e["case"])
    keep = main_structure(p)
    post = np.where(keep, p, 0).astype(np.uint8)
    m = target_mask(e["kind"], e["comp"], r, p, keep)
    img, r, p, post, m = (ras(a, aff) for a in (img, r, p, post, m))
    sp = spacing_of(reorient_array(np.zeros((1, 1, 1)), aff)[1])  # RAS spacing
    k = int(np.argmax(m.sum(axis=(0, 1))))
    panels = []
    for lab in (None, r, p, post):
        i2, l2, asp = plane(img, lab, sp, "axial", k)
        panels.append((i2, l2))
    _, m2, asp = plane(img, m.astype(np.uint8), sp, "axial", k)
    ys, xs = np.nonzero(m2)
    cy, cx = ys.mean(), xs.mean()
    ext = max((ys.max() - ys.min() + 1) * sp[1], (xs.max() - xs.min() + 1) * sp[0]) / 2
    half = max(ZOOM_MM, ext + 15)
    hy, hx = half / sp[1], half / sp[0]

    fig, axs = plt.subplots(2, 4, figsize=(15, 8.4), gridspec_kw={"height_ratios": [1, 1]})
    tit = ["CT", "reference", f"{e['arm']} as saved\n{dice_txt(pcs[0])}", f"{e['arm']} after ccPost\n{dice_txt(pcs[1])}"]
    for j, (i2, l2) in enumerate(panels):
        draw(axs[0, j], i2, l2, asp)
        ring(axs[0, j], cy, cx, ext + 8, sp)
        axs[0, j].set_title(tit[j], fontsize=9)
        draw(axs[1, j], crop(i2, cy, cx, hy, hx), None if l2 is None else crop(l2, cy, cx, hy, hx), asp)
        axs[1, j].set_title(f"zoom ±{half:.0f} mm", fontsize=8)
    what = {"pred": "predicted tumour piece", "ref": "reference lesion", "liver": "removed predicted liver piece"}[e["kind"]]
    fig.suptitle(f"{e['key']} · {e['case']} · axial slice {k} (RAS) · {e['arm']} · ring: {what}, {e['ml']:.2f} ml\n"
                 f"{e['why']}", fontsize=11)
    fig.legend(handles=legend_handles("liver", "tumour") + [Patch(fc="none", ec="black", ls="--", label="piece of interest")],
               loc="lower center", bbox_to_anchor=(0.5, 0.022), ncol=3, fontsize=9, frameon=False)
    fig.text(0.5, 0.004, CAPTION + " ccPost = keep only the predicted liver ∪ tumour piece holding the largest predicted "
             "liver piece.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    return fname, k


def render_runs(job):
    e, pcs, nm, fname = job
    img_n = nib.load(f"{pt.IMG}/{e['case']}_0000.nii.gz")
    aff = img_n.affine
    img = ras(np.asarray(img_n.dataobj).astype(np.int16), aff)
    sp = spacing_of(reorient_array(np.zeros((1, 1, 1)), aff)[1])
    preds = {}
    for run in RUNS:
        _, r, p = load(run, e["case"])
        keep = main_structure(p)
        preds[run] = (ras(p, aff), ras(np.where(keep, p, 0).astype(np.uint8), aff))
    rr = ras(r, aff)
    # slice: most reference tumour, else most predicted tumour of `all`
    src = rr == 2 if (rr == 2).any() else preds["101:0"][0] == 2
    k = int(np.argmax(src.sum(axis=(0, 1))))
    fig, axs = plt.subplots(len(RUNS), 4, figsize=(13, 3.4 * len(RUNS)))
    for i, run in enumerate(RUNS):
        for j, lab in enumerate((None, rr, preds[run][0], preds[run][1])):
            i2, l2, asp = plane(img, lab, sp, "axial", k)
            draw(axs[i, j], i2, l2, asp)
        axs[i, 0].set_ylabel(nm[run], fontsize=10)
        axs[i, 0].set_title("CT", fontsize=8)
        axs[i, 1].set_title("reference", fontsize=8)
        axs[i, 2].set_title(f"as saved\n{dice_txt(pcs[run][0])}", fontsize=8)
        axs[i, 3].set_title(f"after ccPost\n{dice_txt(pcs[run][1])}", fontsize=8)
    fig.suptitle(f"{e['key']} · {e['case']} · axial slice {k} (RAS) · every run\n{e['why']} "
                 "(Dice/FP/lesions are whole-case numbers)", fontsize=11)
    fig.legend(handles=legend_handles("liver", "tumour"), loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=2,
               fontsize=9, frameon=False)
    fig.text(0.5, 0.002, CAPTION, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(fname, dpi=90)
    plt.close(fig)
    return fname, k


CATS = {"caught": "A. ccPost catches: false-positive tumour outside the predicted liver, removed",
        "harm": "B. ccPost harms: something real removed",
        "kept": "C. ccPost can't reach: false-positive tumour inside the predicted liver, kept",
        "missed": "D. Missed lesions (false negatives; ccPost can't add tumour)",
        "runs": "E. One case, every run: saved vs ccPost"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    cases = [f"lits_{i:03d}" for i in pt.split()["test"]]
    nm = names()
    jobs = []
    for run in RUNS:
        d = f"{SCAN}/{run.replace(':', '_')}"
        os.makedirs(d, exist_ok=True)
        for c in cases:
            f = f"{d}/{c}.json"
            if os.path.exists(f) and json.load(open(f)).get("version") == VERSION:
                continue
            jobs.append((run, c, f))
    print(f"scan: {len(jobs)} of {len(RUNS) * len(cases)} to do, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for i, (run, c) in enumerate(pool.imap_unordered(scan, jobs), 1):
            if i % 50 == 0 or i == len(jobs):
                print(f"  scan {i}/{len(jobs)} (last {run} {c})", flush=True)

    ex, runs_ex, pc = select(cases, nm)
    os.makedirs(OUT, exist_ok=True)
    rj = []
    for e in ex:
        e["file"] = f"{e['key']}_{e['cat']}_{e['case']}_{e['arm']}.png"
        rj.append((e, (pc[(e["run"], e["case"], "none")], pc[(e["run"], e["case"], "ccPost")]), f"{OUT}/{e['file']}"))
    uj = []
    for e in runs_ex:
        e["file"] = f"{e['key']}_runs_{e['case']}.png"
        pcs = {r: (pc[(r, e["case"], "none")], pc[(r, e["case"], "ccPost")]) for r in RUNS}
        uj.append((e, pcs, nm, f"{OUT}/{e['file']}"))
    print(f"draw: {len(rj) + len(uj)} figures", flush=True)
    with Pool(min(a.workers, len(rj) + len(uj))) as pool:
        r1 = pool.map_async(render, rj, chunksize=1)
        r2 = pool.map_async(render_runs, uj, chunksize=1)
        slices = dict(r1.get() + r2.get())
    for e in ex + runs_ex:
        e["slice"] = slices[f"{OUT}/{e['file']}"]
        print(f"  {e['file']}", flush=True)

    L = ["# ccPost review: what the cc rule on the predictions catches, harms, can't reach, and missed lesions", "",
         "Written by `nnunet/ccpost_figures.py` from the checkpoint_final test predictions (111 LiTS) and "
         "`reports/data/test_eval_liver/fp_location/`. **ccPost** = the training-label cc rule applied to a prediction: "
         "keep the piece of predicted liver ∪ tumour that holds the largest predicted liver piece, everything else → "
         "background. Examples are picked by fixed rules (largest of each kind, a new case per example where possible), "
         "not by eye. Each figure: top = full axial slice (CT | reference | as saved | after ccPost), the dashed ring "
         "marks the piece of interest; bottom = the same zoomed. Panel titles give whole-case tumour Dice, FP pieces "
         "and lesions found. Style: " + CAPTION, ""]
    for cat, head in CATS.items():
        L += [f"## {head}", ""]
        for e in [x for x in ex + runs_ex if x["cat"] == cat]:
            if cat == "runs":
                L.append(f"- **{e['key']}** `{e['file']}`: {e['case']}, slice {e['slice']}: {e['why']}.")
            else:
                L.append(f"- **{e['key']}** `{e['file']}`: {e['case']}, {e['arm']}, {e['ml']:.2f} ml, slice "
                         f"{e['slice']}: {e['why']}.")
        L.append("")
    rd = f"{OUT}/READING.md"  # by-eye notes on the examples, written by hand; appended when present
    if os.path.exists(rd):
        L += [open(rd).read()]
    open(f"{OUT}/README.md", "w").write("\n".join(L))
    json.dump(ex + runs_ex, open(f"{OUT}/examples.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
