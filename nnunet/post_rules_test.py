#!/usr/bin/env python
"""A3 (autopilot/MISSION.md): the frozen post-hoc rules on the pancreas predictions, one evaluation.

The rules, controls and parameters below were frozen in notes/AUTOPILOT.md (s6, 2026-11-24 20:30) after the two
val screens of post_rules.py (whose functions this script reuses, unchanged) and before any test run of them. The
test result is evaluation, not selection: nothing here may be changed after reading it.

Frozen rules (Holm over these 4, main metric per-case tumour Dice; definitions in post_rules.py):
  pos+minLes          position prior (spine-relative, from the real val labels), then minLes (1.19 ml)
  grow0.1+minLes      grow the gland at p(pancreas)+p(tumour) >= 0.1 (10 mm, 26-connected, -50..300 HU), minLes
  grow0.1+tumNear10   grow0.1, then drop tumour pieces > 10 mm from predicted pancreas
  grow0.1+pos+minLes  grow0.1, position prior, minLes
Controls and components (reported, not in the Holm family):
  minLes, pos, grow0.1, tumNear10                          the parts of the frozen combinations
  thr0.1, thr0.1+minLes, thr0.1+tumNear10, thr0.1+pos+minLes   the same with a plain threshold instead of grow
                                                           (no distance / connectivity / HU limit): is the
                                                           anatomical part of grow needed?
  open1, rand_minLes, rand_pos+minLes                      generic opening; random tumour-piece removal matched
                                                           in count to minLes / pos+minLes (seeded per case)
Declared contrasts (secondary): each grow0.1+X vs thr0.1+X (tumour and pancreas Dice); pancreas Dice of every rule.
Parameters: min lesion 1.19 ml and the position prior come from the 43 real val labels only
(reports/data/post_rules/pancreas_val/val_label_params.json, pos bins cache). On val the prior is leave-one-out
(reproduces the screen); on test it is the union of all 43 val labels.

  python nnunet/post_rules_test.py --split val  --workers 12          # sanity: reproduces the screen numbers
  python nnunet/post_rules_test.py --split test --workers 12 --no-probs   # the 2 rules without probabilities now
  python nnunet/post_rules_test.py --split test --workers 12          # all, needs fold_X/test_probs/*.npz
-> reports/data/post_rules/pancreas_<split>_frozen/{summary.md, per_case.csv}. Exploratory.
"""
import argparse
import csv
import json
import os
import sys
import zlib
from multiprocessing import Pool

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import post_rules as pr  # noqa: E402
import predict_test as pt  # noqa: E402
import fp_location as fl  # noqa: E402

VERSION = "frozen1"
FROZEN = ("pos+minLes", "grow0.1+minLes", "grow0.1+tumNear10", "grow0.1+pos+minLes")
CONTROLS = ("minLes", "pos", "grow0.1", "tumNear10", "thr0.1", "thr0.1+minLes", "thr0.1+tumNear10",
            "thr0.1+pos+minLes", "open1", "rand_minLes", "rand_pos+minLes")
PROB = {"grow0.1+minLes", "grow0.1+tumNear10", "grow0.1+pos+minLes", "grow0.1", "thr0.1", "thr0.1+minLes",
        "thr0.1+tumNear10", "thr0.1+pos+minLes"}
CONTRASTS = (("grow0.1+minLes", "thr0.1+minLes"), ("grow0.1+tumNear10", "thr0.1+tumNear10"),
             ("grow0.1+pos+minLes", "thr0.1+pos+minLes"), ("grow0.1", "thr0.1"))
OUT = f"{os.environ['nnUNet_results']}/test_eval_pancreas/post_rules_frozen"


def apply_frozen(p, img, prob, spacing, aff, support, min_ml, seed):
    v = {"none": p}
    c = pr.spine_centre(img, spacing, aff)
    ml = lambda x: pr.min_les(x, spacing, min_ml)  # noqa: E731
    pos = lambda x: pr.position(x, c, spacing, aff, support)  # noqa: E731
    near = lambda x: pr.tum_near(x, spacing, 10.0)[0]  # noqa: E731
    v["minLes"], v["pos"], v["tumNear10"] = ml(p), pos(p), near(p)
    v["pos+minLes"] = ml(v["pos"])
    if prob is not None:
        for base, x in (("grow0.1", pr.grow(p, img, prob, spacing, 0.1)), ("thr0.1", pr.threshold(prob, 0.1))):
            v[base] = x
            v[f"{base}+minLes"] = ml(x)
            v[f"{base}+tumNear10"] = near(x)
            v[f"{base}+pos+minLes"] = ml(pos(x))
    n0 = pr.n_tumour_pieces(p)
    for r in ("minLes", "pos+minLes"):
        v[f"rand_{r}"] = pr.rand_remove(p, n0 - pr.n_tumour_pieces(v[r]), seed)
    v["open1"] = pr.opening(p)
    return v


def measure(args):
    run, case, img_f, ref_f, pred_f, prob_f, cache_f, min_ml, support = args
    pnii = nib.load(pred_f)
    p = np.asarray(pnii.dataobj).astype(np.uint8)
    spacing = pr.spacing_of(pnii.affine)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    img = nib.load(img_f).get_fdata(dtype=np.float32)
    assert img.shape == p.shape, (img_f, img.shape, p.shape)
    prob, agree = None, None
    if prob_f:
        prob = pr.load_probs(prob_f, p.shape)
        agree = float((prob.argmax(0) == p).mean())
        assert agree > 0.995, (prob_f, "probabilities do not match the saved prediction", agree)
    seed = zlib.crc32(f"{run}/{case}".encode())
    variants = apply_frozen(p, img, prob, spacing, pnii.affine, support, min_ml, seed)
    del img, prob
    ref = nib.load(ref_f)  # the reference is read only after every rule has been applied
    r = np.asarray(ref.dataobj).astype(np.uint8)
    assert r.shape == p.shape and np.allclose(ref.affine, pnii.affine, atol=1e-3), ref_f
    out = {"version": VERSION, "run": run, "case": case, "voxel_ml": voxel_ml, "prob_agree": agree,
           "metrics": {k: {**fl.metrics(r, x, voxel_ml), "changed_vox": int((x != p).sum())}
                       for k, x in variants.items()}}
    tmp = cache_f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, cache_f)
    return run, case


def paths(split, run, case):
    did, fold = map(int, run.split(":"))
    fdir = f"{pt.res_dir(did)}/fold_{fold}"
    if split == "val":
        return (f"{pt.RAW}/imagesTr/{case}_0000.nii.gz", f"{pt.RAW}/labelsTr/{case}.nii.gz",
                f"{fdir}/validation/{case}.nii.gz", f"{fdir}/val_probs/{case}.npz")
    return (f"{pt.IMG}/{case}_0000.nii.gz", f"{pt.LAB}/{case}.nii.gz", f"{fdir}/{pt.PRED_SUB}/{case}.nii.gz",
            f"{fdir}/test_probs/{case}.npz")


def tables(split, runs, cases, rules, pub):
    data = {(run, c): json.load(open(f"{OUT}/{split}/{run.replace(':', '_')}/{c}.json")) for run in runs for c in cases}
    assert all(d["version"] == VERSION for d in data.values())
    os.makedirs(pub, exist_ok=True)
    keys = list(data[(runs[0], cases[0])]["metrics"]["none"].keys())
    with open(f"{pub}/per_case.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "case", "rule"] + keys)
        for (run, c), d in data.items():
            for r in ["none"] + rules:
                w.writerow([run, c, r] + [d["metrics"][r][k] for k in keys])
    names = {r: f"D{r.split(':')[0]} f{r.split(':')[1]} {pt.arm_name(*map(int, r.split(':')))}" for r in runs}
    names["mean"] = "mean of the 2 runs"

    def col(rule, key, run):
        if run == "mean":
            return np.nanmean([[data[(r, c)]["metrics"][rule][key] for c in cases] for r in runs], axis=0)
        return np.array([data[(run, c)]["metrics"][rule][key] for c in cases], float)

    res = {}
    for rule in rules:
        for run in list(runs) + ["mean"]:
            res[(rule, run)] = (pt.paired(col("none", "tumour_dice", run), col(rule, "tumour_dice", run)),
                                pt.paired(col("none", f"{pr.TARGET}_dice", run), col(rule, f"{pr.TARGET}_dice", run)))
    fam = [r for r in FROZEN if r in rules]
    holm = {}
    for run in list(runs) + ["mean"]:
        ps = [res[(r, run)][0]["p"] for r in fam]
        for r, a in zip(fam, pt.holm(np.nan_to_num(ps, nan=1.0)) if fam else []):
            holm[(r, run)] = a
    lines = [f"# A3: frozen post-hoc rules, pancreas {split} ({len(cases)} cases)", "",
             "Rules and controls frozen in `notes/AUTOPILOT.md` (s6) before this run; definitions `nnunet/post_rules.py`,"
             " `nnunet/post_rules_test.py`. Each rule vs the saved prediction (`none`), per-case paired difference "
             "(bootstrap 95 % CI, Wilcoxon p, Holm over the frozen rules present, per run and for the per-case mean "
             "of the two runs). Exploratory.", "",
             "| rule | kind | run | tumour Dice | vs none [95% CI] | p | Holm p | pancreas Dice | vs none [95% CI] | "
             "lesion recall | FP pieces / case |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in list(runs) + ["mean"]:
        st, sg = res[(rules[0], run)]
        if run != "mean":
            m = [data[(run, c)]["metrics"]["none"] for c in cases]
            rec = sum(x["les_det"] for x in m) / max(1, sum(x["les_ref"] for x in m))
            fp = np.mean([x["les_fp"] for x in m])
            lines.append(f"| none | – | {names[run]} | {st['x']:.3f} | – | – | – | {sg['x']:.3f} | – | {rec:.3f} | {fp:.1f} |")
        else:
            lines.append(f"| none | – | {names[run]} | {st['x']:.3f} | – | – | – | {sg['x']:.3f} | – | – | – |")
    for rule in rules:
        kind = "frozen" if rule in FROZEN else "control"
        for run in list(runs) + ["mean"]:
            st, sg = res[(rule, run)]
            hp = f"{holm[(rule, run)]:.2g}" if (rule, run) in holm else ""
            if run != "mean":
                m = [data[(run, c)]["metrics"][rule] for c in cases]
                ref = sum(x["les_ref"] for x in m)
                rec = f"{sum(x['les_det'] for x in m) / ref:.3f}" if ref else "–"
                fp = f"{np.mean([x['les_fp'] for x in m]):.1f}"
            else:
                rec = fp = "–"
            lines.append(f"| {rule} | {kind} | {names[run]} | {st['y']:.3f} | {pt.fmt_diff(st, 3)} | {st['p']:.2g} | "
                         f"{hp} | {sg['y']:.3f} | {pt.fmt_diff(sg, 3)} | {rec} | {fp} |")
    con = [(a, b) for a, b in CONTRASTS if a in rules and b in rules]
    if con:
        lines += ["", "Declared contrasts: grow (anatomical limits) vs the same with a plain threshold "
                  "(grow − thr, per case).", "", "| contrast | run | tumour Dice diff [95% CI] | p | "
                  "pancreas Dice diff [95% CI] | p |", "|---|---|---|---|---|---|"]
        for a, b in con:
            for run in list(runs) + ["mean"]:
                st = pt.paired(col(b, "tumour_dice", run), col(a, "tumour_dice", run))
                sg = pt.paired(col(b, f"{pr.TARGET}_dice", run), col(a, f"{pr.TARGET}_dice", run))
                lines.append(f"| {a} − {b} | {names[run]} | {pt.fmt_diff(st, 3)} | {st['p']:.2g} | "
                             f"{pt.fmt_diff(sg, 3)} | {sg['p']:.2g} |")
    ag = [d["prob_agree"] for d in data.values() if d["prob_agree"] is not None]
    if ag:
        lines += ["", f"Probabilities vs saved label maps: argmax agreement min {min(ag):.4f} over {len(ag)} cases."]
    open(f"{pub}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--runs", nargs="+", default=pr.RUNS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="+", help="subset (fake test)")
    ap.add_argument("--no-probs", action="store_true", help="only the rules without probabilities")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    pt.set_target(pr.TARGET)
    fl.ORG = pr.TARGET
    if a.cases:
        cases = a.cases
    elif a.split == "val":
        cases = [f"msd_{i:03d}" for i in pt.split()["val"]]
    else:
        cases = sorted(f[: -len("_0000.nii.gz")] for f in os.listdir(pt.IMG) if f.endswith("_0000.nii.gz"))
    rules = [r for r in FROZEN + CONTROLS if not (a.no_probs and r in PROB)]
    pub = f"{pr.REPO}/reports/data/post_rules/pancreas_{a.split}_frozen" + ("_noprobs" if a.no_probs else "")
    if not a.tables_only:
        params = json.load(open(f"{pr.PUB}/val_label_params.json"))
        bins = np.load(f"{pr.OUT}/pos_bins.npz")
        bin_sets = {c: bins[c] for c in bins.files}
        assert len(bin_sets) == 43 and abs(params["min_lesion_ml"] - 1.19) < 0.01, (len(bin_sets), params)
        full = pr.prior_support(list(bin_sets.values()))
        jobs = []
        for run in a.runs:
            cdir = f"{OUT}/{a.split}/{run.replace(':', '_')}"
            os.makedirs(cdir, exist_ok=True)
            for c in cases:
                img_f, ref_f, pred_f, prob_f = paths(a.split, run, c)
                prob_f = None if a.no_probs else prob_f
                if prob_f and not os.path.exists(prob_f):
                    sys.exit(f"missing {prob_f} (predict the probabilities first, or --no-probs)")
                cache = f"{cdir}/{c}.json"
                if os.path.exists(cache):
                    d = json.load(open(cache))
                    if d.get("version") == VERSION and all(r in d["metrics"] for r in rules):
                        continue
                if a.split == "test":
                    assert c not in bin_sets, c
                support = pr.prior_support([b for k, b in bin_sets.items() if k != c]) if c in bin_sets else full
                jobs.append((run, c, img_f, ref_f, pred_f, prob_f, cache, params["min_lesion_ml"], support))
        print(f"{len(jobs)} (run, case) pairs to measure, {a.workers} workers", flush=True)
        with Pool(a.workers) as pool:
            for k, (run, c) in enumerate(pool.imap_unordered(measure, jobs, chunksize=1), 1):
                print(f"  {k}/{len(jobs)} {run} {c}", flush=True)
    tables(a.split, a.runs, cases, rules, pub)


if __name__ == "__main__":
    main()
