#!/usr/bin/env python
"""Liver L1c (autopilot/MISSION.md "Liver extension", optional; backlog "Contrast phase" (a)): contrast phase of the
111 LiTS test cases and tumour Dice per phase. CPU. DESCRIPTIVE ONLY: nothing is chosen from it. Written in s46
(2026-11-28), before any phase result.

Real LiTS has no vessel labels, so the aorta and the portal + splenic vein come from TotalSegmentator (`total`,
`--roi_subset aorta portal_vein_and_splenic_vein`, run separately in $PROJ/venv-ts into
$PROJ/totalseg_lits_test/<case>/{aorta,portal_vein_and_splenic_vein}.nii.gz). Its liver output is never used.
Per case: median HU of each vessel mask eroded by one voxel (the mask itself if the core has < 20 voxels), as
maisi/liver_metrics.py vessel_hu does for the MAISI pool, then the same rough phase rule (liver_metrics.phase_of:
aorta < 100 HU non-contrast; portal vein < 0.7 x aorta arterial; aorta >= 150 portal venous; else late).
Tumour Dice per phase from the L3 eval's per-case table (reports/data/test_eval_liver/size/cases.csv: baseline,
ccAucRepair and the size arms, per-case mean of each group's runs), saved and ccPost.

  python nnunet/phase_eval.py                       (all 111; cases without masks are listed, not guessed)
  python nnunet/phase_eval.py --cases lits_000 --out /tmp/x   (fake test)
-> $nnUNet_results/test_eval_liver/phase/{phase.csv, summary.md}, copied to reports/data/test_eval_liver/phase/.
"""
import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "maisi"))
import predict_test as pt  # noqa: E402  (paths, test split)
from liver_metrics import (PHASE_ART_RATIO, PHASE_NONCON_AORTA, PHASE_PV_AORTA,  # noqa: E402
                           phase_of)

TS_DIR = f"{os.environ['PROJ']}/totalseg_lits_test"
VESSELS = {"aorta": "aorta", "portal_vein": "portal_vein_and_splenic_vein"}
OUT = f"{os.environ['nnUNet_results']}/test_eval_liver/phase"
PUB = f"{REPO}/reports/data/test_eval_liver/phase"
SIZE_CSV = f"{REPO}/reports/data/test_eval_liver/size/cases.csv"
PHASES = ["non-contrast", "arterial", "portal venous", "late", None]


def vessel_hu(case):
    """{'aorta': {'hu', 'vox'}, 'portal_vein': {...}} or {'error': ...}."""
    img_nii = nib.load(f"{pt.IMG}/{case}_0000.nii.gz")
    img = np.asarray(img_nii.dataobj, dtype=np.float32)
    out = {}
    for key, fname in VESSELS.items():
        f = f"{TS_DIR}/{case}/{fname}.nii.gz"
        if not os.path.exists(f):
            return {"error": f"missing {f}"}
        m = np.asarray(nib.load(f).dataobj) > 0
        if m.shape != img.shape:
            return {"error": f"{fname} {m.shape} vs image {img.shape}"}
        core = ndimage.binary_erosion(m, border_value=0)
        v = img[core if core.sum() >= 20 else m]
        out[key] = {"hu": round(float(np.median(v)), 1) if v.size else None, "vox": int(v.size)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    cases = a.cases or [f"lits_{i:03d}" for i in pt.split()["test"]]
    os.makedirs(a.out, exist_ok=True)

    rows, missing = [], []
    for c in cases:
        vs = vessel_hu(c)
        if "error" in vs:
            missing.append(f"{c}: {vs['error']}")
            continue
        rows.append({"case": c, "aorta_hu": vs["aorta"]["hu"], "aorta_vox": vs["aorta"]["vox"],
                     "pv_hu": vs["portal_vein"]["hu"], "pv_vox": vs["portal_vein"]["vox"], "phase": phase_of(vs)})
    phase = {r["case"]: r["phase"] for r in rows}

    # per-case group scores (mean over each group's runs) from the L3 table
    sc = defaultdict(lambda: defaultdict(list))  # sc[(group, var)][case] -> run values
    tum_ml = {}
    with open(SIZE_CSV) as fh:
        for r in csv.DictReader(fh):
            tum_ml[r["case"]] = float(r["case_tumour_ml"])
            for var in ("none", "ccPost"):
                v = r[f"tumour_dice_{var}"]
                sc[(r["group"], var)][r["case"]].append(float(v) if v not in ("", "nan") else np.nan)
    groups = list(dict.fromkeys(g for g, _ in sc))
    for r in rows:
        r["tumour_ml"] = tum_ml.get(r["case"])
        for g in groups:
            for var in ("none", "ccPost"):
                vals = [x for x in sc[(g, var)].get(r["case"], []) if not np.isnan(x)]
                r[f"{g}_{var}"] = round(float(np.mean(vals)), 4) if vals else None
    if rows:
        with open(f"{a.out}/phase.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def q(xs):
        xs = [x for x in xs if x is not None]
        return " / ".join(f"{v:.0f}" for v in np.percentile(xs, [5, 50, 95])) if xs else "-"

    L = [f"# Contrast phase of the LiTS test cases (L1c, `nnunet/phase_eval.py`, {len(rows)}/{len(cases)} cases)", "",
         "Descriptive, exploratory; nothing is chosen from it. Vessel masks: TotalSegmentator `total` (aorta, portal + "
         "splenic vein; its liver output unused), median HU of the mask eroded by one voxel. Rough phase rule as for "
         f"the MAISI pool (`maisi/liver_metrics.py`): aorta < {PHASE_NONCON_AORTA} HU non-contrast; portal vein < "
         f"{PHASE_ART_RATIO} x aorta arterial; aorta >= {PHASE_PV_AORTA} portal venous; else late. Compare with the "
         "pool's phase mix in `reports/data/liver_check/summary.md` section 5 (pool aorta median 94 HU).", "",
         f"Aorta HU p5 / median / p95: {q([r['aorta_hu'] for r in rows])}; portal vein: "
         f"{q([r['pv_hu'] for r in rows])}.", ""]
    if missing:
        L += [f"Cases without vessel masks ({len(missing)}): " + "; ".join(missing), ""]
    L += ["## Tumour Dice per phase (per-case mean of each group's runs; tumour-free cases score as in L3)", "",
          "| phase | cases | median tumour ml | " + " | ".join(f"{g} saved / ccPost" for g in groups) + " |",
          "|---|---|---|" + "---|" * len(groups)]
    for p in PHASES:
        sub = [r for r in rows if r["phase"] == p]
        if not sub:
            continue
        cells = []
        for g in groups:
            m = [np.nanmean([r[f"{g}_{v}"] if r[f"{g}_{v}"] is not None else np.nan for r in sub])
                 for v in ("none", "ccPost")]
            cells.append(f"{m[0]:.3f} / {m[1]:.3f}")
        ml = [r["tumour_ml"] for r in sub if r["tumour_ml"] is not None]
        L.append(f"| {p or 'undetermined'} | {len(sub)} | {np.median(ml) if ml else float('nan'):.1f} | "
                 + " | ".join(cells) + " |")
    L += ["", "Small phase groups: read as a description, not as a test. Tumour burden differs between phases "
          "(column 3) and drives Dice (L1), so a phase difference here is not a phase effect."]
    with open(f"{a.out}/summary.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    if a.out == OUT:
        os.makedirs(PUB, exist_ok=True)
        for f in ("phase.csv", "summary.md"):
            if os.path.exists(f"{a.out}/{f}"):
                shutil.copy(f"{a.out}/{f}", f"{PUB}/{f}")


if __name__ == "__main__":
    main()
