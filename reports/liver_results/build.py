#!/usr/bin/env python
"""Build the liver test-results report as PDF: charts (make_charts.py) + CT crops + report.html -> Chromium PDF.

Needs matplotlib, pandas, Pillow and a Chromium on PATH; the page loads Inter (Google Fonts) and Mermaid
(jsDelivr) at print time, so it needs internet.

  python reports/liver_results/build.py "<out.pdf>" [--build <dir>]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIG = f"{REPO}/reports/figures/filter_review"
CCP = f"{REPO}/reports/figures/ccpost_review"
PER_CASE = f"{REPO}/reports/data/test_eval_liver/fp_location/per_case.csv"
RUN_ID = {"all": "101:0", "all_rep2": "101:1", "ccReject": "101:2", "ccRepair": "103:0", "ccRepairDel": "103:1",
          "ccAucRepair": "104:0"}

# CT strips for report.html (<!--ct:name-->): panels cut out of nnunet/ccpost_figures.py's figures.
# A-D figures: 2 rows (0 = full slice, 1 = zoom) x 4 columns (0 CT, 1 reference, 2 as saved, 3 after ccPost).
# E figures: 6 rows (all, all_rep2, ccReject, ccRepair, ccRepairDel, ccAucRepair) x the same 4 columns.
# Label tokens {arm:variant} become "Dice x · n FP · lesions a/b" for that run and case (fp_location per_case.csv).
E_CROP = (0.04, 0.10, 0.74, 0.80)  # liver region of an E panel (fractions of the panel)
STRIPS = {
    "caught": [
        ("A1 · lits_115 · <i>all</i>: 32 ml false-positive tumour on a mass in the right chest", "A1_caught_lits_115_all.png",
         [(0, 2, "all as saved, full slice"), (1, 0, "CT, zoom"), (1, 2, "as saved · {all:none}"),
          (1, 3, "after ccPost · {all:ccPost}")]),
        ("A5 · lits_091 · <i>all</i>: 16 ml tumour on stool in the colon of a tumour-free case", "A5_caught_lits_091_all.png",
         [(0, 2, "all as saved, full slice"), (1, 0, "CT, zoom"), (1, 2, "as saved · {all:none}"),
          (1, 3, "after ccPost · {all:ccPost}")]),
    ],
    "harm": [
        ("A3 · lits_043 · <i>all_rep2</i>: the liver is called tumour, the largest 'liver' piece is the spleen",
         "A3_caught_lits_043_all_rep2.png",
         [(0, 0, "CT"), (0, 1, "reference"), (0, 2, "as saved · {all_rep2:none}"),
          (0, 3, "after ccPost · {all_rep2:ccPost}")]),
        ("B1 · lits_130 · <i>all_rep2</i>: a correctly found 25 ml tumour beyond a gap in the predicted liver",
         "B1_harm_lits_130_all_rep2.png",
         [(0, 1, "reference, full slice"), (1, 1, "reference, zoom"), (1, 2, "as saved · {all_rep2:none}"),
          (1, 3, "after ccPost · {all_rep2:ccPost}")]),
    ],
    "kept": [
        ("C2 · lits_120 · ccAucRepair: 118 ml pleural effusion called tumour, touching the predicted liver",
         "C2_kept_lits_120_ccAucRepair.png",
         [(0, 2, "as saved, full slice"), (1, 0, "CT, zoom"), (1, 1, "reference"),
          (1, 3, "after ccPost · {ccAucRepair:ccPost}")]),
        ("C1 · lits_020 · <i>all</i>: 149 ml of a heterogeneous right lobe called tumour (reference: liver)",
         "C1_kept_lits_020_all.png",
         [(0, 1, "reference, full slice"), (1, 0, "CT, zoom"), (1, 1, "reference"),
          (1, 3, "after ccPost · {all:ccPost}")]),
    ],
    "lits088": [
        ("lits_088 · large tumour at the liver hilum, every run as saved (cropped to the liver)",
         "E2_runs_lits_088.png",
         [((0, 1), "reference"), ((0, 2), "all · {all:none}"), ((2, 2), "ccReject · {ccReject:none}"),
          ((3, 2), "ccRepair · {ccRepair:none}"), ((5, 2), "ccAucRepair · {ccAucRepair:none}")]),
    ],
    "missed": [
        ("D4 · lits_074 · 125 ml isodense lesion: both <i>all</i> runs find it, ccAucRepair does not",
         "D4_missed_lits_074_ccAucRepair.png",
         [(0, 1, "reference, full slice"), (1, 0, "CT, zoom"), (1, 1, "reference"),
          (1, 2, "ccAucRepair · {ccAucRepair:none}")]),
        ("D5 · lits_046 · 9 ml hypodense lesion: ccAucRepair finds it, both <i>all</i> runs leave a hole",
         "D5_missed_lits_046_all.png",
         [(0, 1, "reference, full slice"), (1, 0, "CT, zoom"), (1, 1, "reference"), (1, 2, "all · {all:none}")]),
    ],
}


def run_stats():
    import csv
    st = {}
    for r in csv.DictReader(open(PER_CASE)):
        st[(r["run"], r["case"], r["variant"])] = (f"Dice {float(r['tumour_dice']):.2f} · {r['les_fp']} FP · "
                                                   f"lesions {r['les_det']}/{r['les_ref']}")
    return st


def grid(path):
    """Panel boxes (x0, y0, x1, y1) of a ccpost_figures.py figure, keyed (row, col)."""
    import numpy as np
    a = np.asarray(Image.open(path).convert("L")) < 200

    def runs(b, minlen=100):
        out, s0 = [], None
        for i, v in enumerate(list(b) + [False]):
            if v and s0 is None:
                s0 = i
            if not v and s0 is not None:
                if i - s0 >= minlen:
                    out.append((s0, i))
                s0 = None
        return out
    cols = runs(a.sum(0) > a.shape[0] * 0.25)
    rows = runs(a.sum(1) > a.shape[1] * 0.25)
    if len(rows) == 1:  # A-D: full slice and zoom rows touch; same layout in every figure (split 316 of 642 px)
        y0, y1 = rows[0]
        cut = y0 + round((y1 - y0) * 316 / 642)
        rows = [(y0, cut), (cut, y1)]
    assert len(cols) == 4, (path, cols)
    return {(i, j): (c[0], r[0], c[1], r[1]) for i, r in enumerate(rows) for j, c in enumerate(cols)}


def ct_strips(b):
    st = run_stats()
    html = {}
    for name, rows in STRIPS.items():
        parts = []
        ncol = max(len(p) for _, _, p in rows)
        for k, (head, fn, picks) in enumerate(rows):
            case = re.search(r"lits_\d+", fn).group(0)
            g = grid(f"{CCP}/{fn}")
            im = Image.open(f"{CCP}/{fn}").convert("RGB")
            parts.append(f'<div class="hd">{head}</div>')
            for j, pick in enumerate(picks):
                if isinstance(pick[0], tuple):  # E figure: ((row, col), label), cropped to the liver
                    (r, c), lab = pick
                    x0, y0, x1, y1 = g[(r, c)]
                    w, h = x1 - x0, y1 - y0
                    box = (x0 + int(E_CROP[0] * w), y0 + int(E_CROP[1] * h), x0 + int(E_CROP[2] * w),
                           y0 + int(E_CROP[3] * h))
                else:
                    r, c, lab = pick
                    x0, y0, x1, y1 = g[(r, c)]
                    box = (x0 + 1, y0 + 1, x1 - 1, y1 - 1)
                out = f"ct_{name}_{k}_{j}.jpg"
                im.crop(box).save(f"{b}/{out}", quality=92)
                lab = re.sub(r"\{(\w+):(\w+)\}", lambda m: st[(RUN_ID[m.group(1)], case, m.group(2))], lab)
                parts.append(f'<div class="p"><div class="lb">{lab}</div><img src="{out}" alt="{lab}"></div>')
        html[name] = f'<div class="ct" style="--n:{ncol}">' + "".join(parts) + "</div>"
    return html


def crop_rows(src, rows, out):
    """Rows of a review_figures.py sheet (title + 4 panels each), stacked."""
    im = Image.open(src).convert("RGB")
    k = im.width / 1534  # sheet geometry measured at 1534 px width
    parts = [im.crop((int(30 * k), int((146 + 367.6 * i) * k), int(1475 * k), int((510 + 367.6 * i) * k)))
             for i in rows]
    gap = 12
    o = Image.new("RGB", (parts[0].width, sum(p.height for p in parts) + gap * (len(parts) - 1)), "white")
    y = 0
    for p in parts:
        o.paste(p, (0, y))
        y += p.height + gap
    o.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--build", default=f"{HERE}/build")
    a = ap.parse_args()
    b = a.build
    os.makedirs(f"{b}/style", exist_ok=True)
    subprocess.run([sys.executable, f"{HERE}/make_charts.py", b], check=True)
    crop_rows(f"{FIG}/maisi_off_liver.png", [0, 4], f"{b}/off_liver.jpg")       # liver_0048, liver_0883
    crop_rows(f"{FIG}/maisi_invisible_label.png", [0, 3], f"{b}/invisible.jpg")  # liver_0735, liver_0305
    for f in ("report.css", "diagrams.js"):
        shutil.copy(f"{REPO}/reports/style/{f}", f"{b}/style/{f}")

    html = open(f"{HERE}/report.html").read()

    def svg(m):
        s = open(f"{b}/{m.group(1)}.svg").read()
        return s[s.index("<svg"):]
    html = re.sub(r"<!--svg:(\w+)-->", svg, html)
    strips = ct_strips(b)
    html = re.sub(r"<!--ct:(\w+)-->", lambda m: strips[m.group(1)], html)
    page = f"{b}/report.html"
    open(page, "w").write(html)

    chrome = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    # file:// pages may import local ES modules (style/diagrams.js) only with file access allowed
    flags = [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
             "--virtual-time-budget=30000", "--run-all-compositor-stages-before-draw"]
    dom = subprocess.run(flags + ["--dump-dom", page], check=True, capture_output=True, text=True).stdout
    if 'data-ready="1"' not in dom:
        sys.exit("diagrams did not render (no data-ready on <body>); check Mermaid syntax / network")
    subprocess.run(flags + ["--no-pdf-header-footer", f"--print-to-pdf={os.path.abspath(a.pdf)}", page],
                   check=True, capture_output=True)
    print("wrote", a.pdf)


if __name__ == "__main__":
    main()
