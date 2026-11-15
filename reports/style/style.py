"""One visual style for every chart, diagram and report page (CT images keep their own rule: maisi/viz.py).

Everything a figure needs comes from here, so charts, diagrams and CT figures read as one system:

  * tokens   INK / INK2 / MUTED / GRID / SURFACE / PANEL: text, axes and backgrounds (same values as
             reports/style/report.css and reports/style/diagrams.js)
  * anatomy  ORGAN green and TUMOUR red, the exact colours of maisi/viz.py's CT overlays, plus light tints
             for filled shapes in diagrams (ORGAN_TINT, TUMOUR_TINT)
  * arms     ARM_COLOR: one fixed colour per training arm (the validated default categorical palette,
             slots 1-5 in order; adjacent pairs pass the CVD check). A repeat run (`<arm>_rep2`) reuses its
             arm's colour with a dashed line. Never re-colour an arm in another chart.
  * charts   use() sets matplotlib rcParams (thin marks, recessive grid, no top/right spines);
             save_svg() keeps text as text so the report page renders it in the page font (Inter).

    import sys; sys.path.insert(0, "reports/style"); import style
    style.use()
    fig, ax = plt.subplots(figsize=style.size(1.0, 0.45))
    ...
    style.save_svg(fig, "out.svg")

Only five categorical colours exist for arms. With more than three series in a scatter (all pairs on screen),
don't rely on colour: put arms on labelled rows (dot plot) or facet instead.
"""
import logging

INK, INK2, MUTED, GRID, SURFACE, PANEL = "#1f1f1e", "#5f5e58", "#8a8985", "#e6e5e0", "#ffffff", "#f6f5f2"
FONT = ["Inter", "DejaVu Sans"]  # Inter in the report page; DejaVu Sans where Inter is missing (PNG export)

ORGAN, TUMOUR = "#00d900", "#ff1a1a"  # = maisi/viz.py COLORS (0, 0.85, 0) and (1, 0.1, 0.1)
ORGAN_DARK, TUMOUR_DARK = "#138a13", "#c81414"  # strokes / text on light backgrounds
ORGAN_TINT, TUMOUR_TINT = "#e3f7e3", "#fde4e4"  # fills in diagrams

ARM_COLOR = {"all": "#2a78d6", "ccReject": "#eb6834", "ccRepair": "#1baf7a", "ccRepairDel": "#eda100",
             "ccAucRepair": "#e87ba4"}
ARM_EXTRA = ["#008300", "#6250d6", "#e34948"]  # follow-up arms, in order of first appearance
ARM_ORDER = ["all", "ccReject", "ccRepair", "ccRepairDel", "ccAucRepair"]

TEXT_WIDTH_IN = 6.7  # A4 text width of report.css (210 mm - 2 x 18 mm margins)


def arm_color(arm):
    base = arm.split("_rep")[0]
    return ARM_COLOR.get(base, ARM_EXTRA[0])


def size(width_frac=1.0, aspect=0.5):
    """Figure size in inches for a fraction of the report text width and a height/width ratio."""
    w = TEXT_WIDTH_IN * width_frac
    return w, w * aspect


def use():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 8.5,
        "axes.titlesize": 9, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlecolor": INK,
        "axes.labelsize": 8.5, "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8.5,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 6,
        "lines.linewidth": 1.6, "lines.markersize": 6, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 100,
        "svg.fonttype": "none",  # text stays text: the page's font renders it
    })


def save_svg(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.04)
    import matplotlib.pyplot as plt
    plt.close(fig)
