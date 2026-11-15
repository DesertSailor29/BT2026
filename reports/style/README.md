# One style for charts, diagrams and report pages

CT images have their own rule (`maisi/viz.py`: W400/L40, green organ / red tumour, radiological). Everything else
uses this folder, so a chart, a diagram and a CT figure look like parts of the same document.

| file | what | how |
|---|---|---|
| `style.py` | colour tokens, the fixed arm colours, matplotlib settings | `style.use()`, then plot; `style.save_svg()` keeps text as text |
| `diagrams.js` | Mermaid theme + shared node classes | `<script type="text/x-mermaid">` blocks, then `renderDiagrams()` |
| `report.css` | A4 report page: type, stat tiles, boxes, tables, figures, CT strips (`.ct`) | `<link rel="stylesheet" href="style/report.css">` |

**Colours.** Ink `#1f1f1e`, secondary `#5f5e58`, muted `#8a8985`, grid `#e6e5e0`, panel `#f6f5f2`, white surface.
Anatomy colours are the CT overlay colours: organ green `#00d900`, tumour red `#ff1a1a` (darker `#138a13` /
`#c81414` for strokes and text, tints `#e3f7e3` / `#fde4e4` for filled shapes). Each arm keeps one colour
everywhere: all `#2a78d6`, ccReject `#eb6834`, ccRepair `#1baf7a`, ccRepairDel `#eda100`, ccAucRepair `#e87ba4`
(the validated default categorical palette in order; adjacent pairs pass the colour-blind check). A repeat run
(`_rep2`) uses its arm's colour, dashed or hollow. Follow-up arms take `#008300`, `#6250d6`, `#e34948` in order.

**Charts.** Matplotlib, saved as SVG and inlined in the page, so the text uses the page font (Inter). Thin marks,
recessive grid, no top/right spines, direct labels. Arms on labelled rows (dot plots) rather than colour-only
scatters: four or more colours on one scatter can't be told apart reliably.

**Diagrams.** Mermaid 11.4.1 flowcharts. Node classes: `data`, `step`, `decision`, `muted`, `organ`, `tumour`,
`result`, and one per arm (`all`, `ccReject`, ...). Never colour a node inline; add a class here instead.

**PDF.** Headless Chromium prints the page (see `reports/liver_results/build.py`); it needs internet for the font
and Mermaid. Local ES modules from `file://` need `--allow-file-access-from-files`.
