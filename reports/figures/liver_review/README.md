# Liver review figures

Made by `maisi/liver_metrics.py` (version `liver2 phase cc26 core=2mm grad_sigma=1mm fat<-30 air<-200 bone>300`). Shared figure style (`maisi/viz.py`): W400/L40, radiological, green liver, red tumour. Numbers: `summary.md` next to the caches on UCloud (`/work/Research/maisi_runs/pool/filter_metrics/liver/liver_check`), copied into NOTES.md.

## 1. Blind review first (`blind/`)

60 pool volumes under codes B01..: 30 flagged by the liver measures (outside the real val range on >= 1 measure; 537 in the pool) and 30 not flagged (of 462), shuffled. Each row: axial slice of the liver's largest cross-section and coronal slice through its centroid, image alone and with the label.

Fill `blind/marks.csv`, one verdict per code: **ok** (the green label is a liver, roughly where the image shows it), **bad** (it covers bowel, fat, other organs or no visible liver; or the anatomy is broken), **unsure**. Then on UCloud: `python maisi/liver_metrics.py --score reports/figures/liver_review/blind/marks.csv`. The key is in `blind_key.json` on UCloud, not in the repo. **Don't open `open/` before marking.**

## 2. Open figures (`open/`)

- `detached_liver_pieces_pool.jpg` / `_real.jpg`: the largest detached piece of liver label per volume (not connected to the main liver, not even through tumour), largest volumes first. Real livers show what annotation specks look like.
- `tumour_on_detached_piece.jpg`: tumour components that touch only a detached piece of liver label (3 components; pass touch v1, fail v2), largest per volume.
- `liver_extremes_*.jpg`: per liver measure and side, the most extreme pool volume beyond the real val range, next to the most extreme real val liver on the same side.
