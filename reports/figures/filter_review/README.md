# Filter review figures

Made by `maisi/review_figures.py` from `/work/Research/maisi_runs/pool/filter_metrics/liver` (pool `/work/Research/maisi_runs/pool`). Shared figure style (`maisi/viz.py`): W400/L40, radiological, green liver, red tumour. Each example: image alone, then image + label. sep = separability of tumour core vs surrounding HU (0.5 = invisible, 1 = fully separable); real val lesions: median 0.90, synthetic components: median 0.71.

## real_invisible_lesions.png

The 9 least visible real validation lesions (sep < 0.65; cases: lits_021, lits_040, lits_049). If nothing is visible under the label, the lesion is a label error and can go into `maisi/real_reference_exclusions.json` (it sets the AUC cut).

## maisi_off_liver.png

6 examples (largest component per volume) of tumour components >= 30 mm from the liver; 534/999 volumes have at least one. LiTS labels only liver tumours, so these are wrong labels (rule touch: rejectTouch drops the volume, repairTouch sets them to background).

## maisi_invisible_label.png

6 examples of tumour components with sep < 0.547 (the least visible real lesion outside lits_049) that sit at the liver (touch it, contact >= 0.5); 75/999 volumes have one at the liver, 269/999 anywhere (many big off-liver blobs in maisi_off_liver.png are invisible too). The label says tumour, the image shows tissue like its surroundings (rule AUC).

## maisi_stray_island.png

3 examples (largest first, one per volume) of the 3/90 touching components with contact < 0.1 whose only liver contact is a small liver-label island separate from the main liver. They pass the touch rule although they sit outside the liver: a loophole of the rule, kept as found (rules were fixed in advance).
