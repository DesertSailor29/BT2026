# What the cc / ccAuc rules do: example figures

Made by `maisi/cc_review_figures.py` from `/work/Research/maisi_runs/pool/filter_metrics/pancreas` (pool `/work/Research/maisi_runs/pool`, 999 pancreas volumes). Rules unchanged from liver (`maisi/cc_rules.py`); AUC cut **0.509** = the least visible real val lesion (msd_025). Shared figure style (`maisi/viz.py`): W400/L40, radiological, green pancreas, red tumour. Each row: image alone, image + label, whole axial + coronal slice with the object circled. Examples: one per volume, ranked by size, 6 at evenly spaced ranks (largest ... smallest), so typical cases show too.

## cc_tumour_far.png

27 components in 27 volumes are outside the main structure and >= 30 mm from the pancreas (38.7 ml in total). cc sets them to background.

## cc_tumour_near.png

48 components in 48 volumes are outside the main structure but closer than 30 mm; 0 of them touch pancreas label, i.e. only a detached piece (they pass the old touch rule). Near misses (not touching, <= 3 mm): 1.

## cc_organ_pieces.png

14/999 volumes have pancreas label outside the main structure (total per volume: median 0.010 ml, p95 0.183, max 0.41; >= 0.1 ml in 1, >= 1 ml in 0). The row shows each volume's largest piece.

## ccauc_invisible.png

30 components in 30 volumes (168.5 ml); ccAuc relabels 27 to pancreas (>= 50% of the outer shell is pancreas) and 3 to background.

## ccauc_borderline.png

The 6 least visible components that pass (one per volume): what 'barely visible' looks like. Scored in-structure components: 921, unscored (pass): 3.

## real_least_visible.png

The 6 least visible of 43 scored real val lesions (43 cases). The first non-excluded one sets the cut; if nothing is visible under its label it may be a label error (real_reference_exclusions.json, with a reason).

## real_cc_changes.png

31/281 real cases: cc would remove pancreas label (detached pieces) in 31 and tumour outside the main structure in 0. Real labels are the reference, so this is a cost of the rule (it would delete real anatomy), not a repair. Largest changes first.
