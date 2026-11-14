# Liver checks: counts and measures (descriptive only)

`maisi/liver_metrics.py`, version `liver3 border band=6mm shell=1-4mm local=3mm phase cc26 core=2mm grad_sigma=1mm fat<-30 air<-200 bone>300`. Pool: 999 volumes; real LiTS: 20 val (the reference) + 111 test (descriptive only). Nothing here changes a filter decision.

Cross-check with the filter_metrics cache: component numbering, voxels and touch v1 identical.

## 1. Tumour rules: what repair and reject would do

Per volume: *unchanged* = no component fails; *repaired* = some fail, tumour left; *tumour-free* = every component fails (repair leaves no tumour: kept as a tumour-free volume by repair-keep, deleted by repair-delete); *no tumour* = none generated. Reject keeps unchanged + no tumour.

| rule | no tumour | unchanged | repaired | tumour-free after repair | reject keeps | repair-keep keeps | repair-delete keeps |
|---|---|---|---|---|---|---|---|
| touch v1 (any liver voxel; as fixed) | 3 | 398 | 441 | 157 | 401 | 999 | 842 |
| touch v2 (main structure) | 3 | 398 | 441 | 157 | 401 | 999 | 842 |
| AUC sep < 0.504 | 3 | 965 | 29 | 2 | 968 | 999 | 997 |
| AUC sep < 0.547 | 3 | 729 | 244 | 23 | 732 | 999 | 976 |
| touch v2 or AUC < 0.504 | 3 | 393 | 444 | 159 | 396 | 999 | 840 |
| touch v2 or AUC < 0.547 | 3 | 349 | 471 | 176 | 352 | 999 | 823 |

Tumour components: 3873. Touch v1 passes but v2 fails (touches only a detached liver piece): **3 components in 3 volumes** (0.2 ml). In the main structure but not touching the main liver itself (joined via another liver piece / tumour): 1.

Real val lesions in the main structure (pass touch v2): 199/199 (touch v1: 199/199).
Real test lesions in the main structure (pass touch v2): 707/709 (touch v1: 707/709).

## 2. "Keep the main structure" (liver + tumour connected to the main liver; the rest -> background)

| volumes | count |
|---|---|
| unchanged | 343 |
| liver only | 58 |
| tumour only | 477 |
| liver+tumour | 121 |
| tumour-free after this repair (had tumour) | 157 |
| reject (keep only unchanged) keeps | 343 |

## 3. Liver label pieces (the liver is one organ)

| set | volumes | > 1 piece | with detached piece | pieces / volume (max) | detached ml p50 / p95 / max | largest detached piece (ml) | min main share | attached only via tumour |
|---|---|---|---|---|---|---|---|---|
| pool | 999 | 210 | 179 | 11 | 0.008 / 0.596 / 1.991 | 1.99 (liver_0784) | 0.9980 | 71 |
| real val | 20 | 11 | 3 | 98 | 0.002 / 0.005 / 0.005 | 0.00 (lits_018) | 0.9998 | 124 |
| real test | 111 | 62 | 19 | 447 | 0.001 / 0.079 / 0.965 | 0.96 (lits_002) | 0.9974 | 1515 |
- pool volumes with a detached piece >= 0.1 ml: 49; real val 0, real test 1
- pool volumes with a detached piece >= 1 ml: 7; real val 0, real test 0
- pool volumes with a detached piece >= 10 ml: 0; real val 0, real test 0

## 4. Is the liver label liver? Measures vs the 20 real val livers

A pool volume is *outside* if a measure is below the min or above the max of the real val livers (same principle as the AUC cut: unlike every real liver), counted only in the less liver-like direction: hu_median (both), hu_iqr (high), frac_fat (high), frac_air (high), frac_bone (high), border_step_hu (low), edge_offset_mm (abs), edge_peak_ratio (low) (abs = |x| beyond the largest real |x|). MAISI's edge sits exactly on its label and is sharper than on real labels (drawn from the mask), so a low edge offset or a high peak ratio does not flag. 'pool below / above' count every value beyond the range.

| measure | real val min / max | real test p5 / median / p95 | pool p5 / median / p95 | pool below | pool above | real test outside |
|---|---|---|---|---|---|---|
| liver volume (ml, all pieces) | 983 / 2256 | 1168 / 1592 / 2240 | 914 / 1282 / 2092 | 92 | 40 | 6/111 |
| main piece / all liver label | 0.9998 / 1.0000 | 0.9995 / 1.0000 / 1.0000 | 0.9999 / 1.0000 / 1.0000 | 25 | 0 | 8/111 |
| core HU median [flag: both] | 50.0 / 145.0 | 73.5 / 101.0 / 132.5 | 59.0 / 78.0 / 98.0 | 10 | 0 | 3/111 |
| core HU IQR [flag: high] | 16.0 / 45.0 | 18.0 / 30.0 / 48.5 | 19.0 / 23.0 / 39.0 | 0 | 23 | 7/111 |
| core share < -30 HU (fat) [flag: high] | 0.0000 / 0.0012 | 0.0000 / 0.0001 / 0.0026 | 0.0000 / 0.0000 / 0.0030 | 0 | 60 | 16/111 |
| core share < -200 HU (air) [flag: high] | 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0 | 10 | 2/111 |
| core share > 300 HU (bone) [flag: high] | 0.0000 / 0.0007 | 0.0000 / 0.0000 / 0.0002 | 0.0000 / 0.0000 / 0.0000 | 0 | 0 | 2/111 |
| edge gradient median (HU/mm) | 13.8 / 51.9 | 13.1 / 28.3 / 41.8 | 14.8 / 20.9 / 28.9 | 37 | 0 | 8/111 |
| edge / core gradient | 3.358 / 12.011 | 2.958 / 4.498 / 7.154 | 3.696 / 5.492 / 7.290 | 21 | 0 | 20/111 |
| border: median local |HU inside - HU outside| [flag: low] | 56.6 / 182.5 | 62.3 / 107.2 / 148.4 | 48.2 / 72.8 / 102.2 | 138 | 0 | 0/111 |
| border: strongest edge vs label border (mm, + = outside) [flag: abs] | 1.05 / 4.79 | 0.93 / 2.07 / 4.39 | -0.50 / 1.71 / 2.89 | 109 | 0 | 1/111 |
| border: edge profile peak / flank (|d| 4-6 mm) [flag: low] | 1.693 / 3.635 | 1.663 / 2.420 / 3.409 | 1.906 / 3.153 / 6.119 | 15 | 374 | 7/111 |
| liver centroid right of body centroid (mm) | -94.3 / -50.5 | -92.8 / -68.5 / -41.1 | -86.5 / -66.0 / -43.6 | 20 | 163 | 22/111 |
| main liver craniocaudal extent (mm) | 134.4 / 184.0 | 138.9 / 168.7 / 206.2 | 114.0 / 158.0 / 203.0 | 164 | 130 | 30/111 |

**Pool volumes outside on >= 1 flag measure: 214/999** (by number of measures: 1: 179, 2: 29, 3: 5, 4: 1). Real test livers outside on >= 1: 30/111 (how often a real liver would be flagged).

Overlap with the tumour rules (volumes):

| | touch v2 ok | touch v2 fails |
|---|---|---|
| liver ok | 324 | 461 |
| liver flagged | 77 | 137 |

Most extreme pool volumes per flag measure (beyond the real val range):

- hu_median low (10): liver_0406 35, liver_0936 36, liver_0057 42, liver_0392 43, liver_0531 47, liver_0778 47
- hu_iqr high (23): liver_0406 64, liver_0654 59, liver_0573 59, liver_0768 56, liver_0311 54, liver_0936 53
- frac_fat high (60): liver_0406 0.1017, liver_0936 0.0596, liver_0654 0.043, liver_0573 0.0423, liver_0768 0.04, liver_0057 0.0301
- frac_air high (10): liver_0938 0.0004, liver_0810 0.0004, liver_0224 0.0003, liver_0639 0.0002, liver_0889 0.0001, liver_0666 0.0001
- border_step_hu low (138): liver_0761 31.3, liver_0529 35.2, liver_0954 36.2, liver_0355 36.6, liver_0675 37.2, liver_0080 37.4
- edge_peak_ratio low (15): liver_0455 1.346, liver_0873 1.438, liver_0013 1.476, liver_0617 1.478, liver_0212 1.497, liver_0526 1.509

## 5. Contrast phase of the MAISI volumes (pool only)

MAISI has no phase input (we generate with modality 1 = generic CT), so the phase is whatever it draws. Median HU per structure from MAISI's raw 132-class mask (eroded by one voxel). Raw masks measured: 999/999; raw liver / tumour identical to the pair labels: 999/999. Real LiTS has no vessel masks, so the real side needs a segmenter (backlog).

| structure | volumes with the mask | median HU p5 / median / p95 |
|---|---|---|
| aorta | 999 | 59 / 94 / 123 |
| ivc | 999 | 55 / 79 / 110 |
| portal_vein | 999 | 52 / 88 / 131 |
| hepatic_vessel | 0 | - |
| spleen | 998 | 45 / 61 / 75 |
| kidney | 999 | 63 / 91 / 114 |
| liver core (pair label) | 999 | 59 / 78 / 98 |

Portal vein / aorta HU ratio: p5 / median / p95 0.64 / 0.95 / 1.39 (about 1 in portal venous, clearly below 1 in arterial).

Rough phase rule: aorta < 100 HU non-contrast; portal vein < 0.7 x aorta arterial; aorta >= 150 portal venous; else late. Descriptive only; thresholds are textbook values, not fitted.

| phase | volumes | aorta HU p50 | portal vein HU p50 | liver core HU p5 / p50 / p95 | liver flagged (any) | hu_median outside real val |
|---|---|---|---|---|---|---|
| non-contrast | 612 | 84 | 82 | 55 / 74 / 89 | 108 | 10 |
| arterial | 45 | 114 | 71 | 64 / 84 / 99 | 10 | 0 |
| portal venous | 2 | 152 | 154 | 89 / 96 / 103 | 1 | 0 |
| late | 340 | 109 | 102 | 71 / 88 / 102 | 95 | 0 |
