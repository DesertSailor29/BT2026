# Test set: true / false positives and negatives per run

111 real LiTS test cases, checkpoint_final. Sums over all cases. Written by `predict_test.py eval`.

## Tumour, voxel-level (label 2; millions of voxels)

| run | TP | FP | FN | TN | sensitivity | precision | FP / TP |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 9.54 | 7.17 | 6.54 | 12,922.2 | 0.593 | 0.571 | 0.75 |
| D101 f1 all_rep2 | 10.30 | 8.52 | 5.77 | 12,920.9 | 0.641 | 0.547 | 0.83 |
| D101 f2 ccReject | 7.34 | 1.28 | 8.74 | 12,928.1 | 0.456 | 0.851 | 0.18 |
| D103 f0 ccRepair | 9.50 | 3.61 | 6.58 | 12,925.8 | 0.591 | 0.725 | 0.38 |
| D103 f1 ccRepairDel | 9.79 | 4.20 | 6.28 | 12,925.2 | 0.609 | 0.700 | 0.43 |
| D104 f0 ccAucRepair | 9.66 | 2.87 | 6.42 | 12,926.5 | 0.601 | 0.771 | 0.30 |

## Liver, voxel-level (label 1 only, without tumour; millions of voxels)

| run | TP | FP | FN | TN | sensitivity | precision | FP / TP |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 243.72 | 25.48 | 40.28 | 12,636.0 | 0.858 | 0.905 | 0.10 |
| D101 f1 all_rep2 | 252.05 | 37.41 | 31.95 | 12,624.0 | 0.888 | 0.871 | 0.15 |
| D101 f2 ccReject | 238.54 | 26.16 | 45.46 | 12,635.3 | 0.840 | 0.901 | 0.11 |
| D103 f0 ccRepair | 242.60 | 27.57 | 41.39 | 12,633.9 | 0.854 | 0.898 | 0.11 |
| D103 f1 ccRepairDel | 236.71 | 27.45 | 47.28 | 12,634.0 | 0.834 | 0.896 | 0.12 |
| D104 f0 ccAucRepair | 246.36 | 31.51 | 37.64 | 12,629.9 | 0.867 | 0.887 | 0.13 |

## Tumour, lesion-level (26-connected components; no TN at this level)

| run | reference lesions | TP (found) | FN (missed) | FP (predicted pieces on no lesion) | FP per case |
|---|---|---|---|---|---|
| D101 f0 all | 709 | 299 | 410 | 7416 | 66.8 |
| D101 f1 all_rep2 | 709 | 314 | 395 | 5137 | 46.3 |
| D101 f2 ccReject | 709 | 251 | 458 | 1052 | 9.5 |
| D103 f0 ccRepair | 709 | 295 | 414 | 1201 | 10.8 |
| D103 f1 ccRepairDel | 709 | 286 | 423 | 1332 | 12.0 |
| D104 f0 ccAucRepair | 709 | 293 | 416 | 1047 | 9.4 |

## Tumour, case-level

| run | TP (tumour case, >= 1 lesion found) | FN (tumour case, none found) | FP (tumour-free case, tumour predicted) | TN (tumour-free case, clean) |
|---|---|---|---|---|
| D101 f0 all | 85 | 16 | 10 | 0 |
| D101 f1 all_rep2 | 87 | 14 | 10 | 0 |
| D101 f2 ccReject | 78 | 23 | 10 | 0 |
| D103 f0 ccRepair | 85 | 16 | 10 | 0 |
| D103 f1 ccRepairDel | 82 | 19 | 10 | 0 |
| D104 f0 ccAucRepair | 86 | 15 | 10 | 0 |

Voxel TN outnumbers everything else ~1000x (specificity > 0.999 for every run), so compare runs on sensitivity, precision and FP / TP.
