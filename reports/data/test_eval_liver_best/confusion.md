# Test set: true / false positives and negatives per run

111 real LiTS test cases, checkpoint_best. Sums over all cases. Written by `predict_test.py eval`.

## Tumour, voxel-level (label 2; millions of voxels)

| run | TP | FP | FN | TN | sensitivity | precision | FP / TP |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 9.58 | 7.79 | 6.49 | 12,921.6 | 0.596 | 0.552 | 0.81 |
| D101 f1 all_rep2 | 9.85 | 6.26 | 6.22 | 12,923.1 | 0.613 | 0.611 | 0.64 |
| D101 f2 ccReject | 6.92 | 1.80 | 9.16 | 12,927.6 | 0.430 | 0.794 | 0.26 |
| D103 f0 ccRepair | 8.77 | 3.67 | 7.31 | 12,925.7 | 0.545 | 0.705 | 0.42 |
| D103 f1 ccRepairDel | 10.02 | 4.57 | 6.06 | 12,924.8 | 0.623 | 0.687 | 0.46 |
| D104 f0 ccAucRepair | 9.21 | 2.97 | 6.87 | 12,926.4 | 0.573 | 0.756 | 0.32 |
| D104 f1 ccAucRepair_rep2 | 9.86 | 3.83 | 6.22 | 12,925.6 | 0.613 | 0.720 | 0.39 |

## Liver, voxel-level (label 1 only, without tumour; millions of voxels)

| run | TP | FP | FN | TN | sensitivity | precision | FP / TP |
|---|---|---|---|---|---|---|---|
| D101 f0 all | 248.11 | 26.03 | 35.88 | 12,635.4 | 0.874 | 0.905 | 0.10 |
| D101 f1 all_rep2 | 252.00 | 37.54 | 31.99 | 12,623.9 | 0.887 | 0.870 | 0.15 |
| D101 f2 ccReject | 265.47 | 47.42 | 18.52 | 12,614.0 | 0.935 | 0.848 | 0.18 |
| D103 f0 ccRepair | 237.89 | 32.23 | 46.11 | 12,629.2 | 0.838 | 0.881 | 0.14 |
| D103 f1 ccRepairDel | 242.93 | 23.82 | 41.07 | 12,637.6 | 0.855 | 0.911 | 0.10 |
| D104 f0 ccAucRepair | 263.23 | 58.59 | 20.76 | 12,602.9 | 0.927 | 0.818 | 0.22 |
| D104 f1 ccAucRepair_rep2 | 247.85 | 32.67 | 36.14 | 12,628.8 | 0.873 | 0.884 | 0.13 |

## Tumour, lesion-level (26-connected components; no TN at this level)

| run | reference lesions | TP (found) | FN (missed) | FP (predicted pieces on no lesion) | FP per case |
|---|---|---|---|---|---|
| D101 f0 all | 709 | 332 | 377 | 4434 | 39.9 |
| D101 f1 all_rep2 | 709 | 324 | 385 | 6760 | 60.9 |
| D101 f2 ccReject | 709 | 233 | 476 | 2626 | 23.7 |
| D103 f0 ccRepair | 709 | 276 | 433 | 1832 | 16.5 |
| D103 f1 ccRepairDel | 709 | 280 | 429 | 2232 | 20.1 |
| D104 f0 ccAucRepair | 709 | 298 | 411 | 2284 | 20.6 |
| D104 f1 ccAucRepair_rep2 | 709 | 294 | 415 | 1386 | 12.5 |

## Tumour, case-level

| run | TP (tumour case, >= 1 lesion found) | FN (tumour case, none found) | FP (tumour-free case, tumour predicted) | TN (tumour-free case, clean) |
|---|---|---|---|---|
| D101 f0 all | 89 | 12 | 10 | 0 |
| D101 f1 all_rep2 | 87 | 14 | 10 | 0 |
| D101 f2 ccReject | 79 | 22 | 10 | 0 |
| D103 f0 ccRepair | 82 | 19 | 10 | 0 |
| D103 f1 ccRepairDel | 81 | 20 | 10 | 0 |
| D104 f0 ccAucRepair | 87 | 14 | 10 | 0 |
| D104 f1 ccAucRepair_rep2 | 85 | 16 | 10 | 0 |

Voxel TN outnumbers everything else ~1000x (specificity > 0.999 for every run), so compare runs on sensitivity, precision and FP / TP.
