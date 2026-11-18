# Test set: true / false positives and negatives per run

238 real MSD07 test cases, checkpoint_final. Sums over all cases. Written by `predict_test.py eval`.

## Tumour, voxel-level (label 2; millions of voxels)

| run | TP | FP | FN | TN | sensitivity | precision | FP / TP |
|---|---|---|---|---|---|---|---|
| D102 f0 all | 0.26 | 0.93 | 1.50 | 5,998.3 | 0.150 | 0.221 | 3.52 |
| D102 f1 all_rep2 | 0.24 | 0.82 | 1.52 | 5,998.4 | 0.136 | 0.225 | 3.44 |

## Pancreas, voxel-level (label 1 only, without tumour; millions of voxels)

| run | TP | FP | FN | TN | sensitivity | precision | FP / TP |
|---|---|---|---|---|---|---|---|
| D102 f0 all | 1.63 | 1.14 | 9.52 | 5,988.7 | 0.146 | 0.587 | 0.70 |
| D102 f1 all_rep2 | 1.30 | 0.83 | 9.85 | 5,989.0 | 0.117 | 0.612 | 0.63 |

## Tumour, lesion-level (26-connected components; no TN at this level)

| run | reference lesions | TP (found) | FN (missed) | FP (predicted pieces on no lesion) | FP per case |
|---|---|---|---|---|---|
| D102 f0 all | 240 | 132 | 108 | 4034 | 16.9 |
| D102 f1 all_rep2 | 240 | 125 | 115 | 1268 | 5.3 |

## Tumour, case-level

| run | TP (tumour case, >= 1 lesion found) | FN (tumour case, none found) | FP (tumour-free case, tumour predicted) | TN (tumour-free case, clean) |
|---|---|---|---|---|
| D102 f0 all | 132 | 106 | 0 | 0 |
| D102 f1 all_rep2 | 125 | 113 | 0 | 0 |

Voxel TN outnumbers everything else ~1000x (specificity > 0.999 for every run), so compare runs on sensitivity, precision and FP / TP.
