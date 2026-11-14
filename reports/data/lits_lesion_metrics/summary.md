# Real LiTS lesions: filter metrics (descriptive only)

Same code as `filter_metrics.py` (version `cc3 touch=adj26 auc: core=erode1 shell=2-10mm organ if contact>=0.5 or organ>=0.25 min=10`), all 131 LiTS cases: 20 val (the filter reference) + 111 test. Nothing here sets a filter rule: the cut comes from the val cases only.

## Touch and separability per split

| split | cases | cases with tumour | lesions | touch liver | scored | sep min | sep p5 / p25 / median / p75 / p95 |
|---|---|---|---|---|---|---|---|
| val | 20 | 17 | 199 | 199/199 | 190 | 0.504 | 0.667 / 0.825 / 0.898 / 0.951 / 0.997 |
| test | 111 | 101 | 709 | 707/709 | 681 | 0.550 | 0.663 / 0.817 / 0.903 / 0.962 / 0.993 |
| all | 131 | 118 | 908 | 906/908 | 871 | 0.504 | 0.663 / 0.818 / 0.900 / 0.961 / 0.993 |

Lesions not touching the liver: 2: lits_027 c6 (0.045 ml, 2.32 mm), lits_079 c2 (0.3391 ml, 4.66 mm)

## Separability by lesion size (all 131)

| size | lesions scored | sep p5 / median | below 0.547 |
|---|---|---|---|
| < 0.1 ml | 116 | 0.752 / 0.955 | 0 |
| 0.1-1 ml | 406 | 0.677 / 0.904 | 1 |
| 1-10 ml | 236 | 0.653 / 0.886 | 0 |
| >= 10 ml | 113 | 0.639 / 0.862 | 0 |

## Least visible lesions: one case or many?

| case | split | lesions scored | min sep | median sep | lesions < 0.60 |
|---|---|---|---|---|---|
| lits_049 | val | 22 | 0.504 | 0.689 | 2 |
| lits_021 | val | 5 | 0.547 | 0.720 | 1 |
| lits_028 | test | 7 | 0.550 | 0.895 | 1 |
| lits_048 | test | 4 | 0.559 | 0.665 | 1 |
| lits_116 | test | 1 | 0.577 | 0.577 | 1 |
| lits_030 | test | 1 | 0.585 | 0.585 | 1 |
| lits_084 | test | 10 | 0.594 | 0.648 | 1 |
| lits_094 | test | 14 | 0.603 | 0.698 | 0 |
| lits_051 | test | 9 | 0.604 | 0.756 | 0 |
| lits_128 | test | 11 | 0.605 | 0.675 | 0 |
| lits_085 | test | 2 | 0.608 | 0.624 | 0 |
| lits_013 | test | 9 | 0.624 | 0.806 | 0 |
| lits_033 | test | 15 | 0.625 | 0.739 | 0 |
| lits_097 | test | 12 | 0.626 | 0.853 | 0 |
| lits_074 | test | 2 | 0.632 | 0.815 | 0 |

## Candidate AUC cuts: real lesions below vs synthetic pool

Real: share of scored lesions below the cut (a filter at this cut would call them 'invisible'). Synthetic: components below, and volumes rejectAUC would keep.

| cut | real val below | real test below | real cases with one below | synthetic comps below | rejectAUC keeps |
|---|---|---|---|---|---|
| 0.504 | 0/190 | 0/681 | 0/131 | 32 | 968 |
| 0.547 | 1/190 | 0/681 | 1/131 | 349 | 732 |
| 0.56 | 3/190 | 2/681 | 4/131 | 431 | 683 |
| 0.6 | 3/190 | 5/681 | 7/131 | 700 | 556 |
| 0.637 | 6/190 | 22/681 | 15/131 | 979 | 460 |
| 0.65 | 9/190 | 27/681 | 17/131 | 1068 | 434 |
| 0.7 | 15/190 | 54/681 | 30/131 | 1409 | 325 |
