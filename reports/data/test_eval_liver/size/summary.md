# Liver tumour-size arms (L3, `nnunet/size_eval.py`, 111 LiTS test cases, checkpoint_final)

Exploratory. Comparisons declared in autopilot/MISSION.md L3 (2026-11-26) before any result. Group score per case = mean of its available runs. Variant ccPost = one-structure rule on the prediction.

| group | runs |
|---|---|
| baseline | all (101:0), all_rep2 (101:1) |
| ccAucRepair | ccAucRepair (104:0), ccAucRepair_rep2 (104:1) |
| sizeLarge | ccAucRepair_sizeLarge (104:2) |
| sizeRand | ccAucRepair_sizeRand (104:3) |
| sizeSmall | ccAucRepair_sizeSmall (104:4) |

## 1. Mean per-case tumour Dice (liver Dice, lesion recall, FP pieces / case)

| group / run | variant | tumour Dice | liver Dice | lesion recall | FP pieces / case |
|---|---|---|---|---|---|
| **baseline** (mean of 2) | none | 0.3027 | 0.8728 | 0.432 | 56.5 |
| **baseline** (mean of 2) | ccPost | 0.3810 | 0.8942 | 0.422 | 7.5 |
| all | none | 0.3006 | 0.8702 | 0.422 | 66.8 |
| all | ccPost | 0.3790 | 0.8852 | 0.408 | 7.7 |
| all_rep2 | none | 0.3048 | 0.8755 | 0.443 | 46.3 |
| all_rep2 | ccPost | 0.3831 | 0.9033 | 0.437 | 7.4 |
| **ccAucRepair** (mean of 2) | none | 0.3921 | 0.8715 | 0.413 | 10.8 |
| **ccAucRepair** (mean of 2) | ccPost | 0.4032 | 0.8944 | 0.407 | 7.1 |
| ccAucRepair | none | 0.4028 | 0.8700 | 0.413 | 9.4 |
| ccAucRepair | ccPost | 0.4117 | 0.8924 | 0.410 | 6.3 |
| ccAucRepair_rep2 | none | 0.3814 | 0.8731 | 0.413 | 12.2 |
| ccAucRepair_rep2 | ccPost | 0.3947 | 0.8964 | 0.403 | 7.9 |
| **sizeLarge** (mean of 1) | none | 0.4193 | 0.8725 | 0.430 | 2.8 |
| **sizeLarge** (mean of 1) | ccPost | 0.4420 | 0.8956 | 0.422 | 1.6 |
| **sizeRand** (mean of 1) | none | 0.4256 | 0.8716 | 0.429 | 9.5 |
| **sizeRand** (mean of 1) | ccPost | 0.4384 | 0.8927 | 0.422 | 6.5 |
| **sizeSmall** (mean of 1) | none | 0.1060 | 0.8898 | 0.269 | 8.3 |
| **sizeSmall** (mean of 1) | ccPost | 0.1074 | 0.9113 | 0.261 | 6.8 |

## 2. Declared comparisons (per-case tumour Dice, B − A, 95 % bootstrap CI, Wilcoxon p)

| role | B vs A | runs B / A | variant | A | B | diff [CI] | p | verdict |
|---|---|---|---|---|---|---|---|---|
| primary | sizeLarge vs sizeRand | 1 / 1 | none | 0.4256 | 0.4193 | -0.0063 [-0.0268, +0.0134] | 0.89 | not met |
| primary | sizeLarge vs sizeRand | 1 / 1 | ccPost | 0.4384 | 0.4420 | +0.0035 [-0.0148, +0.0220] | 0.38 | not met |
| secondary | sizeSmall vs sizeRand | 1 / 1 | none | 0.4256 | 0.1060 | -0.3196 [-0.3860, -0.2543] | 4.3e-12 |  |
| secondary | sizeSmall vs sizeRand | 1 / 1 | ccPost | 0.4384 | 0.1074 | -0.3311 [-0.3971, -0.2651] | 9.9e-13 |  |
| secondary | sizeLarge vs ccAucRepair | 1 / 2 | none | 0.3921 | 0.4193 | +0.0272 [+0.0053, +0.0493] | 0.0089 |  |
| secondary | sizeLarge vs ccAucRepair | 1 / 2 | ccPost | 0.4032 | 0.4420 | +0.0387 [+0.0189, +0.0607] | 0.00066 |  |
| secondary | sizeLarge vs baseline | 1 / 2 | none | 0.3027 | 0.4193 | +0.1166 [+0.0861, +0.1490] | 1.3e-11 |  |
| secondary | sizeLarge vs baseline | 1 / 2 | ccPost | 0.3810 | 0.4420 | +0.0609 [+0.0354, +0.0897] | 2.5e-05 |  |
| secondary | sizeRand vs ccAucRepair | 1 / 2 | none | 0.3921 | 0.4256 | +0.0335 [+0.0188, +0.0508] | 3.3e-05 |  |
| secondary | sizeRand vs ccAucRepair | 1 / 2 | ccPost | 0.4032 | 0.4384 | +0.0352 [+0.0208, +0.0522] | 1.6e-06 |  |
| secondary | sizeRand vs baseline | 1 / 2 | none | 0.3027 | 0.4256 | +0.1229 [+0.0942, +0.1531] | 4.1e-12 |  |
| secondary | sizeRand vs baseline | 1 / 2 | ccPost | 0.3810 | 0.4384 | +0.0574 [+0.0374, +0.0794] | 1.7e-06 |  |
| secondary | sizeSmall vs ccAucRepair | 1 / 2 | none | 0.3921 | 0.1060 | -0.2860 [-0.3541, -0.2195] | 7.9e-11 |  |
| secondary | sizeSmall vs ccAucRepair | 1 / 2 | ccPost | 0.4032 | 0.1074 | -0.2959 [-0.3630, -0.2302] | 1.3e-11 |  |
| secondary | sizeSmall vs baseline | 1 / 2 | none | 0.3027 | 0.1060 | -0.1967 [-0.2661, -0.1299] | 1.1e-06 |  |
| secondary | sizeSmall vs baseline | 1 / 2 | ccPost | 0.3810 | 0.1074 | -0.2737 [-0.3419, -0.2066] | 4.5e-10 |  |

Single size runs vs the baseline (descriptive; runs of the same arm varied 0.02-0.04 before):

| run | variant | baseline | run | diff [CI] | p |
|---|---|---|---|---|---|
| ccAucRepair_sizeLarge (104:2) | none | 0.3027 | 0.4193 | +0.1166 [+0.0861, +0.1490] | 1.3e-11 |
| ccAucRepair_sizeLarge (104:2) | ccPost | 0.3810 | 0.4420 | +0.0609 [+0.0354, +0.0897] | 2.5e-05 |
| ccAucRepair_sizeRand (104:3) | none | 0.3027 | 0.4256 | +0.1229 [+0.0942, +0.1531] | 4.1e-12 |
| ccAucRepair_sizeRand (104:3) | ccPost | 0.3810 | 0.4384 | +0.0574 [+0.0374, +0.0794] | 1.7e-06 |
| ccAucRepair_sizeSmall (104:4) | none | 0.3027 | 0.1060 | -0.1967 [-0.2661, -0.1299] | 1.1e-06 |
| ccAucRepair_sizeSmall (104:4) | ccPost | 0.3810 | 0.1074 | -0.2737 [-0.3419, -0.2066] | 4.5e-10 |

## 3. Lesion recall / mean coverage by real lesion size (per group: per-lesion share of its runs)

| group / run | variant | 0-0.1 ml | 0.1-1 ml | 1-10 ml | 10-100 ml | 100+ ml |
|---|---|---|---|---|---|---|
| n lesions | | 113 | 308 | 192 | 74 | 22 |
| **baseline** | none | 0.02 / 0.02 | 0.26 / 0.16 | 0.68 / 0.48 | 0.97 / 0.70 | 0.98 / 0.58 |
| **baseline** | ccPost | 0.02 / 0.02 | 0.25 / 0.15 | 0.67 / 0.48 | 0.96 / 0.70 | 0.91 / 0.58 |
| **ccAucRepair** | none | 0.02 / 0.01 | 0.23 / 0.15 | 0.67 / 0.46 | 0.95 / 0.70 | 0.89 / 0.56 |
| **ccAucRepair** | ccPost | 0.02 / 0.01 | 0.23 / 0.15 | 0.66 / 0.46 | 0.95 / 0.70 | 0.82 / 0.56 |
| **sizeLarge** | none | 0.01 / 0.01 | 0.27 / 0.16 | 0.70 / 0.50 | 0.95 / 0.68 | 0.82 / 0.55 |
| **sizeLarge** | ccPost | 0.01 / 0.01 | 0.25 / 0.16 | 0.69 / 0.50 | 0.95 / 0.68 | 0.82 / 0.55 |
| **sizeRand** | none | 0.02 / 0.01 | 0.26 / 0.18 | 0.67 / 0.47 | 0.97 / 0.71 | 0.91 / 0.56 |
| **sizeRand** | ccPost | 0.02 / 0.01 | 0.25 / 0.18 | 0.67 / 0.47 | 0.97 / 0.71 | 0.82 / 0.56 |
| **sizeSmall** | none | 0.02 / 0.01 | 0.21 / 0.12 | 0.38 / 0.15 | 0.45 / 0.01 | 0.77 / 0.00 |
| **sizeSmall** | ccPost | 0.02 / 0.01 | 0.21 / 0.12 | 0.38 / 0.15 | 0.41 / 0.01 | 0.68 / 0.00 |

Declared (MISSION.md L3, 2026-11-26): "large tumours help" if sizeLarge − sizeRand > +0.01 with the CI above 0, on 2 runs each; everything else is secondary / descriptive.
