# Filter metrics: liver (999 synthetic volumes, 20 real val cases)

Rules: PLAN_training.md "Filter rules". A component fails touch if no voxel is 26-adjacent to the organ; it fails auc if its sep < the minimum sep of the real val lesions; both = either. reject = keep volumes without a failing component; repair = relabel the failing components (fails touch -> background, otherwise auc -> surroundings' label).

## Real reference (sanity checks 1 + 2)

Real lesions: 199 (0 excluded by real_reference_exclusions.json), **touching the organ: 199/199**.

Scored real lesions: 190 (unscored, < 10 measurable voxels: 9). sep p5/p25/median/p75/p95: 0.667 / 0.825 / 0.898 / 0.951 / 0.997.

Least visible real lesions (view them; a label error goes into real_reference_exclusions.json with a reason):

| case | component | ml | sep | core HU | shell HU | shell |
|---|---|---|---|---|---|---|
| lits_049 | 13 | 0.1323 | 0.504 | 116.0 | 118.0 | organ |
| lits_021 | 3 | 0.0077 | 0.5475 | 115.0 | 90.0 | organ |
| lits_049 | 3 | 0.2714 | 0.5598 | 113.0 | 120.0 | organ |
| lits_049 | 15 | 0.2782 | 0.6061 | 104.5 | 111.0 | organ |
| lits_049 | 16 | 0.0753 | 0.6214 | 103.0 | 109.0 | organ |

**AUC cut = 0.504** (from lits_049 component 13).

## Decisions per criterion (sanity checks 3 + 4)

| criterion | no tumour | clean (reject keeps) | repaired | tumour-free after repair | reject keeps | repair changes |
|---|---|---|---|---|---|---|
| touch | 3 | 398 | 441 | 157 | 401/999 | 598/999 |
| auc | 3 | 965 | 29 | 2 | 968/999 | 31/999 |
| both | 3 | 393 | 444 | 159 | 396/999 | 603/999 |

touch: 1701/3873 components fail, 1061.9 ml (median 0.02 ml, max 95.4); repair relabels 1701 to background, 0 to organ.

auc: 32/3873 components fail, 62.5 ml (median 0.07 ml, max 52.0); repair relabels 26 to background, 6 to organ.

both: 1717/3873 components fail, 1070.5 ml (median 0.02 ml, max 95.4); repair relabels 1711 to background, 6 to organ.

Near misses (fail touch but within 3 mm of the organ): 59 components in 48 volumes, 1.4 ml. Many of them would mean the no-tolerance assumption is wrong (view them).
Non-touching components: median distance to the organ 162 mm (IQR 92-187).

Volumes failing touch x failing auc:

| | auc ok | auc fails |
|---|---|---|
| touch ok | 396 | 5 |
| touch fails | 572 | 26 |

| sep | n | p5 / p25 / median / p75 / p95 |
|---|---|---|
| real val lesions | 190 | 0.667 / 0.825 / 0.898 / 0.951 / 0.997 |
| synthetic components | 2952 | 0.520 / 0.606 / 0.711 / 0.832 / 0.951 |
| synthetic touching | 1815 | 0.529 / 0.629 / 0.751 / 0.867 / 0.966 |

Unscored synthetic components (pass auc): 921.

## Descriptive

Contact share (organ part of the outer shell), touching components: real 0.770 / 0.903 / 1.000 / 1.000 / 1.000, synthetic 0.124 / 0.356 / 0.500 / 0.756 / 1.000 (p5/p25/median/p75/p95).

Thesis rule (volume-level, no arm) vs rejectTouch:

| | rejectTouch keeps | rejectTouch drops |
|---|---|---|
| thesis pass | 385 | 236 |
| thesis fail | 13 | 362 |
| no tumour | 3 | 0 |

Cross-check thesis fraction vs result.json: 996 volumes, max |diff| 0.0000.
MAISI QC pass 993, fail 6, unknown 0.

## Sanity check 4

- rejectTouch keeps 401 (ok)
- repairTouch changes 598 (ok)
- rejectAUC keeps 968 (ok)
- repairAUC changes 31 (ok)
- rejectBoth keeps 396 (ok)
- repairBoth changes 603 (ok)
