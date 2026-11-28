# Contrast phase of the LiTS test cases (L1c, `nnunet/phase_eval.py`, 111/111 cases)

Descriptive, exploratory; nothing is chosen from it. Vessel masks: TotalSegmentator `total` (aorta, portal + splenic vein; its liver output unused), median HU of the mask eroded by one voxel. Rough phase rule as for the MAISI pool (`maisi/liver_metrics.py`): aorta < 100 HU non-contrast; portal vein < 0.7 x aorta arterial; aorta >= 150 portal venous; else late. Compare with the pool's phase mix in `reports/data/liver_check/summary.md` section 5 (pool aorta median 94 HU).

Aorta HU p5 / median / p95: 106 / 143 / 206; portal vein: 121 / 156 / 224.

## Tumour Dice per phase (per-case mean of each group's runs; tumour-free cases score as in L3)

| phase | cases | median tumour ml | baseline saved / ccPost | ccAucRepair saved / ccPost | sizeLarge saved / ccPost | sizeRand saved / ccPost | sizeSmall saved / ccPost |
|---|---|---|---|---|---|---|---|
| arterial | 3 | 1.1 | 0.166 / 0.175 | 0.171 / 0.172 | 0.166 / 0.167 | 0.253 / 0.254 | 0.019 / 0.019 |
| portal venous | 45 | 22.4 | 0.332 / 0.383 | 0.392 / 0.396 | 0.426 / 0.432 | 0.412 / 0.416 | 0.068 / 0.068 |
| late | 63 | 7.8 | 0.288 / 0.389 | 0.403 / 0.419 | 0.430 / 0.453 | 0.417 / 0.434 | 0.138 / 0.140 |

Small phase groups: read as a description, not as a test. Tumour burden differs between phases (column 3) and drives Dice (L1), so a phase difference here is not a phase effect.
